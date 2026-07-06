import contextlib
import logging
import os

from transformers import (
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    CohereAsrForConditionalGeneration,
    VoxtralForConditionalGeneration,
)


@contextlib.contextmanager
def suppress_output():
    """Context manager to temporarily suppress stdout and stderr."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)


# Suppress OneLogger/NeMo output at runtime (model loading)
@contextlib.contextmanager
def suppress_nemo():
    """Temporarily disable NeMo's OneLogger (bypasses Python logging)."""
    if os.environ.get("FASTER_WHISPER_HOTKEY_DEBUG", "0") == "1":
        yield
        return

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)

        # Also patch Python-level logger functions
        patched: list[tuple] = []
        try:
            from nemo.utils import logging as nemo_logging
        except ImportError:
            nemo_logging = None

        if nemo_logging is not None:
            for attr in ("log_info", "log_warn", "log_error", "log_debug",
                         "info", "warn", "error", "debug"):
                orig = getattr(nemo_logging, attr, None)
                if orig is not None and callable(orig):
                    setattr(nemo_logging, attr, lambda *a, **k: None)
                    patched.append((nemo_logging, attr, orig))

        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        for obj, attr, orig in patched:
            setattr(obj, attr, orig)


# Suppress OneLogger/NeMo initialization warnings at import time
with suppress_output():
    import tempfile

    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel
    from nemo.collections.asr.models import ASRModel, EncDecMultiTaskModel

    # Patch SentencePieceTokenizer.eos_id for canary models.
    # Canary's tokenizer has <s> (token 3) as EOS, but SentencePiece doesn't
    # flag it as a special token, so tokenizer.eos_id() returns -1. The NeMo
    # canary2 prompt formatter asserts answer_ids[-1] == tokenizer.eos, which
    # fails (3 != -1). We detect canary by its unique <|startoftranscript|> token.
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import (
        SentencePieceTokenizer,
    )

    _original_eos_id = SentencePieceTokenizer.eos_id

    @property  # type: ignore[misc]
    def _patched_eos_id(self):
        try:
            if (
                hasattr(self, "tokenizer")
                and self.tokenizer.piece_to_id("<|startoftranscript|>") == 4
            ):
                return 3  # CANARY_EOS = "<s>"
        except Exception:
            pass
        return _original_eos_id.fget(self)

    SentencePieceTokenizer.eos_id = _patched_eos_id

logger = logging.getLogger(__name__)


def _materialize_weights(model):
    """Force all model parameters and buffers into RAM.

    safetensors uses memory-mapped files by default. Even with
    low_cpu_mem_usage=False and no device_map, tensors loaded from
    safetensors remain mmap'd — data is read from disk on-demand
    during inference, causing severe slowdowns. This function clones
    every parameter and buffer to break the mmap and ensure weights
    reside in actual RAM.
    """
    for p in model.parameters():
        p.data = p.data.clone()
    for b in model.buffers():
        b.data = b.data.clone()

# Optional types import (already available in Python 3.9+)


def _check_transformers_version():
    """Check that transformers version is compatible with granite model."""
    import transformers as tf_lib
    from packaging import version as pkg_version

    if pkg_version.parse(tf_lib.__version__) < pkg_version.parse("5.5.3"):
        raise ImportError(
            f"Granite model requires transformers>=5.5.3, "
            f"but {tf_lib.__version__} is installed. "
            f"Upgrade with: pip install 'transformers>=5.5.3'"
        )


class ModelWrapper:
    """
    Encapsulates loading and running different model types (whisper, parakeet, canary, voxtral, cohere).
    """

    def __init__(self, settings):
        self.settings = settings
        self.model_type = settings.model_type.lower()
        self.model = None
        self.processor = None
        self.TranscriptionRequest = None
        self._model_ref = None
        self._load_model()

    def _load_model(self):
        mt = self.model_type
        device = self.settings.device
        compute_type = getattr(self.settings, "compute_type", None)

        if mt == "whisper":
            self.model = WhisperModel(
                model_size_or_path=self.settings.model_name,
                device=device,
                compute_type=compute_type,
            )

        elif mt == "parakeet":
            with suppress_nemo():
                if compute_type in ("int8", "int4") and device == "cuda":
                    quant_cfg = BitsAndBytesConfig(
                        load_in_8bit=True if compute_type == "int8" else False,
                        load_in_4bit=True if compute_type == "int4" else False,
                    )
                    self.model = ASRModel.from_pretrained(
                        model_name=self.settings.model_name,
                        map_location=self.settings.device,
                    ).eval()
                else:
                    self.model = ASRModel.from_pretrained(
                        model_name=self.settings.model_name,
                        map_location=self.settings.device,
                    ).eval()
                self._model_ref = self.model

            if compute_type and compute_type not in ("int8", "int4"):
                self.model = self.model.to(
                    {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
                        compute_type, torch.float32
                    )
                )

        elif mt == "canary":
            with suppress_nemo():
                if compute_type in ("int8", "int4") and device == "cuda":
                    quant_cfg = BitsAndBytesConfig(
                        load_in_8bit=True if compute_type == "int8" else False,
                        load_in_4bit=True if compute_type == "int4" else False,
                    )
                    self.model = EncDecMultiTaskModel.from_pretrained(
                        self.settings.model_name, map_location=self.settings.device
                    ).eval()
                else:
                    self.model = EncDecMultiTaskModel.from_pretrained(
                        self.settings.model_name, map_location=self.settings.device
                    ).eval()
                self._model_ref = self.model

            if compute_type and compute_type not in ("int8", "int4"):
                self.model = self.model.to(
                    {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
                        compute_type, torch.float32
                    )
                )

        elif mt == "voxtral":
            repo_id = self.settings.model_name
            self.processor = AutoProcessor.from_pretrained(repo_id)
            device_map = {"": device}

            if self.settings.compute_type == "int8":
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    repo_id,
                    quantization_config=quant_cfg,
                    device_map=device_map,
                ).eval()

            elif self.settings.compute_type == "int4":
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    repo_id,
                    quantization_config=quant_cfg,
                    device_map=device_map,
                ).eval()

            else:
                compute_dtype = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }.get(self.settings.compute_type, torch.float16)

                if device == "cpu":
                    self.model = VoxtralForConditionalGeneration.from_pretrained(
                        repo_id,
                        dtype=compute_dtype,
                        low_cpu_mem_usage=False,
                    )
                    _materialize_weights(self.model)
                else:
                    self.model = VoxtralForConditionalGeneration.from_pretrained(
                        repo_id,
                        dtype=compute_dtype,
                        device_map=device_map,
                    )
                self.model = self.model.eval()

        elif mt == "cohere":
            repo_id = self.settings.model_name
            device_map = {"": self.settings.device}

            self.processor = AutoProcessor.from_pretrained(repo_id)

            if compute_type in ("int8", "int4") and device == "cuda":
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=True if compute_type == "int8" else False,
                    load_in_4bit=True if compute_type == "int4" else False,
                )
                self.model = CohereAsrForConditionalGeneration.from_pretrained(
                    repo_id,
                    device_map=device_map,
                    quantization_config=quant_cfg,
                )
            else:
                _dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
                    compute_type, torch.bfloat16 if device == "cuda" else torch.float32
                ) if compute_type and compute_type not in ("int8", "int4") else torch.float32

                if device == "cpu":
                    self.model = CohereAsrForConditionalGeneration.from_pretrained(
                        repo_id,
                        torch_dtype=_dtype,
                        low_cpu_mem_usage=False,
                    )
                    _materialize_weights(self.model)
                else:
                    self.model = CohereAsrForConditionalGeneration.from_pretrained(
                        repo_id,
                        device_map=device_map,
                    )
                    self.model = self.model.to(dtype=_dtype)
                self.model = self.model.eval()

        elif mt == "granite":
            repo_id = self.settings.model_name
            device_map = {"": self.settings.device}

            _check_transformers_version()

            self.processor = AutoProcessor.from_pretrained(
                repo_id, trust_remote_code=True
            )

            if compute_type in ("int8", "int4") and device == "cuda":
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=True if compute_type == "int8" else False,
                    load_in_4bit=True if compute_type == "int4" else False,
                )
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    repo_id,
                    device_map=device_map,
                    quantization_config=quant_cfg,
                    trust_remote_code=True,
                ).eval()
            else:
                _dtype = compute_type and {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
                    compute_type
                ) or (torch.bfloat16 if device == "cuda" else torch.float32)

                if device == "cpu":
                    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        repo_id,
                        torch_dtype=_dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=False,
                    )
                    _materialize_weights(self.model)
                else:
                    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        repo_id,
                        device_map=device_map,
                        torch_dtype=_dtype,
                        trust_remote_code=True,
                    )
                self.model = self.model.eval()

        elif mt == "granite-nar":
            repo_id = self.settings.model_name
            device_map = {"": self.settings.device}

            _check_transformers_version()

            self.processor = AutoProcessor.from_pretrained(
                repo_id, trust_remote_code=True
            )

            if compute_type in ("int8", "int4") and device == "cuda":
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=True if compute_type == "int8" else False,
                    load_in_4bit=True if compute_type == "int4" else False,
                )
                self.model = AutoModel.from_pretrained(
                    repo_id,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    device_map=device_map,
                    quantization_config=quant_cfg,
                ).eval()
            else:
                _dtype = compute_type and {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
                    compute_type
                ) or (torch.bfloat16 if device == "cuda" else torch.float32)

                if device == "cpu":
                    self.model = AutoModel.from_pretrained(
                        repo_id,
                        trust_remote_code=True,
                        attn_implementation="sdpa",
                        torch_dtype=_dtype,
                        low_cpu_mem_usage=False,
                    )
                    _materialize_weights(self.model)
                else:
                    self.model = AutoModel.from_pretrained(
                        repo_id,
                        trust_remote_code=True,
                        attn_implementation="flash_attention_2",
                        device_map=device_map,
                        torch_dtype=_dtype,
                    )
                self.model = self.model.eval()

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def transcribe(
        self, audio_data, sample_rate: int = 16000, language: str | None = None
    ) -> str:
        """
        Transcribe a numpy array of audio samples and return transcribed text.
        For some models (canary, voxtral) we write to a temp file and call model utilities requiring a file.
        For Voxtral-Mini-3B-2507, handles potential input size limits by chunking.
        """
        mt = self.model_type
        try:
            if mt == "whisper":
                segments, _ = self.model.transcribe(
                    audio_data,
                    beam_size=5,
                    condition_on_previous_text=False,
                    language=(language if language and language != "auto" else None),
                )
                return " ".join(segment.text.strip() for segment in segments)

            elif mt == "parakeet":
                with torch.inference_mode():
                    out = list(self.model.transcribe([audio_data]))
                if not out:
                    return ""
                result = out[0]
                if hasattr(result, "text"):
                    return result.text
                if isinstance(result, str):
                    return result
                if isinstance(result, list) and result:
                    first = result[0]
                    if hasattr(first, "text"):
                        return first.text
                    if isinstance(first, str):
                        return first
                return ""

            elif mt == "canary":
                lang = language or "en-en"
                lang_parts = lang.split("-")
                if len(lang_parts) != 2:
                    source_lang, target_lang = "en", "en"
                else:
                    source_lang, target_lang = lang_parts

                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_path = f.name
                    sf.write(temp_path, audio_data, sample_rate)
                    with suppress_output():
                        out = self.model.transcribe(
                            audio=[temp_path],
                            source_lang=source_lang,
                            target_lang=target_lang,
                        )
                    if not out or len(out) == 0:
                        return ""
                    result = out[0]
                    return result.strip() if isinstance(result, str) else result.text.strip()
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            elif mt == "voxtral":
                return self._transcribe_voxtral(audio_data, sample_rate, language)

            elif mt == "cohere":
                return self._transcribe_cohere(audio_data, sample_rate, language)
            elif mt == "granite":
                device = self.settings.device
                waveform = torch.from_numpy(audio_data).to(device)
                # Full language names expected by granite-speech-4.1-2b prompts
                granite_lang_names: dict[str, str] = {
                    "en": "English",
                    "fr": "French",
                    "de": "German",
                    "es": "Spanish",
                    "ja": "Japanese",
                    "it": "Italian",
                    "zh": "Mandarin",
                }
                lang = language or "en-en"
                lang_parts = lang.split("-") if lang else ["en", "en"]
                if len(lang_parts) == 2 and lang_parts[0] != lang_parts[1]:
                    target_name = granite_lang_names.get(
                        lang_parts[1], lang_parts[1]
                    )
                    action = f"translate the speech to {target_name}"
                else:
                    action = "transcribe the speech"
                user_prompt = (
                    f"<|audio|>{action} with proper punctuation "
                    "and capitalization."
                )
                chat = [{"role": "user", "content": user_prompt}]
                prompt = self.processor.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                model_inputs = self.processor(
                    prompt, waveform, device=device, return_tensors="pt"
                ).to(device)
                model_outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    num_beams=1,
                )
                num_input_tokens = model_inputs["input_ids"].shape[-1]
                new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
                output_text = self.processor.tokenizer.batch_decode(
                    new_tokens, add_special_tokens=False, skip_special_tokens=True
                )
                return output_text[0] if output_text else ""

            elif mt == "granite-nar":
                device = self.settings.device
                waveform = torch.from_numpy(audio_data).to(device)
                inputs = self.processor([waveform], device=device)
                with torch.no_grad():
                    output = self.model.transcribe(**inputs)
                transcriptions = self.processor.batch_decode(
                    output.preds, skip_special_tokens=True
                )
                return transcriptions[0] if transcriptions else ""

            else:
                raise ValueError(f"Unknown model type: {mt}")

        except Exception as e:
            logger.error(f"Error during model.transcribe: {e}")
            return ""

    def _transcribe_cohere(
        self, audio_data, sample_rate: int, language: str | None
    ) -> str:
        """Transcribe audio for cohere-transcribe-03-2026 with native chunking."""
        lang = language or "en"
        inputs = self.processor(
            audio_data, sampling_rate=sample_rate, return_tensors="pt", language=lang
        )
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=lang,
        )
        if isinstance(text, list):
            return text[0].strip() if text else ""
        return text.strip() if text else ""

    def _transcribe_voxtral(
        self, audio_data, sample_rate: int, language: str | None
    ) -> str:
        """Transcribe audio using Voxtral with native chunking via apply_transcription_request."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            sf.write(tmp_audio.name, audio_data, sample_rate)
            audio_path = tmp_audio.name

        try:
            inputs = self.processor.apply_transcription_request(
                audio=audio_path,
                model_id=self.settings.model_name,
                language=language if language and language != "auto" else None,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    num_beams=1,
                )

            decoded = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            return decoded
        finally:
            with contextlib.suppress(Exception):
                os.unlink(audio_path)
