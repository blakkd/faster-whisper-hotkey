import contextlib
import logging
import os
import time

import numpy as np
from transformers import (
    AutoModel,
    AutoModelForCTC,
    AutoModelForMultimodalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    CohereAsrForConditionalGeneration,
    VoxtralForConditionalGeneration,
)

from .hf_offline import enable_offline_if_unreachable


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
            for attr in ("log_info", "log_warn", "log_error", "log_debug", "info", "warn", "error", "debug"):
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
            if hasattr(self, "tokenizer") and self.tokenizer.piece_to_id("<|startoftranscript|>") == 4:
                return 3  # CANARY_EOS = "<s>"
        except Exception:  # noqa: BLE001, S110
            pass
        return _original_eos_id.fget(self)

    SentencePieceTokenizer.eos_id = _patched_eos_id

logger = logging.getLogger(__name__)

# The first forward on a fresh CUDA context pays a one-time cost (lazy CUDA
# kernel loading, cuBLAS/cuDNN handle init, SDPA backend selection/autotune,
# bitsandbytes init for int8/int4). Pay it at load time with a short silent
# clip so the first real recording isn't slowed down. This is most visible for
# models whose whole inference is a single forward pass (e.g. granite-turboctc),
# where the warmup cost otherwise IS the transcription delay; for
# autoregressive models it is amortized over the decode loop.
CUDA_WARMUP_DURATION_S = 2.0


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


def _cuda_int8_supported() -> bool:
    """
    CTranslate2 int8 CUDA GEMM fails with CUBLAS_STATUS_NOT_SUPPORTED on Blackwell
    (sm_120/121) when a dimension is not divisible by 4 (whisper vocab is 51865/51866) —
    OpenNMT/CTranslate2#1865. Upstream only auto-selects a different type (PR #1937);
    an explicit int8 request still loads but fails at inference, so fall back.
    """
    if not torch.cuda.is_available():
        return True
    major, _minor = torch.cuda.get_device_capability()
    return major < 12


def _patch_bnb_int8_noncontiguous():
    """Work around a bitsandbytes bug: Linear8bitLt returns wrong results for
    non-contiguous (strided) inputs.

    MatMul8bitLt reshapes 3-D inputs with ``A.reshape(-1, K)``; for batch size 1
    that reshape is a view (no copy), so strided layouts — e.g. the
    ``hidden_states.transpose(1, 2)`` fed to GraniteSpeech5's encoder conv block
    (pointwise_lin2) — reach the int8 GEMM kernel, which reads the memory as if
    it were row-major. The result is garbage activations and, for CTC models, an
    all-pad transcription. Contiguifying is a no-op for already-contiguous inputs.
    """
    import bitsandbytes.nn as bnb_nn

    cls = getattr(bnb_nn, "Linear8bitLt")
    if getattr(cls, "_fwh_contiguous_patched", False):
        return
    orig = cls.forward

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        return orig(self, x)

    cls.forward = forward
    cls._fwh_contiguous_patched = True


def _check_transformers_version(min_version: str, model_label: str):
    """Check that the installed transformers version supports a model."""
    import transformers as tf_lib
    from packaging import version as pkg_version

    if pkg_version.parse(tf_lib.__version__) < pkg_version.parse(min_version):
        raise ImportError(
            f"{model_label} model requires transformers>={min_version}, "
            f"but {tf_lib.__version__} is installed. "
            f"Upgrade with: pip install 'transformers>={min_version}'"
        )


class ModelWrapper:
    """
    Encapsulates loading and running different model types
    (whisper, parakeet, canary, voxtral, cohere, granite, granite-nar, granite-turboctc, qwen3-asr).
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
        enable_offline_if_unreachable(self.settings.model_name)

        mt = self.model_type
        device = self.settings.device
        compute_type = getattr(self.settings, "compute_type", None)

        if mt != "whisper" and device == "cuda" and compute_type == "int8":
            _patch_bnb_int8_noncontiguous()

        if mt == "whisper":
            if device == "cuda" and compute_type == "int8" and not _cuda_int8_supported():
                raise ValueError(
                    "CUDA int8 is not supported on this GPU (Blackwell sm_120/121, "
                    "CTranslate2/cuBLAS limitation). Pick a different whisper precision "
                    "(e.g. float16) in the settings screen."
                )
            self.model = WhisperModel(
                model_size_or_path=self.settings.model_name,
                device=device,
                compute_type=compute_type,
            )

        elif mt == "parakeet":
            with suppress_nemo():
                if compute_type in ("int8", "int4") and device == "cuda":
                    quant_cfg = BitsAndBytesConfig(
                        load_in_8bit=compute_type == "int8",
                        load_in_4bit=compute_type == "int4",
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
                        load_in_8bit=compute_type == "int8",
                        load_in_4bit=compute_type == "int4",
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
                    load_in_8bit=compute_type == "int8",
                    load_in_4bit=compute_type == "int4",
                )
                self.model = CohereAsrForConditionalGeneration.from_pretrained(
                    repo_id,
                    device_map=device_map,
                    quantization_config=quant_cfg,
                )
            else:
                _dtype = (
                    {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
                        compute_type, torch.bfloat16 if device == "cuda" else torch.float32
                    )
                    if compute_type and compute_type not in ("int8", "int4")
                    else torch.float32
                )

                if device == "cpu":
                    self.model = CohereAsrForConditionalGeneration.from_pretrained(
                        repo_id,
                        dtype=_dtype,
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

            # granite_speech loads natively (no remote code) from transformers 5.17 on
            _check_transformers_version("5.17.0", "Granite")

            self.processor = AutoProcessor.from_pretrained(repo_id)

            if compute_type in ("int8", "int4") and device == "cuda":
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=compute_type == "int8",
                    load_in_4bit=compute_type == "int4",
                )
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    repo_id,
                    device_map=device_map,
                    quantization_config=quant_cfg,
                ).eval()
            else:
                _dtype = (
                    compute_type
                    and {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
                        compute_type
                    )
                    or (torch.bfloat16 if device == "cuda" else torch.float32)
                )

                if device == "cpu":
                    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        repo_id,
                        dtype=_dtype,
                        low_cpu_mem_usage=False,
                    )
                    _materialize_weights(self.model)
                else:
                    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        repo_id,
                        device_map=device_map,
                        dtype=_dtype,
                    )
                self.model = self.model.eval()

        elif mt == "granite-nar":
            repo_id = self.settings.model_name
            device_map = {"": self.settings.device}

            _check_transformers_version("5.5.3", "Granite")

            self.processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)

            if compute_type in ("int8", "int4") and device == "cuda":
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=compute_type == "int8",
                    load_in_4bit=compute_type == "int4",
                )
                self.model = AutoModel.from_pretrained(
                    repo_id,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                    device_map=device_map,
                    quantization_config=quant_cfg,
                ).eval()
            else:
                _dtype = (
                    compute_type
                    and {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
                        compute_type
                    )
                    or (torch.bfloat16 if device == "cuda" else torch.float32)
                )

                if device == "cpu":
                    self.model = AutoModel.from_pretrained(
                        repo_id,
                        trust_remote_code=True,
                        attn_implementation="sdpa",
                        dtype=_dtype,
                        low_cpu_mem_usage=False,
                    )
                    _materialize_weights(self.model)
                else:
                    self.model = AutoModel.from_pretrained(
                        repo_id,
                        trust_remote_code=True,
                        attn_implementation="sdpa",
                        device_map=device_map,
                        dtype=_dtype,
                    )
                self.model = self.model.eval()

        elif mt == "granite-turboctc":
            repo_id = self.settings.model_name
            device_map = {"": self.settings.device}

            _check_transformers_version("5.16.0", "Granite Speech 5.0 TurboCTC")

            self.processor = AutoProcessor.from_pretrained(repo_id)

            if compute_type in ("int8", "int4") and device == "cuda":
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=(compute_type == "int8"),
                    load_in_4bit=(compute_type == "int4"),
                    # GraniteSpeech5Encoder casts mel features to input_linear.weight.dtype
                    # before projecting. A bnb-quantized input_linear has an int8/uint8
                    # weight dtype, which crashes bitsandbytes (bias cast to a non-float
                    # dtype). An explicit skip list replaces the defaults, so the tied
                    # ctc_head/encoder.out pair is kept in float too, as the defaults do.
                    llm_int8_skip_modules=["input_linear", "ctc_head", r"encoder\.out$"],
                )
                self.model = AutoModelForCTC.from_pretrained(
                    repo_id,
                    device_map=device_map,
                    quantization_config=quant_cfg,
                ).eval()
            else:
                # Weights are stored natively as bf16 (see HF repo config.json)
                _dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(compute_type, torch.bfloat16)

                if device == "cpu":
                    self.model = AutoModelForCTC.from_pretrained(
                        repo_id,
                        dtype=_dtype,
                        low_cpu_mem_usage=False,
                    )
                    _materialize_weights(self.model)
                else:
                    self.model = AutoModelForCTC.from_pretrained(
                        repo_id,
                        dtype=_dtype,
                        device_map=device_map,
                    )
                self.model = self.model.eval()

        elif mt == "qwen3-asr":
            repo_id = self.settings.model_name
            device_map = {"": self.settings.device}

            _check_transformers_version("5.13.0", "Qwen3-ASR")

            self.processor = AutoProcessor.from_pretrained(repo_id)

            if compute_type in ("int8", "int4") and device == "cuda":
                quant_cfg = BitsAndBytesConfig(
                    load_in_8bit=(compute_type == "int8"),
                    load_in_4bit=(compute_type == "int4"),
                )
                self.model = AutoModelForMultimodalLM.from_pretrained(
                    repo_id,
                    device_map=device_map,
                    quantization_config=quant_cfg,
                ).eval()
            else:
                # Weights are stored natively as bf16 (see HF repo config.json)
                _dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(compute_type, torch.bfloat16)

                if device == "cpu":
                    self.model = AutoModelForMultimodalLM.from_pretrained(
                        repo_id,
                        dtype=_dtype,
                        low_cpu_mem_usage=False,
                    )
                    _materialize_weights(self.model)
                else:
                    self.model = AutoModelForMultimodalLM.from_pretrained(
                        repo_id,
                        dtype=_dtype,
                        device_map=device_map,
                    )
                self.model = self.model.eval()

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if device == "cuda" and torch.cuda.is_available():
            self._warmup_cuda()

    def _warmup_cuda(self):
        """Run one short silent transcription through the production path to pay the one-time CUDA warmup cost."""
        dummy = np.zeros(int(CUDA_WARMUP_DURATION_S * 16000), dtype=np.float32)
        start = time.perf_counter()
        self.transcribe(dummy, sample_rate=16000, language=self.settings.language)
        logger.info(f"CUDA warmup finished in {time.perf_counter() - start:.1f} s")

    def transcribe(self, audio_data, sample_rate: int = 16000, language: str | None = None) -> str:
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
                    target_name = granite_lang_names.get(lang_parts[1], lang_parts[1])
                    action = f"translate the speech to {target_name}"
                else:
                    action = "transcribe the speech"
                user_prompt = f"<|audio|>{action} with proper punctuation and capitalization."
                chat = [{"role": "user", "content": user_prompt}]
                prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                model_inputs = self.processor(prompt, waveform, device=device, return_tensors="pt").to(device)
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
                transcriptions = self.processor.batch_decode(output.preds, skip_special_tokens=True)
                return transcriptions[0] if transcriptions else ""

            elif mt == "granite-turboctc":
                device = self.settings.device
                inputs = self.processor(audio_data, sampling_rate=sample_rate, device=device, return_tensors="pt")
                inputs = inputs.to(device, dtype=self.model.dtype)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs)
                transcriptions = self.processor.batch_decode(outputs, skip_special_tokens=True)
                return transcriptions[0] if transcriptions else ""

            elif mt == "qwen3-asr":
                return self._transcribe_qwen3_asr(audio_data, sample_rate, language)

            else:
                raise ValueError(f"Unknown model type: {mt}")

        except Exception:
            logger.exception(
                f"Error during model.transcribe for {self.model_type}/{self.settings.model_name} "
                f"({self.settings.device}/{getattr(self.settings, 'compute_type', None)})"
            )
            return ""

    def _transcribe_cohere(self, audio_data, sample_rate: int, language: str | None) -> str:
        """Transcribe audio for cohere-transcribe-03-2026 with native chunking."""
        lang = language or "en"
        inputs = self.processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", language=lang)
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

    def _transcribe_voxtral(self, audio_data, sample_rate: int, language: str | None) -> str:
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
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)

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

    def _transcribe_qwen3_asr(self, audio_data, sample_rate: int, language: str | None) -> str:
        """Transcribe audio using Qwen3-ASR via apply_transcription_request."""
        inputs = self.processor.apply_transcription_request(
            audio=audio_data,
            language=language if language and language != "auto" else None,
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        duration = len(audio_data) / sample_rate if sample_rate else 0
        max_new_tokens = min(8192, max(512, int(duration * 8)))

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.decode(new_tokens, return_format="transcription_only")
        if isinstance(decoded, list):
            return decoded[0].strip() if decoded else ""
        return decoded.strip() if decoded else ""
