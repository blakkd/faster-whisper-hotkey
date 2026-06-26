import contextlib
import logging
import os


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

    @property
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

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    CohereAsrForConditionalGeneration,
    VoxtralForConditionalGeneration,
)


logger = logging.getLogger(__name__)

# Optional types import (already available in Python 3.9+)
from typing import Optional


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
            self.model = ASRModel.from_pretrained(
                model_name=self.settings.model_name,
                map_location=self.settings.device,
            ).eval()
            self._model_ref = self.model

        elif mt == "canary":
            self.model = EncDecMultiTaskModel.from_pretrained(
                self.settings.model_name, map_location=self.settings.device
            ).eval()
            self._model_ref = self.model

        elif mt == "voxtral":
            from mistral_common.protocol.transcription.request import (
                TranscriptionRequest as _TR,
            )
            from pydantic_extra_types.language_code import LanguageAlpha2

            class TranscriptionRequest(_TR):
                language: Optional[LanguageAlpha2] = None

            repo_id = self.settings.model_name
            self.processor = AutoProcessor.from_pretrained(repo_id)

            if self.settings.compute_type == "int8":
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    repo_id,
                    quantization_config=quant_cfg,
                    device_map="cuda",
                ).eval()

            elif self.settings.compute_type == "int4":
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    repo_id,
                    quantization_config=quant_cfg,
                    device_map="cuda",
                ).eval()

            else:
                compute_dtype = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }.get(self.settings.compute_type, torch.float16)

                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    repo_id,
                    dtype=compute_dtype,
                    device_map="cuda",
                ).eval()

            self.TranscriptionRequest = TranscriptionRequest
            self.max_duration = (
                getattr(self.processor.feature_extractor, "max_duration", None)
                or 30
            )

        elif mt == "cohere":
            repo_id = self.settings.model_name

            self.processor = AutoProcessor.from_pretrained(repo_id)

            self.model = CohereAsrForConditionalGeneration.from_pretrained(
                repo_id, device_map={"": self.settings.device}
            )
            # float32 is ~8x faster than bfloat16 on CPU; bf16 is fine on GPU
            if self.settings.device == "cpu":
                self.model = self.model.float()
            self.model = self.model.eval()
            self.max_duration = (
                getattr(self.processor.feature_extractor, "max_duration", None)
                or 30
            )

        elif mt == "granite":
            repo_id = self.settings.model_name
            device_map = {"": self.settings.device}

            _check_transformers_version()

            self.processor = AutoProcessor.from_pretrained(
                repo_id, trust_remote_code=True
            )

            if device == "cuda":
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    repo_id,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).eval()
            else:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    repo_id,
                    device_map=device_map,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                ).eval()

        elif mt == "granite-nar":
            repo_id = self.settings.model_name
            device_map = {"": self.settings.device}

            _check_transformers_version()

            self.processor = AutoProcessor.from_pretrained(
                repo_id, trust_remote_code=True
            )

            if device == "cuda":
                self.model = AutoModel.from_pretrained(
                    repo_id,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                ).eval()
            else:
                self.model = AutoModel.from_pretrained(
                    repo_id,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                    device_map=device_map,
                    torch_dtype=torch.float32,
                ).eval()

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def transcribe(
        self, audio_data, sample_rate: int = 16000, language: Optional[str] = None
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
                # Voxtral-specific transcription with chunking based on feature extractor max_duration
                samples_per_second = sample_rate
                max_samples = self.max_duration * samples_per_second

                if len(audio_data) > max_samples:
                    logger.warning(
                        f"Audio length ({len(audio_data) / samples_per_second:.2f}s) exceeds Voxtral-Mini-3B-2507's input limit ({self.max_duration}s). "
                        "Processing in chunks."
                    )
                    chunks = []
                    for i in range(0, len(audio_data), max_samples):
                        chunk = audio_data[i : i + max_samples]
                        if len(chunk) < 1000:  # Skip very short chunks (likely noise)
                            continue
                        chunks.append(chunk)

                    # Process each chunk and concatenate results
                    full_text = ""
                    for i, chunk in enumerate(chunks):
                        try:
                            result = self._transcribe_single_chunk_voxtral(
                                chunk, sample_rate, language
                            )
                            if result.strip():
                                full_text += result + " "
                        except Exception as e:
                            logger.error(f"Failed to transcribe chunk {i}: {e}")
                            # Optionally add a placeholder or skip
                            pass

                    return full_text.strip()
                else:
                    # If audio is within limits, process it directly
                    return self._transcribe_single_chunk_voxtral(
                        audio_data, sample_rate, language
                    )

            elif mt == "cohere":
                # Cohere feature extractor has max_duration; chunk longer audio
                samples_per_second = sample_rate
                max_samples = self.max_duration * samples_per_second

                if len(audio_data) > max_samples:
                    logger.warning(
                        f"Audio length ({len(audio_data) / samples_per_second:.2f}s) exceeds "
                        f"cohere-transcribe-03-2026's input limit ({self.max_duration}s). "
                        "Processing in chunks."
                    )
                    chunks = []
                    for i in range(0, len(audio_data), max_samples):
                        chunk = audio_data[i : i + max_samples]
                        if len(chunk) < 1000:
                            continue
                        chunks.append(chunk)

                    full_text = ""
                    for i, chunk in enumerate(chunks):
                        try:
                            result = self._transcribe_single_chunk_cohere(
                                chunk, sample_rate, language
                            )
                            if result.strip():
                                full_text += result + " "
                        except Exception as e:
                            logger.error(f"Failed to transcribe chunk {i}: {e}")
                            pass

                    return full_text.strip()
                else:
                    return self._transcribe_single_chunk_cohere(
                        audio_data, sample_rate, language
                    )
            elif mt == "granite":
                device = self.settings.device
                waveform = torch.from_numpy(audio_data).to(device)
                lang = language or "en-en"
                lang_parts = lang.split("-") if lang else ["en", "en"]
                if len(lang_parts) == 2 and lang_parts[0] != lang_parts[1]:
                    action = f"translate the speech to {lang_parts[1]}"
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

    def _transcribe_single_chunk_cohere(
        self, audio_data, sample_rate: int, language: Optional[str]
    ) -> str:
        """Transcribe a single chunk of audio for cohere-transcribe-03-2026."""
        lang = language or "en"
        inputs = self.processor(
            audio_data, sampling_rate=sample_rate, return_tensors="pt", language=lang
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(outputs, skip_special_tokens=True)
        return text if isinstance(text, str) else (text[0] if text else "")

    def _transcribe_single_chunk_voxtral(
        self, audio_data, sample_rate: int, language: Optional[str]
    ) -> str:
        """
        Internal helper to transcribe a single chunk of audio for Voxtral-Mini-3B-2507.
        This handles the file I/O and model call.
        """
        # Write chunk to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            sf.write(tmp_audio.name, audio_data, sample_rate)
            audio_path = tmp_audio.name

        try:
            # Create a wrapper class to mimic what the processor expects
            class FileWrapper:
                def __init__(self, file_obj):
                    self.file = file_obj

            with open(audio_path, "rb") as f:
                wrapped_file = FileWrapper(f)

                # Prepare request similar to test_voxtral.py
                openai_req = {
                    "model": self.settings.model_name,
                    "file": wrapped_file,
                }
                if language and language != "auto":
                    openai_req["language"] = language

                tr = self.TranscriptionRequest.from_openai(openai_req)

                # Get tokens from the processor's tokenizer
                tok = self.processor.tokenizer.tokenizer.encode_transcription(tr)

                try:
                    input_features = self.processor.feature_extractor(
                        audio_data,
                        sampling_rate=sample_rate,
                        return_tensors="pt",
                    ).input_features.to(self.model.device)

                    # Get the tokens correctly (they should be in tok.tokens)
                    if hasattr(tok, "tokens") and tok.tokens is not None:
                        token_ids = torch.tensor([tok.tokens], device=self.model.device)
                    else:
                        logger.warning("Token IDs might be invalid")
                        return ""

                except Exception as e:
                    logger.error(f"Feature extraction failed: {e}")
                    raise

                # Generate using the model
                with torch.no_grad():
                    ids = self.model.generate(
                        input_features=input_features,
                        input_ids=token_ids,
                        max_new_tokens=500,
                        num_beams=1,
                    )
                decoded = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
                return decoded

        except Exception as e:
            logger.error(f"Voxtral-Mini-3B-2507 transcription error in chunk: {e}")
            raise
        finally:
            try:
                os.unlink(audio_path)
            except Exception:
                pass
