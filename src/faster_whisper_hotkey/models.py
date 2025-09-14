import os
import tempfile
import logging
import torch
import soundfile as sf

from typing import Optional

from transformers import (
    VoxtralForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from nemo.collections.asr.models import ASRModel, EncDecMultiTaskModel
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Encapsulates loading and running different model types (whisper, parakeet, canary, voxtral).
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
            from typing import Optional
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
                    out = self.model.transcribe([audio_data])
                return out[0].text if out else ""

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
                    out = self.model.transcribe(
                        audio=[temp_path],
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                    return out[0].text.strip() if out and len(out) > 0 else ""
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            elif mt == "voxtral":
                # --- Voxtral-Mini-3B-2507-specific transcription with chunking ---
                # Based on documentation and typical behavior, 30s is a safe limit for the encoder.
                MAX_DURATION_SECONDS = 30
                samples_per_second = sample_rate
                max_samples = MAX_DURATION_SECONDS * samples_per_second

                if len(audio_data) > max_samples:
                    logger.warning(
                        f"Audio length ({len(audio_data) / samples_per_second:.2f}s) exceeds Voxtral-Mini-3B-2507's recommended input limit ({MAX_DURATION_SECONDS}s). "
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

            else:
                raise ValueError(f"Unknown model type: {mt}")

        except Exception as e:
            logger.error(f"Error during model.transcribe: {e}")
            return ""

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
