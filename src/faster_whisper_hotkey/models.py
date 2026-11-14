"""All inference in this app goes through one engine: transcribe.cpp.

Models are single-file GGUF checkpoints (see gguf_models.py), downloaded on
first use (see downloader.py) and run through the pure-Python
``transcribe-cpp`` binding (ctypes over the C library). One ggml backend —
CPU, CUDA or Vulkan — serves every model family, which is what previously
took four engines (CTranslate2, NeMo/Photon, transformers, bitsandbytes).
"""

import logging
import time
from typing import Any

import numpy as np
import transcribe_cpp
from transcribe_cpp.errors import InputTooLong, InputTooShort, OutputTruncated, TranscribeError

from .capitalization import add_capitalization
from .config import accepted_languages_cohere
from .downloader import ensure_gguf
from .gguf_models import VARIANTS, gguf_filename, repo_and_filename, resolve_quant
from .settings import Settings

logger = logging.getLogger(__name__)

# The first forward on a fresh GPU context pays a one-time cost (kernel module
# load, backend init). Pay it at load time with a short silent clip so the
# first real recording isn't slowed down. This is most visible for models
# whose whole inference is a single forward pass (e.g. granite-turboctc),
# where the warmup cost otherwise IS the transcription delay.
GPU_WARMUP_DURATION_S = 2.0
_GPU_BACKENDS = ("cuda", "vulkan", "metal", "rocm")

# Settings values that mean "detect the language automatically".
_AUTO_LANGUAGES = ("", "auto", None)

# Families whose settings language is a "source-target" pair ("en-en",
# "en-de"); same source and target means transcription, different means
# translation.
_PAIR_LANGUAGE_TYPES = ("canary", "granite")

# device (settings) -> ggml backend name
_BACKEND_BY_DEVICE: dict[str, "transcribe_cpp.Backend"] = {"cpu": "cpu", "cuda": "cuda", "vulkan": "vulkan"}


def _backend_for(device: str) -> "transcribe_cpp.Backend":
    return _BACKEND_BY_DEVICE.get((device or "").lower(), "auto")


def _is_auto(language: str | None) -> bool:
    return language in _AUTO_LANGUAGES or str(language).strip() in _AUTO_LANGUAGES


def pcm(audio_data) -> Any:
    """Contiguous float32 mono PCM as the binding expects it (buffer protocol)."""
    return np.ascontiguousarray(audio_data, dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared support models (VAD, language identification)
# ---------------------------------------------------------------------------

_support_models: dict[str, transcribe_cpp.Model] = {}


def _load_support_model(variant_id: str, backend: "transcribe_cpp.Backend") -> transcribe_cpp.Model:
    """Load a small support model (VAD/lang-id), cached per backend."""
    key = f"{variant_id}:{backend}"
    model = _support_models.get(key)
    if model is None:
        variant = VARIANTS[variant_id]
        path = ensure_gguf(variant.hf_repo, gguf_filename(variant, resolve_quant(variant, "")))
        model = transcribe_cpp.Model(str(path), backend=backend)
        _support_models[key] = model
    return model


def load_vad_model(device: str) -> transcribe_cpp.Model:
    """Load the Silero VAD GGUF on the same backend as the main model."""
    return _load_support_model("silero-vad-v6.2", _backend_for(device))


class ModelWrapper:
    """
    Loads one transcribe.cpp GGUF model (whisper, parakeet, parakeet-ultra,
    canary, voxtral, cohere, granite, granite-nar, granite-turboctc, qwen3-asr)
    and transcribes 16 kHz mono float32 PCM with it.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_type = settings.model_type.lower()

        repo, filename, variant, quant = repo_and_filename(self.model_type, settings.model_name, settings.compute_type)
        self.variant = variant
        self.quant = quant
        self.backend: transcribe_cpp.Backend = _backend_for(settings.device)

        if not transcribe_cpp.backend_available(self.backend):
            raise ValueError(
                f"The {self.backend!r} backend is not available with the installed transcribe_cpp "
                f"library. Pick 'cpu' (or a supported GPU backend) in the settings screen."
            )

        path = ensure_gguf(repo, filename)
        logger.info(f"Loading {variant.variant} ({quant}, {self.backend}) from {path}")
        self.model = transcribe_cpp.Model(str(path), backend=self.backend)
        self.capabilities = self.model.capabilities
        self.max_audio_ms = self.capabilities.max_audio_ms
        if self.max_audio_ms:
            logger.info(f"{variant.variant} rejects input longer than {self.max_audio_ms / 1000:.0f} s")

        self.session = self.model.session()
        self._langid_model: transcribe_cpp.Model | None = None  # cohere "auto": separate lang-id model

        if self.backend in _GPU_BACKENDS:
            self._warmup_gpu()

    def _warmup_gpu(self):
        """Run one short silent transcription through the production path to pay the one-time GPU warmup cost."""
        dummy = np.zeros(int(GPU_WARMUP_DURATION_S * 16000), dtype=np.float32)
        start = time.perf_counter()
        self.transcribe(dummy, sample_rate=16000, language=self.settings.language)
        logger.info(f"GPU warmup finished in {time.perf_counter() - start:.1f} s")

    # ------------------------------------------------------------------
    # Language/task selection per family
    # ------------------------------------------------------------------

    def _run_kwargs(self, audio_data: np.ndarray, language: str | None) -> dict:
        """Translate the settings language into transcribe.cpp run kwargs.

        ``task`` is "transcribe" everywhere except the translation-capable
        families (whisper's own translation, canary and granite src-tgt
        pairs), which the TUI encodes as an "en-de" style pair.
        """
        mt = self.model_type

        if mt in _PAIR_LANGUAGE_TYPES:
            lang = language or "en-en"
            parts = lang.split("-")
            source, target = (parts[0], parts[1]) if len(parts) == 2 else (lang, lang)
            if source == target:
                return {"task": "transcribe", "language": source, "target_language": None}
            return {"task": "translate", "language": source, "target_language": target}

        if mt == "granite-turboctc":
            return {"task": "transcribe", "language": None, "target_language": None}  # English-only

        if mt == "cohere":
            # No built-in language detection: the engine needs an explicit hint.
            # Resolve "auto" (or an unusable value) with the VoxLingua107 model.
            lang = "" if _is_auto(language) else str(language).strip()
            if lang not in accepted_languages_cohere:
                lang = self._detect_cohere_language(audio_data)
            return {"task": "transcribe", "language": lang, "target_language": None}

        # whisper, parakeet, parakeet-ultra, voxtral, qwen3-asr, granite-nar:
        # explicit hint or internal auto-detection.
        return {
            "task": "transcribe",
            "language": None if _is_auto(language) else str(language).strip(),
            "target_language": None,
        }

    def _detect_cohere_language(self, audio_data: np.ndarray) -> str:
        """Best-effort language for cohere-transcribe, which needs an explicit hint."""
        try:
            if self._langid_model is None:
                self._langid_model = _load_support_model("lang-id-voxlingua107-ecapa", self.backend)
            with self._langid_model.langid_session() as lid:
                result = lid.run(pcm(audio_data), allowed=list(accepted_languages_cohere))
            code = result.code or ""
            if code in accepted_languages_cohere:
                logger.info(f"Language for cohere auto-detected as '{code}'")
                return code
            logger.info(f"Language auto-detection returned unusable {code!r}; defaulting to 'en'")
        except (TranscribeError, InputTooShort):  # e.g. clip below the lang-id minimum length
            logger.info("Language auto-detection unavailable for this clip; defaulting to 'en'")
        return "en"

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio_data, sample_rate: int = 16000, language: str | None = None) -> str:
        """Transcribe a numpy array of float32 PCM samples and return the text.

        Length-bounded models (canary, cohere, ...) reject over-long input up
        front instead of degrading silently; we surface that as a clear log
        line and an empty result, like any other transcription failure.
        """
        mt = self.model_type
        duration_ms = 0.0
        try:
            duration_ms = len(audio_data) / sample_rate * 1000 if sample_rate else 0
            if self.max_audio_ms and duration_ms > self.max_audio_ms:
                logger.error(
                    f"Recording is {duration_ms / 1000:.0f} s but {self.variant.variant} accepts at most "
                    f"{self.max_audio_ms / 1000:.0f} s — shortening the recording or switching model."
                )
                return ""

            kwargs = self._run_kwargs(audio_data, language)
            result = self.session.run(pcm(audio_data), **kwargs)
            text = result.text or ""

            if mt == "granite-nar":
                # The NAR model emits lowercase, unpunctuated text; restore
                # sentence casing (the engine does not, for any quant).
                text = add_capitalization(text, "en" if _is_auto(language) else str(language))
            return text

        except OutputTruncated as exc:
            # The engine hit its output budget: keep what it managed instead
            # of discarding the whole partial transcript.
            partial = getattr(exc, "partial_result", None)
            logger.warning(f"{self.variant.variant} output truncated at its token budget", exc_info=True)
            return (partial.text if partial is not None else "").strip()
        except InputTooLong:
            logger.error(
                f"{self.variant.variant} rejected the {duration_ms / 1000:.0f} s recording "
                f"(limit {self.max_audio_ms / 1000:.0f} s)"
            )
            return ""
        except Exception:
            logger.exception(
                f"Error during model.transcribe for {self.model_type}/{self.settings.model_name} "
                f"({self.settings.device}/{self.quant})"
            )
            return ""
