"""All inference in this app goes through one engine: transcribe.cpp.

Models are single-file GGUF checkpoints on the Hugging Face hub (see
``catalog.py``), downloaded on first use (see ``downloader.py``) and run through
the pure-Python ``transcribe-cpp`` binding (ctypes over the C library). One ggml
backend — CPU, CUDA or Vulkan — serves every model family.

The app carries no model- or language-specific tables: which languages a model
transcribes, whether it detects language, whether it translates and to where,
and its input length limit all come from the loaded GGUF's ``capabilities``.
"""

import logging
import re
import time
from typing import Any

import numpy as np
import soundfile as sf
import transcribe_cpp
from transcribe_cpp.errors import OutOfMemory, OutputTruncated, TranscribeError

from . import catalog
from .capitalization import add_capitalization
from .downloader import ensure_gguf
from .settings import Settings

logger = logging.getLogger(__name__)

# Warmup: one short clip through the production path at load time, on EVERY
# backend. A GPU pays the one-time context/kernel-module cost, a CPU pages in
# the mmap'd weights — either way the first real recording isn't slowed down.
# The clip is deliberately REAL audio (2 s of a certain 1987 hit) instead of
# silence: silence warms the encoder but barely feeds the decoder, while audio
# with transients and structure exercises both.
WARMUP_CLIP = "assets/never_gonna_give_you_up_2s.wav"

# Settings values that mean "detect the language automatically".
_AUTO_LANGUAGES = ("", "auto", None)

# device (settings) -> ggml backend name
_BACKEND_BY_DEVICE: dict[str, "transcribe_cpp.Backend"] = {"cpu": "cpu", "cuda": "cuda", "vulkan": "vulkan"}


def resolve_backend(device: str) -> "transcribe_cpp.Backend":
    """Map the settings device to a backend, strictly.

    No silent substitution: an explicit device the installed library cannot
    serve (e.g. "cuda" with the CPU+Vulkan PyPI wheel) raises with a clear
    remedy instead of quietly running somewhere else. The TUI only offers
    serviceable backends, so this only fires on stale settings files — where
    startup then stops with the instruction to fix them.
    """
    backend = _BACKEND_BY_DEVICE.get((device or "").lower(), "auto")
    if backend != "auto" and not transcribe_cpp.backend_available(backend):
        raise ValueError(
            f"The {backend!r} backend is not available with the installed transcribe_cpp library. "
            "Pick a listed device in the settings screen (the CUDA provider installs with "
            "`pip install 'transcribe-cpp[cu12]' --extra-index-url https://handy-computer.github.io/transcribe.cpp/whl/cu12`)."
        )
    return backend


def _is_auto(language: Any) -> bool:
    return language in _AUTO_LANGUAGES or str(language).strip() in _AUTO_LANGUAGES


def pcm(audio_data) -> Any:
    """Contiguous float32 mono PCM as the binding expects it (buffer protocol)."""
    return np.ascontiguousarray(audio_data, dtype=np.float32)


def _needs_sentence_case(filename: str) -> bool:
    """Whether a checkpoint emits punctuated but all-lowercase text.

    This is an intrinsic output quirk of the IBM granite *non-autoregressive*
    checkpoints (file name carries a ``nar`` token): unlike the autoregressive
    variant they emit punctuation but no capitalization, so the engine returns
    lowercase and we restore sentence case. The ``turboctc`` checkpoints are
    intentionally NOT post-processed — they emit neither punctuation nor case,
    so sentence boundaries are undetectable (use LLM correction for those).
    """
    tokens = re.split(r"[-_.]", filename.lower())
    return "nar" in tokens


# ---------------------------------------------------------------------------
# Shared support models (VAD), loaded lazily and cached per backend
# ---------------------------------------------------------------------------

_support_models: dict[str, transcribe_cpp.Model] = {}


def _load_support_model(repo: str, filename: str, backend: "transcribe_cpp.Backend") -> transcribe_cpp.Model:
    """Load a small support model (VAD), cached per (file, backend)."""
    key = f"{filename}:{backend}"
    model = _support_models.get(key)
    if model is None:
        path = ensure_gguf(repo, filename)
        model = transcribe_cpp.Model(str(path), backend=backend)
        _support_models[key] = model
    return model


def load_vad_model(device: str) -> transcribe_cpp.Model:
    """Load the Silero VAD GGUF on the given device/backend name ("auto" = best available)."""
    return _load_support_model(catalog.VAD_REPO, catalog.VAD_FILE, resolve_backend(device))


class ModelWrapper:
    """
    Loads one transcribe.cpp GGUF model (chosen by ``model_repo`` + ``model_file``)
    and transcribes 16 kHz mono float32 PCM with it. Language and translation
    behavior come from the model's ``capabilities``.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.repo = settings.model_repo
        self.filename = settings.model_file
        self.name = catalog.short_name(self.repo)

        self.backend: transcribe_cpp.Backend = resolve_backend(settings.device)

        path = ensure_gguf(self.repo, self.filename)
        logger.info(f"Loading {self.name} ({self.filename}, {self.backend}) from {path}")
        try:
            self.model = transcribe_cpp.Model(str(path), backend=self.backend)
        except OutOfMemory as exc:
            # No device substitution: a full GPU means a full GPU. Report it
            # actionably; main() turns this into one clean error line + exit.
            remedy = (
                "the GPU is likely full — free VRAM, use a smaller quantization, or pick 'cpu' in the settings screen"
                if self.backend != "cpu"
                else "system RAM is full — close applications or use a smaller quantization"
            )
            raise OutOfMemory(f"Failed to load {self.name} on {self.backend!r}: out of memory ({remedy}).") from exc

        self.capabilities = self.model.capabilities
        self.variant = self.model.variant
        self.arch = self.model.arch
        self.max_audio_ms = self.capabilities.max_audio_ms
        if self.max_audio_ms:
            logger.info(f"{self.name} rejects input longer than {self.max_audio_ms / 1000:.0f} s")

        self._sentence_case = _needs_sentence_case(self.filename)
        self.session = self.model.session()
        self._closed = False

        self._warmup()

    # ------------------------------------------------------------------
    # Capabilities -> menu options (the UI builds its language screens from these)
    # ------------------------------------------------------------------

    @property
    def supports_translate(self) -> bool:
        return bool(self.capabilities.supports_translate)

    def transcription_languages(self) -> list[str]:
        """Selectable transcription languages; "auto" first when the model can detect language."""
        langs = list(self.capabilities.languages)
        if self.capabilities.supports_language_detect:
            return ["auto", *langs]
        return langs

    def translation_source_languages(self) -> list[str]:
        """Valid translation sources (an explicit language the model transcribes)."""
        return list(self.capabilities.languages) or self.transcription_languages()

    def translation_target_languages(self) -> list[str]:
        """Valid translation targets, as advertised by the model."""
        return list(self.capabilities.translate_target_languages)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self):
        """Run the 2 s warmup clip through the production path once.

        Never fatal: a warmup failure (odd backend) is logged and the app
        still starts — worst case the first real recording pays the cost.
        """
        from .config import get_resource_path

        audio, sr = sf.read(get_resource_path(WARMUP_CLIP), dtype="float32", always_2d=True)
        dummy = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
        start = time.perf_counter()
        try:
            text = self.transcribe(dummy, sample_rate=sr)
            logger.info(f"Warmup finished in {time.perf_counter() - start:.1f} s")
            heard = (text or "").strip()[:60]
            logger.debug(f"Warmup heard: {heard!r}" if heard else "Warmup heard nothing — too well-behaved for the easter egg")
        except Exception as exc:  # noqa: BLE001 — warmup must never block startup
            logger.warning(f"Warmup failed ({exc}); first real transcription may be slower")

    # ------------------------------------------------------------------
    # Task/language selection from capabilities
    # ------------------------------------------------------------------

    def _run_kwargs(self, language: Any, target: Any) -> dict:
        """Translate the chosen language/target into transcribe.cpp run kwargs.

        A non-empty target switches to translation (source -> target); the
        model's own capabilities determine which pairs it actually accepts, and
        the UI only offers those — an invalid pair would be surfaced by the
        engine at run time, never guessed at here.
        """
        lang = None if _is_auto(language) else str(language).strip()
        tgt = None if _is_auto(target) else str(target).strip() or None
        task = "translate" if tgt else "transcribe"
        return {"task": task, "language": lang, "target_language": tgt}

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio_data, sample_rate: int = 16000, language: Any = None, target: Any = "") -> str:
        """Transcribe (or translate) a numpy array of float32 PCM samples.

        Length-bounded models reject over-long input up front; we surface that
        as a clear log line and an empty result, like any other failure.
        """
        duration_ms = 0.0
        try:
            duration_ms = len(audio_data) / sample_rate * 1000 if sample_rate else 0
            if self.max_audio_ms and duration_ms > self.max_audio_ms:
                logger.error(
                    f"Recording is {duration_ms / 1000:.0f} s but {self.name} accepts at most "
                    f"{self.max_audio_ms / 1000:.0f} s — shortening the recording or switching model."
                )
                return ""

            kwargs = self._run_kwargs(language, target)
            result = self.session.run(pcm(audio_data), **kwargs)
            text = result.text or ""

            if self._sentence_case:
                # The NAR checkpoint emits lowercase; restore sentence casing.
                lang = "en" if _is_auto(language) else str(language)
                text = add_capitalization(text, lang)
            return text

        except OutputTruncated as exc:
            # The engine hit its output budget: keep what it managed instead
            # of discarding the whole partial transcript.
            partial = getattr(exc, "partial_result", None)
            logger.warning(f"{self.name} output truncated at its token budget", exc_info=True)
            return (partial.text if partial is not None else "").strip()
        except TranscribeError:
            # Covers InputTooLong and every other engine failure: log and return
            # empty (mirrors the length guard above) instead of crashing.
            logger.exception(f"Engine error transcribing {self.name} ({self.filename} on {self.settings.device})")
            return ""
        except Exception:
            logger.exception(f"Unexpected error running {self.name} ({self.filename} on {self.settings.device})")
            return ""

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self):
        """Free the loaded model (used when the UI discards a loaded model)."""
        if getattr(self, "_closed", False) or getattr(self, "model", None) is None:
            return
        self._closed = True
        try:
            self.model.close()
        except Exception as exc:  # noqa: BLE001 — teardown is best-effort
            logger.debug(f"Error closing model: {exc}")
