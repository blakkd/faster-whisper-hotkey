"""Tests for models.py (ModelWrapper on the transcribe.cpp engine).

The engine itself is exercised by tests/test_model_all_configs.py; these unit
tests verify the settings -> engine plumbing: variant/quant/backend selection,
language/task mapping per family, and the error paths.
"""

import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from faster_whisper_hotkey.settings import Settings

_started_patches: list = []


@pytest.fixture(autouse=True)
def _clear_support_model_cache():
    """Keep the per-backend support-model cache (VAD/lang-id) from leaking across tests."""
    import faster_whisper_hotkey.models as m

    m._support_models.clear()
    yield
    for p in _started_patches:
        p.stop()
    _started_patches.clear()
    m._support_models.clear()


class MockSettings:
    """Simple mock settings object for testing."""

    def __init__(self, model_type, model_name, device, compute_type=None, language: str | None = "auto"):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language


def make_engine(**attrs):
    """A MagicMock standing in for the transcribe_cpp module.

    Model(...) returns engine.model; its session().run returns an object with
    .text (engine.text_result). backend_available is True for all backends.
    """
    engine = MagicMock()
    engine.backend_available.return_value = True

    session = MagicMock()
    session.run.return_value = SimpleNamespace(text=attrs.get("text", "mock transcription"), language="en")
    engine.model = MagicMock()
    engine.model.session.return_value = session
    engine.model.capabilities = SimpleNamespace(max_audio_ms=attrs.get("max_audio_ms", 0), languages=())
    engine.Model.return_value = engine.model

    # _load_support_model also goes through transcribe_cpp.Model, so the
    # langid session hangs off the same model mock.
    langid_session = MagicMock()
    langid_session.run.return_value = SimpleNamespace(code=attrs.get("lid_code", "de"))
    engine.model.langid_session.return_value.__enter__.return_value = langid_session
    return engine


def build(settings, engine=None):
    """Instantiate ModelWrapper against a fully mocked engine.

    The engine/download mocks stay active for the rest of the test (released
    by the autouse fixture), so wrapper.transcribe() never reaches the real
    engine either.
    """
    from faster_whisper_hotkey.models import ModelWrapper

    eng = engine or make_engine()
    _started_patches.extend(
        [
            patch("faster_whisper_hotkey.models.ensure_gguf", return_value="/tmp/fake.gguf"),
            patch("faster_whisper_hotkey.models.transcribe_cpp", eng),
        ]
    )
    for p in _started_patches:
        p.start()
    wrapper = ModelWrapper(settings)
    return wrapper, eng


class TestModelLoad:
    """ModelWrapper construction: variant, quant, backend."""

    def test_whisper_maps_variant_and_quant(self):
        settings = MockSettings("whisper", "small", "cpu", "int8")
        wrapper, eng = build(settings)

        assert wrapper.model_type == "whisper"
        assert wrapper.variant.variant == "whisper-small"
        assert wrapper.quant == "Q8_0"
        eng.Model.assert_called_once_with("/tmp/fake.gguf", backend="cpu")

    def test_whisper_large_v1_maps_to_plain_large_files(self):
        from faster_whisper_hotkey.gguf_models import repo_and_filename

        repo, filename, _, _ = repo_and_filename("whisper", "large-v1", "float16")
        assert repo == "handy-computer/whisper-large-gguf"
        assert filename == "whisper-large-F16.gguf"

    def test_unknown_compute_type_falls_back_to_default_quant(self):
        wrapper, _ = build(MockSettings("parakeet-ultra", "parakeet-ultra", "cpu", ""))
        assert wrapper.quant == "Q8_0"

    def test_bfloat16_on_nemo_family_falls_back_to_f16(self):
        wrapper, _ = build(MockSettings("canary", "canary-1b-v2", "cpu", "bfloat16"))
        assert wrapper.quant == "F16"

    def test_off_menu_quants_resolve_to_real_files(self):
        # q6_k/q5_k_m are published by every repo but not offered in the TUI
        # menu; hand-written settings / the benchmark matrix request them.
        wrapper, _ = build(MockSettings("whisper", "small", "cpu", "q6_k"))
        assert wrapper.quant == "Q6_K"
        wrapper, _ = build(MockSettings("voxtral", "voxtral-mini-3b-2507", "cpu", "q5_k_m"))
        assert wrapper.quant == "Q5_K_M"

    def test_whisper_small_float32_is_real_f32(self):
        wrapper, _ = build(MockSettings("whisper", "small", "cpu", "float32"))
        assert wrapper.quant == "F32"

    def test_whisper_large_v3_float32_falls_back_to_f16(self):
        # The large-v3 repo publishes no F32 file; float32 maps to F16.
        wrapper, _ = build(MockSettings("whisper", "large-v3", "cpu", "float32"))
        assert wrapper.quant == "F16"

    def test_float32_on_llm_family_falls_back_to_bf16(self):
        # The LLM-head ports ship no F32 files; float32 maps to the native BF16.
        wrapper, _ = build(MockSettings("voxtral", "voxtral-mini-3b-2507", "cpu", "float32"))
        assert wrapper.quant == "BF16"

    def test_vulkan_device_selects_vulkan_backend(self):
        wrapper, eng = build(MockSettings("parakeet", "parakeet-tdt-0.6b-v3", "vulkan", "int8"))
        assert wrapper.backend == "vulkan"
        eng.Model.assert_called_once_with("/tmp/fake.gguf", backend="vulkan")

    def test_unavailable_backend_raises_actionably(self):
        """No silent substitution: a device the library can't serve stops with
        a clear remedy instead of quietly running on another backend."""
        engine = make_engine()
        engine.backend_available.side_effect = lambda b: b != "cuda"
        from faster_whisper_hotkey.models import ModelWrapper

        with patch("faster_whisper_hotkey.models.transcribe_cpp", engine), pytest.raises(ValueError, match="not available"):
            ModelWrapper(cast(Settings, MockSettings("whisper", "small", "cuda", "float16")))
        engine.Model.assert_not_called()

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown whisper model"):
            build(MockSettings("whisper", "nonexistent", "cpu", "int8"))

    def test_gpu_oom_raises_actionably_without_retry(self):
        """A full GPU stops the app with a remedy — no CPU masquerade."""
        from transcribe_cpp.errors import OutOfMemory

        engine = make_engine()
        engine.Model.side_effect = OutOfMemory("out of memory (status 7)")
        with pytest.raises(OutOfMemory, match="GPU is likely full|free VRAM"):
            build(MockSettings("parakeet-ultra", "parakeet-ultra", "cuda", "int4"), engine)
        engine.Model.assert_called_once()

    def test_cpu_oom_raises_actionably(self):
        from transcribe_cpp.errors import OutOfMemory

        engine = make_engine()
        engine.Model.side_effect = OutOfMemory("out of memory (status 7)")
        with pytest.raises(OutOfMemory, match="RAM is full"):
            build(MockSettings("whisper", "small", "cpu", "int8"), engine)
        engine.Model.assert_called_once()

    def test_warmup_always_runs_with_real_clip(self):
        # Warmup is device-independent (GPU: kernel-module init; CPU: pages in
        # the mmap'd weights) and feeds the real bundled clip — silence would
        # warm only the encoder.
        for device in ("cuda", "cpu"):
            with patch("faster_whisper_hotkey.models.ModelWrapper.transcribe") as mock_tr:
                build(MockSettings("whisper", "small", device, "int8"))
            mock_tr.assert_called_once()
            dummy = mock_tr.call_args[0][0]
            assert dummy.dtype == np.float32 and dummy.any()
            assert abs(len(dummy) - 2 * 16000) < 2000  # ~2 s @ 16 kHz

    def test_warmup_failure_is_not_fatal(self):
        with patch("faster_whisper_hotkey.models.ModelWrapper.transcribe", side_effect=RuntimeError("boom")):
            wrapper, _ = build(MockSettings("whisper", "small", "cpu", "int8"))
        assert wrapper.model is not None  # app starts regardless

    def test_warmup_debug_logs_what_was_heard(self, caplog):
        """--debug must always reveal the easter egg, hallucinated or not."""
        with (
            caplog.at_level(logging.DEBUG, logger="faster_whisper_hotkey.models"),
            patch("faster_whisper_hotkey.models.ModelWrapper.transcribe", return_value="  never gonna give you up "),
        ):
            build(MockSettings("whisper", "small", "cpu", "int8"))
        assert "Warmup heard: 'never gonna give you up'" in caplog.text

    def test_warmup_debug_logs_clean_models_too(self, caplog):
        with (
            caplog.at_level(logging.DEBUG, logger="faster_whisper_hotkey.models"),
            patch("faster_whisper_hotkey.models.ModelWrapper.transcribe", return_value=""),
        ):
            build(MockSettings("whisper", "small", "cpu", "int8"))
        assert "heard nothing" in caplog.text


class TestLanguageTaskMapping:
    """Settings language -> engine run kwargs, per family."""

    def _run_kwargs(self, model_type, model_name, language):
        wrapper, eng = build(MockSettings(model_type, model_name, "cpu", "int8", language=language))
        wrapper.transcribe(np.ones(1600, dtype=np.float32), 16000, language=language)
        return eng.model.session.return_value.run.call_args[1]

    def test_whisper_auto_language_passes_none(self):
        kwargs = self._run_kwargs("whisper", "small", "auto")
        assert kwargs == {"task": "transcribe", "language": None, "target_language": None}

    def test_whisper_explicit_language(self):
        kwargs = self._run_kwargs("whisper", "small", "fr")
        assert kwargs["language"] == "fr"

    def test_parakeet_empty_language_is_auto(self):
        kwargs = self._run_kwargs("parakeet", "parakeet-tdt-0.6b-v3", "")
        assert kwargs["language"] is None

    def test_canary_same_pair_transcribes(self):
        kwargs = self._run_kwargs("canary", "canary-1b-v2", "en-en")
        assert kwargs == {"task": "transcribe", "language": "en", "target_language": None}

    def test_canary_mixed_pair_translates(self):
        kwargs = self._run_kwargs("canary", "canary-1b-v2", "en-de")
        assert kwargs == {"task": "translate", "language": "en", "target_language": "de"}

    def test_canary_missing_pair_defaults_to_en(self):
        kwargs = self._run_kwargs("canary", "canary-1b-v2", None)
        assert kwargs["language"] == "en" and kwargs["task"] == "transcribe"

    def test_granite_mixed_pair_translates(self):
        kwargs = self._run_kwargs("granite", "granite-speech-4.1-2b", "en-it")
        assert kwargs == {"task": "translate", "language": "en", "target_language": "it"}

    def test_turboctc_ignores_language(self):
        kwargs = self._run_kwargs("granite-turboctc", "granite-speech-5.0-470m-turboctc-nc", "en")
        assert kwargs == {"task": "transcribe", "language": None, "target_language": None}

    def test_qwen3_asr_explicit_and_auto(self):
        assert self._run_kwargs("qwen3-asr", "qwen3-asr-1.7b", "de")["language"] == "de"
        assert self._run_kwargs("qwen3-asr", "qwen3-asr-1.7b", "auto")["language"] is None


class TestCohereLanguageDetection:
    """cohere needs an explicit hint; "auto" resolves via the VoxLingua107 model."""

    def test_explicit_language_used_directly(self):
        wrapper, eng = build(MockSettings("cohere", "cohere-transcribe-03-2026", "cpu", "int8"))
        eng.model.langid_session.reset_mock()  # warmup ran auto-detect with the "" settings language; irrelevant here
        wrapper.transcribe(np.ones(1600, dtype=np.float32), 16000, language="fr")
        kwargs = eng.model.session.return_value.run.call_args[1]
        assert kwargs["language"] == "fr"
        eng.model.langid_session.assert_not_called()

    def test_auto_resolves_via_langid(self):
        wrapper, eng = build(MockSettings("cohere", "cohere-transcribe-03-2026", "cpu", "int8", language="auto"))
        wrapper.transcribe(np.ones(16000, dtype=np.float32), 16000, language="auto")
        kwargs = eng.model.session.return_value.run.call_args[1]
        assert kwargs["language"] == "de"  # engine mock lid_code
        lid = eng.model.langid_session.return_value.__enter__.return_value
        assert set(lid.run.call_args[1]["allowed"]) >= {"en", "de", "zh"}

    def test_unusable_detection_falls_back_to_en(self):
        engine = make_engine(lid_code="sw")  # swahili: not among cohere's 14
        wrapper, eng = build(MockSettings("cohere", "cohere-transcribe-03-2026", "cpu", "int8", language="auto"), engine)
        wrapper.transcribe(np.ones(16000, dtype=np.float32), 16000, language="auto")
        assert eng.model.session.return_value.run.call_args[1]["language"] == "en"

    def test_langid_failure_falls_back_to_en(self):
        from transcribe_cpp.errors import InputTooShort

        engine = make_engine()
        lid = engine.model.langid_session.return_value.__enter__.return_value
        lid.run.side_effect = InputTooShort("too short")
        wrapper, eng = build(MockSettings("cohere", "cohere-transcribe-03-2026", "cpu", "int8", language="auto"), engine)
        wrapper.transcribe(np.ones(160, dtype=np.float32), 16000, language="auto")
        assert eng.model.session.return_value.run.call_args[1]["language"] == "en"


class TestGraniteNarCapitalization:
    """The NAR model emits lowercase; the post-processing pass restores casing."""

    def test_capitalization_applied(self):
        engine = make_engine(text="hello world. this is a test")
        wrapper, _ = build(MockSettings("granite-nar", "granite-speech-4.1-2b-nar", "cpu", "int8"), engine)
        result = wrapper.transcribe(np.ones(1600, dtype=np.float32), 16000, language="en")
        assert result == "Hello world. This is a test"


class TestTranscribeErrors:
    """Error paths return an empty string (and clear logs), never raise."""

    def test_generic_error_returns_empty(self):
        engine = make_engine()
        engine.model.session.return_value.run.side_effect = RuntimeError("boom")
        wrapper, _ = build(MockSettings("whisper", "small", "cpu", "int8"), engine)
        assert wrapper.transcribe(np.ones(1600, dtype=np.float32), 16000) == ""

    def test_over_long_input_rejected_before_engine_call(self):
        engine = make_engine(max_audio_ms=400_000)  # canary-class 400 s cap
        wrapper, _ = build(MockSettings("canary", "canary-1b-v2", "cpu", "int8"), engine)
        engine.model.session.return_value.run.reset_mock()  # the load-time warmup ran once legitimately
        long_audio = np.ones(401 * 16000, dtype=np.float32)
        assert wrapper.transcribe(long_audio, 16000, language="en-en") == ""
        engine.model.session.return_value.run.assert_not_called()

    def test_input_too_long_from_engine_returns_empty(self):
        from transcribe_cpp.errors import InputTooLong

        engine = make_engine()
        engine.model.session.return_value.run.side_effect = InputTooLong("too long")
        wrapper, _ = build(MockSettings("canary", "canary-1b-v2", "cpu", "int8"), engine)
        assert wrapper.transcribe(np.ones(16000, dtype=np.float32), 16000, language="en-en") == ""

    def test_output_truncated_keeps_partial_text(self):
        from transcribe_cpp.errors import OutputTruncated

        engine = make_engine()
        exc = OutputTruncated("budget")
        exc.partial_result = SimpleNamespace(text="partial transcript")  # pyright: ignore[reportAttributeAccessIssue] — engine attaches it dynamically
        engine.model.session.return_value.run.side_effect = exc
        wrapper, _ = build(MockSettings("whisper", "small", "cpu", "int8"), engine)
        assert wrapper.transcribe(np.ones(16000, dtype=np.float32), 16000) == "partial transcript"


class TestLoadVadModel:
    """VAD GGUF loading is cached per backend."""

    def test_cached_per_backend(self):
        engine = make_engine()
        with patch("faster_whisper_hotkey.models.transcribe_cpp", engine), patch(
            "faster_whisper_hotkey.models.ensure_gguf", return_value="/tmp/vad.gguf"
        ):
            import faster_whisper_hotkey.models as m

            m.load_vad_model("cpu")
            m.load_vad_model("cpu")
            m.load_vad_model("cuda")
        assert engine.Model.call_count == 2  # cpu once, cuda once
