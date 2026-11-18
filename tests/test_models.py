"""Tests for models.py (ModelWrapper on the transcribe.cpp engine).

The engine itself is exercised by tests/test_model_all_configs.py; these unit
tests verify the settings -> engine plumbing: repo/file load, strict backend
selection, capabilities-driven language/translation mapping, the output
quirks (sentence casing), and the error paths.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from faster_whisper_hotkey.settings import Settings

_started_patches: list = []


@pytest.fixture(autouse=True)
def _clear_support_model_cache():
    import faster_whisper_hotkey.models as m

    m._support_models.clear()
    yield
    for p in _started_patches:
        p.stop()
    _started_patches.clear()
    m._support_models.clear()


def make_settings(**over) -> Settings:
    data = {
        "device_name": "",
        "model_repo": "handy-computer/parakeet-ultra-gguf",
        "model_file": "parakeet-ultra-Q8_0.gguf",
        "device": "cpu",
        "language": "auto",
    }
    data.update(over)
    return Settings(**data)


def make_engine(**attrs):
    """A MagicMock standing in for the transcribe_cpp module."""
    engine = MagicMock()
    engine.backend_available.return_value = True

    session = MagicMock()
    session.run.return_value = SimpleNamespace(text=attrs.get("text", "mock transcription"), language="en")
    engine.model = MagicMock()
    engine.model.session.return_value = session
    engine.model.variant = attrs.get("variant", "tdt-0.6b-ultra")
    engine.model.arch = attrs.get("arch", "parakeet")
    engine.model.capabilities = SimpleNamespace(
        languages=attrs.get("languages", ("en", "de", "fr")),
        supports_language_detect=attrs.get("supports_language_detect", True),
        supports_translate=attrs.get("supports_translate", False),
        translate_target_languages=attrs.get("translate_target_languages", ()),
        max_audio_ms=attrs.get("max_audio_ms", 0),
        native_sample_rate=16000,
    )
    engine.Model.return_value = engine.model
    return engine


def build(settings: Settings, engine: MagicMock | None = None):
    """Instantiate ModelWrapper against a fully mocked engine (stays patched for the test)."""
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
    return ModelWrapper(settings), eng


class TestModelLoad:
    def test_loads_by_repo_and_file(self):
        from faster_whisper_hotkey.models import ModelWrapper

        engine = make_engine()
        settings = make_settings(model_repo="handy-computer/whisper-small-gguf", model_file="whisper-small-Q8_0.gguf")
        with (
            patch("faster_whisper_hotkey.models.ensure_gguf", return_value="/tmp/fake.gguf") as dl,
            patch("faster_whisper_hotkey.models.transcribe_cpp", engine),
            patch("faster_whisper_hotkey.models.ModelWrapper._warmup"),
        ):
            wrapper = ModelWrapper(settings)

        dl.assert_called_once_with("handy-computer/whisper-small-gguf", "whisper-small-Q8_0.gguf")
        engine.Model.assert_called_once_with("/tmp/fake.gguf", backend="cpu")
        assert wrapper.name == "whisper-small"
        assert wrapper.variant == "tdt-0.6b-ultra"  # echoed from the engine strings
        assert wrapper.arch == "parakeet"

    def test_sentence_case_true_only_for_nar_files(self):
        from faster_whisper_hotkey.models import _needs_sentence_case

        assert _needs_sentence_case("granite-speech-4.1-2b-nar-BF16.gguf") is True
        assert _needs_sentence_case("parakeet-ultra-Q8_0.gguf") is False
        assert _needs_sentence_case("granite-speech-5.0-470m-turboctc-nc-Q4_K_M.gguf") is False  # "nc", not "nar"

    def test_max_audio_ms_recorded(self):
        wrapper, _ = build(make_settings(), make_engine(max_audio_ms=400_000))
        assert wrapper.max_audio_ms == 400_000

    def test_backend_maps_device(self):
        for device in ("cpu", "cuda", "vulkan"):
            wrapper, eng = build(make_settings(device=device))
            assert wrapper.backend == device
            eng.Model.assert_called_once_with("/tmp/fake.gguf", backend=device)

    def test_unavailable_backend_raises_actionably(self):
        from faster_whisper_hotkey.models import ModelWrapper

        engine = make_engine()
        engine.backend_available.side_effect = lambda b: b != "cuda"
        with patch("faster_whisper_hotkey.models.transcribe_cpp", engine), pytest.raises(ValueError, match="not available"):
            ModelWrapper(make_settings(device="cuda"))
        engine.Model.assert_not_called()

    def test_gpu_oom_raises_actionably_without_retry(self):
        from transcribe_cpp.errors import OutOfMemory

        engine = make_engine()
        engine.Model.side_effect = OutOfMemory("out of memory (status 7)")
        with pytest.raises(OutOfMemory, match="GPU is likely full|free VRAM"):
            build(make_settings(device="cuda", model_file="parakeet-ultra-Q4_K_M.gguf"), engine)
        engine.Model.assert_called_once()

    def test_cpu_oom_raises_actionably(self):
        from transcribe_cpp.errors import OutOfMemory

        engine = make_engine()
        engine.Model.side_effect = OutOfMemory("out of memory (status 7)")
        with pytest.raises(OutOfMemory, match="RAM is full"):
            build(make_settings(device="cpu"), engine)
        engine.Model.assert_called_once()

    def test_warmup_runs_real_clip_language_independent(self):
        for device in ("cuda", "cpu"):
            with patch("faster_whisper_hotkey.models.ModelWrapper.transcribe") as mock_tr:
                build(make_settings(device=device))
            mock_tr.assert_called_once()
            args, kwargs = mock_tr.call_args
            dummy = args[0]
            assert dummy.dtype == np.float32 and dummy.any()
            assert abs(len(dummy) - 2 * 16000) < 2000
            # warmup does not depend on a chosen language (model not configured yet)
            assert "language" not in kwargs and (len(args) == 1 or args[1] != "auto")

    def test_warmup_failure_is_not_fatal(self):
        with patch("faster_whisper_hotkey.models.ModelWrapper.transcribe", side_effect=RuntimeError("boom")):
            wrapper, _ = build(make_settings())
        assert wrapper.model is not None

    def test_warmup_debug_logs_heard_and_clean(self, caplog):
        with (
            caplog.at_level(logging.DEBUG, logger="faster_whisper_hotkey.models"),
            patch("faster_whisper_hotkey.models.ModelWrapper.transcribe", return_value=" never gonna give you up "),
        ):
            build(make_settings())
        assert "Warmup heard: 'never gonna give you up'" in caplog.text
        caplog.clear()
        with (
            caplog.at_level(logging.DEBUG, logger="faster_whisper_hotkey.models"),
            patch("faster_whisper_hotkey.models.ModelWrapper.transcribe", return_value=""),
        ):
            build(make_settings())
        assert "heard nothing" in caplog.text


class TestCapabilitiesMenu:
    def test_auto_offered_when_model_detects_language(self):
        wrapper, _ = build(make_settings(), make_engine(languages=("en", "de"), supports_language_detect=True))
        assert wrapper.transcription_languages() == ["auto", "en", "de"]

    def test_auto_not_offered_without_detection(self):
        wrapper, _ = build(make_settings(), make_engine(languages=("en",), supports_language_detect=False))
        assert wrapper.transcription_languages() == ["en"]

    def test_translation_targets_and_translate_flag(self):
        engine = make_engine(supports_translate=True, translate_target_languages=("en", "de"), languages=("en", "de", "fr"))
        wrapper, _ = build(make_settings(), engine)
        assert wrapper.supports_translate is True
        assert wrapper.translation_target_languages() == ["en", "de"]
        assert wrapper.translation_source_languages() == ["en", "de", "fr"]

    def test_no_translate(self):
        wrapper, _ = build(make_settings(), make_engine())
        assert wrapper.supports_translate is False
        assert wrapper.translation_target_languages() == []


class TestRunKwargs:
    def test_transcribe_auto(self):
        wrapper, _ = build(make_settings())
        assert wrapper._run_kwargs("auto", "") == {"task": "transcribe", "language": None, "target_language": None}
        assert wrapper._run_kwargs("", "") == {"task": "transcribe", "language": None, "target_language": None}

    def test_transcribe_explicit(self):
        wrapper, _ = build(make_settings())
        assert wrapper._run_kwargs("fr", "") == {"task": "transcribe", "language": "fr", "target_language": None}

    def test_translate(self):
        wrapper, _ = build(make_settings())
        assert wrapper._run_kwargs("en", "de") == {"task": "translate", "language": "en", "target_language": "de"}


class TestTranscribe:
    def _wrapper_text(self, **engine_attrs):
        engine = make_engine(**engine_attrs)
        wrapper, eng = build(make_settings(), engine)
        eng.model.session.return_value.run.reset_mock()  # drop the load-time warmup call
        return wrapper, eng

    def test_returns_text(self):
        wrapper, _ = self._wrapper_text(text="hello there")
        assert wrapper.transcribe(np.ones(1600, dtype=np.float32), 16000) == "hello there"

    def test_nar_sentence_casing_applied(self):
        wrapper, _ = self._wrapper_text(
            text="hello world. this is a test",
        )
        # force the nar path via a nar file name
        wrapper._sentence_case = True
        assert wrapper.transcribe(np.ones(1600, dtype=np.float32), 16000, language="en") == "Hello world. This is a test"

    def test_over_long_input_rejected_before_run(self):
        wrapper, eng = self._wrapper_text(max_audio_ms=400_000)
        assert wrapper.transcribe(np.ones(401 * 16000, dtype=np.float32), 16000) == ""
        eng.model.session.return_value.run.assert_not_called()

    def test_input_too_long_returns_empty(self):
        from transcribe_cpp.errors import InputTooLong

        wrapper, eng = self._wrapper_text()
        eng.model.session.return_value.run.side_effect = InputTooLong("too long")
        assert wrapper.transcribe(np.ones(16000, dtype=np.float32), 16000) == ""

    def test_output_truncated_keeps_partial(self):
        from transcribe_cpp.errors import OutputTruncated

        wrapper, eng = self._wrapper_text()
        exc = OutputTruncated("budget")
        exc.partial_result = SimpleNamespace(text="partial transcript")  # pyright: ignore[reportAttributeAccessIssue]
        eng.model.session.return_value.run.side_effect = exc
        assert wrapper.transcribe(np.ones(16000, dtype=np.float32), 16000) == "partial transcript"

    def test_generic_error_returns_empty(self):
        wrapper, eng = self._wrapper_text()
        eng.model.session.return_value.run.side_effect = RuntimeError("boom")
        assert wrapper.transcribe(np.ones(1600, dtype=np.float32), 16000) == ""


class TestClose:
    def test_close_is_idempotent(self):
        wrapper, eng = build(make_settings())
        wrapper.close()
        wrapper.close()
        eng.model.close.assert_called_once()


class TestLoadVadModel:
    def test_uses_catalog_constants_and_caches_per_backend(self):
        from faster_whisper_hotkey import catalog

        engine = make_engine()
        with patch("faster_whisper_hotkey.models.transcribe_cpp", engine), patch(
            "faster_whisper_hotkey.models.ensure_gguf", return_value="/tmp/vad.gguf"
        ) as mock_dl:
            import faster_whisper_hotkey.models as m

            m.load_vad_model("cpu")
            m.load_vad_model("cpu")
            m.load_vad_model("cuda")
        assert engine.Model.call_count == 2  # cpu once, cuda once
        mock_dl.assert_any_call(catalog.VAD_REPO, catalog.VAD_FILE)
