import json
from unittest.mock import patch

from faster_whisper_hotkey.settings import (
    Settings,
    load_settings,
    save_settings,
)


def _new(**over):
    """A full current-schema settings dict."""
    data = {
        "device_name": "test_device",
        "model_repo": "handy-computer/parakeet-ultra-gguf",
        "model_file": "parakeet-ultra-Q8_0.gguf",
        "device": "cpu",
        "language": "auto",
    }
    data.update(over)
    return data


class TestSettingsDataclass:
    def test_creation_and_defaults(self):
        s = Settings(device_name="d", model_repo="r", model_file="f.gguf", device="cpu", language="en")
        assert s.hotkey == "pause"
        assert s.translate_target == ""
        assert s.vad_enabled is True
        assert s.llm_correction_enabled is False
        assert s.llm_api_key == ""


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        f = str(tmp_path / "s.json")
        save_settings(_new(device="cuda", hotkey="f8"), settings_file=f)
        s = load_settings(settings_file=f)
        assert s is not None
        assert s.model_repo == "handy-computer/parakeet-ultra-gguf"
        assert s.device == "cuda"
        assert s.hotkey == "f8"

    def test_save_failure_logs_not_raises(self):
        with patch("faster_whisper_hotkey.settings.open", side_effect=OSError("Permission denied")):
            save_settings({"device_name": "test"})  # must not raise

    def test_load_not_found(self):
        assert load_settings(settings_file="/nonexistent/path/settings.json") is None

    def test_load_invalid_json(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text("invalid json {")
        assert load_settings(settings_file=str(f)) is None

    def test_env_prefixed_api_key_preserved(self, tmp_path):
        f = str(tmp_path / "s.json")
        save_settings(_new(llm_correction_enabled=True, llm_api_key="env:MY_API_KEY"), settings_file=f)
        s = load_settings(settings_file=f)
        assert s is not None and s.llm_api_key == "env:MY_API_KEY" and s.llm_correction_enabled is True


class TestLegacyMigration:
    """Pre-catalog settings files (model_type/model_name/compute_type) are rewritten on load."""

    def _load_legacy(self, tmp_path, data):
        f = str(tmp_path / "legacy.json")
        with open(f, "w") as fh:
            json.dump(data, fh)
        return load_settings(settings_file=f)

    def test_whisper_short_name(self, tmp_path):
        s = self._load_legacy(tmp_path, {"device_name": "d", "model_type": "whisper", "model_name": "large-v3", "compute_type": "int8", "device": "cpu", "language": "en"})
        assert s is not None
        assert s.model_repo == "handy-computer/whisper-large-v3-gguf"
        assert s.model_file == "whisper-large-v3-F16.gguf"
        assert s.language == "en"

    def test_upstream_repo_id_migrates(self, tmp_path):
        s = self._load_legacy(tmp_path, {"device_name": "d", "model_type": "cohere", "model_name": "CohereLabs/cohere-transcribe-03-2026", "compute_type": "bfloat16", "device": "cuda", "language": "en"})
        assert s is not None
        assert s.model_repo == "handy-computer/cohere-transcribe-03-2026-gguf"

    def test_distil_maps_to_full_checkpoint(self, tmp_path):
        s = self._load_legacy(tmp_path, {"device_name": "d", "model_type": "whisper", "model_name": "distil-large-v3", "compute_type": "int8", "device": "cpu", "language": "en"})
        assert s is not None
        assert s.model_repo == "handy-computer/whisper-large-v3-gguf"

    def test_pair_language_split_translation(self, tmp_path):
        s = self._load_legacy(tmp_path, {"device_name": "d", "model_type": "canary", "model_name": "canary-1b-v2", "compute_type": "int8", "device": "cpu", "language": "en-de"})
        assert s is not None
        assert s.language == "en"
        assert s.translate_target == "de"

    def test_same_pair_is_transcription(self, tmp_path):
        s = self._load_legacy(tmp_path, {"device_name": "d", "model_type": "canary", "model_name": "canary-1b-v2", "compute_type": "int8", "device": "cpu", "language": "en-en"})
        assert s is not None
        assert s.language == "en"
        assert s.translate_target == ""

    def test_parakeet_legacy_id(self, tmp_path):
        s = self._load_legacy(tmp_path, {"device_name": "d", "model_type": "parakeet", "model_name": "nvidia/parakeet-tdt-0.6b-v3", "compute_type": "float16", "device": "cpu", "language": ""})
        assert s is not None
        assert s.model_repo == "handy-computer/parakeet-tdt-0.6b-v3-gguf"

    def test_unknown_legacy_model_returns_none(self, tmp_path):
        s = self._load_legacy(tmp_path, {"device_name": "d", "model_type": "whisper", "model_name": "totally-made-up", "compute_type": "int8", "device": "cpu", "language": "en"})
        assert s is None

    def test_legacy_file_lands_in_current_schema_on_disk(self, tmp_path):
        # The in-memory Settings is new-schema; that's what matters for booting.
        s = self._load_legacy(tmp_path, {"device_name": "d", "model_type": "whisper", "model_name": "small", "compute_type": "int4", "device": "cuda", "language": "fr"})
        assert s is not None
        assert {s.model_repo, s.model_file, s.device, s.language} == {
            "handy-computer/whisper-small-gguf",
            "whisper-small-F16.gguf",
            "cuda",
            "fr",
        }


class TestNewSchemaRobustness:
    def test_new_schema_passes_through(self, tmp_path):
        f = str(tmp_path / "s.json")
        save_settings(_new(model_repo="handy-computer/medasr-gguf"), settings_file=f)
        s = load_settings(settings_file=f)
        assert s is not None and s.model_repo == "handy-computer/medasr-gguf"

    def test_unknown_extra_keys_are_dropped(self, tmp_path):
        f = str(tmp_path / "s.json")
        save_settings(_new(some_future_field="x", model_type="whisper"), settings_file=f)
        s = load_settings(settings_file=f)  # stale model_type must not break Settings(**fields)
        assert s is not None
        assert not hasattr(s, "model_type")

    def test_missing_required_field_returns_none(self, tmp_path):
        f = str(tmp_path / "s.json")
        with open(f, "w") as fh:
            json.dump({"model_repo": "r", "model_file": "f"}, fh)  # no device_name/device
        assert load_settings(settings_file=f) is None

    def test_file_with_no_model_info_returns_none(self, tmp_path):
        f = str(tmp_path / "s.json")
        with open(f, "w") as fh:
            json.dump({"device_name": "d", "device": "cpu"}, fh)
        assert load_settings(settings_file=f) is None
