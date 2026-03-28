import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from faster_whisper_hotkey.settings import (
    SETTINGS_FILE,
    Settings,
    load_settings,
    save_settings,
)


class TestSettingsDataclass:
    def test_settings_creation(self):
        settings = Settings(
            device_name="test_device",
            model_type="whisper",
            model_name="small",
            compute_type="float16",
            device="cuda",
            language="en",
            hotkey="pause",
        )
        assert settings.device_name == "test_device"
        assert settings.model_type == "whisper"
        assert settings.model_name == "small"
        assert settings.compute_type == "float16"
        assert settings.device == "cuda"
        assert settings.language == "en"
        assert settings.hotkey == "pause"

    def test_settings_defaults(self):
        settings = Settings(
            device_name="test",
            model_type="whisper",
            model_name="small",
            compute_type="float16",
            device="cuda",
            language="en",
        )
        assert settings.hotkey == "pause"


class TestSaveSettings:
    @patch("faster_whisper_hotkey.settings.SETTINGS_FILE", "/tmp/mock_settings.json")
    def test_save_settings_success(self, tmp_path):
        test_file = str(tmp_path / "test_settings.json")
        with patch("faster_whisper_hotkey.settings.SETTINGS_FILE", test_file):
            settings_dict = {
                "device_name": "test_device",
                "model_type": "whisper",
                "model_name": "small",
                "compute_type": "float16",
                "device": "cuda",
                "language": "en",
                "hotkey": "pause",
            }
            save_settings(settings_dict)

            assert os.path.exists(test_file)
            with open(test_file, "r") as f:
                loaded = json.load(f)
            assert loaded == settings_dict

    @patch(
        "faster_whisper_hotkey.settings.open", side_effect=IOError("Permission denied")
    )
    def test_save_settings_failure(self, mock_open):
        settings_dict = {"device_name": "test"}
        save_settings(settings_dict)


class TestLoadSettings:
    @pytest.fixture
    def temp_settings_file(self, tmp_path):
        test_file = str(tmp_path / "test_settings.json")
        original_file = SETTINGS_FILE

        test_data = {
            "device_name": "test_device",
            "model_type": "whisper",
            "model_name": "small",
            "compute_type": "float16",
            "device": "cuda",
            "language": "en",
        }
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        yield test_file

    def test_load_settings_success(self, temp_settings_file):
        with patch("faster_whisper_hotkey.settings.SETTINGS_FILE", temp_settings_file):
            settings = load_settings()

        assert isinstance(settings, Settings)
        assert settings.device_name == "test_device"
        assert settings.model_type == "whisper"
        assert settings.hotkey == "pause"

    def test_load_settings_with_hotkey(self, tmp_path):
        test_file = str(tmp_path / "test_settings.json")
        test_data = {
            "device_name": "test_device",
            "model_type": "whisper",
            "model_name": "small",
            "compute_type": "float16",
            "device": "cuda",
            "language": "en",
            "hotkey": "f4",
        }
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        with patch("faster_whisper_hotkey.settings.SETTINGS_FILE", test_file):
            settings = load_settings()

        assert settings.hotkey == "f4"

    def test_load_settings_not_found(self):
        with patch(
            "faster_whisper_hotkey.settings.SETTINGS_FILE",
            "/nonexistent/path/settings.json",
        ):
            settings = load_settings()

        assert settings is None

    def test_load_settings_invalid_json(self, tmp_path):
        test_file = str(tmp_path / "test_settings.json")
        with open(test_file, "w") as f:
            f.write("invalid json {")

        with patch("faster_whisper_hotkey.settings.SETTINGS_FILE", test_file):
            settings = load_settings()

        assert settings is None


class TestSettingsRoundTrip:
    def test_save_and_load(self, tmp_path):
        test_file = str(tmp_path / "test_settings.json")

        original_data = {
            "device_name": "my_device",
            "model_type": "parakeet",
            "model_name": "nvidia/parakeet-tdt-0.6b-v3",
            "compute_type": "float16",
            "device": "cpu",
            "language": "",
            "hotkey": "f8",
        }

        with patch("faster_whisper_hotkey.settings.SETTINGS_FILE", test_file):
            save_settings(original_data)
            settings = load_settings()

        assert isinstance(settings, Settings)
        assert settings.device_name == "my_device"
        assert settings.model_type == "parakeet"
        assert settings.hotkey == "f8"
