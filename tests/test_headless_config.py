"""Tests for --headless and --config CLI options."""

import json
import os
from unittest.mock import MagicMock, patch


class TestHeadlessMode:
    """Test headless mode in transcribe.main()."""

    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.settings.load_settings")
    def test_headless_with_saved_settings(self, mock_load, mock_transcriber_cls):
        """Headless mode with valid saved settings starts transcriber."""
        from faster_whisper_hotkey.settings import Settings
        from faster_whisper_hotkey.transcribe import main

        expected_settings = Settings(
            device_name="test_dev",
            model_type="whisper",
            model_name="small",
            compute_type="int8",
            device="cpu",
            language="en",
            hotkey="pause",
        )
        mock_load.return_value = expected_settings
        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber

        main(headless=True)

        mock_load.assert_called_once_with(None)
        mock_transcriber_cls.assert_called_once_with(expected_settings)
        mock_transcriber.run.assert_called_once()

    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.settings.load_settings")
    def test_headless_without_saved_settings(self, mock_load, mock_transcriber_cls):
        """Headless mode without saved settings returns without error."""
        from faster_whisper_hotkey.transcribe import main

        mock_load.return_value = None

        # Should return cleanly, not raise
        main(headless=True)

        mock_load.assert_called_once_with(None)
        mock_transcriber_cls.assert_not_called()

    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.settings.load_settings")
    def test_headless_with_custom_config_path(self, mock_load, mock_transcriber_cls):
        """Headless mode passes custom config path to load_settings."""
        from faster_whisper_hotkey.settings import Settings
        from faster_whisper_hotkey.transcribe import main

        expected_settings = Settings(
            device_name="test_dev",
            model_type="whisper",
            model_name="small",
            compute_type="int8",
            device="cpu",
            language="en",
            hotkey="pause",
        )
        mock_load.return_value = expected_settings
        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber

        main(headless=True, settings_file="/custom/path/settings.json")

        mock_load.assert_called_once_with("/custom/path/settings.json")
        mock_transcriber_cls.assert_called_once_with(expected_settings)
        mock_transcriber.run.assert_called_once()

    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_normal_mode_ignores_headless_false(self, mock_wrapper, mock_transcriber_cls):
        """Non-headless mode still runs curses UI even with settings_file."""
        from faster_whisper_hotkey.settings import Settings
        from faster_whisper_hotkey.transcribe import main

        expected_settings = Settings(
            device_name="test_dev",
            model_type="whisper",
            model_name="small",
            compute_type="int8",
            device="cpu",
            language="en",
        )
        mock_wrapper.return_value = expected_settings
        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber

        main(headless=False, settings_file="/custom/path.json")

        mock_wrapper.assert_called_once()


class TestConfigOption:
    """Test --config option with custom settings file path."""

    def test_load_settings_with_custom_path(self, tmp_path):
        """load_settings reads from custom path when provided."""
        from faster_whisper_hotkey.settings import load_settings

        test_file = str(tmp_path / "custom_settings.json")
        test_data = {
            "device_name": "custom_device",
            "model_type": "parakeet",
            "model_name": "nvidia/parakeet-tdt-0.6b-v3",
            "compute_type": "float16",
            "device": "cpu",
            "language": "en",
            "hotkey": "f8",
        }
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        settings = load_settings(settings_file=test_file)

        assert settings is not None
        assert settings.device_name == "custom_device"
        assert settings.model_type == "parakeet"
        assert settings.hotkey == "f8"

    def test_save_settings_with_custom_path(self, tmp_path):
        """save_settings writes to custom path when provided."""
        from faster_whisper_hotkey.settings import save_settings

        test_file = str(tmp_path / "custom_settings.json")
        settings_dict = {
            "device_name": "saved_device",
            "model_type": "whisper",
            "model_name": "large-v3",
            "compute_type": "float16",
            "device": "cuda",
            "language": "fr",
            "hotkey": "insert",
        }

        save_settings(settings_dict, settings_file=test_file)

        assert os.path.exists(test_file)
        with open(test_file) as f:
            loaded = json.load(f)
        assert loaded == settings_dict

    def test_load_custom_path_not_found(self):
        """load_settings returns None for nonexistent custom path."""
        from faster_whisper_hotkey.settings import load_settings

        settings = load_settings(settings_file="/nonexistent/custom/path.json")
        assert settings is None

    def test_roundtrip_custom_path(self, tmp_path):
        """Save and load from custom path works as roundtrip."""
        from faster_whisper_hotkey.settings import load_settings, save_settings

        test_file = str(tmp_path / "my_custom_config.json")
        original_data = {
            "device_name": "roundtrip_device",
            "model_type": "granite",
            "model_name": "ibm/granite-speech-4.1-2b",
            "compute_type": "bfloat16",
            "device": "cuda",
            "language": "auto",
            "hotkey": "f4",
            "llm_correction_enabled": True,
            "llm_endpoint": "http://localhost:8080",
            "llm_model_name": "llama3",
        }

        save_settings(original_data, settings_file=test_file)
        settings = load_settings(settings_file=test_file)

        assert settings is not None
        assert settings.device_name == "roundtrip_device"
        assert settings.model_type == "granite"
        assert settings.hotkey == "f4"
        assert settings.llm_correction_enabled is True
        assert settings.llm_endpoint == "http://localhost:8080"


class TestCLIParsing:
    """Test that __main__.py correctly parses --headless and --config."""

    def test_headless_flag_parsed(self):
        """--headless flag is parsed correctly."""
        from faster_whisper_hotkey.__main__ import main as cli_main

        with patch("sys.argv", ["faster-whisper-hotkey", "--headless"]), patch(
            "faster_whisper_hotkey.transcribe.main"
        ) as mock_transcribe_main, patch(
            "faster_whisper_hotkey.settings.load_settings", return_value=None
        ):
            cli_main()

            mock_transcribe_main.assert_called_once_with(
                headless=True, settings_file=None
            )

    def test_config_flag_parsed(self):
        """--config flag is parsed and passed correctly."""
        from faster_whisper_hotkey.__main__ import main as cli_main

        custom_path = "/my/custom/config.json"
        with patch(
            "sys.argv",
            ["faster-whisper-hotkey", "--config", custom_path],
        ), patch(
            "faster_whisper_hotkey.transcribe.main"
        ) as mock_transcribe_main:
            cli_main()

            mock_transcribe_main.assert_called_once_with(
                headless=False, settings_file=custom_path
            )

    def test_headless_and_config_combined(self):
        """--headless and --config work together."""
        from faster_whisper_hotkey.__main__ import main as cli_main

        custom_path = "/my/custom/config.json"
        with patch(
            "sys.argv",
            ["faster-whisper-hotkey", "--headless", "--config", custom_path],
        ), patch(
            "faster_whisper_hotkey.transcribe.main"
        ) as mock_transcribe_main:
            cli_main()

            mock_transcribe_main.assert_called_once_with(
                headless=True, settings_file=custom_path
            )

    def test_headless_config_with_debug(self):
        """--debug, --headless, and --config all work together."""
        from faster_whisper_hotkey.__main__ import main as cli_main

        custom_path = "/my/custom/config.json"
        with patch(
            "sys.argv",
            [
                "faster-whisper-hotkey",
                "--debug",
                "--headless",
                "--config",
                custom_path,
            ],
        ), patch(
            "faster_whisper_hotkey.transcribe.main"
        ) as mock_transcribe_main:
            cli_main()

            assert os.environ.get("FASTER_WHISPER_HOTKEY_DEBUG") == "1"
            mock_transcribe_main.assert_called_once_with(
                headless=True, settings_file=custom_path
            )

        os.environ.pop("FASTER_WHISPER_HOTKEY_DEBUG", None)
