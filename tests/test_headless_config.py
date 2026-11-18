"""Tests for --headless and --config CLI options."""

import os
from unittest.mock import MagicMock, patch


def _new_settings():
    from faster_whisper_hotkey.settings import Settings

    return Settings(
        device_name="test_dev",
        model_repo="handy-computer/whisper-small-gguf",
        model_file="whisper-small-Q8_0.gguf",
        device="cpu",
        language="en",
        hotkey="pause",
    )


class TestHeadlessMode:
    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.settings.load_settings")
    def test_headless_with_saved_settings(self, mock_load, mock_transcriber_cls):
        from faster_whisper_hotkey.transcribe import main

        expected = _new_settings()
        mock_load.return_value = expected
        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber

        main(headless=True)

        mock_load.assert_called_once_with(None)
        # headless has no preloaded model -> the transcriber loads from settings
        mock_transcriber_cls.assert_called_once_with(expected, None)
        mock_transcriber.run.assert_called_once()

    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.settings.load_settings")
    def test_headless_without_saved_settings(self, mock_load, mock_transcriber_cls):
        from faster_whisper_hotkey.transcribe import main

        mock_load.return_value = None
        main(headless=True)
        mock_load.assert_called_once_with(None)
        mock_transcriber_cls.assert_not_called()

    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.settings.load_settings")
    def test_headless_with_custom_config_path(self, mock_load, mock_transcriber_cls):
        from faster_whisper_hotkey.transcribe import main

        expected = _new_settings()
        mock_load.return_value = expected
        mock_transcriber_cls.return_value = MagicMock()

        main(headless=True, settings_file="/custom/path/settings.json")
        mock_load.assert_called_once_with("/custom/path/settings.json")
        mock_transcriber_cls.assert_called_once_with(expected, None)

    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_normal_mode_runs_curses_then_transcriber(self, mock_wrapper, mock_transcriber_cls):
        from faster_whisper_hotkey.transcribe import main

        expected = _new_settings()
        loaded = object()
        mock_wrapper.return_value = (expected, loaded)
        mock_transcriber_cls.return_value = MagicMock()

        main(headless=False, settings_file="/x.json")

        mock_wrapper.assert_called_once()
        mock_transcriber_cls.assert_called_once_with(expected, loaded)


class TestConfigOption:
    def test_load_settings_with_custom_path(self, tmp_path):
        from faster_whisper_hotkey.settings import load_settings

        f = tmp_path / "custom_settings.json"
        f.write_text(
            '{"device_name": "custom_device", "model_repo": "handy-computer/parakeet-tdt-0.6b-v3-gguf", '
            '"model_file": "parakeet-tdt-0.6b-v3-F32.gguf", "device": "cpu", "language": "en", "hotkey": "f8"}'
        )
        settings = load_settings(settings_file=str(f))
        assert settings is not None
        assert settings.device_name == "custom_device"
        assert settings.model_repo == "handy-computer/parakeet-tdt-0.6b-v3-gguf"
        assert settings.hotkey == "f8"

    def test_save_settings_with_custom_path(self, tmp_path):
        import json

        from faster_whisper_hotkey.settings import save_settings

        f = str(tmp_path / "custom_settings.json")
        data = {
            "device_name": "saved_device",
            "model_repo": "handy-computer/whisper-large-v3-gguf",
            "model_file": "whisper-large-v3-Q8_0.gguf",
            "device": "cuda",
            "language": "fr",
            "hotkey": "insert",
        }
        save_settings(data, settings_file=f)
        assert os.path.exists(f)
        with open(f) as fh:
            assert json.load(fh) == data

    def test_load_custom_path_not_found(self):
        from faster_whisper_hotkey.settings import load_settings

        assert load_settings(settings_file="/nonexistent/custom/path.json") is None

    def test_roundtrip_custom_path(self, tmp_path):
        from faster_whisper_hotkey.settings import load_settings, save_settings

        f = str(tmp_path / "my_custom_config.json")
        data = {
            "device_name": "roundtrip_device",
            "model_repo": "handy-computer/granite-speech-4.1-2b-gguf",
            "model_file": "granite-speech-4.1-2b-Q8_0.gguf",
            "device": "cuda",
            "language": "en",
            "translate_target": "de",
            "hotkey": "f4",
            "llm_correction_enabled": True,
            "llm_endpoint": "http://localhost:8080",
            "llm_model_name": "llama3",
        }
        save_settings(data, settings_file=f)
        settings = load_settings(settings_file=f)
        assert settings is not None
        assert settings.translate_target == "de"
        assert settings.hotkey == "f4"
        assert settings.llm_correction_enabled is True
        assert settings.llm_endpoint == "http://localhost:8080"


class TestCLIParsing:
    def test_headless_flag_parsed(self):
        from faster_whisper_hotkey.__main__ import main as cli_main

        with (
            patch("sys.argv", ["faster-whisper-hotkey", "--headless"]),
            patch("faster_whisper_hotkey.transcribe.main") as mock_transcribe_main,
            patch("faster_whisper_hotkey.settings.load_settings", return_value=None),
        ):
            cli_main()
            mock_transcribe_main.assert_called_once_with(headless=True, settings_file=None)

    def test_config_flag_parsed(self):
        from faster_whisper_hotkey.__main__ import main as cli_main

        custom_path = "/my/custom/config.json"
        with (
            patch("sys.argv", ["faster-whisper-hotkey", "--config", custom_path]),
            patch("faster_whisper_hotkey.transcribe.main") as mock_transcribe_main,
        ):
            cli_main()
            mock_transcribe_main.assert_called_once_with(headless=False, settings_file=custom_path)

    def test_headless_and_config_combined(self):
        from faster_whisper_hotkey.__main__ import main as cli_main

        custom_path = "/my/custom/config.json"
        with (
            patch("sys.argv", ["faster-whisper-hotkey", "--headless", "--config", custom_path]),
            patch("faster_whisper_hotkey.transcribe.main") as mock_transcribe_main,
        ):
            cli_main()
            mock_transcribe_main.assert_called_once_with(headless=True, settings_file=custom_path)

    def test_headless_config_with_debug(self):
        from faster_whisper_hotkey.__main__ import main as cli_main

        custom_path = "/my/custom/config.json"
        with (
            patch("sys.argv", ["faster-whisper-hotkey", "--debug", "--headless", "--config", custom_path]),
            patch("faster_whisper_hotkey.transcribe.main") as mock_transcribe_main,
        ):
            cli_main()
            assert os.environ.get("FASTER_WHISPER_HOTKEY_DEBUG") == "1"
            mock_transcribe_main.assert_called_once_with(headless=True, settings_file=custom_path)

        os.environ.pop("FASTER_WHISPER_HOTKEY_DEBUG", None)
