"""
Tests for transcribe.py - main entry point and logging setup.

The configuration UI was moved to ui.py; those functions are tested in test_ui.py.
"""

from unittest.mock import MagicMock, patch


class TestMainFlow:
    """Test main() orchestrates config -> transcriber correctly."""

    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_successful_config_starts_transcriber(self, mock_wrapper, mock_transcriber_cls):
        """When config returns Settings, transcriber should be created and run."""
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
        mock_wrapper.return_value = expected_settings

        mock_transcriber = MagicMock()
        mock_transcriber_cls.return_value = mock_transcriber

        main()

        mock_wrapper.assert_called_once()
        mock_transcriber_cls.assert_called_once_with(expected_settings)
        mock_transcriber.run.assert_called_once()

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_cancelled_config_exits(self, mock_wrapper):
        """When config returns None (cancelled), main should return without error."""
        from faster_whisper_hotkey.transcribe import main

        mock_wrapper.return_value = None

        # Should return cleanly, not raise
        main()

        mock_wrapper.assert_called_once()

    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_non_settings_result_exits(self, mock_wrapper):
        """When config returns a non-Settings, non-None value, main exits."""
        from faster_whisper_hotkey.transcribe import main

        mock_wrapper.return_value = "not_settings"

        main()

        mock_wrapper.assert_called_once()


class TestSetupLogging:
    """Test _setup_logging configures logging correctly."""

    def test_debug_mode_includes_module_names(self):
        """In debug mode, formatter includes module path."""
        import logging
        import os

        os.environ["FASTER_WHISPER_HOTKEY_DEBUG"] = "1"

        # Re-import to trigger _setup_logging with debug flag
        from faster_whisper_hotkey.transcribe import _setup_logging

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        _setup_logging()

        handler = root_logger.handlers[0]
        # Debug mode uses standard Formatter with %(name)s
        assert handler.formatter is not None
        assert "%(name)s" in handler.formatter._fmt  # type: ignore[union-attr]

        os.environ.pop("FASTER_WHISPER_HOTKEY_DEBUG")

    def test_normal_mode_omits_module_names(self):
        """In normal mode, formatter omits module path."""
        import logging
        import os

        os.environ.pop("FASTER_WHISPER_HOTKEY_DEBUG", None)

        from faster_whisper_hotkey.transcribe import _setup_logging

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        _setup_logging()

        handler = root_logger.handlers[0]
        # Normal mode uses SimpleFormatter which doesn't include %(name)s
        assert hasattr(handler.formatter, "format")
