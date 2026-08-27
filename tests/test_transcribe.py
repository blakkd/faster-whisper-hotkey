"""
Tests for transcribe.py - main entry point and logging setup.

The configuration UI was moved to ui.py; those functions are tested in test_ui.py.
"""

import curses
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

    @patch("faster_whisper_hotkey.transcriber.MicrophoneTranscriber")
    @patch("faster_whisper_hotkey.transcribe.curses.wrapper")
    def test_failed_transcriber_startup_exits_cleanly(self, mock_wrapper, mock_transcriber_cls):
        """A transcriber construction error (e.g. unsupported compute type) logs and exits, no traceback."""
        from faster_whisper_hotkey.settings import Settings
        from faster_whisper_hotkey.transcribe import main

        mock_wrapper.return_value = Settings(
            device_name="test_dev",
            model_type="whisper",
            model_name="small",
            compute_type="int8",
            device="cuda",
            language="en",
            hotkey="pause",
        )
        mock_transcriber_cls.side_effect = ValueError("CUDA int8 is not supported on this GPU")

        main()  # must not raise

        mock_transcriber_cls.assert_called_once()

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


class TestReadPendingInput:
    """Test _read_pending_input captures keystrokes that arrived before curses started."""

    def test_non_tty_stdin_returns_empty(self):
        """With non-tty stdin (e.g. pytest capture), must return empty without raising."""
        from faster_whisper_hotkey.transcribe import _read_pending_input

        assert _read_pending_input() == b""

    def test_captures_stale_keypresses(self):
        import os
        import pty
        import termios

        from faster_whisper_hotkey.transcribe import _read_pending_input

        master, slave = pty.openpty()
        # Mirror curses' cbreak mode so a partial line (no newline) is deliverable
        attrs = termios.tcgetattr(slave)
        attrs[3] &= ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(slave, termios.TCSANOW, attrs)

        fake_stdin = os.fdopen(slave, "rb", closefd=False)
        try:
            os.write(master, b"\x1b[B\n")  # stale down-arrow sequence + Enter

            with patch("sys.stdin", fake_stdin):
                data = _read_pending_input()

            assert data == b"\x1b[B\n"
        finally:
            fake_stdin.close()
            os.close(master)
            os.close(slave)

    def test_restores_file_flags(self):
        import fcntl
        import os
        import pty

        from faster_whisper_hotkey.transcribe import _read_pending_input

        master, slave = pty.openpty()
        fake_stdin = os.fdopen(slave, "rb", closefd=False)
        try:
            with patch("sys.stdin", fake_stdin):
                _read_pending_input()

            flags = fcntl.fcntl(fake_stdin, fcntl.F_GETFL)
            assert not (flags & os.O_NONBLOCK)
        finally:
            fake_stdin.close()
            os.close(master)
            os.close(slave)


class TestParseKeySequence:
    """Test _parse_key_sequence translates raw bytes into curses key codes."""

    def _parse(self, data: bytes):
        from faster_whisper_hotkey.transcribe import _parse_key_sequence

        return _parse_key_sequence(data)

    def test_empty(self):
        assert self._parse(b"") == []

    def test_printables(self):
        assert self._parse(b"abc") == [ord("a"), ord("b"), ord("c")]

    def test_enter_variants(self):
        assert self._parse(b"\n") == [10]
        assert self._parse(b"\r") == [13]

    def test_backspace(self):
        assert self._parse(b"\x7f") == [127]

    def test_bare_escape(self):
        assert self._parse(b"\x1b") == [27]

    def test_escape_then_printable(self):
        assert self._parse(b"\x1bx") == [27, ord("x")]

    def test_arrow_keys_csi_mode(self):
        assert self._parse(b"\x1b[A") == [curses.KEY_UP]
        assert self._parse(b"\x1b[B") == [curses.KEY_DOWN]
        assert self._parse(b"\x1b[C") == [curses.KEY_RIGHT]
        assert self._parse(b"\x1b[D") == [curses.KEY_LEFT]

    def test_arrow_keys_application_mode(self):
        assert self._parse(b"\x1bOA") == [curses.KEY_UP]
        assert self._parse(b"\x1bOB") == [curses.KEY_DOWN]
        assert self._parse(b"\x1bOC") == [curses.KEY_RIGHT]
        assert self._parse(b"\x1bOD") == [curses.KEY_LEFT]

    def test_parameterized_sequences_dropped_whole(self):
        assert self._parse(b"\x1b[1~") == []  # Home
        assert self._parse(b"\x1b[3~") == []  # Delete
        assert self._parse(b"\x1b[5~") == []  # Page Up

    def test_mixed_stream(self):
        assert self._parse(b"\x1b[B\n") == [curses.KEY_DOWN, 10]
        assert self._parse(b"A\x1b[BA") == [ord("A"), curses.KEY_DOWN, ord("A")]

    def test_control_chars_dropped(self):
        assert self._parse(b"\x01\x02\x03") == []

    def test_truncated_sequence_emits_no_spurious_escape(self):
        assert self._parse(b"\x1b[") == []
        assert self._parse(b"\x1b[1") == []
        assert self._parse(b"\x1bO") == []


class TestReplayWindow:
    """Test _ReplayWindow serves stale keys before delegating to the real window."""

    def test_replays_queued_keys_then_delegates(self):
        from faster_whisper_hotkey.transcribe import _ReplayWindow

        window = MagicMock()
        window.getch.side_effect = [99]  # a live 'c' after the queued keys
        proxy = _ReplayWindow(window, [curses.KEY_DOWN, 13])

        assert proxy.getch() == curses.KEY_DOWN
        assert proxy.getch() == 13
        assert proxy.getch() == 99
        assert window.getch.call_count == 1

    def test_delegates_other_methods(self):
        from faster_whisper_hotkey.transcribe import _ReplayWindow

        window = MagicMock()
        window.getmaxyx.return_value = (24, 80)
        proxy = _ReplayWindow(window, [])

        assert proxy.getmaxyx() == (24, 80)
        proxy.addstr(0, 0, "x")
        window.addstr.assert_called_once_with(0, 0, "x")


class TestRunConfigScreen:
    """Test the curses.wrapper callback wiring."""

    @patch("faster_whisper_hotkey.ui.config_screen_main")
    def test_returns_ui_result(self, mock_screen_main):
        from faster_whisper_hotkey.transcribe import _run_config_screen

        stdscr = MagicMock()
        mock_screen_main.return_value = "settings"

        assert _run_config_screen(stdscr, "/tmp/settings.json") == "settings"
        mock_screen_main.assert_called_once_with(stdscr, "/tmp/settings.json")

    @patch("faster_whisper_hotkey.transcribe._read_pending_input", return_value=b"")
    @patch("faster_whisper_hotkey.ui.config_screen_main")
    def test_no_stale_keys_passes_window_through(self, mock_screen_main, mock_read):
        from faster_whisper_hotkey.transcribe import _run_config_screen

        stdscr = MagicMock()
        _run_config_screen(stdscr, "/tmp/settings.json")

        assert mock_screen_main.call_args[0][0] is stdscr

    @patch("faster_whisper_hotkey.transcribe._read_pending_input", return_value=b"\x1b[B\n")
    @patch("faster_whisper_hotkey.ui.config_screen_main")
    def test_stale_keys_replayed_through_proxy(self, mock_screen_main, mock_read):
        from faster_whisper_hotkey.transcribe import _ReplayWindow, _run_config_screen

        stdscr = MagicMock()
        _run_config_screen(stdscr, "/tmp/settings.json")

        passed_window = mock_screen_main.call_args[0][0]
        assert isinstance(passed_window, _ReplayWindow)
        assert passed_window.getch() == curses.KEY_DOWN
        assert passed_window.getch() == 10  # \n


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
