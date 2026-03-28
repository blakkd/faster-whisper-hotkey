"""Tests for paste.py operations."""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestPasteX11:
    """Test X11 paste functionality."""

    @patch("faster_whisper_hotkey.paste.keyboard_controller")
    def test_paste_x11_terminal(self, mock_kb):
        """Test Ctrl+Shift+V for terminal on X11."""
        from faster_whisper_hotkey.paste import paste_x11

        with patch("faster_whisper_hotkey.paste.time.sleep"):
            paste_x11(is_terminal=True)

        # Verify the sequence of key presses and releases
        press_calls = mock_kb.press.call_args_list
        release_calls = mock_kb.release.call_args_list

        assert len(press_calls) == 3  # ctrl_l, shift, v
        assert len(release_calls) == 3  # v, shift, ctrl_l

    @patch("faster_whisper_hotkey.paste.keyboard_controller")
    def test_paste_x11_gui(self, mock_kb):
        """Test Ctrl+V for GUI on X11."""
        from faster_whisper_hotkey.paste import paste_x11

        with patch("faster_whisper_hotkey.paste.time.sleep"):
            paste_x11(is_terminal=False)

        press_calls = mock_kb.press.call_args_list
        release_calls = mock_kb.release.call_args_list

        assert len(press_calls) == 2  # ctrl_l, v
        assert len(release_calls) == 2  # v, ctrl_l


class TestPasteWayland:
    """Test Wayland paste functionality."""

    @patch("faster_whisper_hotkey.paste.shutil.which")
    def test_send_key_wayland_wtype_not_found(self, mock_which):
        """Test fallback when wtype is not available."""
        from faster_whisper_hotkey.paste import _send_key_wayland

        mock_which.return_value = None

        result = _send_key_wayland("ctrl+v")

        assert result is False

    @patch("faster_whisper_hotkey.paste.subprocess.run")
    @patch("faster_whisper_hotkey.paste.shutil.which")
    def test_send_key_wayland_success(self, mock_which, mock_run):
        """Test successful wtype execution."""
        from faster_whisper_hotkey.paste import _send_key_wayland

        mock_which.return_value = "/usr/bin/wtype"
        mock_run.return_value = MagicMock(returncode=0)

        result = _send_key_wayland("ctrl+shift+v")

        assert result is True
        mock_run.assert_called_once_with(["/usr/bin/wtype", "ctrl+shift+v"], check=True)

    @patch("faster_whisper_hotkey.paste.subprocess.run")
    @patch("faster_whisper_hotkey.paste.shutil.which")
    def test_send_key_wayland_failure(self, mock_which, mock_run):
        """Test wtype execution failure."""
        from faster_whisper_hotkey.paste import _send_key_wayland

        mock_which.return_value = "/usr/bin/wtype"
        mock_run.side_effect = subprocess.SubprocessError("wtype failed")

        result = _send_key_wayland("ctrl+v")

        assert result is False

    @patch("faster_whisper_hotkey.paste.paste_x11")
    @patch("faster_whisper_hotkey.paste._send_key_wayland")
    def test_paste_wayland_terminal(self, mock_send_key, mock_paste_x11):
        """Test Wayland paste for terminal window."""
        from faster_whisper_hotkey.paste import paste_wayland

        mock_send_key.return_value = True

        paste_wayland(is_terminal=True)

        mock_send_key.assert_called_once_with("ctrl+shift+v")
        mock_paste_x11.assert_not_called()

    @patch("faster_whisper_hotkey.paste.paste_x11")
    @patch("faster_whisper_hotkey.paste._send_key_wayland")
    def test_paste_wayland_gui(self, mock_send_key, mock_paste_x11):
        """Test Wayland paste for GUI window."""
        from faster_whisper_hotkey.paste import paste_wayland

        mock_send_key.return_value = True

        paste_wayland(is_terminal=False)

        mock_send_key.assert_called_once_with("ctrl+v")
        mock_paste_x11.assert_not_called()

    @patch("faster_whisper_hotkey.paste.paste_x11")
    @patch("faster_whisper_hotkey.paste._send_key_wayland")
    def test_paste_wayland_fallback_to_x11(self, mock_send_key, mock_paste_x11):
        """Test Wayland paste falling back to X11 on failure."""
        from faster_whisper_hotkey.paste import paste_wayland

        mock_send_key.return_value = False

        paste_wayland(is_terminal=True)

        mock_send_key.assert_called_once_with("ctrl+shift+v")
        mock_paste_x11.assert_called_once_with(True)


class TestPasteToActiveWindow:
    """Test active window detection and paste routing."""

    @patch("faster_whisper_hotkey.paste.paste_x11")
    @patch("faster_whisper_hotkey.paste.terminal.is_terminal_window_x11")
    @patch("faster_whisper_hotkey.paste.terminal.get_active_window_class_x11")
    def test_paste_to_active_window_x11(
        self, mock_get_class, mock_is_terminal, mock_paste
    ):
        """Test paste routing on X11."""
        from faster_whisper_hotkey.paste import paste_to_active_window

        # Clear WAYLAND_DISPLAY environment variable
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": ""}, clear=False):
            mock_get_class.return_value = ["alacritty", "Alacritty"]
            mock_is_terminal.return_value = True

            paste_to_active_window()

            mock_get_class.assert_called_once()
            mock_is_terminal.assert_called_once_with(["alacritty", "Alacritty"])
            mock_paste.assert_called_once_with(True)

    @patch("faster_whisper_hotkey.paste.paste_wayland")
    @patch("faster_whisper_hotkey.paste.terminal.is_terminal_window_wayland")
    @patch("faster_whisper_hotkey.paste.terminal.get_focused_container_wayland")
    def test_paste_to_active_window_wayland(
        self, mock_get_container, mock_is_terminal, mock_paste
    ):
        """Test paste routing on Wayland."""
        from faster_whisper_hotkey.paste import paste_to_active_window

        # Set WAYLAND_DISPLAY environment variable
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
            mock_get_container.return_value = "terminal.1234"
            mock_is_terminal.return_value = False

            paste_to_active_window()

            mock_get_container.assert_called_once()
            mock_is_terminal.assert_called_once_with("terminal.1234")
            mock_paste.assert_called_once_with(False)

    @patch("faster_whisper_hotkey.paste.paste_x11")
    @patch("faster_whisper_hotkey.paste.terminal.is_terminal_window_x11")
    @patch("faster_whisper_hotkey.paste.terminal.get_active_window_class_x11")
    def test_paste_to_active_window_default_x11(
        self, mock_get_class, mock_is_terminal, mock_paste_x11
    ):
        """Test paste defaults to X11 when WAYLAND_DISPLAY not set."""
        from faster_whisper_hotkey.paste import paste_to_active_window

        mock_get_class.return_value = ["firefox", "Firefox"]
        mock_is_terminal.return_value = False

        with patch.dict(os.environ, {}, clear=True):
            paste_to_active_window()

        mock_get_class.assert_called_once()
        mock_is_terminal.assert_called_once()
        mock_paste_x11.assert_called_once_with(False)
