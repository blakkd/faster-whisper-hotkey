from unittest.mock import patch

import pytest

from faster_whisper_hotkey.terminal import (
    TERMINAL_IDENTIFIERS,
    get_active_window_class_x11,
    get_focused_container_wayland,
    is_terminal_window_wayland,
    is_terminal_window_x11,
)


class TestTerminalIdentifiers:
    def test_identifiers_is_list(self):
        assert isinstance(TERMINAL_IDENTIFIERS, list)

    def test_identifiers_not_empty(self):
        assert len(TERMINAL_IDENTIFIERS) > 0

    def test_common_terminals_present(self):
        common = ["kitty", "alacritty", "xterm"]
        for terminal in common:
            assert terminal in TERMINAL_IDENTIFIERS


class TestIsActiveTerminalWindowX11:
    def test_is_terminal_window_with_kitty(self):
        classes = ["kitty", "Kitty"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_alacritty(self):
        classes = ["Alacritty", "alacritty"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_xterm(self):
        classes = ["xterm", "XTerm"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_term_substring(self):
        classes = ["gnome-terminal-server", "Gnome-terminal"]
        assert is_terminal_window_x11(classes) is True

    def test_is_not_terminal_window(self):
        classes = ["firefox", "Google-chrome"]
        assert is_terminal_window_x11(classes) is False

    def test_empty_classes_list(self):
        assert is_terminal_window_x11([]) is False


class TestGetActiveWindowClassX11:
    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_active_window_class_success(self, mock_check_output):
        mock_check_output.side_effect = [
            b"12345678",
            b'WM_CLASS(STRING) = "kitty", "Kitty"',
        ]

        result = get_active_window_class_x11()

        assert result == ["kitty", "Kitty"]

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_active_window_class_exception(self, mock_check_output):
        mock_check_output.side_effect = Exception("xdotool not found")

        result = get_active_window_class_x11()

        assert result == []

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_active_window_class_multiple_classes(self, mock_check_output):
        mock_check_output.side_effect = [
            b"87654321",
            b'WM_CLASS(STRING) = "gnome-terminal-server", "Gnome-terminal"',
        ]

        result = get_active_window_class_x11()

        assert len(result) == 2


class TestGetFocusedContainerWayland:
    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_focused_container_success(self, mock_check_output):
        mock_tree = {
            "type": "root",
            "nodes": [
                {
                    "type": "workspace",
                    "nodes": [
                        {
                            "type": "window",
                            "app_id": "kitty",
                            "name": "shell",
                            "focused": True,
                        },
                    ],
                }
            ],
        }
        mock_check_output.return_value = b'{"type": "root", "nodes": [{"type": "workspace", "nodes": [{"type": "window", "app_id": "kitty", "name": "shell", "focused": true}] }]}'

        result = get_focused_container_wayland()

        assert result is not None
        assert result["app_id"] == "kitty"

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_focused_container_nested(self, mock_check_output):
        mock_tree = {
            "type": "root",
            "nodes": [
                {
                    "type": "workspace",
                    "nodes": [
                        {
                            "type": "container",
                            "nodes": [
                                {
                                    "type": "window",
                                    "app_id": "alacritty",
                                    "name": "bash",
                                    "focused": True,
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        mock_check_output.return_value = (
            b'{"type": "root", "nodes": [{"type": "workspace", "nodes": [{"type": "container"'
            b', "nodes": [{"type": "window", "app_id": "alacritty", "focused": true}]}]}]}'
        )

        result = get_focused_container_wayland()

        assert result is not None

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_focused_container_exception(self, mock_check_output):
        mock_check_output.side_effect = Exception("swaymsg not found")

        result = get_focused_container_wayland()

        assert result is None

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_focused_container_no_focused(self, mock_check_output):
        mock_check_output.return_value = b'{"type": "root", "nodes": []}'

        result = get_focused_container_wayland()

        assert result is None


class TestIsTerminalWindowWayland:
    def test_is_terminal_with_kitty_app_id(self):
        container = {"app_id": "kitty", "name": ""}
        assert is_terminal_window_wayland(container) is True

    def test_is_terminal_with_alacritty_name(self):
        container = {"app_id": "", "name": "Alacritty"}
        assert is_terminal_window_wayland(container) is True

    def test_is_not_terminal(self):
        container = {"app_id": "firefox", "name": "Web Browser"}
        assert is_terminal_window_wayland(container) is False

    def test_none_container(self):
        assert is_terminal_window_wayland(None) is False

    def test_empty_container(self):
        container = {}
        assert is_terminal_window_wayland(container) is False


class TestTerminalDetectionWorkflow:
    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_x11_terminal_detection_workflow(self, mock_check_output):
        mock_check_output.side_effect = [
            b"12345",
            b'WM_CLASS(STRING) = "alacritty", "Alacritty"',
        ]

        classes = get_active_window_class_x11()
        is_terminal = is_terminal_window_x11(classes)

        assert is_terminal is True

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_wayland_terminal_detection_workflow(self, mock_check_output):
        mock_tree = {
            "type": "root",
            "nodes": [{"type": "window", "app_id": "kitty", "focused": True}],
        }
        mock_check_output.return_value = b'{"type": "root", "nodes": [{"type": "window", "app_id": "kitty", "focused": true}]}'

        container = get_focused_container_wayland()
        is_terminal = is_terminal_window_wayland(container)

        assert is_terminal is True
