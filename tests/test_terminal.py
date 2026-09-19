from unittest.mock import patch

from faster_whisper_hotkey.terminal import (
    TERMINAL_EXACT_IDENTIFIERS,
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
        common = ["kitty", "alacritty", "ghostty", "xterm", "putty", "sakura"]
        for terminal in common:
            assert terminal in TERMINAL_IDENTIFIERS

    def test_short_identifiers_are_word_tiered(self):
        # Short names must be in the word-boundary tier, not the substring tier
        for terminal in ["st", "foot", "tabby", "hyper", "rio"]:
            assert terminal in TERMINAL_EXACT_IDENTIFIERS
            assert terminal not in TERMINAL_IDENTIFIERS


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

    def test_is_terminal_window_with_ghostty(self):
        # Ghostty X11 WM_CLASS (see src/apprt/gtk/winproto/x11.zig in ghostty-org/ghostty):
        # WM_CLASS(STRING) = "ghostty", "com.mitchellh.ghostty"
        classes = ["ghostty", "com.mitchellh.ghostty"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_ghostty_debug_build(self):
        classes = ["ghostty-debug", "com.mitchellh.ghostty-debug"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_term_substring(self):
        classes = ["gnome-terminal-server", "Gnome-terminal"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_st(self):
        classes = ["st", "st"]
        assert is_terminal_window_x11(classes) is True

    def test_st_does_not_match_studio(self):
        classes = ["Studio", "com.obsproject.Studio"]
        assert is_terminal_window_x11(classes) is False

    def test_is_terminal_window_with_foot(self):
        classes = ["foot", "foot"]
        assert is_terminal_window_x11(classes) is True

    def test_foot_does_not_match_footnotes(self):
        classes = ["footnotes", "Footnotes"]
        assert is_terminal_window_x11(classes) is False

    def test_is_terminal_window_with_putty(self):
        classes = ["putty", "PuTTY"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_tabby(self):
        classes = ["tabby", "Tabby"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_hyper(self):
        classes = ["hyper", "Hyper"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_sakura(self):
        classes = ["sakura", "Sakura"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_black_box(self):
        classes = ["blackbox", "Black Box"]
        assert is_terminal_window_x11(classes) is True

    def test_is_terminal_window_with_rio(self):
        # Rio X11 WM_CLASS (rio sets instance=lowercased app id, class="Rio")
        classes = ["rio", "Rio"]
        assert is_terminal_window_x11(classes) is True

    def test_rio_does_not_match_riotclient(self):
        classes = ["riotclient", "RiotClient"]
        assert is_terminal_window_x11(classes) is False

    def test_is_not_terminal_window(self):
        classes = ["firefox", "Google-chrome"]
        assert is_terminal_window_x11(classes) is False

    def test_vscode_window_is_not_terminal(self):
        # WM_CLASS only carries the app class, never the project/file name
        classes = ["codium", "VSCodium"]
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

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_active_window_class_ghostty(self, mock_check_output):
        mock_check_output.side_effect = [
            b"99999999",
            b'WM_CLASS(STRING) = "ghostty", "com.mitchellh.ghostty"',
        ]

        result = get_active_window_class_x11()

        assert result == ["ghostty", "com.mitchellh.ghostty"]


class TestGetFocusedContainerWayland:
    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_focused_container_success(self, mock_check_output):
        _mock_tree = {
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
        mock_check_output.return_value = (
            b'{"type": "root", "nodes": [{"type": "workspace", "nodes": '
            b'[{"type": "window", "app_id": "kitty", "name": "shell", '
            b'"focused": true}] }]}'
        )

        result = get_focused_container_wayland()

        assert result is not None
        assert result["app_id"] == "kitty"

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_get_focused_container_nested(self, mock_check_output):
        _mock_tree = {
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

    def test_is_terminal_with_alacritty_app_id(self):
        container = {"app_id": "alacritty", "name": ""}
        assert is_terminal_window_wayland(container) is True

    def test_is_terminal_with_ghostty_app_id(self):
        # Ghostty Wayland app_id comes from its `class` config option (default: com.mitchellh.ghostty)
        container = {"app_id": "com.mitchellh.ghostty", "name": "Ghostty"}
        assert is_terminal_window_wayland(container) is True

    def test_is_terminal_with_gnome_terminal_app_id(self):
        container = {"app_id": "org.gnome.Terminal", "name": "user@host: /home"}
        assert is_terminal_window_wayland(container) is True

    def test_is_terminal_with_st_app_id(self):
        container = {"app_id": "st", "name": "bash"}
        assert is_terminal_window_wayland(container) is True

    def test_is_terminal_with_rio_app_id(self):
        # Rio Wayland app_id is "Rio" (rio-window's with_name instance, general)
        container = {"app_id": "Rio", "name": "bash"}
        assert is_terminal_window_wayland(container) is True

    def test_rio_does_not_match_riotclient_app_id(self):
        container = {"app_id": "riotclient", "name": "Riot Client"}
        assert is_terminal_window_wayland(container) is False

    def test_st_does_not_match_studio_app_id(self):
        container = {"app_id": "com.obsproject.Studio", "name": "OBS Studio"}
        assert is_terminal_window_wayland(container) is False

    def test_foot_does_not_match_footnotes_app_id(self):
        container = {"app_id": "com.example.footnotes", "name": "Notes"}
        assert is_terminal_window_wayland(container) is False

    def test_window_title_alone_is_never_matched(self):
        # A non-terminal app whose title contains a terminal word (VSCode on a
        # project called "term-calc") must not be detected as a terminal
        container = {"app_id": "com.vscodium.codium", "name": "term-calc - VSCodium"}
        assert is_terminal_window_wayland(container) is False

    def test_is_not_terminal(self):
        container = {"app_id": "firefox", "name": "Web Browser"}
        assert is_terminal_window_wayland(container) is False

    def test_none_container(self):
        assert is_terminal_window_wayland(None) is False

    def test_empty_container(self):
        container = {}
        assert is_terminal_window_wayland(container) is False


class TestIsTerminalWindowWaylandXWayland:
    """XWayland containers carry the underlying X11 window ID in 'window'."""

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_xwayland_xterm(self, mock_check_output):
        mock_check_output.return_value = b'WM_CLASS(STRING) = "xterm", "XTerm"'
        container = {"app_id": "xwayland", "name": "user@host: /home", "window": 62914569}
        assert is_terminal_window_wayland(container) is True

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_xwayland_ghostty(self, mock_check_output):
        mock_check_output.return_value = b'WM_CLASS(STRING) = "ghostty", "com.mitchellh.ghostty"'
        container = {"app_id": "xwayland", "name": "ghostty", "window": 62914570}
        assert is_terminal_window_wayland(container) is True

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_xwayland_vscode_is_not_terminal(self, mock_check_output):
        mock_check_output.return_value = b'WM_CLASS(STRING) = "codium", "VSCodium"'
        container = {"app_id": "xwayland", "name": "term-calc - VSCodium", "window": 62914571}
        assert is_terminal_window_wayland(container) is False

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_xwayland_xprop_failure(self, mock_check_output):
        mock_check_output.side_effect = Exception("xprop not found")
        container = {"app_id": "xwayland", "name": "user@host", "window": 62914572}
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
        _mock_tree = {
            "type": "root",
            "nodes": [{"type": "window", "app_id": "kitty", "focused": True}],
        }
        mock_check_output.return_value = (
            b'{"type": "root", "nodes": [{"type": "window", "app_id": "kitty", "focused": true}]}'
        )

        container = get_focused_container_wayland()
        is_terminal = is_terminal_window_wayland(container)

        assert is_terminal is True

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_xwayland_terminal_detection_workflow(self, mock_check_output):
        # Focused XWayland window: sway tree first, then xprop WM_CLASS
        mock_check_output.side_effect = [
            (
                b'{"type": "root", "nodes": [{"type": "window", "app_id": "xwayland", '
                b'"name": "user@host: /home", "window": 62914569, "focused": true}]}'
            ),
            b'WM_CLASS(STRING) = "kitty", "Kitty"',
        ]

        container = get_focused_container_wayland()
        is_terminal = is_terminal_window_wayland(container)

        assert is_terminal is True

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_x11_ghostty_detection_workflow(self, mock_check_output):
        # Full detection path for a focused Ghostty window on X11
        mock_check_output.side_effect = [
            b"12345",
            b'WM_CLASS(STRING) = "ghostty", "com.mitchellh.ghostty"',
        ]

        classes = get_active_window_class_x11()
        is_terminal = is_terminal_window_x11(classes)

        assert is_terminal is True

    @patch("faster_whisper_hotkey.terminal.subprocess.check_output")
    def test_wayland_ghostty_detection_workflow(self, mock_check_output):
        # Full detection path for a focused Ghostty window on Wayland (Sway)
        mock_check_output.return_value = (
            b'{"type": "root", "nodes": [{"type": "window", '
            b'"app_id": "com.mitchellh.ghostty", "name": "Ghostty", "focused": true}]}'
        )

        container = get_focused_container_wayland()
        is_terminal = is_terminal_window_wayland(container)

        assert is_terminal is True
