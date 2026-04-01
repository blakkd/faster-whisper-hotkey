"""
Edge case tests for ui.py to handle any terminal window size.
These tests simulate exotic scenarios with extremely narrow or short terminals.
"""

import curses
from unittest.mock import MagicMock, patch


class TestGetTextInputNarrowWidths:
    """Test get_text_input with extremely narrow terminal widths."""

    def _create_mock_stdscr(self, height=24, width=80):
        """Helper to create a mock stdscr with given dimensions."""
        mock = MagicMock()
        mock.getmaxyx.return_value = (height, width)
        return mock

    @patch("faster_whisper_hotkey.ui.curses")
    def test_width_1_terminal(self, mock_curses):
        """Terminal with width 1 - absolute minimum."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=1)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, "URL:", "http://localhost:8678/v1")
        assert result == "http://localhost:8678/v1"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_width_2_terminal(self, mock_curses):
        """Terminal with width 2 - barely usable."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=2)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, "Prompt:", "default")
        assert result == "default"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_prompt_longer_than_terminal(self, mock_curses):
        """Prompt longer than terminal width - should truncate gracefully."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=5)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(
            mock_stdscr, "Endpoint URL: ", "http://localhost:8678/v1"
        )
        assert result == "http://localhost:8678/v1"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_prompt_and_default_both_very_long_narrow_terminal(self, mock_curses):
        """Both prompt and default exceed terminal width."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=8)
        mock_stdscr.getch.side_effect = [13]

        long_prompt = "This is an extremely long prompt text: "
        long_default = "http://localhost:8678/v1?param=value&other=data"
        result = get_text_input(mock_stdscr, long_prompt, long_default)
        assert result == long_default

    @patch("faster_whisper_hotkey.ui.curses")
    def test_width_equals_prompt_length(self, mock_curses):
        """Terminal width exactly equals prompt length."""
        from faster_whisper_hotkey.ui import get_text_input

        prompt = "Exact:"
        mock_stdscr = self._create_mock_stdscr(width=len(prompt))
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, prompt, "default")
        assert result == "default"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_width_one_more_than_prompt(self, mock_curses):
        """Terminal width is one character more than prompt."""
        from faster_whisper_hotkey.ui import get_text_input

        prompt = "Input:"
        mock_stdscr = self._create_mock_stdscr(width=len(prompt) + 1)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, prompt, "x")
        assert result == "x"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_width_two_more_than_prompt_with_long_default(self, mock_curses):
        """Limited input space but long default value."""
        from faster_whisper_hotkey.ui import get_text_input

        prompt = "Enter:"
        mock_stdscr = self._create_mock_stdscr(width=len(prompt) + 2)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, prompt, "abcdefghijklmnopqrstuvwxyz")
        assert result == "abcdefghijklmnopqrstuvwxyz"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_cursor_position_with_narrow_width(self, mock_curses):
        """Cursor position must be within valid bounds for narrow terminal."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=10)
        mock_stdscr.getch.side_effect = [13]

        get_text_input(mock_stdscr, "Enter: ", "default_value_here")

        move_calls = mock_stdscr.move.call_args_list
        for call in move_calls:
            y, x = call[0]
            assert x >= 0, f"Cursor X should not be negative: {x}"
            assert x < 10, f"Cursor X should not exceed width-1: {x} >= 10"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_cursor_after_typing_in_narrow_terminal(self, mock_curses):
        """After typing, cursor position stays within bounds on narrow terminal."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=12)
        key_sequence = [ord(c) for c in "abc"]
        key_sequence.append(13)
        mock_stdscr.getch.side_effect = key_sequence

        get_text_input(mock_stdscr, "In: ", "")

        move_calls = mock_stdscr.move.call_args_list
        for call in move_calls:
            y, x = call[0]
            assert 0 <= x < 12, f"Cursor X out of bounds after typing: {x}"


class TestGetTextInputShortHeight:
    """Test get_text_input with very short terminal heights."""

    def _create_mock_stdscr(self, height=24, width=80):
        """Helper to create a mock stdscr with given dimensions."""
        mock = MagicMock()
        mock.getmaxyx.return_value = (height, width)
        return mock

    @patch("faster_whisper_hotkey.ui.curses")
    def test_height_1_terminal(self, mock_curses):
        """Terminal with height 1 - extreme case."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(height=1, width=80)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, "Enter:", "default")
        assert result == "default"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_height_2_terminal(self, mock_curses):
        """Terminal with height 2."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(height=2, width=80)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, "Enter:", "default")
        assert result == "default"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_y_prompt_calculation_various_heights(self, mock_curses):
        """y_prompt calculation should work for any height >= 1."""
        from faster_whisper_hotkey.ui import get_text_input

        for h in range(1, 11):
            mock_stdscr = self._create_mock_stdscr(height=h, width=80)
            mock_stdscr.getch.side_effect = [13]

            try:
                result = get_text_input(mock_stdscr, "P:", "d")
                assert result == "d"
            except Exception as e:
                assert False, f"Failed for height={h}: {e}"


class TestGetTextInputCombinedEdgeCases:
    """Combined edge cases with both narrow width and short height."""

    def _create_mock_stdscr(self, height=24, width=80):
        """Helper to create a mock stdscr with given dimensions."""
        mock = MagicMock()
        mock.getmaxyx.return_value = (height, width)
        return mock

    @patch("faster_whisper_hotkey.ui.curses")
    def test_minimal_terminal_1x1(self, mock_curses):
        """Absolute minimal terminal: 1x1 pixel/char."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(height=1, width=1)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, "X", "D")
        assert result == "D"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_tiny_terminal_3x5(self, mock_curses):
        """Very small terminal: 3x5."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(height=3, width=5)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, "UR L:", "htp://loc:8678/v1")
        assert result == "htp://loc:8678/v1"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_ultra_short_wide_terminal(self, mock_curses):
        """Very short but wide terminal."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(height=2, width=120)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(
            mock_stdscr,
            "Full Endpoint URL: ",
            "http://localhost:8678/v1/chat/completions",
        )
        assert result == "http://localhost:8678/v1/chat/completions"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_ultra_narrow_tall_terminal(self, mock_curses):
        """Very narrow but tall terminal."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(height=100, width=4)
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, "L:", "http://localhost:8678/v1")
        assert result == "http://localhost:8678/v1"


class TestGetTextInputCursorMovementEdgeCases:
    """Edge cases for cursor movement with various text lengths and terminal sizes."""

    def _create_mock_stdscr(self, height=24, width=80):
        """Helper to create a mock stdscr with given dimensions."""
        mock = MagicMock()
        mock.getmaxyx.return_value = (height, width)
        return mock

    @patch("faster_whisper_hotkey.ui.curses")
    def test_insert_in_middle_narrow(self, mock_curses):
        """Insert characters in middle of text with narrow terminal."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_curses.KEY_LEFT = 260
        mock_stdscr = self._create_mock_stdscr(width=80)
        mock_stdscr.getch.side_effect = [260, 260, ord("X"), 13]

        result = get_text_input(mock_stdscr, "T:", "hello")
        assert result == "helXlo"


class TestGetTextInputDisplayTruncation:
    """Test that display truncation works correctly while preserving full text."""

    def _create_mock_stdscr(self, height=24, width=80):
        """Helper to create a mock stdscr with given dimensions."""
        mock = MagicMock()
        mock.getmaxyx.return_value = (height, width)
        return mock

    @patch("faster_whisper_hotkey.ui.curses")
    def test_full_text_preserved_when_truncated_display(self, mock_curses):
        """When display is truncated, full text should still be returned."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=20)
        mock_stdscr.getch.side_effect = [13]

        very_long_url = (
            "http://localhost:8678/v1/chat/completions?model=test&extra=params"
        )
        result = get_text_input(mock_stdscr, "URL: ", very_long_url)
        assert result == very_long_url

    @patch("faster_whisper_hotkey.ui.curses")
    def test_addstr_never_exceeds_width(self, mock_curses):
        """addstr calls should never try to write past terminal width."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=15)
        mock_stdscr.getch.side_effect = [13]

        get_text_input(mock_stdscr, "This is LONG: ", "this_is_also_a_very_long_value")

        for call in mock_stdscr.addstr.call_args_list:
            y, x, text = call[0]
            if text:
                assert x >= 0, f"X position should be non-negative: {x}"
                assert x < 15, f"X position should be within width: {x} >= 15"
                assert x + len(text) <= 15, (
                    f"Text would overflow: pos {x} + len {len(text)} > 15"
                )


class TestCursesMenuEdgeCases:
    """Edge case tests for curses_menu function."""

    @patch("faster_whisper_hotkey.ui.curses")
    def test_menu_1x1_terminal(self, mock_curses):
        """Menu on absolute smallest terminal."""
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (1, 1)
        mock_stdscr.getch.side_effect = [27]

        result = curses_menu(mock_stdscr, "T", ["A", "B"])
        assert result is None

    @patch("faster_whisper_hotkey.ui.curses")
    def test_menu_single_row_terminal(self, mock_curses):
        """Menu on single-row terminal."""
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (1, 80)
        mock_stdscr.getch.side_effect = [27]

        result = curses_menu(mock_stdscr, "", ["Opt1"])
        assert result is None

    @patch("faster_whisper_hotkey.ui.curses")
    def test_menu_single_column_terminal(self, mock_curses):
        """Menu on single-column terminal."""
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 1)
        mock_stdscr.getch.side_effect = [27]

        result = curses_menu(mock_stdscr, "", ["A", "B"])
        assert result is None

    @patch("faster_whisper_hotkey.ui.curses")
    def test_menu_zero_dimensions(self, mock_curses):
        """Menu with zero dimensions handled gracefully."""
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (0, 0)
        mock_stdscr.getch.side_effect = [27]

        result = curses_menu(mock_stdscr, "", ["X"])
        assert result is None


class TestVariousWidthScenarios:
    """Test specific width scenarios that have caused issues."""

    def _create_mock_stdscr(self, height=24, width=80):
        """Helper to create a mock stdscr with given dimensions."""
        mock = MagicMock()
        mock.getmaxyx.return_value = (height, width)
        return mock

    @patch("faster_whisper_hotkey.ui.curses")
    def test_exact_lfs_endpoint_width_issue(self, mock_curses):
        """Reproduce the exact error scenario from the bug report."""
        from faster_whisper_hotkey.ui import get_text_input

        prompt = "Endpoint URL: "
        default = "http://localhost:8678/v1"

        for width in range(1, 50):
            mock_stdscr = self._create_mock_stdscr(width=width)
            mock_stdscr.getch.side_effect = [13]

            try:
                result = get_text_input(mock_stdscr, prompt, default)
                assert result == default
            except Exception as e:
                assert False, f"Failed at width={width}: {type(e).__name__}: {e}"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_various_common_term_widths(self, mock_curses):
        """Test common terminal widths that users might have."""
        from faster_whisper_hotkey.ui import get_text_input

        common_widths = [80, 120, 132, 160, 200, 256, 300, 50, 40, 30, 20, 10]
        prompt = "Configure endpoint: "
        default = "http://localhost:8678/v1"

        for width in common_widths:
            mock_stdscr = self._create_mock_stdscr(width=width)
            mock_stdscr.getch.side_effect = [13]

            result = get_text_input(mock_stdscr, prompt, default)
            assert result == default, f"Failed for width={width}"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_uncommon_mobile_widths(self, mock_curses):
        """Test mobile/screen reader terminal widths."""
        from faster_whisper_hotkey.ui import get_text_input

        mobile_widths = [24, 32, 48, 64]
        prompt = "URL: "
        default = "http://localhost:8678/v1"

        for width in mobile_widths:
            mock_stdscr = self._create_mock_stdscr(width=width)
            mock_stdscr.getch.side_effect = [13]

            result = get_text_input(mock_stdscr, prompt, default)
            assert result == default, f"Failed for mobile width={width}"
