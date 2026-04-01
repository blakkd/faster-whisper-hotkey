"""
Tests for curses-based TUI components in ui.py
"""

import curses
from unittest.mock import MagicMock, patch


class TestGetTextInput:
    """Test the get_text_input function for various scenarios."""

    def _create_mock_stdscr(self, height=24, width=80):
        """Helper to create a mock stdscr with given dimensions."""
        mock = MagicMock()
        mock.getmaxyx.return_value = (height, width)
        return mock

    @patch("faster_whisper_hotkey.ui.curses")
    def test_empty_default_no_input(self, mock_curses):
        """Test with empty default and user enters nothing then presses Enter."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [13]  # Enter

        result = get_text_input(mock_stdscr, "Enter text:", "")
        assert result == ""

    @patch("faster_whisper_hotkey.ui.curses")
    def test_default_with_no_modification(self, mock_curses):
        """Test with default value that user accepts without changes."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(mock_stdscr, "Enter URL:", "http://localhost:8678/v1")
        assert result == "http://localhost:8678/v1"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_default_is_displayed_on_startup(self, mock_curses):
        """Verify default text is displayed after prompt on same line.

        This test verifies the fix for the bug where default values were not
        rendered on screen initially, causing the cursor to appear in the middle
        of nowhere and backspace behaving unexpectedly.
        """
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [13]

        default_url = "http://localhost:8678/v1"
        prompt = "Endpoint URL:"
        get_text_input(mock_stdscr, prompt, default_url)

        calls = mock_stdscr.addstr.call_args_list
        addstr_calls_str = str(calls)

        assert default_url in addstr_calls_str, "Default value was not displayed"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_cursor_position_initialized_correctly(self, mock_curses):
        """Test cursor starts at end of default text (after prompt on same line)."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [13]

        default_text = "test123"
        prompt = "Enter:"
        get_text_input(mock_stdscr, prompt, default_text)

        move_calls = mock_stdscr.move.call_args_list
        y_prompt_moves = [call for call in move_calls if call[0][0] == 11]
        assert len(y_prompt_moves) > 0, "move() should be called"

        columns = [call[0][1] for call in y_prompt_moves]
        cursor_after_prompt = len(prompt) + len(default_text)
        assert cursor_after_prompt in columns, (
            f"Cursor at prompt_len({len(prompt)}) + default_len({len(default_text)})"
        )

    @patch("faster_whisper_hotkey.ui.curses")
    def test_backspace_from_end_of_default(self, mock_curses):
        """Test backspace deletes last character from default value."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [127, 13]

        result = get_text_input(mock_stdscr, "Enter:", "hello")
        assert result == "hell"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_backspace_from_empty_raises_no_error(self, mock_curses):
        """Test backspace on empty input does nothing (no crash)."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [127, 13]

        result = get_text_input(mock_stdscr, "Enter:", "")
        assert result == ""

    @patch("faster_whisper_hotkey.ui.curses")
    def test_backspace_from_beginning_jumps_nothing(self, mock_curses):
        """Test backspace at position 0 is ignored."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [127, 127, 127, 13]

        result = get_text_input(mock_stdscr, "Enter:", "")
        assert result == ""

    @patch("faster_whisper_hotkey.ui.curses")
    def test_escape_returns_none(self, mock_curses):
        """Test ESC key returns None."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [27]

        result = get_text_input(mock_stdscr, "Enter:", "default")
        assert result is None

    @patch("faster_whisper_hotkey.ui.curses")
    def test_typing_after_default(self, mock_curses):
        """Test typing additional characters after default value."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [120, 121, 122, 13]

        result = get_text_input(mock_stdscr, "Enter:", "hello")
        assert result == "helloxyz"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_typing_correctly_appends_to_default(self, mock_curses):
        """Verify characters are appended correctly to default value."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [97, 98, 99, 13]

        result = get_text_input(mock_stdscr, "Enter:", "default")
        assert result == "defaultabc"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_replace_default_with_new_input(self, mock_curses):
        """Test deleting default and typing new value."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        backspaces = [127] * 5
        new_text = [ord(c) for c in "world"]
        mock_stdscr.getch.side_effect = backspaces + new_text + [13]

        result = get_text_input(mock_stdscr, "Enter:", "hello")
        assert result == "world"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_left_right_navigation(self, mock_curses):
        """Test cursor navigation with left/right arrow keys."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [
            curses.KEY_LEFT,
            curses.KEY_LEFT,
            127,
            curses.KEY_RIGHT,
            88,
            13,
        ]

        result = get_text_input(mock_stdscr, "Enter:", "abcde")
        assert result == "abcdX"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_input_truncated_to_width(self, mock_curses):
        """Test input longer than terminal width is truncated for display."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=10)
        mock_stdscr.getch.side_effect = [13]

        long_default = "a" * 50
        result = get_text_input(mock_stdscr, "Enter:", long_default)

        assert result == long_default

    @patch("faster_whisper_hotkey.ui.curses")
    def test_prompt_truncated_to_width(self, mock_curses):
        """Test prompt longer than terminal width is truncated."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(width=10)
        mock_stdscr.getch.side_effect = [13]

        long_prompt = "This is a very long prompt that exceeds width"
        result = get_text_input(mock_stdscr, long_prompt, "")

        assert result == ""

    @patch("faster_whisper_hotkey.ui.curses")
    def test_llm_endpoint_url_default(self, mock_curses):
        """Test realistic Endpoint URL input scenario."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [13]

        result = get_text_input(
            mock_stdscr,
            "Endpoint URL (e.g., http://localhost:8678/v1):",
            "http://localhost:8678/v1",
        )
        assert result == "http://localhost:8678/v1"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_llm_model_name_empty_default(self, mock_curses):
        """Test LLM model name input with empty default."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [ord(c) for c in "mistral"] + [13]

        result = get_text_input(mock_stdscr, "LLM model name:", "")
        assert result == "mistral"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_cursor_not_past_end_after_typing(self, mock_curses):
        """Test cursor position never exceeds text length."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [65, curses.KEY_RIGHT, 13]

        result = get_text_input(mock_stdscr, "Enter:", "")
        assert result == "A"


class TestCursesMenu:
    """Test the curses_menu function."""

    @patch("faster_whisper_hotkey.ui.curses")
    def test_simple_selection(self, mock_curses):
        """Test selecting first option with Enter."""
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        mock_stdscr.getch.side_effect = [13]

        options = ["Option 1", "Option 2", "Option 3"]
        result = curses_menu(mock_stdscr, "Title", options)

        assert result == "Option 1"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_select_second_option(self, mock_curses):
        """Test navigating down and selecting."""
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        # KEY_DOWN is 258 in curses; when mocked, we need to use the actual int
        # so that comparison works with the mocked module's KEY_DOWN
        mock_curses.KEY_DOWN = 258
        mock_stdscr.getch.side_effect = [258, 13]

        options = ["Option 1", "Option 2", "Option 3"]
        result = curses_menu(mock_stdscr, "Title", options)

        assert result == "Option 2"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_escape_aborts(self, mock_curses):
        """Test ESC key returns None."""
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        mock_stdscr.getch.side_effect = [27]

        result = curses_menu(mock_stdscr, "Title", ["Opt1", "Opt2"])
        assert result is None

    @patch("faster_whisper_hotkey.ui.curses")
    def test_terminal_too_small(self, mock_curses):
        """Test warning when terminal cannot display menu."""
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (3, 20)
        mock_stdscr.getch.side_effect = [27]

        result = curses_menu(
            mock_stdscr, "Title", ["Opt1", "Opt2"], message="Very long message..."
        )
        assert result is None
