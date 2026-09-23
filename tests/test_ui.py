"""
Tests for curses-based TUI components in ui.py
"""

import curses
from unittest.mock import MagicMock, patch

import pytest


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
        assert cursor_after_prompt in columns, f"Cursor at prompt_len({len(prompt)}) + default_len({len(default_text)})"

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
        """Test a bare ESC (nothing following within the window) returns None."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [27, -1]  # ESC, then nothing follows

        result = get_text_input(mock_stdscr, "Enter:", "default")
        assert result is None

    @patch("faster_whisper_hotkey.ui.curses")
    def test_alt_backspace_deletes_previous_word(self, mock_curses):
        """Alt+Backspace (ESC + backspace) deletes the whole previous word, not the screen."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [27, 127, 13]

        result = get_text_input(mock_stdscr, "Enter:", "hello world")
        assert result == "hello"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_alt_backspace_without_space_clears_before_cursor(self, mock_curses):
        """With no space before the cursor, Alt+Backspace clears everything before it."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [27, 127, 13]

        result = get_text_input(mock_stdscr, "Enter:", "s1-mini-GGUF")
        assert result == ""

    @patch("faster_whisper_hotkey.ui.curses")
    def test_alt_backspace_on_empty_input_does_not_cancel(self, mock_curses):
        """Alt+Backspace with nothing to delete keeps the field open instead of cancelling."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [27, 127, 13]

        result = get_text_input(mock_stdscr, "Enter:", "")
        assert result == ""

    @patch("faster_whisper_hotkey.ui.curses")
    def test_alt_backspace_repeated_deletes_word_by_word(self, mock_curses):
        """Repeated Alt+Backspace removes one whole word per press."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [27, 127, 27, 127, 13]

        result = get_text_input(mock_stdscr, "Enter:", "hello world foo")
        assert result == "hello"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_escape_plus_other_key_does_not_cancel(self, mock_curses):
        """ESC followed by a regular key is treated as that key, not as a cancel."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()
        mock_stdscr.getch.side_effect = [27, 97, 13]  # ESC, 'a', Enter

        result = get_text_input(mock_stdscr, "Enter:", "")
        assert result == "a"

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

    @patch("faster_whisper_hotkey.ui.curses")
    def test_footer_displayed_below_prompt(self, mock_curses):
        """A footer hint should be drawn one line below the input line."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()  # 24x80 -> prompt at y=11, footer at y=12
        mock_stdscr.getch.side_effect = [13]

        get_text_input(mock_stdscr, "API key: ", "", footer="env: hint here")

        assert (12, 0, "env: hint here") in [call.args for call in mock_stdscr.addstr.call_args_list]

    @patch("faster_whisper_hotkey.ui.curses")
    def test_no_footer_by_default(self, mock_curses):
        """Without a footer, nothing should be drawn on the line below the prompt."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr()  # 24x80 -> prompt at y=11
        mock_stdscr.getch.side_effect = [13]

        get_text_input(mock_stdscr, "API key: ", "value")

        rows_used = {call.args[0] for call in mock_stdscr.addstr.call_args_list}
        assert 12 not in rows_used

    @patch("faster_whisper_hotkey.ui.curses")
    def test_footer_not_drawn_offscreen(self, mock_curses):
        """On a one-line terminal the footer line would be out of bounds and must be skipped."""
        from faster_whisper_hotkey.ui import get_text_input

        mock_stdscr = self._create_mock_stdscr(height=1, width=80)
        mock_stdscr.getch.side_effect = [13]

        get_text_input(mock_stdscr, "Key: ", "", footer="hint")

        for call in mock_stdscr.addstr.call_args_list:
            assert 0 <= call.args[0] < 1


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

        result = curses_menu(mock_stdscr, "Title", ["Opt1", "Opt2"], message="Very long message...")
        assert result is None


class TestWhisperPrecisionScreen:
    """Test the whisper precision screen int8/CUDA guard (Blackwell sm_12x)."""

    KEY_DOWN = 258
    KEY_UP = 259
    ENTER = 13
    ESC = 27

    def _make_stdscr(self, keys):
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        mock_stdscr.getch.side_effect = list(keys)
        return mock_stdscr

    def _config(self, device="cuda", compute_type=""):
        from faster_whisper_hotkey.ui import ConfigData

        config = ConfigData()
        config.device = device
        config.compute_type = compute_type
        config.model_name = "small"
        return config

    @patch("faster_whisper_hotkey.models._cuda_int8_supported", return_value=False)
    @patch("faster_whisper_hotkey.ui.curses")
    def test_int8_blocked_on_unsupported_cuda(self, mock_curses, mock_supported):
        """Picking int8 on an unsupported CUDA GPU shows a warning and stays on the screen."""
        from faster_whisper_hotkey.ui import ConfigStep, _screen_whisper_precision

        mock_curses.KEY_DOWN = self.KEY_DOWN
        mock_curses.KEY_UP = self.KEY_UP
        # down -> int8, Enter (blocked); up -> float16, Enter (accepted)
        mock_stdscr = self._make_stdscr([self.KEY_DOWN, self.ENTER, self.KEY_UP, self.ENTER])

        config = self._config(device="cuda")
        result = _screen_whisper_precision(mock_stdscr, config)

        assert result is not None
        next_step, config = result
        assert next_step == ConfigStep.WHISPER_LANGUAGE
        assert config.compute_type == "float16"
        assert "not supported on this CUDA GPU" in str(mock_stdscr.addstr.call_args_list)

    @patch("faster_whisper_hotkey.models._cuda_int8_supported", return_value=False)
    @patch("faster_whisper_hotkey.ui.curses")
    def test_int8_blocked_until_escalated_choice(self, mock_curses, mock_supported):
        """Repeated int8 selections keep showing the warning; ESC still aborts."""
        from faster_whisper_hotkey.ui import ConfigStep, _screen_whisper_precision

        mock_curses.KEY_DOWN = self.KEY_DOWN
        mock_curses.KEY_UP = self.KEY_UP
        # down -> int8, Enter (blocked); down stays, Enter (blocked again); ESC
        mock_stdscr = self._make_stdscr([self.KEY_DOWN, self.ENTER, self.KEY_DOWN, self.ENTER, self.ESC])

        config = self._config(device="cuda")
        result = _screen_whisper_precision(mock_stdscr, config)

        assert result == (ConfigStep.INITIAL, config)
        assert config.compute_type == ""
        warnings_shown = str(mock_stdscr.addstr.call_args_list).count("not supported on this CUDA GPU")
        assert warnings_shown >= 2

    @patch("faster_whisper_hotkey.models._cuda_int8_supported", return_value=True)
    @patch("faster_whisper_hotkey.ui.curses")
    def test_int8_allowed_when_supported(self, mock_curses, mock_supported):
        """int8 is accepted without warning when the GPU supports it."""
        from faster_whisper_hotkey.ui import ConfigStep, _screen_whisper_precision

        mock_curses.KEY_DOWN = self.KEY_DOWN
        mock_curses.KEY_UP = self.KEY_UP
        mock_stdscr = self._make_stdscr([self.KEY_DOWN, self.ENTER])

        config = self._config(device="cuda")
        result = _screen_whisper_precision(mock_stdscr, config)

        next_step, config = result
        assert next_step == ConfigStep.WHISPER_LANGUAGE
        assert config.compute_type == "int8"
        assert "not supported" not in str(mock_stdscr.addstr.call_args_list)

    @patch("faster_whisper_hotkey.models._cuda_int8_supported", return_value=False)
    @patch("faster_whisper_hotkey.ui.curses")
    def test_int8_on_cpu_never_blocked(self, mock_curses, mock_supported):
        """The GPU guard only applies to CUDA; CPU int8 is always accepted."""
        from faster_whisper_hotkey.ui import ConfigStep, _screen_whisper_precision

        mock_curses.KEY_DOWN = self.KEY_DOWN
        mock_curses.KEY_UP = self.KEY_UP
        mock_stdscr = self._make_stdscr([self.ENTER])

        config = self._config(device="cpu")
        result = _screen_whisper_precision(mock_stdscr, config)

        next_step, config = result
        assert next_step == ConfigStep.WHISPER_LANGUAGE
        assert config.compute_type == "int8"


class TestCursesMenuNativeHint:
    """Test the ' (native)' hint rendered next to the model's native precision."""

    def _make_stdscr(self, keys):
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        mock_stdscr.getch.side_effect = list(keys)
        return mock_stdscr

    @patch("faster_whisper_hotkey.ui.curses")
    def test_native_option_shows_hint(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = self._make_stdscr([13])  # Enter on the first option

        result = curses_menu(mock_stdscr, "Precision", ["float32", "bfloat16", "int8", "int4"], native="bfloat16")

        assert result == "float32"
        assert "bfloat16 (native)" in str(mock_stdscr.addstr.call_args_list)

    @patch("faster_whisper_hotkey.ui.curses")
    def test_no_hint_when_native_not_offered(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = self._make_stdscr([13])

        result = curses_menu(mock_stdscr, "Precision", ["int8"], native="float16")

        assert result == "int8"
        assert "(native)" not in str(mock_stdscr.addstr.call_args_list)

    @patch("faster_whisper_hotkey.ui.curses")
    def test_no_hint_without_native(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        mock_stdscr = self._make_stdscr([13])

        result = curses_menu(mock_stdscr, "Precision", ["float32", "bfloat16", "int8", "int4"])

        assert result == "float32"
        assert "(native)" not in str(mock_stdscr.addstr.call_args_list)


class TestPrecisionScreensUniformOrder:
    """Precision screens list options in the same order for every model and mark the native one."""

    CANONICAL_ORDER = ("float32", "bfloat16", "float16", "int8", "int4")

    @pytest.mark.parametrize(
        ("screen_name", "device", "expected_options", "native"),
        [
            ("_screen_whisper_precision", "cuda", ["float16", "int8"], "float16"),
            ("_screen_whisper_precision", "cpu", ["int8"], "float16"),
            ("_screen_parakeet_precision", "cuda", ["float32", "bfloat16", "int8", "int4"], "float32"),
            ("_screen_parakeet_precision", "cpu", ["float32", "bfloat16"], "float32"),
            ("_screen_canary_precision", "cuda", ["float32", "bfloat16", "int8", "int4"], "float32"),
            ("_screen_canary_precision", "cpu", ["float32", "bfloat16"], "float32"),
            ("_screen_voxtral_precision", "cuda", ["float32", "bfloat16", "int8", "int4"], "float32"),
            ("_screen_cohere_precision", "cuda", ["float32", "bfloat16", "int8", "int4"], "bfloat16"),
            ("_screen_cohere_precision", "cpu", ["float32", "bfloat16"], "bfloat16"),
            ("_screen_granite_nar_precision", "cuda", ["float32", "bfloat16", "int8", "int4"], "bfloat16"),
            ("_screen_granite_nar_precision", "cpu", ["float32", "bfloat16"], "bfloat16"),
            ("_screen_granite_precision", "cuda", ["float32", "bfloat16", "int8", "int4"], "bfloat16"),
            ("_screen_granite_precision", "cpu", ["float32", "bfloat16"], "bfloat16"),
            ("_screen_granite_turboctc_precision", "cuda", ["float32", "bfloat16", "int8", "int4"], "bfloat16"),
            ("_screen_granite_turboctc_precision", "cpu", ["float32", "bfloat16"], "bfloat16"),
            ("_screen_qwen3_asr_precision", "cuda", ["float32", "bfloat16", "int8", "int4"], "bfloat16"),
            ("_screen_qwen3_asr_precision", "cpu", ["float32", "bfloat16"], "bfloat16"),
        ],
    )
    def test_options_follow_canonical_order_and_mark_native(self, screen_name, device, expected_options, native):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData()
        config.device = device

        captured = {}

        def fake_menu(stdscr, title, options, **kwargs):
            captured["options"] = options
            captured["native"] = kwargs.get("native")
            return options[0]

        with patch.object(ui, "curses_menu", fake_menu):
            result = getattr(ui, screen_name)(MagicMock(), config)

        assert result is not None
        assert captured["options"] == expected_options
        assert expected_options == [p for p in self.CANONICAL_ORDER if p in expected_options]
        assert captured["native"] == native
        # The raw precision (no "(native)" suffix) is what gets stored
        assert config.compute_type == expected_options[0]


class TestLLMApiKeyScreen:
    """Test the LLM API key screen (env: hint, prefill, save flow)."""

    ENTER = 13
    ESC = 27

    def _make_stdscr(self, keys):
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        mock_stdscr.getch.side_effect = list(keys)
        return mock_stdscr

    @patch("faster_whisper_hotkey.ui.curses")
    def test_shows_env_hint_footer(self, mock_curses):
        """The API key screen should hint at the env: prefix."""
        from faster_whisper_hotkey.ui import ConfigData, _screen_llm_api_key

        # Enter confirms the key input, Enter confirms the final save screen
        mock_stdscr = self._make_stdscr([self.ENTER, self.ENTER])
        config = ConfigData()

        result = _screen_llm_api_key(mock_stdscr, config)

        assert not isinstance(result, tuple)
        assert result.llm_api_key == ""
        footers = [
            call.args[2]
            for call in mock_stdscr.addstr.call_args_list
            if len(call.args) == 3 and isinstance(call.args[2], str)
        ]
        assert any("env:" in f and "environment variable" in f for f in footers)

    @patch("faster_whisper_hotkey.ui.curses")
    def test_env_reference_saved_not_value(self, mock_curses):
        """Typing env:VAR saves the variable name, never the value."""
        from faster_whisper_hotkey.ui import ConfigData, _screen_llm_api_key

        keys = [ord(c) for c in "env:MY_KEY"] + [self.ENTER, self.ENTER]
        mock_stdscr = self._make_stdscr(keys)
        config = ConfigData()

        result = _screen_llm_api_key(mock_stdscr, config)

        assert not isinstance(result, tuple)
        assert result.llm_api_key == "env:MY_KEY"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_saved_env_reference_prefilled(self, mock_curses):
        """A previously saved env: reference is prefilled and displayed."""
        from faster_whisper_hotkey.ui import ConfigData, _screen_llm_api_key

        mock_stdscr = self._make_stdscr([self.ENTER, self.ENTER])
        config = ConfigData()
        config.llm_api_key = "env:MY_KEY"

        result = _screen_llm_api_key(mock_stdscr, config)

        assert "env:MY_KEY" in str(mock_stdscr.addstr.call_args_list)
        assert not isinstance(result, tuple)
        assert result.llm_api_key == "env:MY_KEY"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_escape_returns_to_initial(self, mock_curses):
        """ESC on the API key screen returns to the initial screen."""
        from faster_whisper_hotkey.ui import ConfigData, ConfigStep, _screen_llm_api_key

        mock_stdscr = self._make_stdscr([self.ESC, -1])  # bare ESC
        config = ConfigData()

        result = _screen_llm_api_key(mock_stdscr, config)

        assert result == (ConfigStep.INITIAL, config)


class TestLLMModelScreen:
    """Test the LLM model name screen (the original ALT+BACKSPACE bug scenario)."""

    ENTER = 13

    def _make_stdscr(self, keys):
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        mock_stdscr.getch.side_effect = list(keys)
        return mock_stdscr

    @patch("faster_whisper_hotkey.ui.curses")
    def test_alt_backspace_erases_prefilled_name(self, mock_curses):
        """ALT+BACKSPACE on the model name screen erases the prefilled name instead of exiting."""
        from faster_whisper_hotkey.ui import ConfigData, ConfigStep, _screen_llm_model

        mock_stdscr = self._make_stdscr([27, 127, self.ENTER])  # Alt+Backspace, then Enter
        config = ConfigData()
        config.llm_model_name = "qwen3-asr"

        result = _screen_llm_model(mock_stdscr, config)

        assert result is not None
        assert result == (ConfigStep.LLM_API_KEY, config)
        assert config.llm_model_name == ""
