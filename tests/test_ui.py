"""Tests for the curses TUI in ui.py.

Covers the generic curses widgets (menu, text input) and the new state machine:
hub-driven model/precision selection, the model load step, and the
capabilities-driven language/translation screens.
"""

import curses
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fakes shared by the flow tests
# ---------------------------------------------------------------------------

ENTER = 13
ESC = 27
KEY_DOWN = 258
KEY_UP = 259


def make_stdscr(keys=(), height=24, width=80):
    m = MagicMock()
    m.getmaxyx.return_value = (height, width)
    m.getch.side_effect = list(keys)
    return m


class FakeWrapper:
    """A loaded-model stand-in exposing the capabilities the language screens use."""

    def __init__(self, langs=("auto", "en", "de"), translate=False, targets=()):
        self._langs = list(langs)
        self.supports_translate = translate
        self._targets = list(targets)
        self.closed = False

    def transcription_languages(self):
        return list(self._langs)

    def translation_source_languages(self):
        return [c for c in self._langs if c != "auto"]

    def translation_target_languages(self):
        return list(self._targets)

    def close(self):
        self.closed = True


# ===========================================================================
# Generic widgets: text input
# ===========================================================================


class TestGetTextInput:
    def _stdscr(self, height=24, width=80):
        m = MagicMock()
        m.getmaxyx.return_value = (height, width)
        return m

    @patch("faster_whisper_hotkey.ui.curses")
    def test_default_with_no_modification(self, mock_curses):
        from faster_whisper_hotkey.ui import get_text_input

        s = self._stdscr()
        s.getch.side_effect = [13]
        assert get_text_input(s, "Enter URL:", "http://localhost:8678/v1") == "http://localhost:8678/v1"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_default_is_displayed(self, mock_curses):
        from faster_whisper_hotkey.ui import get_text_input

        s = self._stdscr()
        s.getch.side_effect = [13]
        get_text_input(s, "Endpoint URL:", "http://x/v1")
        assert "http://x/v1" in str(s.addstr.call_args_list)

    @patch("faster_whisper_hotkey.ui.curses")
    def test_backspace_edits_default(self, mock_curses):
        from faster_whisper_hotkey.ui import get_text_input

        s = self._stdscr()
        s.getch.side_effect = [127, 13]
        assert get_text_input(s, "Enter:", "hello") == "hell"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_escape_returns_none(self, mock_curses):
        from faster_whisper_hotkey.ui import get_text_input

        s = self._stdscr()
        s.getch.side_effect = [27, -1]
        assert get_text_input(s, "Enter:", "default") is None

    @patch("faster_whisper_hotkey.ui.curses")
    def test_alt_backspace_deletes_word(self, mock_curses):
        from faster_whisper_hotkey.ui import get_text_input

        s = self._stdscr()
        s.getch.side_effect = [27, 127, 13]
        assert get_text_input(s, "Enter:", "hello world") == "hello"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_typing_appends(self, mock_curses):
        from faster_whisper_hotkey.ui import get_text_input

        s = self._stdscr()
        s.getch.side_effect = [ord(c) for c in "abc"] + [13]
        assert get_text_input(s, "Enter:", "") == "abc"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_left_right_navigation(self, mock_curses):
        from faster_whisper_hotkey.ui import get_text_input

        s = self._stdscr()
        s.getch.side_effect = [curses.KEY_LEFT, curses.KEY_LEFT, 127, curses.KEY_RIGHT, 88, 13]
        assert get_text_input(s, "Enter:", "abcde") == "abcdX"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_footer_drawn_below_prompt(self, mock_curses):
        from faster_whisper_hotkey.ui import get_text_input

        s = self._stdscr()
        s.getch.side_effect = [13]
        get_text_input(s, "API key: ", "", footer="env: hint")
        assert (12, 0, "env: hint") in [c.args for c in s.addstr.call_args_list]


# ===========================================================================
# Generic widgets: menu
# ===========================================================================


class TestCursesMenu:
    @patch("faster_whisper_hotkey.ui.curses")
    def test_simple_selection(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        s = MagicMock()
        s.getmaxyx.return_value = (24, 80)
        s.getch.side_effect = [13]
        assert curses_menu(s, "T", ["A", "B", "C"]) == "A"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_navigate_down_select(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        mock_curses.KEY_DOWN = 258
        s = MagicMock()
        s.getmaxyx.return_value = (24, 80)
        s.getch.side_effect = [258, 13]
        assert curses_menu(s, "T", ["A", "B", "C"]) == "B"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_escape_aborts(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        s = MagicMock()
        s.getmaxyx.return_value = (24, 80)
        s.getch.side_effect = [27]
        assert curses_menu(s, "T", ["A", "B"]) is None

    @patch("faster_whisper_hotkey.ui.curses")
    def test_native_hint(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        s = MagicMock()
        s.getmaxyx.return_value = (24, 80)
        s.getch.side_effect = [13]
        curses_menu(s, "Precision", ["float32", "bfloat16"], native="bfloat16")
        assert "bfloat16 (native)" in str(s.addstr.call_args_list)


# ===========================================================================
# Fuzzy search (model list is long; typing filters it)
# ===========================================================================


class TestFuzzyIndices:
    def test_prefix_beats_substring_beats_subsequence(self):
        from faster_whisper_hotkey.ui import _fuzzy_indices

        opts = ["a-b-c", "zzabc", "abc-first"]
        assert _fuzzy_indices(opts, "abc") == [2, 1, 0]

    def test_case_insensitive(self):
        from faster_whisper_hotkey.ui import _fuzzy_indices

        assert _fuzzy_indices(["Parakeet-Ultra", "whisper-small"], "PARA") == [0]

    def test_non_match_excluded(self):
        from faster_whisper_hotkey.ui import _fuzzy_indices

        assert _fuzzy_indices(["parakeet-ultra", "whisper-small"], "zzz") == []
        # subsequence across the whole label, ignoring order of the rest
        assert _fuzzy_indices(["parakeet-ultra", "whisper-small"], "prtu") == [0]

    def test_word_boundary_preferred_over_midword(self):
        from faster_whisper_hotkey.ui import _fuzzy_indices

        # neither contains "ma" contiguously: both are subsequences; the option
        # whose match starts at a word boundary must win despite catalog order
        assert _fuzzy_indices(["zmba", "mxa"], "ma") == [1, 0]

    def test_ties_keep_original_order(self):
        from faster_whisper_hotkey.ui import _fuzzy_indices

        assert _fuzzy_indices(["foo-x", "bar-x", "baz-y"], "x") == [0, 1]

    def test_empty_query_returns_all(self):
        from faster_whisper_hotkey.ui import _fuzzy_indices

        assert _fuzzy_indices(["a", "b"], "") == [0, 1]


class TestSearchableMenu:
    OPTS = ("parakeet-ultra (6)", "whisper-small (6)", "qwen3-asr-1.7b (3)")

    def _stdscr(self, keys):
        s = MagicMock()
        s.getmaxyx.return_value = (24, 80)
        s.getch.side_effect = keys
        return s

    @patch("faster_whisper_hotkey.ui.curses")
    def test_typing_filters_and_enter_selects_top_match(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        # "ul" matches only parakeet-ultra
        s = self._stdscr([ord("u"), ord("l"), 13])
        assert curses_menu(s, "Model", self.OPTS, searchable=True) == self.OPTS[0]
        assert "Search: ul" in str(s.addstr.call_args_list)

    @patch("faster_whisper_hotkey.ui.curses")
    def test_navigation_moves_within_filtered_view(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        mock_curses.KEY_DOWN = 258
        opts = ["x-ray", "delta-x"]  # both match "x"; prefix first
        s = self._stdscr([ord("x"), 258, 13])
        assert curses_menu(s, "Model", opts, searchable=True) == "delta-x"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_backspace_edits_the_query(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        # "ulq" matches nothing; backspace -> "ul" matches parakeet-ultra again
        s = self._stdscr([ord("u"), ord("l"), ord("q"), 127, 13])
        assert curses_menu(s, "Model", self.OPTS, searchable=True) == self.OPTS[0]

    @patch("faster_whisper_hotkey.ui.curses")
    def test_enter_on_empty_result_does_nothing_then_esc_restores(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        # "zz": no matches -> Enter is a no-op; ESC clears the filter (not cancel);
        # Enter then picks the top of the restored full list.
        s = self._stdscr([ord("z"), ord("z"), 13, 27, 13])
        assert curses_menu(s, "Model", self.OPTS, searchable=True) == self.OPTS[0]

    @patch("faster_whisper_hotkey.ui.curses")
    def test_esc_clears_query_then_cancels(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        s = self._stdscr([ord("u"), ord("l"), 27, 27])  # filter, clear filter, cancel
        assert curses_menu(s, "Model", self.OPTS, searchable=True) is None

    @patch("faster_whisper_hotkey.ui.curses")
    def test_search_line_separated_from_results(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        s = self._stdscr([ord("u"), ord("l"), 27, 27])
        curses_menu(s, "Model", self.OPTS, searchable=True)
        drawn = [c.args for c in s.addstr.call_args_list if len(c.args) == 3]
        i = next(n for n, (_y, _x, t) in enumerate(drawn) if t.startswith("Search: ul"))
        j = next(n for n in range(i + 1, len(drawn)) if drawn[n][2] == self.OPTS[0])
        assert drawn[j][0] - drawn[i][0] >= 3  # search line, two blank lines, first result

    @patch("faster_whisper_hotkey.ui.curses")
    def test_bottom_section_gaps(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        opts = [f"model-{i:02d} (6)" for i in range(20)]  # longer than the visible window
        h, w = 24, 80
        s = MagicMock()
        s.getmaxyx.return_value = (h, w)
        s.getch.side_effect = [27]
        assert curses_menu(s, "Model", opts, footer="Type to search.", searchable=True) is None

        drawn = [c.args for c in s.addstr.call_args_list if len(c.args) == 3]
        option_rows = {y for y, _x, t in drawn if t in opts}
        elevator_rows = {y for y, _x, t in drawn if t == "█"}
        footer_rows = {y for y, _x, t in drawn if t == "Type to search."}
        assert elevator_rows == {h - 3}
        assert footer_rows == {h - 1}
        assert option_rows and max(option_rows) <= h - 5  # blank line between list and elevator

    @patch("faster_whisper_hotkey.ui.curses")
    def test_non_searchable_menu_ignores_typed_chars(self, mock_curses):
        from faster_whisper_hotkey.ui import curses_menu

        # "z" would filter everything out if search were on -> Enter would stall.
        s = self._stdscr([ord("z"), 13])
        assert curses_menu(s, "T", self.OPTS) == self.OPTS[0]


# ===========================================================================
# Settings <-> config round trip
# ===========================================================================


class TestConfigConversion:
    def test_settings_from_config(self):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData(
            device_name="mic", model_repo="handy-computer/x-gguf", model_file="x-Q8_0.gguf",
            device="cpu", language="fr", translate_target="de", hotkey="f8", vad_enabled=False,
        )
        s = ui._settings_from_config(config)
        assert (s.model_repo, s.model_file, s.language, s.translate_target, s.hotkey, s.vad_enabled) == (
            "handy-computer/x-gguf", "x-Q8_0.gguf", "fr", "de", "f8", False,
        )

    def test_config_from_last_settings_populates_model_name(self):
        from faster_whisper_hotkey import ui
        from faster_whisper_hotkey.settings import Settings

        last = Settings(device_name="mic", model_repo="handy-computer/parakeet-ultra-gguf",
                        model_file="parakeet-ultra-Q8_0.gguf", device="cpu", language="auto")
        config = ui._config_from_last_settings(last)
        assert config.model_name == "parakeet-ultra"
        assert config.model_repo == last.model_repo

    def test_discard_model_closes_wrapper(self):
        from faster_whisper_hotkey import ui

        w = FakeWrapper()
        config = ui.ConfigData(model_wrapper=w)
        config.discard_model()
        assert w.closed is True and config.model_wrapper is None


# ===========================================================================
# Initial screen
# ===========================================================================


class TestInitialScreen:
    def _initial(self, has_saved: bool, choice):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData()
        if has_saved:
            config.model_repo = "handy-computer/x-gguf"
            config.model_file = "x-Q8_0.gguf"
        captured = {}

        def fake_menu(stdscr, title, options, **kw):
            captured["options"] = options
            return choice

        with patch.object(ui, "curses_menu", fake_menu):
            result = ui._screen_initial(MagicMock(), config)
        return result, config, captured

    def test_first_run_only_configures_new(self):
        result, _, captured = self._initial(has_saved=False, choice="Configure New Settings")
        assert captured["options"] == ["Configure New Settings"]
        assert result[0].name == "AUDIO_DEVICE"

    def test_use_last_returns_settings_directly(self):
        from faster_whisper_hotkey.settings import Settings

        result, _, captured = self._initial(has_saved=True, choice="Use Last Settings")
        assert captured["options"] == ["Use Last Settings", "Configure New Settings"]
        assert isinstance(result, Settings)

    def test_escape_at_initial_returns_none(self):
        result, _, _ = self._initial(has_saved=True, choice=None)
        assert result is None


# ===========================================================================
# Model / precision / device screens (hub + capabilities driven)
# ===========================================================================


class TestModelScreen:
    def _entries(self):
        from faster_whisper_hotkey import catalog

        return [
            catalog.ModelEntry("handy-computer/parakeet-ultra-gguf", "parakeet-ultra",
                               (catalog.GGUFFile("parakeet-ultra-Q8_0.gguf", "Q8_0"),)),
            catalog.ModelEntry("handy-computer/whisper-small-gguf", "whisper-small",
                               (catalog.GGUFFile("whisper-small-F16.gguf", "F16"),)),
        ]

    def _run(self, choice, model_repo=""):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData()
        config.model_repo = model_repo
        captured = {}

        def fake_menu(stdscr, title, options, **kw):
            captured["options"] = options
            captured.update(kw)
            return choice

        with (
            patch("faster_whisper_hotkey.catalog.list_models", return_value=self._entries()),
            patch.object(ui, "curses_menu", fake_menu),
        ):
            result = ui._screen_model(MagicMock(), config)
        return result, config, captured

    def test_selects_model_and_advances(self):
        result, config, _ = self._run("whisper-small (1)")
        assert result[0].name == "QUANT"
        assert config.model_repo == "handy-computer/whisper-small-gguf"
        assert config.model_name == "whisper-small"

    def test_escape_returns_initial(self):
        result, _, _ = self._run(None)
        assert result[0].name == "INITIAL"

    def test_list_is_searchable_and_has_no_refresh_entry(self):
        _, _, captured = self._run("whisper-small (1)")
        assert captured["searchable"] is True
        assert not any("refresh" in o.lower() for o in captured["options"])


class TestQuantScreen:
    def _files(self):
        from faster_whisper_hotkey import catalog

        return [
            catalog.GGUFFile("whisper-small-Q8_0.gguf", "Q8_0", size_bytes=200_000_000),
            catalog.GGUFFile("whisper-small-F16.gguf", "F16", size_bytes=500_000_000),
        ]

    def _run(self, choice, files):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData(model_repo="handy-computer/whisper-small-gguf", model_name="whisper-small")
        captured = {}

        def fake_menu(stdscr, title, options, **kw):
            captured["options"] = options
            return choice

        with patch("faster_whisper_hotkey.catalog.files_for", return_value=files), patch.object(ui, "curses_menu", fake_menu):
            result = ui._screen_quant(MagicMock(), config)
        return result, config, captured

    def test_selects_file_with_size_label(self):
        result, config, captured = self._run("int8 — 200 MB", self._files())
        assert result[0].name == "COMPUTE_DEVICE"
        assert config.model_file == "whisper-small-Q8_0.gguf"
        assert any("MB" in o for o in captured["options"])

    def test_escape_returns_initial(self):
        result, _, _ = self._run(None, self._files())
        assert result[0].name == "INITIAL"

    def test_no_files_returns_to_model(self):
        result, _, _ = self._run(None, [])
        assert result[0].name == "MODEL"


class TestComputeDeviceScreen:
    def _run(self, choice, device="", available=("cuda", "vulkan", "cpu")):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData()
        config.device = device
        captured = {}

        def fake_menu(stdscr, title, options, **kw):
            captured["options"] = options
            captured["initial_idx"] = kw.get("initial_idx", 0)
            return choice

        with patch.object(ui, "curses_menu", fake_menu), patch("transcribe_cpp.backend_available", lambda b: b in available):
            result = ui._screen_compute_device(MagicMock(), config)
        return result, config, captured

    def test_select_device_advances_to_load(self):
        result, config, captured = self._run("cpu")
        assert result[0].name == "LOAD"
        assert config.device == "cpu"
        assert captured["options"] == ["cuda", "vulkan", "cpu"]

    def test_preselect_saved_device(self):
        _, _, captured = self._run("vulkan", device="vulkan")
        assert captured["initial_idx"] == 1

    def test_hides_unavailable_backends(self):
        _, _, captured = self._run("cpu", available=("vulkan", "cpu"))
        assert captured["options"] == ["vulkan", "cpu"]


# ===========================================================================
# Load screen (model loaded during configuration)
# ===========================================================================


class TestLoadScreen:
    def test_success_loads_model(self):
        from faster_whisper_hotkey import ui

        fake_wrapper = FakeWrapper()
        config = ui.ConfigData(model_repo="handy-computer/parakeet-ultra-gguf", model_file="parakeet-ultra-Q8_0.gguf", model_name="parakeet-ultra", device="cpu")
        with patch("faster_whisper_hotkey.models.ModelWrapper", return_value=fake_wrapper):
            s = make_stdscr()
            result = ui._screen_load(s, config)
        assert result[0].name == "LANGUAGE"
        assert config.model_wrapper is fake_wrapper

    def test_failure_returns_to_model(self):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData(model_repo="handy-computer/parakeet-ultra-gguf", model_file="parakeet-ultra-Q8_0.gguf", model_name="parakeet-ultra", device="cpu")
        with patch("faster_whisper_hotkey.models.ModelWrapper", side_effect=RuntimeError("disk full")), patch.object(
            ui, "curses_menu", lambda *a, **k: "Back to model"
        ):
            s = make_stdscr()
            result = ui._screen_load(s, config)
        assert result[0].name == "MODEL"
        assert config.model_wrapper is None


# ===========================================================================
# Language + translation screens (from capabilities)
# ===========================================================================


class TestLanguageScreen:
    def _run(self, wrapper, choice):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData(model_wrapper=wrapper, language="en")
        captured = {}

        def fake_menu(stdscr, title, options, **kw):
            captured["options"] = options
            return choice

        with patch.object(ui, "curses_menu", fake_menu):
            result = ui._screen_language(MagicMock(), config)
        return result, config, captured

    def test_pick_language_transcribes(self):
        result, config, _ = self._run(FakeWrapper(langs=["auto", "en", "de"]), "de")
        assert result[0].name == "HOTKEY"
        assert config.language == "de" and config.translate_target == ""

    def test_no_translate_option_when_unsupported(self):
        _, _, captured = self._run(FakeWrapper(langs=["auto", "en"], translate=False), "en")
        assert not any("translate" in o.lower() for o in captured["options"])

    def test_translate_option_when_supported(self):
        result, _, captured = self._run(FakeWrapper(langs=["auto", "en"], translate=True, targets=["de"]), "(translate to another language...)")
        assert result[0].name == "TRANSLATE_SOURCE"
        assert any("translate" in o.lower() for o in captured["options"])

    def test_escape_returns_initial(self):
        result, _, _ = self._run(FakeWrapper(langs=["auto", "en"]), None)
        assert result[0].name == "INITIAL"

    def test_translate_source_then_target(self):
        from faster_whisper_hotkey import ui

        wrapper = FakeWrapper(langs=["auto", "en", "fr"], translate=True, targets=["de", "it"])
        config = ui.ConfigData(model_wrapper=wrapper, language="en")
        with patch.object(ui, "curses_menu", lambda *a, **k: "fr"):
            result = ui._screen_translate_source(MagicMock(), config)
        assert result[0].name == "TRANSLATE_TARGET" and config.language == "fr"
        with patch.object(ui, "curses_menu", lambda *a, **k: "it"):
            result = ui._screen_translate_target(MagicMock(), config)
        assert result[0].name == "HOTKEY" and config.translate_target == "it"


# ===========================================================================
# Hotkey + LLM screens (ESC returns to INITIAL, env: handling)
# ===========================================================================


class TestHotkeyScreen:
    def _run(self, choice, hotkey=""):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData(hotkey=hotkey)
        with patch.object(ui, "curses_menu", lambda *a, **k: choice):
            result = ui._screen_hotkey(MagicMock(), config)
        return result, config

    def test_select_hotkey(self):
        result, config = self._run("F4")
        assert result[0].name == "LLM_ENABLE" and config.hotkey == "f4"

    def test_escape_returns_initial(self):
        result, _ = self._run(None)
        assert result[0].name == "INITIAL"


class TestLLMApiKeyScreen:
    @patch("faster_whisper_hotkey.ui.curses")
    def test_shows_env_hint_and_saves(self, mock_curses):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData(llm_api_key="")
        s = make_stdscr([ENTER, ENTER])  # confirm key (empty), confirm save
        result = ui._screen_llm_api_key(s, config)
        assert not isinstance(result, tuple)
        assert result.llm_api_key == ""
        footers = [c.args[2] for c in s.addstr.call_args_list if len(c.args) == 3 and isinstance(c.args[2], str)]
        assert any("env:" in f for f in footers)

    @patch("faster_whisper_hotkey.ui.curses")
    def test_env_reference_saved_verbatim(self, mock_curses):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData(llm_api_key="")
        s = make_stdscr([ord(c) for c in "env:MY_KEY"] + [ENTER, ENTER])
        result = ui._screen_llm_api_key(s, config)
        assert result.llm_api_key == "env:MY_KEY"

    @patch("faster_whisper_hotkey.ui.curses")
    def test_escape_returns_initial(self, mock_curses):
        from faster_whisper_hotkey import ui

        config = ui.ConfigData()
        s = make_stdscr([ESC, -1])
        result = ui._screen_llm_api_key(s, config)
        assert result[0].name == "INITIAL"
