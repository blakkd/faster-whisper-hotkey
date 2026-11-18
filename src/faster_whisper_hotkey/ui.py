import contextlib
import curses
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

from pulsectl import Pulse

from .settings import Settings, load_settings, save_settings


class ConfigStep(Enum):
    """Configuration steps in order."""

    INITIAL = auto()
    AUDIO_DEVICE = auto()
    VAD_ENABLE = auto()
    MODEL = auto()
    QUANT = auto()
    COMPUTE_DEVICE = auto()
    LOAD = auto()
    LANGUAGE = auto()
    TRANSLATE_SOURCE = auto()
    TRANSLATE_TARGET = auto()
    HOTKEY = auto()
    LLM_ENABLE = auto()
    LLM_ENDPOINT = auto()
    LLM_MODEL = auto()
    LLM_API_KEY = auto()


@dataclass
class ConfigData:
    """Holds configuration state across all steps."""

    device_name: str | None = None
    model_repo: str = ""
    model_file: str = ""
    model_name: str = ""  # display name of the chosen model
    device: str = ""
    language: str = ""
    translate_target: str = ""
    hotkey: str = ""
    vad_enabled: bool = True
    llm_correction_enabled: bool = False
    llm_endpoint: str = ""
    llm_model_name: str = ""
    llm_api_key: str = ""
    model_wrapper: Any = None  # the loaded model (ModelWrapper), once the LOAD step runs

    def discard_model(self) -> None:
        """Free a loaded model when its selection is changed or abandoned."""
        if self.model_wrapper is not None:
            self.model_wrapper.close()
            self.model_wrapper = None


# Menu sentinel for entering translation mode.
_TRANSLATE_OPTION = "(translate to another language...)"


def _fuzzy_indices(options: list[str], query: str) -> list[int]:
    """Indices of options matching `query` as a case-insensitive fuzzy subsequence.

    Best matches first: prefix match, then plain substring, then a scattered
    subsequence with word-boundary hits preferred; ties keep the original order.
    """
    q = query.lower()
    if not q:
        return list(range(len(options)))

    scored = []
    for i, opt in enumerate(options):
        text = opt.lower()
        pos = text.find(q)
        if pos == 0:
            score = 3_000_000_000  # prefix match wins over everything
        elif pos > 0:
            score = 2_000_000_000 - pos  # substring, earlier match first
        else:
            score, j = 1_000_000_000, 0
            for at, ch in enumerate(text):  # subsequence: all query chars in order
                if j < len(q) and ch == q[j]:
                    if at == 0 or not text[at - 1].isalnum():
                        score += 5_000  # matched at a word boundary
                    score -= at // 16  # prefer early matches
                    j += 1
            if j < len(q):
                continue
        scored.append((score, i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in scored]


def curses_menu(
    stdscr,
    title: str,
    options: list[str],
    message: str = "",
    initial_idx: int = 0,
    footer: str = "",
    native: str = "",
    searchable: bool = False,
):
    """
    Display a scrollable list of `options` in a curses window.
    A `message` (e.g. "Source language") can be shown directly above the list.
    A `footer` is shown at the bottom of the screen.
    If `native` matches one of the options, it is shown with a " (native)" suffix.

    With `searchable`, typing any printable character starts a fuzzy filter
    (see _fuzzy_indices): the query and match count show above the list,
    Backspace edits the query, and ESC clears the query before it cancels.

    Returns the selected option (without any suffix), or None if the user aborts with ESC.
    """
    query = ""
    view: list[int] = list(range(len(options)))
    sel = min(initial_idx, len(view) - 1) if view else 0
    h, w = stdscr.getmaxyx()

    def set_query(q: str) -> None:
        nonlocal query, view, sel
        kept = view[sel] if view else None
        query = q
        view = _fuzzy_indices(options, q) if q else list(range(len(options)))
        sel = view.index(kept) if kept in view else 0

    def draw_menu():
        nonlocal h, w
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        if h == 0 or w == 0:
            stdscr.refresh()
            return

        message_lines = message.split("\n") if message else []
        if searchable:
            # search status above the results, with a visual gap between the two
            message_lines = message_lines + [f"Search: {query}  ({len(view)} of {len(options)} matches)", "", ""]
        space_for_message = len(message_lines)
        # bottom rows reserved: blank, elevator, blank, footer
        bottom_reserved = 4
        top_reserved = 1 if title else 0
        region_h = h - top_reserved - bottom_reserved

        if region_h - space_for_message <= 0:
            warning = "Terminal too small for this menu"
            stdscr.addstr(0, 0, warning[: w - 1] if w > 0 else warning)
            stdscr.refresh()
            return

        if title:
            title_x = max(0, (w - len(title)) // 2)
            stdscr.addstr(0, title_x, title)

        max_visible = min(region_h - space_for_message, max(1, len(view)))
        # center the message + options block as one unit inside the region
        block_h = space_for_message + max_visible
        options_y_start = top_reserved + (region_h - block_h) // 2 + space_for_message

        for i, line in enumerate(message_lines):
            truncated_line = line[: w - 1] if w > 0 else line
            x = max(0, (w - len(truncated_line)) // 2)
            y_msg = options_y_start - len(message_lines) + i
            if 0 <= y_msg < h:
                stdscr.addstr(y_msg, x, truncated_line)

        start = max(0, sel - (max_visible // 2))
        end = min(start + max_visible, len(view))

        if not view:
            hint = f"No matches for '{query}'"
            y = max(0, min(options_y_start, h - 1))
            stdscr.addstr(y, max(0, min((w - len(hint)) // 2, w - 1)), hint[: w - 1])
        for p in range(start, end):
            text = options[view[p]]
            if native and text == native:
                text = f"{text} (native)"
            x = w // 2 - len(text) // 2
            x = max(0, min(x, w - 1)) if w > 0 else 0
            y = options_y_start + (p - start)
            if y < 0 or y >= h:
                continue
            truncated_text = text[: w - 1] if w > 0 else text
            if p == sel:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, truncated_text)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, truncated_text)

        if max_visible < len(view) and h > 0 and w > 0:
            ratio = (sel + 1) / len(view)
            y_scroll = h - 3
            x_start = w // 4
            length = w // 2
            x_start = max(0, min(x_start, w - 1))
            length = max(1, min(length, w))
            stdscr.addstr(y_scroll, x_start, "[")
            end_pos = int(ratio * (length - 2)) + x_start + 1
            end_pos = max(x_start + 1, min(end_pos, x_start + length - 1))
            stdscr.addstr(y_scroll, x_start + 1, " " * (length - 2))
            stdscr.addstr(y_scroll, end_pos, "█")
            stdscr.addstr(y_scroll, x_start + length - 1, "]")

        if footer and h > 1:
            max_line = max(w - 1, 1)
            footer_lines = []
            words = footer.split()
            current = ""
            for word in words:
                if not current:
                    current = word
                elif len(current) + 1 + len(word) <= max_line:
                    current = current + " " + word
                else:
                    footer_lines.append(current)
                    current = word
            if current:
                footer_lines.append(current)
            if not footer_lines:
                footer_lines = [""]
            for i, line in enumerate(footer_lines):
                y = h - len(footer_lines) + i
                if 0 <= y < h:
                    x = max(0, (w - len(line)) // 2)
                    stdscr.addstr(y, x, line)

        stdscr.refresh()

    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    draw_menu()

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP and sel > 0:
            sel -= 1
        elif key == curses.KEY_DOWN and sel < len(view) - 1:
            sel += 1
        elif key in [curses.KEY_ENTER, 10, 13]:
            if view:
                return options[view[sel]]
        elif key == 27:
            if query:
                set_query("")  # ESC clears an active filter before it cancels
            else:
                return None
        elif searchable and query and key in (curses.KEY_BACKSPACE, 127, 8):
            set_query(query[:-1])
        elif searchable and 32 <= key <= 126:
            set_query(query + chr(key))

        new_h, new_w = stdscr.getmaxyx()
        if new_h != h or new_w != w:
            h, w = new_h, new_w
        draw_menu()


# How long (ms) to wait for a byte following an ESC before treating it as a bare ESC.
# Terminals send Alt+<key> as ESC + <key>, so without this window a bare ESC (cancel)
# and e.g. Alt+Backspace are indistinguishable on the first byte.
ESC_FOLLOWUP_TIMEOUT_MS = 50


def _delete_previous_word(text: str, cursor_pos: int) -> tuple[str, int]:
    """Delete the word before the cursor (and any space preceding it)."""
    if cursor_pos == 0:
        return text, cursor_pos
    space_idx = text[:cursor_pos].rfind(" ")
    word_start = 0 if space_idx == -1 else space_idx
    return text[:word_start] + text[cursor_pos:], word_start


def get_text_input(stdscr, prompt: str, default: str = "", footer: str = "") -> str | None:
    """
    Prompt the user for text input using curses.
    An optional `footer` hint line is shown just below the input.
    Returns the entered text, or None if a bare ESC is pressed.
    Alt+Backspace (ESC + backspace) deletes the previous word instead of cancelling.
    """
    current_text = default
    cursor_pos = len(current_text)

    def draw():
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        h = max(1, h)
        w = max(1, w)
        y_prompt = max(0, min((h - 1) // 2, h - 1))

        avail_width = max(1, w - 2)
        prompt_to_show = prompt[:avail_width]
        prompt_len = len(prompt_to_show)

        stdscr.addstr(y_prompt, 0, prompt_to_show)

        display_width = max(0, w - prompt_len - 1)
        display_text = current_text[:display_width]
        if display_width > 0:
            stdscr.addstr(y_prompt, prompt_len, " " * display_width)
            stdscr.addstr(y_prompt, prompt_len, display_text)

        if footer:
            y_footer = y_prompt + 1
            if y_footer < h:
                stdscr.addstr(y_footer, 0, footer[: w - 1])

        safe_cursor = min(cursor_pos, max(0, display_width))
        final_col = max(0, min(prompt_len + safe_cursor, w - 1))
        final_row = max(0, min(y_prompt, h - 1))
        stdscr.move(final_row, final_col)
        stdscr.refresh()

    draw()
    curses.curs_set(1)

    while True:
        key = stdscr.getch()

        if key == 27:
            # Distinguish a bare ESC (cancel) from an ESC-prefixed combo such as
            # Alt+Backspace: only a bare ESC cancels.
            stdscr.timeout(ESC_FOLLOWUP_TIMEOUT_MS)
            next_key = stdscr.getch()
            stdscr.timeout(-1)
            if next_key == -1:
                curses.curs_set(0)
                return None
            if next_key in (curses.KEY_BACKSPACE, 127):
                # Alt+Backspace: delete the previous word (and the space before it)
                current_text, cursor_pos = _delete_previous_word(current_text, cursor_pos)
                draw()
                continue
            # Other ESC+key combos (e.g. Alt+letter): treat as the key itself
            key = next_key
        if key in (curses.KEY_ENTER, 10, 13):
            curses.curs_set(0)
            return current_text
        if key in (curses.KEY_BACKSPACE, 127) and cursor_pos > 0:
            current_text = current_text[: cursor_pos - 1] + current_text[cursor_pos:]
            cursor_pos -= 1
        if key == curses.KEY_LEFT and cursor_pos > 0:
            cursor_pos -= 1
        if key == curses.KEY_RIGHT and cursor_pos < len(current_text):
            cursor_pos += 1
        if 32 <= key <= 126:
            current_text = current_text[:cursor_pos] + chr(key) + current_text[cursor_pos:]
            cursor_pos += 1

        draw()


# ---------------------------------------------------------------------------
# Device menu (compute backend)
# ---------------------------------------------------------------------------
# Every model runs on the transcribe.cpp ggml runtime, which offers CUDA
# (NVIDIA), Vulkan (AMD/Intel GPUs) and plain CPU. CUDA and Vulkan are optional
# ggml modules — the PyPI wheel has CPU+Vulkan, CUDA needs a source build
# (dev/onboarding.md); the menu lists what the installed provider can serve.
_DEVICE_OPTIONS: list[Literal["cuda", "vulkan", "cpu"]] = ["cuda", "vulkan", "cpu"]


def _available_devices() -> list[str]:
    """Device menu entries the installed engine can actually serve."""
    import transcribe_cpp

    return [d for d in _DEVICE_OPTIONS if d == "cpu" or transcribe_cpp.backend_available(d)]


# ---------------------------------------------------------------------------
# Settings <-> ConfigData
# ---------------------------------------------------------------------------


def _config_from_last_settings(last: Settings | None) -> ConfigData:
    config = ConfigData()
    if last:
        config.device_name = last.device_name
        config.model_repo = last.model_repo
        config.model_file = last.model_file
        config.device = last.device
        config.language = last.language
        config.translate_target = last.translate_target
        config.hotkey = last.hotkey
        config.vad_enabled = last.vad_enabled
        config.llm_correction_enabled = last.llm_correction_enabled
        config.llm_endpoint = last.llm_endpoint
        config.llm_model_name = last.llm_model_name
        config.llm_api_key = last.llm_api_key
        if last.model_repo:
            from . import catalog

            config.model_name = catalog.short_name(last.model_repo)
    return config


def _settings_from_config(config: ConfigData) -> Settings:
    return Settings(
        device_name=config.device_name or "",
        model_repo=config.model_repo,
        model_file=config.model_file,
        device=config.device,
        language=config.language,
        translate_target=config.translate_target,
        hotkey=config.hotkey,
        vad_enabled=config.vad_enabled,
        llm_correction_enabled=config.llm_correction_enabled,
        llm_endpoint=config.llm_endpoint,
        llm_model_name=config.llm_model_name,
        llm_api_key=config.llm_api_key,
    )


# ---------------------------------------------------------------------------
# Config screen main loop
# ---------------------------------------------------------------------------


def config_screen_main(stdscr, settings_file: str | None = None) -> tuple[Settings, Any] | None:
    """
    Unified configuration screen with consistent ESC behavior.

    ESC at any point returns to the initial screen. The user can then start with
    the last settings or reconfigure. Returns (Settings, loaded_model_or_None) on
    success, or None if cancelled. A model loaded during configuration is reused
    by the caller; it is closed here when the user abandons the run.
    """
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    config = _config_from_last_settings(load_settings(settings_file))
    current_step = ConfigStep.INITIAL

    while True:
        result = _handle_key_transition(stdscr, current_step, config)

        if result is None:
            config.discard_model()  # cancelled: free anything we loaded
            return None

        if isinstance(result, Settings):
            save_settings(result.__dict__, settings_file)
            return result, config.model_wrapper

        current_step, config = result


def _handle_key_transition(stdscr, current_step: ConfigStep, config: ConfigData) -> "tuple[ConfigStep, ConfigData] | Settings | None":
    """
    Handle the current configuration step.

    Returns:
        - None: cancel (only possible at the initial screen)
        - (ConfigStep, ConfigData): move to another step
        - Settings: configuration complete
    """
    handlers = {
        ConfigStep.INITIAL: _screen_initial,
        ConfigStep.AUDIO_DEVICE: _screen_audio_device,
        ConfigStep.VAD_ENABLE: _screen_vad_enable,
        ConfigStep.MODEL: _screen_model,
        ConfigStep.QUANT: _screen_quant,
        ConfigStep.COMPUTE_DEVICE: _screen_compute_device,
        ConfigStep.LOAD: _screen_load,
        ConfigStep.LANGUAGE: _screen_language,
        ConfigStep.TRANSLATE_SOURCE: _screen_translate_source,
        ConfigStep.TRANSLATE_TARGET: _screen_translate_target,
        ConfigStep.HOTKEY: _screen_hotkey,
        ConfigStep.LLM_ENABLE: _screen_llm_enable,
        ConfigStep.LLM_ENDPOINT: _screen_llm_endpoint,
        ConfigStep.LLM_MODEL: _screen_llm_model,
        ConfigStep.LLM_API_KEY: _screen_llm_api_key,
    }
    return handlers[current_step](stdscr, config)


def _back_to_initial(config: ConfigData):
    """Return to initial screen (ESC behavior)."""
    return (ConfigStep.INITIAL, config)


# ============================================================================
# Screen handlers
# ============================================================================


def _screen_initial(stdscr, config: ConfigData):
    """Initial choice: use last settings or configure new ones."""
    has_saved = bool(config.model_repo)

    while True:
        options = ["Use Last Settings", "Configure New Settings"] if has_saved else ["Configure New Settings"]
        choice = curses_menu(stdscr, "", options)

        if choice is None:
            return None

        if choice == "Use Last Settings":
            # No model loaded on this path; the transcriber loads from settings.
            return _settings_from_config(config)

        if choice == "Configure New Settings":
            # Starting a fresh selection: drop any model loaded in a previous pass.
            config.discard_model()
            config.model_repo = ""
            config.model_file = ""
            config.model_name = ""
            config.language = ""
            config.translate_target = ""
            return (ConfigStep.AUDIO_DEVICE, config)


def _screen_audio_device(stdscr, config: ConfigData):
    """Select audio input device."""
    with Pulse() as pulse:
        sources = pulse.source_list()
        source_map = {src.description: src.name for src in sources}
        device_names = list(source_map.keys())

        initial_idx = 0
        if config.device_name:
            for i, (_desc, name) in enumerate(source_map.items()):
                if name == config.device_name:
                    initial_idx = i
                    break

        selected_desc = curses_menu(stdscr, "Audio Device", device_names, initial_idx=initial_idx)

        if selected_desc is None:
            return _back_to_initial(config)

        config.device_name = source_map[selected_desc]
        return (ConfigStep.VAD_ENABLE, config)


def _screen_vad_enable(stdscr, config: ConfigData):
    """Enable or disable Silero VAD speech detection."""
    options = ["Yes", "No"]
    initial_idx = 0 if config.vad_enabled else 1

    selected = curses_menu(
        stdscr,
        "Enable Silero VAD (skip silent recordings)?",
        options,
        footer="Performant, but may miss very quiet whispers",
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.vad_enabled = selected == "Yes"
    return (ConfigStep.MODEL, config)


def _screen_model(stdscr, config: ConfigData):
    """Select a model from the hub catalog (type to fuzzy-search the long list)."""
    from . import catalog

    entries = catalog.list_models()
    options = [f"{e.name} ({len(e.files)})" for e in entries]
    by_label = dict(zip(options, entries, strict=True))

    initial_idx = 0
    for i, e in enumerate(entries):
        if e.repo_id == config.model_repo:
            initial_idx = i
            break

    selected = curses_menu(
        stdscr,
        f"Model ({len(entries)} available)",
        options,
        footer="Pick a model. You can type to search.",
        initial_idx=initial_idx,
        searchable=True,
    )
    if selected is None:
        return _back_to_initial(config)

    entry = by_label[selected]
    config.model_repo = entry.repo_id
    config.model_name = entry.name
    config.model_file = ""  # force a fresh quant choice
    return (ConfigStep.QUANT, config)


def _screen_quant(stdscr, config: ConfigData):
    """Select a GGUF file (quantization) for the chosen model, with sizes."""
    from . import catalog

    while True:
        files = catalog.files_for(config.model_repo)
        if not files:
            # The hub gave nothing and we have nothing cached.
            message = f"Could not list files for {config.model_name}."
            curses_menu(stdscr, "Error", ["Back"], message=message)
            return (ConfigStep.MODEL, config)

        options = []
        for f in files:
            size = "" if f.size_mb is None else f" — {f.size_mb:.0f} MB"
            options.append(f"{f.label}{size}")
        by_label = {opt: f for opt, f in zip(options, files, strict=True)}

        initial_idx = 0
        if config.model_file:
            for i, f in enumerate(files):
                if f.filename == config.model_file:
                    initial_idx = i
                    break

        selected = curses_menu(
            stdscr,
            f"Precision — {config.model_name}",
            options,
            footer="Pick a quantization. Sizes shown are the download.",
            initial_idx=initial_idx,
        )
        if selected is None:
            return _back_to_initial(config)

        chosen = by_label.get(selected)
        if chosen is None:
            continue

        config.model_file = chosen.filename
        return (ConfigStep.COMPUTE_DEVICE, config)


def _screen_compute_device(stdscr, config: ConfigData):
    """Select the compute backend."""
    options = _available_devices()
    initial_idx = options.index(config.device) if config.device in options else 0
    selected = curses_menu(stdscr, "Compute Device", options, initial_idx=initial_idx)
    if selected is None:
        return _back_to_initial(config)

    config.device = selected
    return (ConfigStep.LOAD, config)


def _screen_load(stdscr, config: ConfigData):
    """Download (first use) and load the chosen model, with a live status screen.

    The model must be loaded here so its capabilities can drive the language
    menu. Runs in a background thread while the screen animates; ESC is ignored
    (a download in progress is not safely interruptible). On failure, offers a
    retry or a return to model selection.
    """
    import threading

    from .models import ModelWrapper

    def worker(holder: dict):
        try:
            holder["model"] = ModelWrapper(_settings_from_config(config))
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            holder["error"] = exc

    while True:
        holder: dict = {}
        status = {"line": ""}
        _attach_status_handler(status)
        thread = threading.Thread(target=worker, args=(holder,), daemon=True)
        thread.start()
        start = time.time()
        frames = "|/-\\"
        i = 0
        while thread.is_alive():
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            elapsed = time.time() - start
            lines = [f"Loading {config.model_name} ({config.model_file})", f"{frames[i % len(frames)]} {elapsed:.0f}s elapsed", status["line"][: max(0, w - 2)]]
            for row, line in enumerate(lines):
                y = max(0, (h - len(lines)) // 2 + row)
                if 0 <= y < h and line:
                    try:
                        stdscr.addstr(y, max(0, (w - len(line)) // 2), line[: w - 1])
                    except curses.error:
                        pass
            stdscr.refresh()
            stdscr.timeout(120)
            stdscr.getch()
            stdscr.timeout(-1)
            i += 1
        _detach_status_handler()
        thread.join()

        if "model" in holder and holder["model"] is not None:
            config.model_wrapper = holder["model"]
            return (ConfigStep.LANGUAGE, config)

        # Load failed: show the reason and let the user retry or go back.
        error = holder.get("error")
        chosen = curses_menu(
            stdscr,
            "Could not load model",
            ["Try again", "Back to model"],
            message=_short_error(error),
        )
        if chosen != "Try again":
            config.discard_model()
            return (ConfigStep.MODEL, config)


def _short_error(error: Exception | None) -> str:
    if error is None:
        return "Unknown error."
    text = str(error).strip() or error.__class__.__name__
    return text if len(text) <= 300 else text[:297] + "..."


# A logging handler the loader screen installs so its thread can show the
# latest progress line (from the downloader/model INFO logs) on the curses screen.
_STATUS_HANDLERS: list[logging.Handler] = []


class _StatusLogHandler(logging.Handler):
    def __init__(self, sink: dict):
        super().__init__()
        self._sink = sink

    def emit(self, record):
        # logging must never raise back into the loading thread
        with contextlib.suppress(Exception):
            self._sink["line"] = record.getMessage()[:120]


def _attach_status_handler(sink: dict) -> None:
    handler = _StatusLogHandler(sink)
    logging.getLogger("faster_whisper_hotkey").addHandler(handler)
    _STATUS_HANDLERS.append(handler)


def _detach_status_handler() -> None:
    app_logger = logging.getLogger("faster_whisper_hotkey")
    while _STATUS_HANDLERS:
        app_logger.removeHandler(_STATUS_HANDLERS.pop())


def _screen_language(stdscr, config: ConfigData):
    """Select transcription language (or enter translation) from the model's capabilities."""
    wrapper = config.model_wrapper
    languages = wrapper.transcription_languages() if wrapper is not None else ["auto"]
    can_translate = bool(wrapper and wrapper.supports_translate and wrapper.translation_target_languages())

    options = list(languages)
    if can_translate:
        options.append(_TRANSLATE_OPTION)

    initial_idx = 0
    if config.language in options:
        initial_idx = options.index(config.language)

    selected = curses_menu(
        stdscr,
        "Language",
        options,
        footer="Pick a language to transcribe (auto = detect), or translate." if can_translate else "Pick a language to transcribe.",
        initial_idx=initial_idx,
    )
    if selected is None:
        return _back_to_initial(config)

    if selected == _TRANSLATE_OPTION:
        config.translate_target = ""
        return (ConfigStep.TRANSLATE_SOURCE, config)

    config.language = selected if selected != "auto" else "auto"
    config.translate_target = ""
    return (ConfigStep.HOTKEY, config)


def _screen_translate_source(stdscr, config: ConfigData):
    """Pick the source language for translation."""
    wrapper = config.model_wrapper
    sources = wrapper.translation_source_languages() if wrapper is not None else []
    initial_idx = sources.index(config.language) if config.language in sources else 0

    selected = curses_menu(stdscr, "Translate: source language", sources, initial_idx=initial_idx)
    if selected is None:
        return _back_to_initial(config)

    config.language = selected
    return (ConfigStep.TRANSLATE_TARGET, config)


def _screen_translate_target(stdscr, config: ConfigData):
    """Pick the target language for translation."""
    wrapper = config.model_wrapper
    targets = wrapper.translation_target_languages() if wrapper is not None else []
    initial_idx = targets.index(config.translate_target) if config.translate_target in targets else 0

    selected = curses_menu(stdscr, "Translate: target language", targets, initial_idx=initial_idx)
    if selected is None:
        return _back_to_initial(config)

    config.translate_target = selected
    return (ConfigStep.HOTKEY, config)


def _screen_hotkey(stdscr, config: ConfigData):
    """Select hotkey."""
    hotkey_options = ["Pause", "F4", "F8", "INSERT"]

    initial_idx = 0
    if config.hotkey and config.hotkey.upper() in hotkey_options:
        initial_idx = hotkey_options.index(config.hotkey.upper())

    selected = curses_menu(stdscr, "Hotkey", hotkey_options, initial_idx=initial_idx)
    if selected is None:
        return _back_to_initial(config)

    config.hotkey = selected.lower()
    return (ConfigStep.LLM_ENABLE, config)


def _screen_llm_enable(stdscr, config: ConfigData):
    """Enable or disable LLM correction."""
    options = ["Yes", "No"]
    initial_idx = 0 if config.llm_correction_enabled else 1

    selected = curses_menu(stdscr, "Enable LLM Correction?", options, initial_idx=initial_idx)
    if selected is None:
        return _back_to_initial(config)

    config.llm_correction_enabled = selected == "Yes"

    if not config.llm_correction_enabled:
        # Don't clear endpoint/model - preserve previously saved values
        return _final_save(stdscr, config)

    return (ConfigStep.LLM_ENDPOINT, config)


def _screen_llm_endpoint(stdscr, config: ConfigData):
    """Enter LLM endpoint URL."""
    default = config.llm_endpoint or "http://localhost:8080/v1"

    result = get_text_input(stdscr, "Endpoint URL: ", default)

    if result is None:
        return _back_to_initial(config)

    config.llm_endpoint = result
    return (ConfigStep.LLM_MODEL, config)


def _screen_llm_model(stdscr, config: ConfigData):
    """Enter LLM model name."""
    result = get_text_input(stdscr, "Model name: ", config.llm_model_name or "")

    if result is None:
        return _back_to_initial(config)

    config.llm_model_name = result
    return (ConfigStep.LLM_API_KEY, config)


def _screen_llm_api_key(stdscr, config: ConfigData):
    """Enter LLM API key (plain text, or "env:VAR" to read it from an environment variable)."""
    footer = "For environment variable, use this format 'env:OPENAI_API_KEY'"

    result = get_text_input(stdscr, "API key: ", config.llm_api_key or "", footer=footer)

    if result is None:
        return _back_to_initial(config)

    config.llm_api_key = result
    return _final_save(stdscr, config)


def _final_save(stdscr, config: ConfigData):
    """Display confirmation and return the finished settings (the caller saves)."""
    confirmation_msg = "All settings configured.\nPress ENTER to save and continue."

    selected = curses_menu(stdscr, "Configuration Complete", ["Save & Continue"], message=confirmation_msg)

    if selected is None:
        return _back_to_initial(config)

    return _settings_from_config(config)
