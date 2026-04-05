import curses
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum, auto

from .settings import Settings, load_settings, save_settings
from pulsectl import Pulse


class ConfigStep(Enum):
    """Configuration steps in order."""

    INITIAL = auto()
    DEVICE = auto()
    MODEL_TYPE = auto()
    WHISPER_MODEL = auto()
    WHISPER_DEVICE = auto()
    WHISPER_PRECISION = auto()
    WHISPER_LANGUAGE = auto()
    PARAKEET_DEVICE = auto()
    CANARY_SOURCE_LANG = auto()
    CANARY_TARGET_LANG = auto()
    VOXTRAL_DEVICE = auto()
    VOXTRAL_PRECISION = auto()
    VOXTRAL_INFO = auto()
    COHERE_DEVICE = auto()
    COHERE_LANGUAGE = auto()
    HOTKEY = auto()
    LLM_ENABLE = auto()
    LLM_ENDPOINT = auto()
    LLM_MODEL = auto()


@dataclass
class ConfigData:
    """Holds configuration state across all steps."""

    device_name: str | None = None
    model_type: str | None = None
    model_name: str = ""
    compute_type: str = ""
    device: str = ""
    language: str = ""
    hotkey: str = ""
    llm_correction_enabled: bool = False
    llm_endpoint: str = ""
    llm_model_name: str = ""


def curses_menu(
    stdscr, title: str, options: List[str], message: str = "", initial_idx: int = 0
):
    """
    Display a scrollable list of `options` in a curses window.
    A `message` (e.g. "Source language") can be shown directly above the list.

    Returns the selected option, or None if the user aborts with ESC.
    """
    current_row = initial_idx
    h, w = stdscr.getmaxyx()

    def draw_menu():
        nonlocal h, w
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        if h == 0 or w == 0:
            stdscr.refresh()
            return

        message_lines = message.split("\n") if message else []
        space_for_message = len(message_lines)
        space_for_options = h - 2

        if space_for_options - space_for_message <= 0:
            warning = "Terminal too small for this menu"
            stdscr.addstr(0, 0, warning[: w - 1] if w > 0 else warning)
            stdscr.refresh()
            return

        max_visible = min(space_for_options - space_for_message, len(options))
        start = max(0, current_row - (max_visible // 2))
        end = min(start + max_visible, len(options))

        if title:
            title_x = max(0, (w - len(title)) // 2)
            stdscr.addstr(0, title_x, title)

        options_y_start = (h - max_visible) // 2
        if title:
            options_y_start += 1

        for i, line in enumerate(message_lines):
            truncated_line = line[: w - 1] if w > 0 else line
            x = max(0, (w - len(truncated_line)) // 2)
            y_msg = options_y_start - len(message_lines) + i
            if 0 <= y_msg < h:
                stdscr.addstr(y_msg, x, truncated_line)

        for i in range(start, end):
            text = options[i]
            x = w // 2 - len(text) // 2
            x = max(0, min(x, w - 1)) if w > 0 else 0
            y = options_y_start + (i - start)
            if y < 0 or y >= h:
                continue
            truncated_text = text[: w - 1] if w > 0 else text
            if i == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, truncated_text)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, truncated_text)

        if max_visible < len(options) and h > 0 and w > 0:
            ratio = (current_row + 1) / len(options)
            y_scroll = h - 2
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

        stdscr.refresh()

    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    draw_menu()

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(options) - 1:
            current_row += 1
        elif key in [curses.KEY_ENTER, 10, 13]:
            return options[current_row]
        elif key == 27:
            return None

        new_h, new_w = stdscr.getmaxyx()
        if new_h != h or new_w != w:
            h, w = new_h, new_w
        draw_menu()


def get_text_input(stdscr, prompt: str, default: str = "") -> Optional[str]:
    """
    Prompt the user for text input using curses.
    Returns the entered text, or None if ESC is pressed.
    """
    stdscr.clear()
    h, w = stdscr.getmaxyx()

    h = max(1, h)
    w = max(1, w)
    y_prompt = max(0, min((h - 1) // 2, h - 1))

    avail_width = max(1, w - 2)
    prompt_to_show = prompt[:avail_width]
    prompt_len = len(prompt_to_show)

    stdscr.addstr(y_prompt, 0, prompt_to_show)

    current_text = default
    cursor_pos = len(current_text)

    display_width = max(0, w - prompt_len - 1)
    display_text = current_text[:display_width]
    if display_width > 0:
        stdscr.addstr(y_prompt, prompt_len, " " * display_width)
        stdscr.addstr(y_prompt, prompt_len, display_text)

    safe_cursor = min(cursor_pos, max(0, display_width))
    final_col = max(0, min(prompt_len + safe_cursor, w - 1))
    final_row = max(0, min(y_prompt, h - 1))
    stdscr.move(final_row, final_col)

    curses.curs_set(1)
    stdscr.refresh()

    while True:
        key = stdscr.getch()

        h, w = stdscr.getmaxyx()

        if key == 27:
            curses.curs_set(0)
            return None
        elif key in [curses.KEY_ENTER, 10, 13]:
            curses.curs_set(0)
            return current_text
        elif key == curses.KEY_BACKSPACE or key == 127:
            if cursor_pos > 0:
                current_text = (
                    current_text[: cursor_pos - 1] + current_text[cursor_pos:]
                )
                cursor_pos -= 1
        elif key == curses.KEY_LEFT and cursor_pos > 0:
            cursor_pos -= 1
        elif key == curses.KEY_RIGHT and cursor_pos < len(current_text):
            cursor_pos += 1
        elif 32 <= key <= 126:
            current_text = (
                current_text[:cursor_pos] + chr(key) + current_text[cursor_pos:]
            )
            cursor_pos += 1

        avail_width = max(1, w - 2)
        prompt_to_show = prompt[:avail_width]
        prompt_len = len(prompt_to_show)
        stdscr.addstr(y_prompt, 0, prompt_to_show)

        display_width = max(0, w - prompt_len - 1)
        display_text = current_text[:display_width]
        if display_width > 0:
            stdscr.addstr(y_prompt, prompt_len, " " * display_width)
            stdscr.addstr(y_prompt, prompt_len, display_text)

        safe_cursor = min(cursor_pos, max(0, display_width))
        final_col = max(0, min(prompt_len + safe_cursor, w - 1))
        final_row = max(0, min(y_prompt, h - 1))
        stdscr.move(final_row, final_col)
        stdscr.refresh()


def config_screen_main(stdscr):
    """
    Unified configuration screen with consistent ESC behavior.

    ESC at any point returns to the initial screen (initial choice).
    User can then:
    - Start transcriber with current settings
    - Cancel and exit without saving

    Returns Settings if successfully configured, None if cancelled.
    """
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    # Load last settings for defaults
    last_settings = load_settings()

    # Initialize config data with last settings as defaults
    config = ConfigData()
    if last_settings:
        config.device_name = last_settings.device_name
        config.model_type = last_settings.model_type
        config.model_name = last_settings.model_name
        config.compute_type = last_settings.compute_type
        config.device = last_settings.device
        config.language = last_settings.language
        config.hotkey = last_settings.hotkey
        config.llm_correction_enabled = last_settings.llm_correction_enabled
        config.llm_endpoint = last_settings.llm_endpoint
        config.llm_model_name = last_settings.llm_model_name

    current_step = ConfigStep.INITIAL

    while True:
        key_handle_result = _handle_key_transition(stdscr, current_step, config)

        if key_handle_result is None:
            # ESC pressed at initial step - exit without saving
            return None

        elif isinstance(key_handle_result, tuple) and len(key_handle_result) == 2:
            current_step, config = key_handle_result

        else:
            # Success - all steps completed
            result_settings = _create_settings_from_config(config, last_settings)
            return result_settings


def _handle_key_transition(stdscr, current_step: ConfigStep, config: ConfigData):
    """
    Handle the current configuration step.

    Returns:
        - None: ESC pressed at initial screen, exit without saving
        - tuple(new_step, new_config): Continue to next (or previous) screen
        - Settings: Configuration complete
    """
    if current_step == ConfigStep.INITIAL:
        return _screen_initial(stdscr, config)

    elif current_step == ConfigStep.DEVICE:
        return _screen_device(stdscr, config)

    elif current_step == ConfigStep.MODEL_TYPE:
        return _screen_model_type(stdscr, config)

    # Whisper sub-steps
    elif current_step == ConfigStep.WHISPER_MODEL:
        return _screen_whisper_model(stdscr, config)
    elif current_step == ConfigStep.WHISPER_DEVICE:
        return _screen_whisper_device(stdscr, config)
    elif current_step == ConfigStep.WHISPER_PRECISION:
        return _screen_whisper_precision(stdscr, config)
    elif current_step == ConfigStep.WHISPER_LANGUAGE:
        return _screen_whisper_language(stdscr, config)

    # Parakeet sub-steps
    elif current_step == ConfigStep.PARAKEET_DEVICE:
        return _screen_parakeet_device(stdscr, config)

    # Canary sub-steps
    elif current_step == ConfigStep.CANARY_SOURCE_LANG:
        return _screen_canary_source_lang(stdscr, config)
    elif current_step == ConfigStep.CANARY_TARGET_LANG:
        return _screen_canary_target_lang(stdscr, config)

    # Voxtral sub-steps
    elif current_step == ConfigStep.VOXTRAL_DEVICE:
        return _screen_voxtral_device(stdscr, config)
    elif current_step == ConfigStep.VOXTRAL_PRECISION:
        return _screen_voxtral_precision(stdscr, config)
    elif current_step == ConfigStep.VOXTRAL_INFO:
        return _screen_voxtral_info(stdscr, config)

    # Cohere sub-steps
    elif current_step == ConfigStep.COHERE_DEVICE:
        return _screen_cohere_device(stdscr, config)
    elif current_step == ConfigStep.COHERE_LANGUAGE:
        return _screen_cohere_language(stdscr, config)

    # Common final steps
    elif current_step == ConfigStep.HOTKEY:
        return _screen_hotkey(stdscr, config)

    elif current_step == ConfigStep.LLM_ENABLE:
        return _screen_llm_enable(stdscr, config)
    elif current_step == ConfigStep.LLM_ENDPOINT:
        return _screen_llm_endpoint(stdscr, config)
    elif current_step == ConfigStep.LLM_MODEL:
        return _screen_llm_model(stdscr, config)

    # Should never reach here
    return (current_step, config)


def _back_to_initial(config: ConfigData):
    """Return to initial screen (ESC behavior)."""
    return (ConfigStep.INITIAL, config)


# ============================================================================
# Screen Handlers - Each implements a single configuration step
# ============================================================================


def _screen_initial(stdscr, config: ConfigData):
    """Initial choice: use last settings or configure new ones."""
    while True:
        choice = curses_menu(
            stdscr,
            "",
            ["Use Last Settings", "Configure New Settings"],
        )

        if choice is None:
            # ESC at initial screen - exit without saving
            return None

        elif choice == "Use Last Settings":
            # Skip all screens, save unchanged settings, and start immediately
            return _create_settings_from_config(config, None)

        elif choice == "Configure New Settings":
            # Start fresh configuration from device selection
            return (ConfigStep.DEVICE, config)


def _screen_device(stdscr, config: ConfigData):
    """Select audio input device."""
    with Pulse() as pulse:
        sources = pulse.source_list()
        source_map = {src.description: src.name for src in sources}
        device_names = list(source_map.keys())

        # Find current selection index
        initial_idx = 0
        if config.device_name:
            for i, (desc, name) in enumerate(source_map.items()):
                if name == config.device_name:
                    initial_idx = i
                    break

        selected_desc = curses_menu(
            stdscr,
            "Audio Device",
            device_names,
            initial_idx=initial_idx,
        )

        if selected_desc is None:
            return _back_to_initial(config)

        config.device_name = source_map[selected_desc]
        return (ConfigStep.MODEL_TYPE, config)


def _screen_model_type(stdscr, config: ConfigData):
    """Select model type."""
    model_options = [
        "faster-whisper",
        "parakeet-tdt-0.6b-v3",
        "canary-1b-v2",
        "Voxtral-Mini-3B-2507",
        "cohere-transcribe-03-2026",
    ]

    initial_idx = 0
    type_mapping = {
        "whisper": 0,
        "parakeet": 1,
        "canary": 2,
        "voxtral": 3,
        "cohere": 4,
    }
    if config.model_type and config.model_type in type_mapping:
        initial_idx = type_mapping[config.model_type]

    selected = curses_menu(stdscr, "Model Type", model_options, initial_idx=initial_idx)

    if selected is None:
        return _back_to_initial(config)

    # Map display name to internal type
    type_map = {
        "faster-whisper": "whisper",
        "parakeet-tdt-0.6b-v3": "parakeet",
        "canary-1b-v2": "canary",
        "Voxtral-Mini-3B-2507": "voxtral",
        "cohere-transcribe-03-2026": "cohere",
    }

    config.model_type = type_map[selected]

    # Route to model-specific configuration
    if config.model_type == "whisper":
        return (ConfigStep.WHISPER_MODEL, config)
    elif config.model_type == "parakeet":
        return (ConfigStep.PARAKEET_DEVICE, config)
    elif config.model_type == "canary":
        return (ConfigStep.CANARY_SOURCE_LANG, config)
    elif config.model_type == "voxtral":
        return (ConfigStep.VOXTRAL_DEVICE, config)
    elif config.model_type == "cohere":
        return (ConfigStep.COHERE_DEVICE, config)

    return _back_to_initial(config)


# ============================================================================
# Faster-Whisper Configuration Screens
# ============================================================================


def _screen_whisper_model(stdscr, config: ConfigData):
    """Select Whisper model."""
    from .config import accepted_models_whisper

    initial_idx = 0
    if config.model_name and config.model_name in accepted_models_whisper:
        initial_idx = accepted_models_whisper.index(config.model_name)

    selected = curses_menu(
        stdscr,
        "Whisper Model",
        accepted_models_whisper,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.model_name = selected
    return (ConfigStep.WHISPER_DEVICE, config)


def _screen_whisper_device(stdscr, config: ConfigData):
    """Select compute device for Whisper."""
    options = ["cuda", "cpu"]

    initial_idx = 0
    if config.device == "cpu":
        initial_idx = 1

    selected = curses_menu(
        stdscr,
        "Compute Device",
        options,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.device = selected
    return (ConfigStep.WHISPER_PRECISION, config)


def _screen_whisper_precision(stdscr, config: ConfigData):
    """Select precision for Whisper."""
    from .config import english_only_models_whisper

    if config.device == "cpu":
        options = ["int8"]
    else:
        options = ["float16", "int8"]

    initial_idx = 0
    if config.compute_type == "int8" and config.device != "cpu":
        initial_idx = 1
    selected = curses_menu(
        stdscr,
        "Precision",
        options,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.compute_type = selected

    # Check if model is English-only

    if config.model_name in english_only_models_whisper:
        config.language = "en"
        return (ConfigStep.HOTKEY, config)

    return (ConfigStep.WHISPER_LANGUAGE, config)


def _screen_whisper_language(stdscr, config: ConfigData):
    """Select language for Whisper."""
    from .config import accepted_languages_whisper

    initial_idx = 0 if config.language == "" else -1
    if config.language and config.language in accepted_languages_whisper:
        initial_idx = accepted_languages_whisper.index(config.language)

    selected = curses_menu(
        stdscr,
        "Language",
        accepted_languages_whisper,
        initial_idx=initial_idx if initial_idx >= 0 else 0,
    )

    if selected is None:
        return _back_to_initial(config)

    config.language = selected
    return (ConfigStep.HOTKEY, config)


# ============================================================================
# Parakeet Configuration Screens
# ============================================================================


def _screen_parakeet_device(stdscr, config: ConfigData):
    """Select compute device for Parakeet."""
    options = ["cuda", "cpu"]

    initial_idx = 0
    if config.device == "cpu":
        initial_idx = 1

    selected = curses_menu(
        stdscr,
        "Compute Device",
        options,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.model_name = "nvidia/parakeet-tdt-0.6b-v3"
    config.device = selected
    config.compute_type = "float16"
    config.language = ""  # Parakeet supports auto-detection

    return (ConfigStep.HOTKEY, config)


# ============================================================================
# Canary Configuration Screens
# ============================================================================


def _screen_canary_source_lang(stdscr, config: ConfigData):
    """Select source language for Canary."""
    from .config import canary_source_target_languages

    initial_idx = 0
    if config.language and "-" in config.language:
        src = config.language.split("-")[0]
        if src in canary_source_target_languages:
            initial_idx = canary_source_target_languages.index(src)

    selected = curses_menu(
        stdscr,
        "Source Language",
        canary_source_target_languages,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    # Store temporarily for next step
    config.language = selected  # Will become source part of final language
    return (ConfigStep.CANARY_TARGET_LANG, config)


def _screen_canary_target_lang(stdscr, config: ConfigData):
    """Select target language for Canary."""
    from .config import canary_allowed_language_pairs

    src = config.language  # Set in previous step

    allowed_targets = {
        p.split("-")[1]
        for p in canary_allowed_language_pairs
        if p.startswith(src + "-")
    }
    target_options = sorted(allowed_targets)

    initial_idx = 0
    if config.language and "-" in config.language:
        final_lang = config.language
        parts = final_lang.split("-")
        if len(parts) >= 2 and parts[1] in target_options:
            initial_idx = target_options.index(parts[1])

    selected = curses_menu(
        stdscr,
        "Target Language (same as source for transcription)",
        target_options,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.model_name = "nvidia/canary-1b-v2"
    config.device = "cpu"  # Canary only supports CPU
    config.compute_type = "float16"
    config.language = f"{src}-{selected}"

    return (ConfigStep.HOTKEY, config)


# ============================================================================
# Voxtral Configuration Screens
# ============================================================================


def _screen_voxtral_device(stdscr, config: ConfigData):
    """Select compute device for Voxtral (GPU only)."""
    options = ["cuda"]  # Voxtral is GPU-only

    selected = curses_menu(stdscr, "Compute Device (CUDA only)", options)

    if selected is None:
        return _back_to_initial(config)

    config.device = selected
    return (ConfigStep.VOXTRAL_PRECISION, config)


def _screen_voxtral_precision(stdscr, config: ConfigData):
    """Select precision for Voxtral."""
    options = ["float16", "int8", "int4"]

    initial_idx = 0
    if config.compute_type in options:
        initial_idx = options.index(config.compute_type)

    selected = curses_menu(stdscr, "Precision", options, initial_idx=initial_idx)

    if selected is None:
        return _back_to_initial(config)

    config.compute_type = selected
    return (ConfigStep.VOXTRAL_INFO, config)


def _screen_voxtral_info(stdscr, config: ConfigData):
    """Display info about Voxtral audio length limit."""
    info_message = (
        "For Voxtral-Mini-3B-2507, keep the audio <30s to avoid\n"
        "chunking inconsistencies.\n\n"
        "Press ENTER to continue."
    )

    selected = curses_menu(
        stdscr,
        "Voxtral Info",
        ["Continue"],
        message=info_message,
    )

    if selected is None:
        return _back_to_initial(config)

    config.model_name = "mistralai/Voxtral-Mini-3B-2507"
    config.language = "auto"  # Voxtral supports auto-detection

    return (ConfigStep.HOTKEY, config)


# ============================================================================
# Cohere Configuration Screens
# ============================================================================


def _screen_cohere_device(stdscr, config: ConfigData):
    """Select compute device for Cohere."""
    options = ["cuda", "cpu"]

    initial_idx = 0
    if config.device == "cpu":
        initial_idx = 1

    selected = curses_menu(
        stdscr,
        "Compute Device",
        options,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.device = selected
    return (ConfigStep.COHERE_LANGUAGE, config)


def _screen_cohere_language(stdscr, config: ConfigData):
    """Select language for Cohere (no auto-detection)."""
    from .config import accepted_languages_cohere

    initial_idx = 0
    if config.language and config.language in accepted_languages_cohere:
        initial_idx = accepted_languages_cohere.index(config.language)

    selected = curses_menu(
        stdscr,
        "Language (no auto-detection)",
        accepted_languages_cohere,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.model_name = "CohereLabs/cohere-transcribe-03-2026"
    config.compute_type = "float16"
    config.language = selected

    return (ConfigStep.HOTKEY, config)


# ============================================================================
# Common Configuration Screens
# ============================================================================


def _screen_hotkey(stdscr, config: ConfigData):
    """Select hotkey."""
    hotkey_options = ["Pause", "F4", "F8", "INSERT"]

    initial_idx = 0
    if config.hotkey and config.hotkey.upper() in hotkey_options:
        initial_idx = hotkey_options.index(config.hotkey.upper())

    selected = curses_menu(
        stdscr,
        "Hotkey",
        hotkey_options,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.hotkey = selected.lower()
    return (ConfigStep.LLM_ENABLE, config)


def _screen_llm_enable(stdscr, config: ConfigData):
    """Enable or disable LLM correction."""
    options = ["Yes", "No"]
    initial_idx = 0 if config.llm_correction_enabled else 1

    selected = curses_menu(
        stdscr,
        "Enable LLM Correction?",
        options,
        initial_idx=initial_idx,
    )

    if selected is None:
        return _back_to_initial(config)

    config.llm_correction_enabled = selected == "Yes"

    if not config.llm_correction_enabled:
        config.llm_endpoint = ""
        config.llm_model_name = ""
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
    default = config.llm_model_name or ""

    result = get_text_input(stdscr, "Model name: ", default)

    if result is None:
        return _back_to_initial(config)

    config.llm_model_name = result
    return _final_save(stdscr, config)


def _final_save(stdscr, config: ConfigData):
    """Display confirmation and save settings."""
    confirmation_msg = "All settings configured.\nPress ENTER to save and continue."

    selected = curses_menu(
        stdscr,
        "Configuration Complete",
        ["Save & Continue"],
        message=confirmation_msg,
    )

    if selected is None:
        return _back_to_initial(config)

    # Create settings and save
    result_settings = _create_settings_from_config(config, None)
    return result_settings


def _create_settings_from_config(
    config: ConfigData, last_settings: Settings | None
) -> Settings:
    """Create a Settings object from ConfigData."""
    settings_dict = {
        "device_name": config.device_name or "",
        "model_type": config.model_type or "",
        "model_name": config.model_name,
        "compute_type": config.compute_type,
        "device": config.device,
        "language": config.language,
        "hotkey": config.hotkey,
        "llm_correction_enabled": config.llm_correction_enabled,
        "llm_endpoint": config.llm_endpoint,
        "llm_model_name": config.llm_model_name,
    }

    save_settings(settings_dict)
    return Settings(**settings_dict)
