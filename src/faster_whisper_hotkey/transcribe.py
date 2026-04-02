# src/faster_whisper_hotkey/transcribe.py
import curses
import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message="invalid escape sequence '\\s'",
    category=SyntaxWarning,
    module="lhotse.recipes.iwslt22_ta",
)
warnings.filterwarnings(
    "ignore",
    message="invalid escape sequence '\\('",
    category=SyntaxWarning,
    module="pydub.utils",
)

import pulsectl

from .config import (
    accepted_languages_cohere,
    accepted_languages_whisper,
    accepted_models_whisper,
    canary_allowed_language_pairs,
    canary_source_target_languages,
    english_only_models_whisper,
)
from .settings import Settings, load_settings, save_settings
from .ui import get_text_input
from .transcriber import MicrophoneTranscriber
from .ui import curses_menu, get_initial_choice

import os


def _setup_logging():
    """Configure logging based on DEBUG environment variable."""
    is_debug = os.environ.get("FASTER_WHISPER_HOTKEY_DEBUG", "0") == "1"
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if is_debug:
        # Full logging with module paths (default behavior)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    else:
        # Simplified logging without module paths
        class SimpleFormatter(logging.Formatter):
            def format(self, record):
                return f"{record.levelname}:{record.getMessage()}"

        handler = logging.StreamHandler()
        handler.setFormatter(SimpleFormatter())
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)


_setup_logging()
logger = logging.getLogger(__name__)

HOTKEY_OPTIONS = ["Pause", "F4", "F8", "INSERT"]


def _get_hotkey() -> str | None:
    """Select hotkey from available options."""
    selected = curses.wrapper(
        lambda stdscr: curses_menu(stdscr, "Hotkey", HOTKEY_OPTIONS)
    )
    return selected.lower() if selected else None


def _get_device_choice(cuda_only: bool = False) -> str | None:
    """Get compute device from user. If cuda_only, only offer CUDA."""
    options = ["cuda"] if cuda_only else ["cuda", "cpu"]
    message = "CUDA only (CPU inference not supported)" if cuda_only else ""
    return curses.wrapper(
        lambda stdscr: curses_menu(stdscr, "Compute Device", options, message=message)
    )


def _configure_llm_correction() -> tuple[bool, str, str] | None:
    """Configure LLM correction settings. Returns (enabled, endpoint, model_name) or None."""
    enabled = curses.wrapper(
        lambda stdscr: curses_menu(stdscr, "Enable LLM correction?", ["Yes", "No"])
    )
    if enabled != "Yes":
        return None

    endpoint = curses.wrapper(
        lambda stdscr: get_text_input(
            stdscr, "Endpoint URL: ", "http://localhost:8678/v1"
        )
    )
    if not endpoint:
        return None

    model_name = curses.wrapper(
        lambda stdscr: get_text_input(stdscr, "Model name: ", "")
    )
    return True, endpoint, model_name or "default"


def _configure_whisper() -> dict | None:
    """Configure faster-whisper model settings."""
    selected_model = curses.wrapper(
        lambda stdscr: curses_menu(stdscr, "Whisper Model", accepted_models_whisper)
    )
    if not selected_model:
        return None

    english_only = selected_model in english_only_models_whisper
    device = _get_device_choice(cuda_only=False)
    if not device:
        return None

    available_compute_types = (
        ["int8"] if device == "cpu" else ["float16", "int8"]
    )
    compute_type = curses.wrapper(
        lambda stdscr: curses_menu(stdscr, "Precision", available_compute_types)
    )
    if not compute_type:
        return None

    language = (
        "en"
        if english_only
        else curses.wrapper(
            lambda stdscr: curses_menu(stdscr, "Language", accepted_languages_whisper)
        )
    )
    if not language:
        return None

    return {
        "model_name": selected_model,
        "compute_type": compute_type,
        "device": device,
        "language": language,
    }


def _configure_canary() -> dict | None:
    """Configure Canary model settings."""
    model_name = "nvidia/canary-1b-v2"
    device = _get_device_choice(cuda_only=True)
    if not device:
        return None

    source_language = curses.wrapper(
        lambda stdscr: curses_menu(
            stdscr, "Source Language", canary_source_target_languages
        )
    )
    if not source_language:
        return None

    allowed_targets = {
        tgt
        for src, tgt in (p.split("-") for p in canary_allowed_language_pairs)
        if src == source_language
    }
    target_options = sorted(allowed_targets)

    target_language = curses.wrapper(
        lambda stdscr: curses_menu(
            stdscr, "Target Language (same as source for transcription)",
            target_options,
        )
    )
    if not target_language:
        return None

    return {
        "model_name": model_name,
        "compute_type": "float16",
        "device": device,
        "language": f"{source_language}-{target_language}",
    }


def _configure_parakeet() -> dict | None:
    """Configure Parakeet model settings."""
    model_name = "nvidia/parakeet-tdt-0.6b-v3"
    device = _get_device_choice(cuda_only=False)
    if not device:
        return None

    return {
        "model_name": model_name,
        "compute_type": "float16",
        "device": device,
        "language": "",
    }


def _configure_voxtral() -> dict | None:
    """Configure Voxtral model settings."""
    model_name = "mistralai/Voxtral-Mini-3B-2507"
    device = _get_device_choice(cuda_only=True)
    if not device:
        return None

    compute_type = curses.wrapper(
        lambda stdscr: curses_menu(stdscr, "Precision", ["float16", "int8", "int4"])
    )
    if not compute_type:
        return None

    info_message = (
        "For Voxtral-Mini-3B-2507, keep the audio <30s to avoid "
        "chunking inconsistencies."
    )
    curses.wrapper(
        lambda stdscr: curses_menu(stdscr, "Info", ["Continue"], message=info_message)
    )

    return {
        "model_name": model_name,
        "compute_type": compute_type,
        "device": device,
        "language": "auto",
    }


def _configure_cohere() -> dict | None:
    """Configure Cohere model settings."""
    model_name = "CohereLabs/cohere-transcribe-03-2026"
    device = _get_device_choice(cuda_only=False)
    if not device:
        return None

    language = curses.wrapper(
        lambda stdscr: curses_menu(
            stdscr, "Language (no auto-detection)", accepted_languages_cohere
        )
    )
    if not language:
        return None

    return {
        "model_name": model_name,
        "compute_type": "float16",
        "device": device,
        "language": language,
    }


def _save_and_create_settings(settings_dict: dict) -> Settings:
    """Save settings and create Settings object."""
    save_settings(settings_dict)
    return Settings(**settings_dict)


def _create_settings_dict(
    device_name: str,
    model_type: str,
    model_name: str,
    compute_type: str,
    device: str,
    language: str,
    hotkey: str,
    llm_result: tuple[bool, str, str],
) -> dict:
    """Create settings dictionary."""
    llm_correction_enabled, llm_endpoint, llm_model_name = llm_result
    return {
        "device_name": device_name,
        "model_type": model_type,
        "model_name": model_name,
        "compute_type": compute_type,
        "device": device,
        "language": language,
        "hotkey": hotkey,
        "llm_correction_enabled": llm_correction_enabled,
        "llm_endpoint": llm_endpoint,
        "llm_model_name": llm_model_name,
    }


def main():
    settings: Settings | None = None
    while True:
        try:
            initial_choice = curses.wrapper(get_initial_choice)

            if initial_choice not in ["Use Last Settings", "Choose New Settings"]:
                continue

            if initial_choice == "Use Last Settings":
                settings = load_settings()
                if not settings:
                    logger.info(
                        "No previous settings found. Proceeding with new settings."
                    )
                    initial_choice = "Choose New Settings"

            if initial_choice == "Choose New Settings":
                # ------------------------------------------------------------------
                # 1️⃣  Audio source selection
                # ------------------------------------------------------------------
                with pulsectl.Pulse() as pulse:
                    sources = pulse.source_list()
                    source_map = {src.description: src.name for src in sources}
                    selected_device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Audio Device", list(source_map.keys())
                        )
                    )
                    device_name = (
                        source_map[selected_device] if selected_device else None
                    )
                    if not device_name:
                        continue

                # ------------------------------------------------------------------
                # 2️⃣  Model type
                # ------------------------------------------------------------------
                model_type_options = [
                    "faster-whisper",
                    "parakeet-tdt-0.6b-v3",
                    "canary-1b-v2",
                    "Voxtral-Mini-3B-2507",
                    "cohere-transcribe-03-2026",
                ]
                model_type = curses.wrapper(
                    lambda stdscr: curses_menu(stdscr, "Model", model_type_options)
                )
                if not model_type:
                    continue

                # ------------------------------------------------------------------
                # 3️⃣  Model-specific configuration
                # ------------------------------------------------------------------
                internal_model_type = None
                model_config = None

                if model_type == "faster-whisper":
                    internal_model_type = "whisper"
                    model_config = _configure_whisper()

                elif model_type == "canary-1b-v2":
                    internal_model_type = "canary"
                    model_config = _configure_canary()

                elif model_type == "parakeet-tdt-0.6b-v3":
                    internal_model_type = "parakeet"
                    model_config = _configure_parakeet()

                elif model_type == "Voxtral-Mini-3B-2507":
                    internal_model_type = "voxtral"
                    model_config = _configure_voxtral()

                elif model_type == "cohere-transcribe-03-2026":
                    internal_model_type = "cohere"
                    model_config = _configure_cohere()

                if model_config is None:
                    continue

                hotkey = _get_hotkey()
                if not hotkey:
                    continue

                llm_result = _configure_llm_correction()
                if llm_result is None:
                    continue

                settings = _save_and_create_settings(
                    _create_settings_dict(
                        device_name=device_name,
                        model_type=internal_model_type,
                        **model_config,
                        hotkey=hotkey,
                        llm_result=llm_result,
                    )
                )

            # ----------------------------------------------------------------------
            # 8️⃣  Launch the transcriber
            # ----------------------------------------------------------------------
            assert settings is not None, (
                "Settings must be defined before launching the transcriber."
            )
            transcriber = MicrophoneTranscriber(settings)
            try:
                transcriber.run()
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                continue

        except KeyboardInterrupt:
            logger.info("Program terminated by user")
            break


if __name__ == "__main__":
    main()
