import logging
import curses
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

from .ui import get_initial_choice, curses_menu
from .config import (
    accepted_models_whisper,
    accepted_languages_whisper,
    english_only_models_whisper,
    canary_source_target_languages,
    canary_allowed_language_pairs,
)
from .settings import save_settings, load_settings, Settings
from .transcriber import MicrophoneTranscriber

import pulsectl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
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
                with pulsectl.Pulse() as pulse:
                    sources = pulse.source_list()
                    source_names = [src.name for src in sources]
                    device_name = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "Audio Device", source_names)
                    )
                    if not device_name:
                        continue

                model_type_options = [
                    "faster-whisper",
                    "parakeet-tdt-0.6b-v3",
                    "canary-1b-v2",
                    "Voxtral-Mini-3B-2507",
                ]
                model_type = curses.wrapper(
                    lambda stdscr: curses_menu(stdscr, "Model", model_type_options)
                )
                if not model_type:
                    continue

                if model_type == "faster-whisper":
                    original_models = accepted_models_whisper
                    display_models = [m for m in original_models]
                    selected_model = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "", display_models)
                    )
                    if not selected_model:
                        continue

                    model_name = selected_model
                    english_only = model_name in english_only_models_whisper

                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Compute Device", ["cuda", "cpu"]
                        )
                    )
                    if not device:
                        continue

                    if device == "cpu":
                        available_compute_types = ["int8"]
                        info_message = ""
                    else:
                        available_compute_types = ["float16", "int8"]
                        info_message = ""

                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Precision",
                            available_compute_types,
                            message=info_message,
                        )
                    )
                    if not compute_type:
                        continue

                    if english_only:
                        language = "en"
                    else:
                        language = curses.wrapper(
                            lambda stdscr: curses_menu(
                                stdscr, "", accepted_languages_whisper
                            )
                        )
                        if not language:
                            continue

                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Select Hotkey", hotkey_options
                        )
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "whisper",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                        }
                    )
                    settings = Settings(
                        device_name=device_name,
                        model_type="whisper",
                        model_name=model_name,
                        compute_type=compute_type,
                        device=device,
                        language=language,
                        hotkey=hotkey,
                    )
                elif model_type == "canary-1b-v2":
                    model_name = "nvidia/canary-1b-v2"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Compute Device", ["cuda", "cpu"]
                        )
                    )
                    if not device:
                        continue

                    available_compute_types = ["float16", "int8"]
                    info_message = ""

                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Precision",
                            available_compute_types,
                            message=info_message,
                        )
                    )
                    if not compute_type:
                        continue

                    source_language = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Source Language",
                            canary_source_target_languages,
                        )
                    )
                    if not source_language:
                        continue

                    allowed_pairs = canary_allowed_language_pairs
                    allowed_targets = set()
                    for pair in allowed_pairs:
                        src, tgt = pair.split("-")
                        if src == source_language:
                            allowed_targets.add(tgt)
                    target_options = sorted(allowed_targets)

                    target_language = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Target Language (same as source for transcription)",
                            target_options,
                        )
                    )
                    if not target_language:
                        continue

                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "Hotkey", hotkey_options)
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "canary",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": f"{source_language}-{target_language}",
                            "hotkey": hotkey,
                        }
                    )
                    settings = Settings(
                        device_name=device_name,
                        model_type="canary",
                        model_name=model_name,
                        compute_type=compute_type,
                        device=device,
                        language=f"{source_language}-{target_language}",
                        hotkey=hotkey,
                    )
                elif model_type == "parakeet-tdt-0.6b-v3":
                    model_name = "nvidia/parakeet-tdt-0.6b-v3"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Compute Device", ["cuda", "cpu"]
                        )
                    )
                    if not device:
                        continue

                    available_compute_types = ["float16", "int8"]
                    info_message = ""

                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Precision",
                            available_compute_types,
                            message=info_message,
                        )
                    )
                    if not compute_type:
                        continue

                    language = ""

                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "Hotkey", hotkey_options)
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "parakeet",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                        }
                    )
                    settings = Settings(
                        device_name=device_name,
                        model_type="parakeet",
                        model_name=model_name,
                        compute_type=compute_type,
                        device=device,
                        language=language,
                        hotkey=hotkey,
                    )
                elif model_type == "Voxtral-Mini-3B-2507":
                    model_name = "mistralai/Voxtral-Mini-3B-2507"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Compute Device",
                            ["cuda"],
                            message="GPU only",
                        )
                    )
                    if not device:
                        continue

                    available_compute_types = ["float16", "int8", "int4"]
                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Precision", available_compute_types
                        )
                    )
                    if not compute_type:
                        continue

                    info_message_voxtral = (
                        "For Voxtral-Mini-3B-2507, audio is chunked in 30s segments."
                    )
                    curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Info", ["Continue"], message=info_message_voxtral
                        )
                    )

                    language = "auto"

                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "Hotkey", hotkey_options)
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "voxtral",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                        }
                    )
                    settings = Settings(
                        device_name=device_name,
                        model_type="voxtral",
                        model_name=model_name,
                        compute_type=compute_type,
                        device=device,
                        language=language,
                        hotkey=hotkey,
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
