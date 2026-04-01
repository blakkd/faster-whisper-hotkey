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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
                # 3️⃣  Model‑specific configuration
                # ------------------------------------------------------------------
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

                    available_compute_types = (
                        ["int8"] if device == "cpu" else ["float16", "int8"]
                    )
                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Precision",
                            available_compute_types,
                        )
                    )
                    if not compute_type:
                        continue

                    language = (
                        "en"
                        if english_only
                        else curses.wrapper(
                            lambda stdscr: curses_menu(
                                stdscr, "", accepted_languages_whisper
                            )
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

                    llm_enabled = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Enable LLM correction?", ["Yes", "No"]
                        )
                    )
                    if not llm_enabled:
                        continue

                    llm_correction_enabled = llm_enabled == "Yes"
                    llm_endpoint = ""
                    llm_model_name = ""

                    if llm_correction_enabled:
                        llm_endpoint = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Endpoint URL: ",
                                "http://localhost:8678/v1",
                            )
                        )
                        if not llm_endpoint:
                            continue

                        llm_model_name = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Model name: ",
                                "",
                            )
                        )
                        if not llm_model_name:
                            llm_model_name = "default"

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "whisper",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                            "llm_correction_enabled": llm_correction_enabled,
                            "llm_endpoint": llm_endpoint,
                            "llm_model_name": llm_model_name,
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
                        llm_correction_enabled=llm_correction_enabled,
                        llm_endpoint=llm_endpoint,
                        llm_model_name=llm_model_name,
                    )

                elif model_type == "canary-1b-v2":
                    # ------------------------------------------------------------------
                    # 4️⃣  Canary (nvidia/canary-1b-v2) – no quantisation support, CUDA only
                    # ------------------------------------------------------------------
                    model_name = "nvidia/canary-1b-v2"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Compute Device",
                            ["cuda"],
                            message="CUDA only (CPU inference not supported)",
                        )
                    )
                    if not device:
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

                    # Build allowed target languages based on the selected source
                    allowed_targets = {
                        tgt
                        for src, tgt in (
                            p.split("-") for p in canary_allowed_language_pairs
                        )
                        if src == source_language
                    }
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

                    # Explicit compute_type to keep compatibility with older Settings
                    compute_type = "float16"

                    llm_enabled = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Enable LLM correction?", ["Yes", "No"]
                        )
                    )
                    if not llm_enabled:
                        continue

                    llm_correction_enabled = llm_enabled == "Yes"
                    llm_endpoint = ""
                    llm_model_name = ""

                    if llm_correction_enabled:
                        llm_endpoint = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Endpoint URL: ",
                                "http://localhost:8678/v1",
                            )
                        )
                        if not llm_endpoint:
                            continue

                        llm_model_name = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Model name: ",
                                "",
                            )
                        )
                        if not llm_model_name:
                            llm_model_name = "default"

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "canary",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": f"{source_language}-{target_language}",
                            "hotkey": hotkey,
                            "llm_correction_enabled": llm_correction_enabled,
                            "llm_endpoint": llm_endpoint,
                            "llm_model_name": llm_model_name,
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
                        llm_correction_enabled=llm_correction_enabled,
                        llm_endpoint=llm_endpoint,
                        llm_model_name=llm_model_name,
                    )

                elif model_type == "parakeet-tdt-0.6b-v3":
                    # ------------------------------------------------------------------
                    # 5️⃣  Parakeet (nvidia/parakeet-tdt-0.6b-v3) – no quantisation support
                    # ------------------------------------------------------------------
                    model_name = "nvidia/parakeet-tdt-0.6b-v3"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Compute Device", ["cuda", "cpu"]
                        )
                    )
                    if not device:
                        continue

                    language = ""  # Parakeet does not require a language flag

                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "Hotkey", hotkey_options)
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

                    # Explicit compute_type to keep compatibility with older Settings
                    compute_type = "float16"

                    llm_enabled = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Enable LLM correction?", ["Yes", "No"]
                        )
                    )
                    if not llm_enabled:
                        continue

                    llm_correction_enabled = llm_enabled == "Yes"
                    llm_endpoint = ""
                    llm_model_name = ""

                    if llm_correction_enabled:
                        llm_endpoint = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Endpoint URL: ",
                                "http://localhost:8678/v1",
                            )
                        )
                        if not llm_endpoint:
                            continue

                        llm_model_name = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Model name: ",
                                "",
                            )
                        )
                        if not llm_model_name:
                            llm_model_name = "default"

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "parakeet",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                            "llm_correction_enabled": llm_correction_enabled,
                            "llm_endpoint": llm_endpoint,
                            "llm_model_name": llm_model_name,
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
                        llm_correction_enabled=llm_correction_enabled,
                        llm_endpoint=llm_endpoint,
                        llm_model_name=llm_model_name,
                    )

                elif model_type == "Voxtral-Mini-3B-2507":
                    # ------------------------------------------------------------------
                    # 6️⃣  Voxtral (mistralai/Voxtral-Mini-3B-2507) – CUDA only
                    # ------------------------------------------------------------------
                    model_name = "mistralai/Voxtral-Mini-3B-2507"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Compute Device",
                            ["cuda"],
                            message="CUDA only (CPU inference not supported)",
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

                    # Inform users about the <30s limit for Voxtral
                    info_message_voxtral = (
                        "For Voxtral-Mini-3B-2507, keep the audio <30s to avoid "
                        "chunking inconsistencies."
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

                    llm_enabled = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Enable LLM correction?", ["Yes", "No"]
                        )
                    )
                    if not llm_enabled:
                        continue

                    llm_correction_enabled = llm_enabled == "Yes"
                    llm_endpoint = ""
                    llm_model_name = ""

                    if llm_correction_enabled:
                        llm_endpoint = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Endpoint URL: ",
                                "http://localhost:8678/v1",
                            )
                        )
                        if not llm_endpoint:
                            continue

                        llm_model_name = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Model name: ",
                                "",
                            )
                        )
                        if not llm_model_name:
                            llm_model_name = "default"

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "voxtral",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                            "llm_correction_enabled": llm_correction_enabled,
                            "llm_endpoint": llm_endpoint,
                            "llm_model_name": llm_model_name,
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
                        llm_correction_enabled=llm_correction_enabled,
                        llm_endpoint=llm_endpoint,
                        llm_model_name=llm_model_name,
                    )

                elif model_type == "cohere-transcribe-03-2026":
                    # ------------------------------------------------------------------
                    # 7️⃣  Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026)
                    # ------------------------------------------------------------------
                    model_name = "CohereLabs/cohere-transcribe-03-2026"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Compute Device", ["cuda", "cpu"]
                        )
                    )
                    if not device:
                        continue

                    cohere_languages = [
                        "en",
                        "de",
                        "fr",
                        "it",
                        "es",
                        "pt",
                        "el",
                        "nl",
                        "pl",
                        "ar",
                        "vi",
                        "zh",
                        "ja",
                        "ko",
                    ]
                    language = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Language (no auto-detection)", cohere_languages
                        )
                    )
                    if not language:
                        continue

                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "Hotkey", hotkey_options)
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

                    compute_type = "float16"

                    llm_enabled = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Enable LLM correction?", ["Yes", "No"]
                        )
                    )
                    if not llm_enabled:
                        continue

                    llm_correction_enabled = llm_enabled == "Yes"
                    llm_endpoint = ""
                    llm_model_name = ""

                    if llm_correction_enabled:
                        llm_endpoint = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Endpoint URL: ",
                                "http://localhost:8678/v1",
                            )
                        )
                        if not llm_endpoint:
                            continue

                        llm_model_name = curses.wrapper(
                            lambda stdscr: get_text_input(
                                stdscr,
                                "Model name: ",
                                "",
                            )
                        )
                        if not llm_model_name:
                            llm_model_name = "default"

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "cohere",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                            "llm_correction_enabled": llm_correction_enabled,
                            "llm_endpoint": llm_endpoint,
                            "llm_model_name": llm_model_name,
                        }
                    )
                    settings = Settings(
                        device_name=device_name,
                        model_type="cohere",
                        model_name=model_name,
                        compute_type=compute_type,
                        device=device,
                        language=language,
                        hotkey=hotkey,
                        llm_correction_enabled=llm_correction_enabled,
                        llm_endpoint=llm_endpoint,
                        llm_model_name=llm_model_name,
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
