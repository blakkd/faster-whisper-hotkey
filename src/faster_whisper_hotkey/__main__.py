"""Entry point."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Push-to-talk transcription tool for Linux"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug logging with full module paths",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="skip config UI and start transcribing with saved settings",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to custom settings file (default: ~/.config/faster_whisper_hotkey/transcriber_settings.json)",
    )
    args = parser.parse_args()

    # Set debug flag before any imports so _setup_logging() picks it up
    if args.debug:
        os.environ["FASTER_WHISPER_HOTKEY_DEBUG"] = "1"

    from faster_whisper_hotkey.transcribe import main as transcribe_main

    transcribe_main(headless=args.headless, settings_file=args.config)


if __name__ == "__main__":
    main()
