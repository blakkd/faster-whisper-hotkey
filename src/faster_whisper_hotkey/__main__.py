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
        help="Enable debug logging with full module paths",
    )
    args = parser.parse_args()

    # Set debug flag before any imports so _setup_logging() picks it up
    if args.debug:
        os.environ["FASTER_WHISPER_HOTKEY_DEBUG"] = "1"

    from faster_whisper_hotkey.transcribe import main as transcribe_main

    transcribe_main()


if __name__ == "__main__":
    main()
