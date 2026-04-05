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

from .settings import Settings
from .transcriber import MicrophoneTranscriber
from .ui import config_screen_main


def _setup_logging():
    """Configure logging based on DEBUG environment variable."""
    import os

    is_debug = os.environ.get("FASTER_WHISPER_HOTKEY_DEBUG", "0") == "1"
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if is_debug:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    else:

        class SimpleFormatter(logging.Formatter):
            def format(self, record):
                return f"{record.levelname}:{record.getMessage()}"

        handler = logging.StreamHandler()
        handler.setFormatter(SimpleFormatter())
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)


_setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Main entry point - runs the unified config screen."""
    settings: Settings | None = None

    while True:
        try:
            result = curses.wrapper(config_screen_main)

            # result is either a Settings object (success) or None (aborted/cancelled)
            if isinstance(result, Settings):
                settings = result
                break

            logger.info("Configuration cancelled. Exiting.")
            return

        except KeyboardInterrupt:
            logger.info("Program terminated by user")
            break

    # Launch the transcriber with configured settings
    assert settings is not None
    transcriber = MicrophoneTranscriber(settings)
    try:
        transcriber.run()
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
