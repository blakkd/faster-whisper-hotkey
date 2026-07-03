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

from .settings import Settings, load_settings
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
        root_logger.setLevel(logging.DEBUG)
    else:
        # Suppress third-party INFO/DEBUG noise; only our package logs at INFO
        class SimpleFormatter(logging.Formatter):
            def format(self, record):
                return f"{record.levelname}:{record.getMessage()}"

        handler = logging.StreamHandler()
        handler.setFormatter(SimpleFormatter())
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.WARNING)

        # Our package logger still shows INFO+
        logging.getLogger("faster_whisper_hotkey").setLevel(logging.INFO)


_setup_logging()
logger = logging.getLogger(__name__)


def main(headless: bool = False, settings_file: str | None = None):
    """Main entry point - runs config screen or starts headless with saved settings."""
    settings: Settings | None = None

    if headless:
        settings = load_settings(settings_file)
        if settings is None:
            logger.error(
                "No saved settings found. Run without --headless to configure first, "
                "or use --config to specify a settings file."
            )
            return
        logger.info(f"Headless mode: loaded settings from {settings_file or 'default path'}")
    else:
        while True:
            try:
                result = curses.wrapper(
                    lambda scr: config_screen_main(scr, settings_file)
                )

                # result is either a Settings object (success) or None (aborted/cancelled)
                if isinstance(result, Settings):
                    settings = result
                    break

                logger.info("Configuration cancelled. Exiting.")
                return

            except KeyboardInterrupt:
                logger.info("Program terminated by user")
                return

    # Launch the transcriber with configured settings
    assert settings is not None
    transcriber = MicrophoneTranscriber(settings)
    try:
        transcriber.run()
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
