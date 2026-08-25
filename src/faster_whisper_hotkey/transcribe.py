# src/faster_whisper_hotkey/transcribe.py
import collections
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


def _read_pending_input() -> bytes:
    """Read all keystrokes buffered in the tty before curses took over.

    While the app is starting up (heavy imports), the terminal is still in the
    shell's echo/cooked mode: any key the user presses is echoed and buffered
    in the tty line queue. The bytes are read out here so they can be parsed
    and replayed to the TUI once the first screen is up.
    """
    import fcntl
    import os
    import select
    import sys

    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return b""

        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except (AttributeError, OSError):
        return b""

    data = bytearray()
    try:
        while select.select([fd], [], [], 0)[0]:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            data.extend(chunk)
    finally:
        try:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        except OSError:
            pass

    return bytes(data)


def _parse_key_sequence(data: bytes) -> list[int]:
    """Translate raw pre-TUI input bytes into a list of curses key codes.

    Handles bare keys (ESC, Enter, backspace, printables) and arrow-key
    escape sequences in both cursor-key modes (ESC [ A-D and ESC O A-D).
    Other escape sequences (home, delete, ...) are dropped as a whole so an
    unknown or truncated sequence can never masquerade as a bare ESC.
    """
    keys: list[int] = []
    arrow_map = {
        ord("A"): curses.KEY_UP,
        ord("B"): curses.KEY_DOWN,
        ord("C"): curses.KEY_RIGHT,
        ord("D"): curses.KEY_LEFT,
    }
    i, n = 0, len(data)

    while i < n:
        b = data[i]
        if b != 27:
            if b in (10, 13, 127) or 32 <= b <= 126:
                keys.append(b)
            i += 1
            continue

        if i + 1 < n and data[i + 1] in (ord("["), ord("O")):
            j = i + 2
            if data[i + 1] == ord("["):  # CSI: skip parameters/intermediates
                while j < n and 0x30 <= data[j] <= 0x3F:
                    j += 1
                while j < n and 0x20 <= data[j] <= 0x2F:
                    j += 1
            if j < n and 0x40 <= data[j] <= 0x7E:
                key = arrow_map.get(data[j])
                if key is not None:
                    keys.append(key)
                i = j + 1
            else:
                break  # truncated sequence at end of buffer: drop it
            continue

        keys.append(27)
        i += 1

    return keys


class _ReplayWindow:
    """Curses window proxy that serves replayed pre-TUI keys before live input."""

    def __init__(self, window, keys: list[int]):
        self._window = window
        self._keys = collections.deque(keys)

    def getch(self):
        if self._keys:
            return self._keys.popleft()
        return self._window.getch()

    def __getattr__(self, name):
        return getattr(self._window, name)


def _run_config_screen(stdscr, settings_file: str | None):
    """curses.wrapper callback: run the config TUI, replaying pre-TUI keystrokes first."""
    from .ui import config_screen_main

    stale_keys = _parse_key_sequence(_read_pending_input())
    if stale_keys:
        stdscr = _ReplayWindow(stdscr, stale_keys)
    return config_screen_main(stdscr, settings_file)


def main(headless: bool = False, settings_file: str | None = None):
    """Main entry point - runs config screen or starts headless with saved settings."""
    from .settings import Settings, load_settings
    from .transcriber import MicrophoneTranscriber

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
                result = curses.wrapper(_run_config_screen, settings_file)

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
