import logging

logger = logging.getLogger(__name__)

try:
    import pyperclip  # type: ignore
except Exception:
    pyperclip = None
    logger.error(
        "pyperclip not found - falling back to typing method - uppercase chars/symbols might fail in some text fields"
    )

def backup_clipboard():
    if pyperclip is None:
        logger.warning("pyperclip unavailable - cannot backup clipboard")
        return None
    try:
        return pyperclip.paste()
    except Exception as e:
        logger.debug(f"Could not read clipboard: {e}")
        return None

def set_clipboard(text: str) -> bool:
    if pyperclip is None:
        logger.warning("pyperclip unavailable - cannot set clipboard")
        return False
    try:
        pyperclip.copy(text)
        return True
    except Exception as e:
        logger.error(f"Could not set clipboard: {e}")
        return False

def restore_clipboard(original_text: str | None):
    if pyperclip is None:
        return
    try:
        if original_text is None:
            # best-effort: do nothing if we don't have a backup
            return
        pyperclip.copy(original_text)
    except Exception as e:
        logger.debug(f"Could not restore clipboard: {e}")
