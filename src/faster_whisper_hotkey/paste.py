import time
import shutil
import subprocess
import logging

from pynput import keyboard

from . import terminal

logger = logging.getLogger(__name__)

keyboard_controller = keyboard.Controller()


def paste_x11(is_terminal: bool):
    """
    Send the paste shortcut on X11.
    """
    time.sleep(0.05)  # give clipboard time to settle
    if is_terminal:
        # Ctrl+Shift+V
        keyboard_controller.press(keyboard.Key.ctrl_l)
        keyboard_controller.press(keyboard.Key.shift)
        keyboard_controller.press("v")
        time.sleep(0.01)
        keyboard_controller.release("v")
        keyboard_controller.release(keyboard.Key.shift)
        keyboard_controller.release(keyboard.Key.ctrl_l)
    else:
        # Ctrl+V
        keyboard_controller.press(keyboard.Key.ctrl_l)
        keyboard_controller.press("v")
        time.sleep(0.01)
        keyboard_controller.release("v")
        keyboard_controller.release(keyboard.Key.ctrl_l)


def _send_key_wayland(combo: str) -> bool:
    wtype_path = shutil.which("wtype")
    if not wtype_path:
        logger.warning("wtype not found - cannot auto-paste on Wayland.")
        return False
    try:
        subprocess.run([wtype_path, combo], check=True)
        return True
    except Exception as e:
        logger.error(f"wtype failed: {e}")
        return False


def paste_wayland(is_terminal: bool):
    combo = "ctrl+shift+v" if is_terminal else "ctrl+v"
    success = _send_key_wayland(combo)
    if not success:
        logger.warning(
            "Auto-paste failed on Wayland; please paste manually (Ctrl+Shift+V)."
        )


def paste_to_active_window():
    """
    Detect the focused window and issue the appropriate paste shortcut.
    """
    if __import__("os").getenv("WAYLAND_DISPLAY"):
        container = terminal.get_focused_container_wayland()
        is_terminal = terminal.is_terminal_window_wayland(container)
        paste_wayland(is_terminal)
    else:
        classes = terminal.get_active_window_class_x11()
        is_terminal = terminal.is_terminal_window_x11(classes)
        paste_x11(is_terminal)
