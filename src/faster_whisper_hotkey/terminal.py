import json
import logging
import re
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)

# Terminal identifiers (shared by X11 and Wayland)
TERMINAL_IDENTIFIERS = [
    "terminal",
    "term",
    "konsole",
    "xterm",
    "rxvt",
    "urxvt",
    "kitty",
    "alacritty",
    "terminator",
]


def get_active_window_class_x11() -> List[str]:
    try:
        win_id = subprocess.check_output(["xdotool", "getactivewindow"])
        win_id = win_id.decode().strip()
        xprop_output = subprocess.check_output(["xprop", "-id", win_id, "WM_CLASS"])
        return re.findall(r'"([^"]+)"', xprop_output.decode())
    except Exception as e:
        logger.debug(f"X11 active window detection failed: {e}")
        return []


def is_terminal_window_x11(classes: List[str]) -> bool:
    for cls in classes:
        if any(t in cls.lower() for t in TERMINAL_IDENTIFIERS):
            return True
    return False


def get_focused_container_wayland() -> Optional[dict]:
    try:
        raw = subprocess.check_output(["swaymsg", "-t", "get_tree"])
        tree = json.loads(raw.decode())
    except Exception as e:
        logger.debug(f"Wayland tree retrieval failed: {e}")
        return None

    def find_focused(node):
        if node.get("focused"):
            return node
        for child in node.get("nodes", []):
            r = find_focused(child)
            if r:
                return r
        for child in node.get("floating_nodes", []):
            r = find_focused(child)
            if r:
                return r
        return None

    return find_focused(tree)


def is_terminal_window_wayland(container: Optional[dict]) -> bool:
    if not container:
        return False
    name = (container.get("app_id", "") + container.get("name", "")).lower()
    return any(t in name for t in TERMINAL_IDENTIFIERS)
