import json
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

# Terminal identifiers, matched against WM_CLASS entries (X11) and app_ids
# (Wayland) - never against window titles, so e.g. a VSCode window titled
# "term-calc - VSCodium" cannot be misdetected as a terminal.
TERMINAL_IDENTIFIERS = [
    "terminal",
    "term",
    "konsole",
    "xterm",
    "rxvt",
    "urxvt",
    "kitty",
    "alacritty",
    "ghostty",
    "terminator",
    "putty",
    "sakura",
    "blackbox",
    "black box",
]

# Short identifiers that must match as a whole word: as bare substrings they
# would false-positive on unrelated names (e.g. "st" inside
# "com.obsproject.Studio" or "Desktop", "foot" inside "footnotes",
# "rio" inside "riotclient").
TERMINAL_EXACT_IDENTIFIERS = [
    "st",
    "foot",
    "tabby",
    "hyper",
    "rio",
]


def _is_terminal_name(name: str) -> bool:
    """Return True if a WM_CLASS entry or app_id identifies a terminal emulator."""
    name = name.lower()
    if any(t in name for t in TERMINAL_IDENTIFIERS):
        return True
    return any(re.search(rf"\b{t}\b", name) for t in TERMINAL_EXACT_IDENTIFIERS)


def _wm_class_of_window_id(window_id: int | str) -> list[str]:
    """Return the WM_CLASS entries of an X11 window (empty list on failure)."""
    try:
        xprop_output = subprocess.check_output(["xprop", "-id", str(window_id), "WM_CLASS"])
        return re.findall(r'"([^"]+)"', xprop_output.decode())
    except Exception as e:  # noqa: BLE001
        logger.debug(f"X11 WM_CLASS lookup failed for window {window_id}: {e}")
        return []


def get_active_window_class_x11() -> list[str]:
    try:
        raw_win_id = subprocess.check_output(["xdotool", "getactivewindow"])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"X11 active window detection failed: {e}")
        return []
    return _wm_class_of_window_id(raw_win_id.decode().strip())


def is_terminal_window_x11(classes: list[str]) -> bool:
    return any(_is_terminal_name(cls) for cls in classes)


def get_focused_container_wayland() -> dict | None:
    try:
        raw = subprocess.check_output(["swaymsg", "-t", "get_tree"])
        tree = json.loads(raw.decode())
    except Exception as e:  # noqa: BLE001
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


def is_terminal_window_wayland(container: dict | None) -> bool:
    """
    Decide whether the focused Wayland container belongs to a terminal emulator.

    Native containers are matched by app_id only (never the window title, which
    is user-dependent, e.g. "term-calc - VSCodium"). XWayland containers
    advertise app_id "xwayland" and carry the underlying X11 window ID in their
    "window" field, so that window's WM_CLASS is looked up with xprop and
    matched like any other X11 window.
    """
    if not container:
        return False
    window_id = container.get("window")
    if window_id:
        return is_terminal_window_x11(_wm_class_of_window_id(window_id))
    return _is_terminal_name(container.get("app_id", ""))
