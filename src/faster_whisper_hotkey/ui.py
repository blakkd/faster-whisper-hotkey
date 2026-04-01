import curses
from typing import List, Optional


def curses_menu(
    stdscr, title: str, options: List[str], message: str = "", initial_idx=0
):
    """
    Display a scrollable list of `options` in a curses window.
    A `message` (e.g. "Source language") can be shown directly above the list.
    If the terminal is too small to show the message and at least one option, a warning
    message is displayed instead of the menu.

    Returns the selected option, or None if the user aborts with ESC.
    """
    current_row = initial_idx
    # We'll recalc height/width every time we draw
    h, w = stdscr.getmaxyx()

    def draw_menu():
        nonlocal h, w
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        if h == 0 or w == 0:
            stdscr.refresh()
            return

        # ----- message handling ------------------------------------------------
        message_lines = message.split("\n") if message else []
        space_for_message = len(message_lines)
        space_for_options = h - 2  # reserve two lines for borders/scroll bar

        # If the terminal can't even fit the message + one option, warn the user
        if space_for_options - space_for_message <= 0:
            warning = "Terminal too small for this menu"
            stdscr.addstr(0, 0, warning[: w - 1] if w > 0 else warning)
            stdscr.refresh()
            return

        max_visible = min(space_for_options - space_for_message, len(options))
        start = max(0, current_row - (max_visible // 2))
        end = min(start + max_visible, len(options))

        # ----- title ----------------------------------------------------------
        if title:
            title_x = max(0, (w - len(title)) // 2)
            stdscr.addstr(0, title_x, title)

        # ----- determine the vertical start of the options block ----------------
        options_y_start = (h - max_visible) // 2
        if title:
            # Reserve an extra line for the title
            options_y_start += 1

        # ----- draw message above the options ---------------------------------
        for i, line in enumerate(message_lines):
            truncated_line = line[: w - 1] if w > 0 else line
            x = max(0, (w - len(truncated_line)) // 2)
            y_msg = options_y_start - len(message_lines) + i
            if 0 <= y_msg < h:
                stdscr.addstr(y_msg, x, truncated_line)

        # ----- draw options ----------------------------------------------------
        for i in range(start, end):
            text = options[i]
            x = w // 2 - len(text) // 2
            x = max(0, min(x, w - 1)) if w > 0 else 0
            y = options_y_start + (i - start)
            if y < 0 or y >= h:
                continue
            truncated_text = text[: w - 1] if w > 0 else text
            if i == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, truncated_text)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, truncated_text)

        # ----- scroll bar ------------------------------------------------------
        if max_visible < len(options) and h > 0 and w > 0:
            ratio = (current_row + 1) / len(options)
            y_scroll = h - 2
            x_start = w // 4
            length = w // 2
            x_start = max(0, min(x_start, w - 1))
            length = max(1, min(length, w))
            stdscr.addstr(y_scroll, x_start, "[")
            end_pos = int(ratio * (length - 2)) + x_start + 1
            end_pos = max(x_start + 1, min(end_pos, x_start + length - 1))
            stdscr.addstr(y_scroll, x_start + 1, " " * (length - 2))
            stdscr.addstr(y_scroll, end_pos, "█")
            stdscr.addstr(y_scroll, x_start + length - 1, "]")

        stdscr.refresh()

    # ----- initialise curses colours -----------------------------------------
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    draw_menu()

    # ----- main key‑handling loop --------------------------------------------
    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(options) - 1:
            current_row += 1
        elif key in [curses.KEY_ENTER, 10, 13]:
            return options[current_row]
        elif key == 27:  # ESC
            return None

        # Detect terminal resize and redraw accordingly
        new_h, new_w = stdscr.getmaxyx()
        if new_h != h or new_w != w:
            h, w = new_h, new_w
        draw_menu()


def get_initial_choice(stdscr):
    """
    Prompt the user to choose whether to use the last saved settings or
    to configure new ones.
    """
    options = ["Use Last Settings", "Choose New Settings"]
    return curses_menu(stdscr, "", options)


def get_text_input(stdscr, prompt: str, default: str = "") -> Optional[str]:
    """
    Prompt the user for text input using curses.
    Returns the entered text, or None if ESC is pressed.
    """
    stdscr.clear()
    h, w = stdscr.getmaxyx()

    # Ensure dimensions are valid
    h = max(1, h)
    w = max(1, w)
    y_prompt = max(0, min((h - 1) // 2, h - 1))

    avail_width = max(1, w - 2)
    prompt_to_show = prompt[:avail_width]
    prompt_len = len(prompt_to_show)

    stdscr.addstr(y_prompt, 0, prompt_to_show)

    current_text = default
    cursor_pos = len(current_text)

    display_width = max(0, w - prompt_len - 1)
    display_text = current_text[:display_width]
    if display_width > 0:
        stdscr.addstr(y_prompt, prompt_len, " " * display_width)
        stdscr.addstr(y_prompt, prompt_len, display_text)

    safe_cursor = min(cursor_pos, max(0, display_width))
    final_col = max(0, min(prompt_len + safe_cursor, w - 1))
    final_row = max(0, min(y_prompt, h - 1))
    stdscr.move(final_row, final_col)

    curses.curs_set(1)
    stdscr.refresh()

    while True:
        key = stdscr.getch()

        # Re-read dimensions in case terminal was resized
        h, w = stdscr.getmaxyx()

        if key == 27:  # ESC
            curses.curs_set(0)
            return None
        elif key in [curses.KEY_ENTER, 10, 13]:
            curses.curs_set(0)
            return current_text
        elif key == curses.KEY_BACKSPACE or key == 127:
            if cursor_pos > 0:
                current_text = (
                    current_text[: cursor_pos - 1] + current_text[cursor_pos:]
                )
                cursor_pos -= 1
        elif key == curses.KEY_LEFT and cursor_pos > 0:
            cursor_pos -= 1
        elif key == curses.KEY_RIGHT and cursor_pos < len(current_text):
            cursor_pos += 1
        elif 32 <= key <= 126:
            current_text = (
                current_text[:cursor_pos] + chr(key) + current_text[cursor_pos:]
            )
            cursor_pos += 1

        # Update prompt display for new width
        avail_width = max(1, w - 2)
        prompt_to_show = prompt[:avail_width]
        prompt_len = len(prompt_to_show)
        stdscr.addstr(y_prompt, 0, prompt_to_show)

        display_width = max(0, w - prompt_len - 1)
        display_text = current_text[:display_width]
        if display_width > 0:
            stdscr.addstr(y_prompt, prompt_len, " " * display_width)
            stdscr.addstr(y_prompt, prompt_len, display_text)

        # Ensure cursor position is valid for current window size
        safe_cursor = min(cursor_pos, max(0, display_width))
        final_col = max(0, min(prompt_len + safe_cursor, w - 1))
        final_row = max(0, min(y_prompt, h - 1))
        stdscr.move(final_row, final_col)
        stdscr.refresh()
