import curses
from typing import List

# UI helpers using curses


def curses_menu(
    stdscr, title: str, options: List[str], message: str = "", initial_idx=0
):
    current_row = initial_idx
    h, w = stdscr.getmaxyx()

    def draw_menu():
        stdscr.clear()
        if h == 0 or w == 0:
            stdscr.refresh()
            return

        max_visible = min(h - 2, len(options))
        start = max(0, current_row - (max_visible // 2))
        end = min(start + max_visible, len(options))

        if message:
            lines = message.split("\n")
            for i, line in enumerate(lines):
                truncated_line = line[: w - 1] if w > 0 else ""
                x = (w - len(truncated_line)) // 2
                y_pos = h // 4 - len(lines) + i
                if 0 <= y_pos < h:
                    stdscr.addstr(y_pos, x, truncated_line)

        for i in range(start, end):
            text = options[i]
            x = w // 2 - len(text) // 2
            x = max(0, min(x, w - 1)) if w > 0 else 0
            y = h // 2 - (max_visible // 2) + (i - start)
            if y < 0 or y >= h:
                continue
            truncated_text = text[: w - 1] if w > 0 else ""
            if i == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, truncated_text)
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, truncated_text)

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

    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    draw_menu()

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(options) - 1:
            current_row += 1
        elif key in [curses.KEY_ENTER, 10, 13]:
            return options[current_row]
        elif key == 27:
            return None

        new_h, new_w = stdscr.getmaxyx()
        if (new_h != h) or (new_w != w):
            h, w = new_h, new_w
        draw_menu()


def get_initial_choice(stdscr):
    options = ["Use Last Settings", "Choose New Settings"]
    return curses_menu(stdscr, "", options)
