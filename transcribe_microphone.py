import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue
from pynput import keyboard
import logging
import pulsectl
import time
import curses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicrophoneTranscriber:
    def __init__(self, device_name, language, model_size="distil-large-v2"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="int8")
        self.text_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_recording = False
        self.sample_rate = 16000
        self.device_name = device_name
        self.audio_buffer = []
        self.buffer_time = 40
        self.max_buffer_samples = self.sample_rate * self.buffer_time
        self.segment_number = 0
        self.keyboard_controller = keyboard.Controller()
        self.language = language

    def set_default_audio_source(self):
        with pulsectl.Pulse("set-default-source") as pulse:
            for source in pulse.source_list():
                if source.name == self.device_name:
                    pulse.source_default_set(source)
                    logger.info(f"Set default source to: {source.name}")
                    return
            logger.warning(f"Source '{self.device_name}' not found")

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Status: {status}")

        audio_data = (
            np.mean(indata, axis=1)
            if indata.ndim > 1 and indata.shape[1] == 2
            else indata.flatten()
        )
        audio_data = audio_data.astype(np.float32)
        if np.abs(audio_data).max() > 0:
            audio_data /= np.abs(audio_data).max()

        self.audio_buffer.extend(audio_data)
        if len(self.audio_buffer) > self.max_buffer_samples:
            excess_samples = len(self.audio_buffer) - self.max_buffer_samples
            self.audio_buffer = self.audio_buffer[excess_samples:]

    def transcribe_and_send(self, audio_data):
        try:
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=5,
                condition_on_previous_text=False,
                language=self.language if self.language != "auto" else None,
            )
            transcribed_text = " ".join(
                segment.text for segment in segments if segment.text.strip()
            )
            if transcribed_text.strip():
                for char in transcribed_text:
                    self.keyboard_controller.press(char)
                    self.keyboard_controller.release(char)
                    time.sleep(0.001)
                logger.info(f"Transcribed text: {transcribed_text}")
        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def start_recording(self):
        if not self.is_recording:
            logger.info("Starting recording...")
            self.stop_event.clear()
            self.is_recording = True
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=4000,
                device="default",
            )
            self.stream.start()

    def stop_recording_and_transcribe(self):
        if self.is_recording:
            logger.info("Stopping recording and starting transcription...")
            self.stop_event.set()
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            audio_data = np.array(self.audio_buffer, dtype=np.float32)
            if len(audio_data) > 0:
                threading.Thread(
                    target=self.transcribe_and_send,
                    args=(audio_data,),
                    daemon=True,
                ).start()
            self.audio_buffer = []

    def on_press(self, key):
        try:
            if key == keyboard.Key.pause:
                if not self.is_recording:
                    self.start_recording()
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.pause:
                if self.is_recording:
                    self.stop_recording_and_transcribe()
        except AttributeError:
            pass

    def run(self):
        self.set_default_audio_source()
        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            logger.info("Press PAUSE to start/stop recording. Press Ctrl+C to exit.")
            try:
                listener.join()
            except KeyboardInterrupt:
                if self.is_recording:
                    self.stop_recording_and_transcribe()
                logger.info("Program terminated by user")


def get_model_choice_curses(stdscr):
    accepted_models = [
        "distil-large-v2",
        "distil-large-v3",
        "distil-small.en",
        "distil-medium.en",
        "large-v2",
        "large-v1",
        "medium.en",
        "medium",
        "base.en",
        "base",
        "small.en",
        "small",
        "tiny.en",
        "tiny",
        "large-v3",
    ]

    current_row = 0

    def print_menu(stdscr, selected_row_idx):
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        max_visible_rows = min(h - 2, len(accepted_models))
        start_row = max(0, selected_row_idx - max_visible_rows // 2)
        end_row = min(start_row + max_visible_rows, len(accepted_models))

        for idx in range(start_row, end_row):
            row_text = accepted_models[idx]
            x = w // 2 - len(row_text) // 2
            y = h // 2 - max_visible_rows // 2 + (idx - start_row)
            if idx == selected_row_idx:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row_text[: w - 1])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row_text[: w - 1])
        stdscr.refresh()

    curses.curs_set(0)

    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    print_menu(stdscr, current_row)

    while 1:
        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(accepted_models) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            return accepted_models[current_row]
        elif key == 27:
            stdscr.clear()
            stdscr.refresh()
            return None

        print_menu(stdscr, current_row)


def get_language_choice_curses(stdscr):
    accepted_languages = [
        "auto",
        "af",
        "am",
        "ar",
        "as",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fo",
        "fr",
        "gl",
        "gu",
        "ha",
        "haw",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jw",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "la",
        "lb",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "ne",
        "nl",
        "nn",
        "no",
        "oc",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "uk",
        "ur",
        "uz",
        "vi",
        "yi",
        "yo",
        "zh",
        "yue",
    ]

    current_row = 0
    current_col = 0

    h, w = stdscr.getmaxyx()
    num_columns = max(1, w // 20)
    num_rows = (len(accepted_languages) + num_columns - 1) // num_columns

    def print_menu(stdscr, selected_row_idx, selected_col_idx):
        stdscr.clear()
        for idx in range(len(accepted_languages)):
            row_text = accepted_languages[idx]
            col_idx = idx % num_columns
            row_idx = idx // num_columns
            x = col_idx * 20 + 5
            y = row_idx + 2
            if idx == selected_row_idx * num_columns + selected_col_idx:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row_text[: w - 1])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row_text[: w - 1])
        stdscr.refresh()

    curses.curs_set(0)

    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    print_menu(stdscr, current_row, current_col)

    while 1:
        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < num_rows - 1:
            current_row += 1
        elif key == curses.KEY_LEFT and current_col > 0:
            current_col -= 1
        elif key == curses.KEY_RIGHT and current_col < num_columns - 1:
            if (current_row * num_columns + current_col + 1) < len(accepted_languages):
                current_col += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            selected_index = current_row * num_columns + current_col
            return accepted_languages[selected_index]
        elif key == 27:
            stdscr.clear()
            stdscr.refresh()
            return None

        print_menu(stdscr, current_row, current_col)


if __name__ == "__main__":
    try:
        device_name = "Broo"
        while True:
            model_size = curses.wrapper(get_model_choice_curses)
            if model_size is None:
                continue
            language = curses.wrapper(get_language_choice_curses)
            if language is None:
                continue
            transcriber = MicrophoneTranscriber(device_name, language, model_size)
            transcriber.run()
            break
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if "transcriber" in locals() and hasattr(transcriber, "keyboard_controller"):
            del transcriber.keyboard_controller
