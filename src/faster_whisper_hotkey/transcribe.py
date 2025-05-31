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
import json
import os
import pkg_resources
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_resource_path(filename):
    """Get the path to a package resource file."""
    return pkg_resources.resource_filename("faster_whisper_hotkey", filename)


try:
    config_path = get_resource_path("available_models_languages.json")
    with open(config_path) as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Configuration error: {e}")
    raise

accepted_models = config.get("accepted_models", [])
accepted_languages = config.get("accepted_languages", [])
accepted_compute_types = ["float16", "int8"]
accepted_devices = ["cuda", "cpu"]

conf_dir = os.path.expanduser("~/.config")
settings_dir = os.path.join(conf_dir, "faster_whisper_hotkey")
os.makedirs(settings_dir, exist_ok=True)
SETTINGS_FILE = os.path.join(settings_dir, "transcriber_settings.json")

ENGLISH_ONLY_MODELS = set(config.get("english_only_models", []))


@dataclass
class Settings:
    device_name: str
    model_size: str
    compute_type: str
    device: str
    language: str
    hotkey: str = "pause"


def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except IOError as e:
        logger.error(f"Failed to save settings: {e}")


def load_settings() -> Settings | None:
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
            data.setdefault("hotkey", "pause")
            return Settings(**data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load settings: {e}")
        return None


def curses_menu(stdscr, title: str, options: list, message: str = "", initial_idx=0):
    current_row = initial_idx
    h, w = stdscr.getmaxyx()

    def draw_menu():
        stdscr.clear()
        max_visible = min(h - 2, len(options))
        start = max(0, current_row - (max_visible // 2))
        end = min(start + max_visible, len(options))

        if message:
            lines = message.split("\n")
            for i, line in enumerate(lines):
                x = w // 2 - len(line) // 2
                y = h // 4 - len(lines) + i
                stdscr.addstr(y, x, line[: w - 1])

        for i in range(start, end):
            text = options[i]
            x = w // 2 - len(text) // 2
            y = h // 2 - (max_visible // 2) + (i - start)

            if i == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, text[: w - 1])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, text[: w - 1])

        if max_visible < len(options):
            ratio = (current_row + 1) / len(options)
            y = h - 2
            x_start = w // 4
            length = w // 2

            stdscr.addstr(y, x_start, "[")
            end_pos = int(ratio * (length - 2)) + x_start + 1
            stdscr.addstr(y, x_start + 1, " " * (length - 2))
            stdscr.addstr(y, end_pos, "█")
            stdscr.addstr(y, x_start + length - 1, "]")

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

        if curses.is_term_resized(h, w):
            h, w = stdscr.getmaxyx()

        draw_menu()


def get_initial_choice(stdscr):
    options = ["Use Last Settings", "Choose New Settings"]
    selected = curses_menu(stdscr, "", options)
    return selected


class MicrophoneTranscriber:
    def __init__(self, settings: Settings):
        self.model = WhisperModel(
            settings.model_size,
            device=settings.device,
            compute_type=settings.compute_type,
        )
        self.text_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_recording = False
        self.sample_rate = 16000
        self.device_name = settings.device_name
        self.audio_buffer = []
        # self.buffer_time = 40
        # self.max_buffer_samples = self.sample_rate * self.buffer_time
        self.segment_number = 0
        self.keyboard_controller = keyboard.Controller()
        self.language = settings.language
        self.hotkey = settings.hotkey
        self.hotkey_key = self._parse_hotkey(self.hotkey)

    def _parse_hotkey(self, hotkey_str):
        key_mapping = {
            "pause": keyboard.Key.pause,
            "f4": keyboard.Key.f4,
            "f8": keyboard.Key.f8,
            "insert": keyboard.Key.insert,
        }
        return key_mapping.get(hotkey_str, keyboard.Key.pause)

    def set_default_audio_source(self):
        with pulsectl.Pulse("set-default-source") as pulse:
            for source in pulse.source_list():
                if source.name == self.device_name:
                    pulse.source_default_set(source)
                    logger.info(f"Default source set to: {source.name}")
                    return
            logger.warning(f"Source '{self.device_name}' not found")

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Status: {status}")

        audio_data = (
            np.mean(indata, axis=1)
            if indata.ndim > 1 and indata.shape[1] == 2
            else indata.flatten()
        ).astype(np.float32)

        if not np.isclose(audio_data.max(), 0):
            audio_data /= np.abs(audio_data).max()

        self.audio_buffer.extend(audio_data)
        # if len(self.audio_buffer) > self.max_buffer_samples:
        #     excess = len(self.audio_buffer) - self.max_buffer_samples
        #     self.audio_buffer = self.audio_buffer[excess:]

    def transcribe_and_send(self, audio_data):
        try:
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=5,
                condition_on_previous_text=False,
                language=self.language if self.language != "auto" else None,
            )

            transcribed_text = " ".join(segment.text.strip() for segment in segments)
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

            if self.audio_buffer:
                threading.Thread(
                    target=self.transcribe_and_send,
                    args=(np.array(self.audio_buffer, dtype=np.float32),),
                    daemon=True,
                ).start()

            self.audio_buffer.clear()

    def on_press(self, key):
        try:
            if key == self.hotkey_key and not self.is_recording:
                self.start_recording()
                return True
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == self.hotkey_key and self.is_recording:
                self.stop_recording_and_transcribe()
                return True
        except AttributeError:
            pass

    def run(self):
        self.set_default_audio_source()

        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            logger.info(
                f"Press {self.hotkey.capitalize()} to start/stop recording. Press Ctrl+C to exit."
            )

            try:
                listener.join()
            except KeyboardInterrupt:
                if self.is_recording:
                    self.stop_recording_and_transcribe()
                logger.info("Program terminated by user")


def main():
    while True:
        try:
            initial_choice = curses.wrapper(get_initial_choice)

            if initial_choice not in ["Use Last Settings", "Choose New Settings"]:
                continue

            if initial_choice == "Use Last Settings":
                settings = load_settings()
                if not settings:
                    logger.info(
                        "No previous settings found. Proceeding with new settings."
                    )
                    initial_choice = "Choose New Settings"

            if initial_choice == "Choose New Settings":
                model_display_mapping = {
                    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2"
                }

                original_models = config.get("accepted_models", [])
                display_models = []
                for model in original_models:
                    if model == "deepdml/faster-whisper-large-v3-turbo-ct2":
                        display_models.append("large-v3-turbo")
                    else:
                        display_models.append(model)

                device_name = curses.wrapper(
                    lambda stdscr: curses_menu(
                        stdscr, "", [src.name for src in pulsectl.Pulse().source_list()]
                    )
                )
                model_size = curses.wrapper(
                    lambda stdscr: curses_menu(stdscr, "", display_models)
                )
                if model_size in model_display_mapping:
                    model_size = model_display_mapping[model_size]

                english_only = model_size in ENGLISH_ONLY_MODELS

                device = curses.wrapper(
                    lambda stdscr: curses_menu(stdscr, "", accepted_devices)
                )

                if device == "cpu":
                    available_compute_types = ["int8"]
                    compute_type_message = (
                        "float16 is not supported on CPU: only int8 is available"
                    )
                else:
                    available_compute_types = accepted_compute_types
                    compute_type_message = ""

                if english_only:
                    compute_type_message += (
                        "\n\nLanguage selection skipped for this English-only model."
                    )

                compute_type = curses.wrapper(
                    lambda stdscr: curses_menu(
                        stdscr,
                        "",
                        available_compute_types,
                        message=compute_type_message,
                    )
                )

                if not english_only:
                    language = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "", accepted_languages)
                    )
                else:
                    language = "en"

                hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                selected_hotkey = curses.wrapper(
                    lambda stdscr: curses_menu(stdscr, "Select Hotkey", hotkey_options)
                )
                if selected_hotkey is None:
                    continue
                hotkey = selected_hotkey.lower()
                if any(
                    [
                        not x
                        for x in [
                            device_name,
                            model_size,
                            compute_type,
                            device,
                            language,
                            hotkey,
                        ]
                    ]
                ):
                    continue

                save_settings(
                    {
                        "device_name": device_name,
                        "model_size": model_size,
                        "compute_type": compute_type,
                        "device": device,
                        "language": language,
                        "hotkey": hotkey,
                    }
                )
                settings = Settings(
                    device_name, model_size, compute_type, device, language, hotkey
                )

            transcriber = MicrophoneTranscriber(settings)
            try:
                transcriber.run()
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                continue

        except KeyboardInterrupt:
            logger.info("Program terminated by user")
            break
