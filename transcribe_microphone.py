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
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration for accepted models, languages, etc.
try:
    with open("available_models_languages.json") as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Configuration error: {e}")
    raise

accepted_models = config.get("accepted_models", [])
accepted_languages = config.get("accepted_languages", [])
accepted_compute_types = ["float16", "int8"]
accepted_devices = ["cpu", "cuda"]
SETTINGS_FILE = "transcriber_settings.json"


@dataclass
class Settings:
    device_name: str
    model_size: str
    compute_type: str
    device: str
    language: str


def save_settings(settings: dict):
    """Save settings to a JSON file."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except IOError as e:
        logger.error(f"Failed to save settings: {e}")


def load_settings() -> Settings | None:
    """Load settings from JSON or return None if unavailable."""
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
            return Settings(**data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load settings: {e}")
        return None


def curses_menu(stdscr, title: str, options: list, initial_idx=0):
    """Generic curses menu function for user interaction."""
    current_row = initial_idx
    h, w = stdscr.getmaxyx()

    def draw_menu():
        stdscr.clear()
        max_visible = min(h - 2, len(options))
        start = max(0, current_row - (max_visible // 2))
        end = min(start + max_visible, len(options))

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
    """Get initial user choice using curses menu between using last settings or choosing new ones."""
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
        self.buffer_time = 40
        self.max_buffer_samples = self.sample_rate * self.buffer_time
        self.segment_number = 0
        self.keyboard_controller = keyboard.Controller()
        self.language = settings.language

    def set_default_audio_source(self):
        """Set the default audio source using pulseaudio."""
        with pulsectl.Pulse("set-default-source") as pulse:
            for source in pulse.source_list():
                if source.name == self.device_name:
                    pulse.source_default_set(source)
                    logger.info(f"Default source set to: {source.name}")
                    return
            logger.warning(f"Source '{self.device_name}' not found")

    def audio_callback(self, indata, frames, time, status):
        """Callback function for processing audio input."""
        if status:
            logger.warning(f"Status: {status}")

        # Convert stereo to mono and normalize
        audio_data = (
            np.mean(indata, axis=1)
            if indata.ndim > 1 and indata.shape[1] == 2
            else indata.flatten()
        ).astype(np.float32)

        if not np.isclose(audio_data.max(), 0):
            audio_data /= np.abs(audio_data).max()

        self.audio_buffer.extend(audio_data)
        # Maintain buffer size within max samples
        if len(self.audio_buffer) > self.max_buffer_samples:
            excess = len(self.audio_buffer) - self.max_buffer_samples
            self.audio_buffer = self.audio_buffer[excess:]

    def transcribe_and_send(self, audio_data):
        """Transcribe audio and simulate typing the text."""
        try:
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=5,
                condition_on_previous_text=False,
                language=self.language if self.language != "auto" else None,
            )

            transcribed_text = " ".join(segment.text for segment in segments)
            if transcribed_text.strip():
                # Simulate typing each character
                for char in transcribed_text:
                    self.keyboard_controller.press(char)
                    self.keyboard_controller.release(char)
                    time.sleep(0.001)
                logger.info(f"Transcribed text: {transcribed_text}")
        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def start_recording(self):
        """Start audio recording using sounddevice."""
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
        """Stop recording, process and transcribe the buffered audio."""
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
        """Handle key press events for starting recording."""
        try:
            if key == keyboard.Key.pause and not self.is_recording:
                self.start_recording()
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events for stopping recording."""
        try:
            if key == keyboard.Key.pause and self.is_recording:
                self.stop_recording_and_transcribe()
        except AttributeError:
            pass

    def run(self):
        """Main loop for the transcriber application."""
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


def main():
    while True:
        initial_choice = curses.wrapper(get_initial_choice)

        if initial_choice not in ["Use Last Settings", "Choose New Settings"]:
            continue

        if initial_choice == "Use Last Settings":
            settings = load_settings()
            if not settings:
                logger.info("No previous settings found. Proceeding with new settings.")
                initial_choice = "Choose New Settings"

        if initial_choice == "Choose New Settings":
            device_name = curses.wrapper(
                lambda stdscr: curses_menu(
                    stdscr, "", [src.name for src in pulsectl.Pulse().source_list()]
                )
            )
            model_size = curses.wrapper(
                lambda stdscr: curses_menu(stdscr, "", accepted_models)
            )
            compute_type = curses.wrapper(
                lambda stdscr: curses_menu(stdscr, "", accepted_compute_types)
            )
            device = curses.wrapper(
                lambda stdscr: curses_menu(stdscr, "", accepted_devices)
            )
            language = curses.wrapper(
                lambda stdscr: curses_menu(stdscr, "", accepted_languages)
            )

            if any(
                [
                    not x
                    for x in [device_name, model_size, compute_type, device, language]
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
                }
            )

            settings = Settings(device_name, model_size, compute_type, device, language)

        transcriber = MicrophoneTranscriber(settings)
        try:
            transcriber.run()
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()