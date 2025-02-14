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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and languages from available_models_languages.json
try:
    with open('available_models_languages.json') as f:
        data = json.load(f)
    accepted_models = data['accepted_models']
    accepted_languages = data['accepted_languages']
except FileNotFoundError:
    logger.error("Configuration file 'available_models_languages.json' not found.")
    raise  # Stop execution if JSON file isn't found
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON format in configuration file: {e}")
    raise  # Stop execution due to invalid JSON

# Keep compute types and devices hardcoded for now
accepted_compute_types = ["float16", "int8"]
accepted_devices = ['cpu', 'cuda']

SETTINGS_FILE = "transcriber_settings.json"

def save_settings(settings):
    """Save the current settings to a JSON file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f)
    except IOError as e:
        logger.error(f"Failed to save settings: {e}")

def load_settings():
    """Load settings from the JSON file if it exists."""
    try:
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load settings: {e}")
        return None
def get_initial_choice_curses(stdscr):
    """Get the user's initial choice between using last settings or new."""
    options = ["Use Last Settings", "Choose New Settings"]
    current_row = 0

    def print_menu():
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        for idx, option in enumerate(options):
            x = w // 2 - len(option) // 2
            y = h // 2 - len(options) // 2 + idx
            if idx == current_row:
                stdscr.attron(curses.color_pair(1))
            else:
                stdscr.attroff(curses.color_pair(1))
            stdscr.addstr(y, x, option)
        stdscr.refresh()

    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    print_menu()
    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(options) - 1:
            current_row += 1
        elif key in [curses.KEY_ENTER, 10, 13]:
            return options[current_row]
        elif key == 27:  # Escape
            return None
        print_menu()

def get_device_choice_curses(stdscr):
    with pulsectl.Pulse("list-sources") as pulse:
        sources = pulse.source_list()
        device_names = [source.name for source in sources]

    current_row = 0
    def print_menu(stdscr, selected_row_idx):
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        max_visible_rows = min(h - 2, len(device_names))
        start_row = max(0, selected_row_idx - (max_visible_rows // 2))
        end_row = min(start_row + max_visible_rows, len(device_names))

        for idx in range(start_row, end_row):
            row_text = device_names[idx]
            x = w // 2 - len(row_text) // 2
            y = h // 2 - (max_visible_rows // 2) + (idx - start_row)

            if idx == selected_row_idx:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row_text[:w-1])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row_text[:w-1])
        stdscr.refresh()

    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    print_menu(stdscr, current_row)

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(device_names) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            return device_names[current_row]
        elif key == 27:
            stdscr.clear()
            stdscr.refresh()
            return None
        print_menu(stdscr, current_row)

def get_compute_type_choice_curses(stdscr):
    current_row = 0
    def print_menu(stdscr, selected_row_idx):
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        max_visible_rows = min(h - 2, len(accepted_compute_types))
        start_row = max(0, selected_row_idx - (max_visible_rows // 2))
        end_row = min(start_row + max_visible_rows, len(accepted_compute_types))

        for idx in range(start_row, end_row):
            row_text = accepted_compute_types[idx]
            x = w // 2 - len(row_text) // 2
            y = h // 2 - (max_visible_rows // 2) + (idx - start_row)

            if idx == selected_row_idx:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row_text[:w-1])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row_text[:w-1])
        stdscr.refresh()

    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    print_menu(stdscr, current_row)

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(accepted_compute_types) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            return accepted_compute_types[current_row]
        elif key == 27:
            stdscr.clear()
            stdscr.refresh()
            return None
        print_menu(stdscr, current_row)

def get_device_type_choice_curses(stdscr):
    current_row = 0
    def print_menu(stdscr, selected_row_idx):
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        max_visible_rows = min(h - 2, len(accepted_devices))
        start_row = max(0, selected_row_idx - (max_visible_rows // 2))
        end_row = min(start_row + max_visible_rows, len(accepted_devices))

        for idx in range(start_row, end_row):
            row_text = accepted_devices[idx]
            x = w // 2 - len(row_text) // 2
            y = h // 2 - (max_visible_rows // 2) + (idx - start_row)

            if idx == selected_row_idx:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row_text[:w-1])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row_text[:w-1])
        stdscr.refresh()

    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    print_menu(stdscr, current_row)

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(accepted_devices) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            return accepted_devices[current_row]
        elif key == 27:
            stdscr.clear()
            stdscr.refresh()
            return None
        print_menu(stdscr, current_row)

class MicrophoneTranscriber:
    def __init__(self, device_name, language, model_size, compute_type, device):
        self.model = WhisperModel(
                model_size,
            device=device,
            compute_type=compute_type
            )
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
    current_row = 0
    def print_menu(stdscr, selected_row_idx):
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        max_visible_rows = min(h - 2, len(accepted_models))
        start_row = max(0, selected_row_idx - (max_visible_rows // 2))
        end_row = min(start_row + max_visible_rows, len(accepted_models))

        for idx in range(start_row, end_row):
            row_text = accepted_models[idx]
            x = w // 2 - len(row_text) // 2
            y = h // 2 - (max_visible_rows // 2) + (idx - start_row)

            if idx == selected_row_idx:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row_text[:w-1])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row_text[:w-1])
        stdscr.refresh()
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    print_menu(stdscr, current_row)
    while True:
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
    selected_index = 0
    h, w = stdscr.getmaxyx()
    max_visible_rows = min(h - 4, len(accepted_languages))
    
    def update_menu():
        stdscr.clear()
        # Calculate visible range
        start_row = selected_index // max_visible_rows * max_visible_rows
        end_row = min(start_row + max_visible_rows, len(accepted_languages))
        
        for idx in range(start_row, end_row):
            row_text = accepted_languages[idx]
            y = (idx - start_row) + 2
            x = w // 2 - len(row_text) // 2
            
            if idx == selected_index:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, row_text[:w-1])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, row_text[:w-1])
                
        # Draw scroll indicators
        if max_visible_rows < len(accepted_languages):
            ratio = (selected_index + 1) / len(accepted_languages)
            progress_bar_y = h - 2
            progress_bar_x_start = w // 4
            progress_bar_length = w // 2
            
            stdscr.addstr(progress_bar_y, progress_bar_x_start, "[")
            end_pos = int(ratio * (progress_bar_length - 2)) + progress_bar_x_start +1
            stdscr.addstr(progress_bar_y, progress_bar_x_start+1, ' '*(progress_bar_length-2))
            stdscr.addstr(progress_bar_y, end_pos, "█")
            stdscr.addstr(progress_bar_y, progress_bar_x_start + progress_bar_length -1, "]")
            
        stdscr.refresh()
    
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    update_menu()

    while True:
        key = stdscr.getch()
        if key == curses.KEY_UP and selected_index > 0:
            selected_index -= 1
        elif key == curses.KEY_DOWN and selected_index < len(accepted_languages) - 1:
            selected_index += 1
        elif key in [curses.KEY_ENTER, 10, 13]:
            return accepted_languages[selected_index]
        elif key == 27:  # Escape
            stdscr.clear()
            stdscr.refresh()
            return None
        
        # Check for window resize
        if curses.is_term_resized(h, w):
            h, w = stdscr.getmaxyx()
            max_visible_rows = min(h -4, len(accepted_languages))
        
        update_menu()
if __name__ == "__main__":
    try:
        while True:
            # Initial choice: use last settings or choose new
            initial_choice = curses.wrapper(get_initial_choice_curses)
            if initial_choice is None:
                continue  # User pressed escape, restart the loop

            if initial_choice == "Use Last Settings":
                saved_settings = load_settings()
                if not saved_settings:
                    logger.info("No previous settings found. Proceeding with new settings.")
                    initial_choice = "Choose New Settings"

            if initial_choice == "Choose New Settings":
                # Original selection process
                device_name = curses.wrapper(get_device_choice_curses)
                if device_name is None:
                    continue

                model_size = curses.wrapper(get_model_choice_curses)
                if model_size is None:
                    continue

                compute_type = curses.wrapper(get_compute_type_choice_curses)
                if compute_type is None:
                    continue

                device = curses.wrapper(get_device_type_choice_curses)
                if device is None:
                    continue

                language = curses.wrapper(get_language_choice_curses)
                if language is None:
                    continue

                # Save current settings for next time
                save_settings({
                    "device_name": device_name,
                    "model_size": model_size,
                    "compute_type": compute_type,
                    "device": device,
                    "language": language
                })

            else:  # Using saved settings
                device_name = saved_settings.get("device_name")
                model_size = saved_settings.get("model_size")
                compute_type = saved_settings.get("compute_type")
                device = saved_settings.get("device")
                language = saved_settings.get("language")

            transcriber = MicrophoneTranscriber(
                device_name,
                language,
                model_size,
                compute_type,
                device
            )
            transcriber.run()
            break

    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if "transcriber" in locals() and hasattr(transcriber, "keyboard_controller"):
            del transcriber.keyboard_controller