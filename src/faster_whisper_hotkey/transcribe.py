import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
from pynput import keyboard
import logging
import pulsectl
import time
import curses
import json
import os
from importlib.resources import files
from dataclasses import dataclass
import torch
from nemo.collections.asr.models import ASRModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_resource_path(filename):
    """Retrieve the path to a package resource file.

    Uses importlib.resources.files to handle paths within the installed package,
    ensuring correct access even when the script is run from different directories.
    """
    return files("faster_whisper_hotkey").joinpath(filename).as_posix()


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
    """Dataclass to hold application settings with type hints.

    Attributes:
        device_name: Name of the audio input device (microphone)
        model_type: Type of ASR model ("whisper" or "parakeet")
        model_name: Specific model identifier/name
        compute_type: Precision for model computation ("float16", "int8")
        device: Hardware device to use ("cuda" for GPU, "cpu" for CPU)
        language: Target language code (e.g., "en", "es")
        hotkey: Keyboard key to trigger recording ("pause", "f4", etc.)
    """

    device_name: str
    model_type: str
    model_name: str
    compute_type: str
    device: str
    language: str
    hotkey: str = "pause"


def save_settings(settings: dict):
    """Save settings dictionary to a JSON file.

    Args:
        settings: Dictionary of configuration parameters to persist
    """
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except IOError as e:
        logger.error(f"Failed to save settings: {e}")


def load_settings() -> Settings | None:
    """Load settings from JSON file into a Settings object.

    Returns:
        Settings instance if file exists and is valid; None otherwise
    """
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
            # Set default values for missing keys to avoid KeyError
            data.setdefault("hotkey", "pause")
            data.setdefault("model_type", "whisper")
            data.setdefault("model_name", "large-v3")
            return Settings(**data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load settings: {e}")
        return None


def curses_menu(stdscr, title: str, options: list, message: str = "", initial_idx=0):
    """Create an interactive text menu using curses library.

    Handles scrolling for long option lists and provides visual feedback.

    Args:
        stdscr: Curses screen object
        title: Menu title (not used in current implementation)
        options: List of strings to display as selectable options
        message: Additional text to show above the options
        initial_idx: Starting selected index

    Returns:
        Selected option string or None if exited with ESC
    """
    current_row = initial_idx
    h, w = stdscr.getmaxyx()

    def draw_menu():
        """Redraw the menu interface based on current state."""
        stdscr.clear()
        # Calculate visible options (up to 2 less than height for status)
        max_visible = min(h - 2, len(options))
        # Determine scroll start/end to center selection
        start = max(0, current_row - (max_visible // 2))
        end = min(start + max_visible, len(options))

        # Display message above options if provided
        if message:
            lines = message.split("\n")
            for i, line in enumerate(lines):
                x = w // 2 - len(line) // 2  # Center horizontally
                y = h // 4 - len(lines) + i  # Position above menu
                stdscr.addstr(y, x, line[: w - 1])  # Truncate to screen width

        # Draw options with highlight on current selection
        for i in range(start, end):
            text = options[i]
            x = w // 2 - len(text) // 2  # Center horizontally
            y = h // 2 - (max_visible // 2) + (i - start)  # Vertical position
            if i == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, text[: w - 1])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, text[: w - 1])

        # Draw scrollbar if options exceed visible area
        if max_visible < len(options):
            ratio = (current_row + 1) / len(options)
            y = h - 2  # Scrollbar position at bottom
            x_start = w // 4
            length = w // 2
            stdscr.addstr(y, x_start, "[")
            # Calculate end position based on ratio (clamped to valid range)
            end_pos = int(ratio * (length - 2)) + x_start + 1
            stdscr.addstr(y, x_start + 1, " " * (length - 2))  # Clear space
            stdscr.addstr(y, end_pos, "█")
            stdscr.addstr(y, x_start + length - 1, "]")

        stdscr.refresh()

    # Initialize curses UI elements
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Highlight color pair
    draw_menu()  # Initial render

    while True:
        key = stdscr.getch()  # Wait for user input
        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(options) - 1:
            current_row += 1
        elif key in [curses.KEY_ENTER, 10, 13]:
            return options[current_row]
        # Handle ESC to exit menu
        elif key == 27:
            return None

        # Detect terminal resize and redraw
        if curses.is_term_resized(h, w):
            h, w = stdscr.getmaxyx()
        draw_menu()


def get_initial_choice(stdscr):
    """Present initial menu to choose between last settings or new settings.

    Args:
        stdscr: Curses screen object

    Returns:
        User's choice ("Use Last Settings", "Choose New Settings", or None)
    """
    options = ["Use Last Settings", "Choose New Settings"]
    return curses_menu(stdscr, "", options)


class MicrophoneTranscriber:
    """Handles audio capture, transcription, and text input via hotkey.

    Manages recording state, buffer handling, model inference, and user interaction.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sample_rate = 16000  # Standard sample rate
        self.max_buffer_length = 10 * 60 * self.sample_rate  # buffer limit
        self.buffer_monitor_thread = None

        # Initialize ASR model based on settings
        if self.settings.model_type == "whisper":
            self.model = WhisperModel(
                model_size_or_path=self.settings.model_name,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
        elif self.settings.model_type == "parakeet":
            self.model = ASRModel.from_pretrained(
                model_name=self.settings.model_name,
                map_location=self.settings.device,
            ).eval()
        else:
            raise ValueError(f"Unknown model type: {self.settings.model_type}")

        self.stop_event = threading.Event()
        self.is_recording = False
        self.device_name = self.settings.device_name
        self.audio_buffer = []
        self.keyboard_controller = keyboard.Controller()
        self.language = self.settings.language
        self.hotkey_key = self._parse_hotkey(self.settings.hotkey)

    def _parse_hotkey(self, hotkey_str):
        """Map hotkey string (e.g., "pause") to pynput's Key constant.

        Args:
            hotkey_str: String representation of the hotkey

        Returns:
            pynput.keyboard.Key instance corresponding to the string
        """
        key_mapping = {
            "pause": keyboard.Key.pause,
            "f4": keyboard.Key.f4,
            "f8": keyboard.Key.f8,
            "insert": keyboard.Key.insert,
        }
        return key_mapping.get(hotkey_str, keyboard.Key.pause)

    def set_default_audio_source(self):
        """Set the specified audio source as default using PulseAudio.

        Uses pulsectl to interact with the system's sound server (Linux only).
        """
        with pulsectl.Pulse("set-default-source") as pulse:
            for source in pulse.source_list():
                if source.name == self.device_name:
                    pulse.source_default_set(source)
                    logger.info(f"Default source set to: {source.name}")
                    return
            logger.warning(f"Source '{self.device_name}' not found")

    def audio_callback(self, indata, frames, time, status):
        """Callback function for sounddevice's InputStream.

        Processes incoming audio data and appends to buffer.

        Args:
            indata: Raw audio data (numpy array)
            frames: Number of samples per block
            status: Stream status (error reporting)
        """
        if status:
            logger.warning(f"Status: {status}")

        # Convert stereo to mono by averaging channels; flatten if already mono
        audio_data = (
            np.mean(indata, axis=1)
            if indata.ndim > 1 and indata.shape[1] == 2
            else indata.flatten()
        ).astype(np.float32)

        # Normalize audio to prevent clipping (scale to [-1, 1])
        if not np.isclose(audio_data.max(), 0):
            audio_data /= np.abs(audio_data).max()

        self.audio_buffer.extend(audio_data)

    def transcribe_and_send(self, audio_data):
        """Transcribe audio data and simulate typing the result.

        Args:
            audio_data: Numpy array of raw audio samples
        """
        try:
            if self.settings.model_type == "whisper":
                # Whisper transcription parameters (beam search for accuracy)
                segments, _ = self.model.transcribe(
                    audio_data,
                    beam_size=5,
                    condition_on_previous_text=False,
                    language=(
                        self.settings.language
                        if self.settings.language != "auto"
                        else None
                    ),
                )
                transcribed_text = " ".join(
                    segment.text.strip() for segment in segments
                )
            elif self.settings.model_type == "parakeet":
                # Parakeet expects input as a list of audio arrays (batch processing)
                with torch.inference_mode():
                    out = self.model.transcribe([audio_data])
                transcribed_text = out[0].text

            # Simulate typing each character
            if transcribed_text.strip():
                for char in transcribed_text:
                    self.keyboard_controller.press(char)
                    self.keyboard_controller.release(char)
                    time.sleep(0.001)
                logger.info(f"Transcribed text: {transcribed_text}")
        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def start_recording(self):
        """Start audio recording and monitoring threads."""
        if not self.is_recording:
            logger.info("Starting recording...")
            self.stop_event.clear()  # Reset stop signal
            self.is_recording = True
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=4000,
                device="default",
            )
            self.stream.start()
            # Start buffer monitor thread to prevent overflow
            self.buffer_monitor_thread = threading.Thread(
                target=self.monitor_buffer, daemon=True
            )
            self.buffer_monitor_thread.start()

    def stop_recording_and_transcribe(self):
        """Stop recording, close stream, and transcribe accumulated audio."""
        if self.is_recording:
            logger.info("Stopping recording and starting transcription...")
            self.stop_event.set()  # Signal buffer monitor to stop
            self.is_recording = False
            self.stream.stop()  # Halt audio capture
            self.stream.close()  # Release resources

            # Transcribe and send if buffer has data
            if self.audio_buffer:
                threading.Thread(
                    target=self.transcribe_and_send,
                    args=(np.array(self.audio_buffer, dtype=np.float32),),
                    daemon=True,
                ).start()

            self.audio_buffer.clear()  # Reset buffer

    def monitor_buffer(self):
        """Thread to monitor audio buffer size and prevent overflow.

        Stops recording if buffer exceeds 10-minute limit to avoid memory issues.
        """
        while not self.stop_event.is_set():  # Continue until stop signal
            time.sleep(0.1)  # Check every 0.1 seconds
            if len(self.audio_buffer) >= self.max_buffer_length:
                logger.info("Audio buffer reached 10-minute limit. Stopping recording.")
                self.stop_recording_and_transcribe()
                break

    def on_press(self, key):
        """Handle hotkey press event.

        Starts recording when the hotkey is pressed (if not already recording).

        Args:
            key: Key pressed (pynput.keyboard.Key or KeyCode)

        Returns:
            True to suppress further processing of this key
        """
        try:
            if key == self.hotkey_key and not self.is_recording:
                self.start_recording()
                return True
        except AttributeError:
            pass  # Ignore non-key objects (e.g., special keys)

    def on_release(self, key):
        """Handle hotkey release event.

        Stops recording and transcribes when the hotkey is released (if recording).

        Args:
            key: Key released

        Returns:
            True to suppress further processing of this key
        """
        try:
            if key == self.hotkey_key and self.is_recording:
                self.stop_recording_and_transcribe()
                return True
        except AttributeError:
            pass

    def run(self):
        """Main execution loop.

        Sets default audio source, starts keyboard listener, and handles termination.
        """
        self.set_default_audio_source()  # Ensure correct microphone is selected

        # Start keyboard listener to detect hotkey presses/releases
        with keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        ) as listener:
            logger.info(
                f"Press {self.settings.hotkey.capitalize()} to start/stop recording. Press Ctrl+C to exit."
            )

            try:
                listener.join()
            except KeyboardInterrupt:
                if self.is_recording:
                    self.stop_recording_and_transcribe()
                logger.info("Program terminated by user")


def main():
    """Main program entry point.

    Handles initial settings selection, creates transcriber instance, and runs it.
    """
    while True:
        try:
            # Use curses to show initial choice menu
            initial_choice = curses.wrapper(get_initial_choice)

            if initial_choice not in ["Use Last Settings", "Choose New Settings"]:
                continue

            # Attempt to load previous settings
            if initial_choice == "Use Last Settings":
                settings = load_settings()
                if not settings:
                    logger.info(
                        "No previous settings found. Proceeding with new settings."
                    )
                    initial_choice = "Choose New Settings"

            if initial_choice == "Choose New Settings":
                # Select audio device (microphone) from available sources
                with pulsectl.Pulse() as pulse:
                    sources = pulse.source_list()
                    source_names = [src.name for src in sources]
                    # Use curses to select device (non-blocking)
                    device_name = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "", source_names)
                    )
                    if not device_name:
                        continue

                # Select ASR model type
                model_type_options = ["Whisper", "Parakeet"]
                model_type = curses.wrapper(
                    lambda stdscr: curses_menu(
                        stdscr, "Select Model Type", model_type_options
                    )
                )
                if not model_type:
                    continue

                # Handle Whisper model configuration
                if model_type == "Whisper":
                    original_models = config.get("accepted_models", [])
                    display_models = []
                    for model in original_models:
                        display_models.append(model)
                    # Let user select model size/variant
                    selected_model = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "", display_models)
                    )
                    if not selected_model:
                        continue

                    model_name = selected_model

                    # Check if model is English-only (affects language UI)
                    english_only = model_name in ENGLISH_ONLY_MODELS

                    # Select hardware device (cuda/cpu)
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "", accepted_devices)
                    )
                    if not device:
                        continue

                    # Determine available compute types based on device
                    if device == "cpu":
                        available_compute_types = ["int8"]
                        compute_type_message = (
                            "float16 is not supported on CPU: only int8 is available"
                        )
                    else:
                        available_compute_types = accepted_compute_types
                        compute_type_message = ""

                    if english_only:
                        compute_type_message += "\n\nLanguage selection skipped for this English-only model."

                    # Let user select compute type (precision)
                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "",
                            available_compute_types,
                            message=compute_type_message,
                        )
                    )
                    if not compute_type:
                        continue

                    if english_only:
                        language = "en"
                    else:
                        language = curses.wrapper(
                            lambda stdscr: curses_menu(stdscr, "", accepted_languages)
                        )
                        if not language:
                            continue

                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Select Hotkey", hotkey_options
                        )
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

                    # Save new settings to file
                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "whisper",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                        }
                    )
                    settings = Settings(
                        device_name=device_name,
                        model_type="whisper",
                        model_name=model_name,
                        compute_type=compute_type,
                        device=device,
                        language=language,
                        hotkey=hotkey,
                    )

                # Handle Parakeet model configuration
                elif model_type == "Parakeet":
                    model_name = "nvidia/parakeet-tdt-0.6b-v2"
                    # Select hardware device
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Select Device", accepted_devices
                        )
                    )
                    if not device:
                        continue

                    # Compute type options depend on device
                    compute_type_message = ""
                    if device == "cuda":
                        available_compute_types = accepted_compute_types
                        compute_type_message = (
                            "This model is English-only. Select compute type."
                        )
                    else:
                        available_compute_types = ["int8"]
                        compute_type_message = (
                            "This model is English-only. Using int8 on CPU."
                        )

                    # Let user select compute type
                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "",
                            available_compute_types,
                            message=compute_type_message,
                        )
                    )
                    if not compute_type:
                        continue

                    language = "en"  # Parakeet is English-only

                    # Select hotkey
                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Select Hotkey", hotkey_options
                        )
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

                    # Save new settings to file
                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "parakeet",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                        }
                    )
                    settings = Settings(
                        device_name=device_name,
                        model_type="parakeet",
                        model_name=model_name,
                        compute_type=compute_type,
                        device=device,
                        language=language,
                        hotkey=hotkey,
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


if __name__ == "__main__":
    main()
