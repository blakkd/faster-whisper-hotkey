import os
import re
import shutil
import subprocess
import time
import tempfile
import json
import logging
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from pynput import keyboard

import torch
from nemo.collections.asr.models import ASRModel, EncDecMultiTaskModel
from transformers import (
    VoxtralForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from faster_whisper import WhisperModel
import pulsectl
import curses

# ----------------------------------------------------------------------
# Optional clipboard helper
# ----------------------------------------------------------------------
try:
    import pyperclip
except ImportError:
    pyperclip = None
    logging.getLogger(__name__).error(
        "pyperclip not found - falling back to typing method - uppercase chars/symbols might fail in some text fields"
    )

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helpers for packaged resources
# ----------------------------------------------------------------------
def get_resource_path(filename):
    return files("faster_whisper_hotkey").joinpath(filename).as_posix()

# ----------------------------------------------------------------------
# Load configuration
# ----------------------------------------------------------------------
try:
    config_path = get_resource_path("available_models_languages.json")
    with open(config_path) as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Configuration error: {e}")
    raise

accepted_models_whisper = config.get("accepted_models_whisper", [])
accepted_languages_whisper = config.get("accepted_languages_whisper", [])
accepted_compute_types = ["float16", "int8"]
accepted_devices = ["cuda", "cpu"]
accepted_device_voxtral = ["cuda"]
english_only_models_whisper = set(config.get("english_only_models_whisper", []))

# ----------------------------------------------------------------------
# Settings persistence
# ----------------------------------------------------------------------
conf_dir = os.path.expanduser("~/.config")
settings_dir = os.path.join(conf_dir, "faster_whisper_hotkey")
os.makedirs(settings_dir, exist_ok=True)
SETTINGS_FILE = os.path.join(settings_dir, "transcriber_settings.json")

@dataclass
class Settings:
    device_name: str
    model_type: str
    model_name: str
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
            data.setdefault("model_type", "whisper")
            data.setdefault("model_name", "large-v3")
            return Settings(**data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load settings: {e}")
        return None

# ----------------------------------------------------------------------
# Curses menu helpers
# ----------------------------------------------------------------------
def curses_menu(stdscr, title: str, options: list, message: str = "", initial_idx=0):
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

# ----------------------------------------------------------------------
# Terminal detection helpers
# ----------------------------------------------------------------------
# X11 terminal identifiers
TERMINAL_IDENTIFIERS_X11 = [
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
# Wayland terminal identifiers (same heuristic)
TERMINAL_IDENTIFIERS_WAYLAND = TERMINAL_IDENTIFIERS_X11

# ----------------------------------------------------------------------
# MicrophoneTranscriber
# ----------------------------------------------------------------------
class MicrophoneTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sample_rate = 16000
        self.max_buffer_length = 10 * 60 * self.sample_rate
        self.audio_buffer = np.zeros(self.max_buffer_length, dtype=np.float32)
        self.buffer_index = 0

        # Load the requested model
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
        elif self.settings.model_type == "canary":
            self.model = EncDecMultiTaskModel.from_pretrained(
                self.settings.model_name, map_location=self.settings.device
            ).eval()
        elif self.settings.model_type == "voxtral":
            from typing import Optional
            from mistral_common.protocol.transcription.request import (
                TranscriptionRequest as _TR,
            )
            from pydantic_extra_types.language_code import LanguageAlpha2

            class TranscriptionRequest(_TR):
                language: Optional[LanguageAlpha2] = None

            repo_id = self.settings.model_name
            self.processor = AutoProcessor.from_pretrained(repo_id)

            if self.settings.compute_type == "int8":
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    repo_id,
                    quantization_config=quant_cfg,
                    device_map="auto",
                ).eval()

            elif self.settings.compute_type == "int4":
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    repo_id,
                    quantization_config=quant_cfg,
                    device_map="auto",
                ).eval()

            else:
                compute_dtype = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }.get(self.settings.compute_type, torch.float16)

                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    repo_id,
                    torch_dtype=compute_dtype,
                    device_map="auto",
                ).eval()

            self.TranscriptionRequest = TranscriptionRequest
        else:
            raise ValueError(f"Unknown model type: {self.settings.model_type}")

        self.stop_event = threading.Event()
        self.is_recording = False
        self.device_name = self.settings.device_name
        self.keyboard_controller = keyboard.Controller()
        self.language = self.settings.language
        self.hotkey_key = self._parse_hotkey(self.settings.hotkey)
        self.is_transcribing = False
        self.last_transcription_end_time = 0.0
        self.transcription_queue = []
        self.timer = None

    # ------------------------------------------------------------------
    # Hotkey mapping
    # ------------------------------------------------------------------
    def _parse_hotkey(self, hotkey_str):
        key_mapping = {
            "pause": keyboard.Key.pause,
            "f4": keyboard.Key.f4,
            "f8": keyboard.Key.f8,
            "insert": keyboard.Key.insert,
        }
        return key_mapping.get(hotkey_str, keyboard.Key.pause)

    # ------------------------------------------------------------------
    # Set default audio source
    # ------------------------------------------------------------------
    def set_default_audio_source(self):
        with pulsectl.Pulse("set-default-source") as pulse:
            for source in pulse.source_list():
                if source.name == self.device_name:
                    pulse.source_default_set(source)
                    logger.info(f"Default source set to: {source.name}")
                    return
            logger.warning(f"Source '{self.device_name}' not found")

    # ------------------------------------------------------------------
    # Audio callback
    # ------------------------------------------------------------------
    def audio_callback(self, indata, frames, time_, status):
        if status:
            logger.warning(f"Status: {status}")
        audio_data = (
            np.mean(indata, axis=1)
            if indata.ndim > 1 and indata.shape[1] == 2
            else indata.flatten()
        ).astype(np.float32)
        if not np.isclose(audio_data.max(), 0):
            audio_data /= np.abs(audio_data).max()

        new_index = self.buffer_index + len(audio_data)
        if new_index > self.max_buffer_length:
            audio_data = audio_data[: self.max_buffer_length - self.buffer_index]
            new_index = self.max_buffer_length
        self.audio_buffer[self.buffer_index : new_index] = audio_data
        self.buffer_index = new_index

    # ------------------------------------------------------------------
    # Clipboard helpers
    # ------------------------------------------------------------------
    def _backup_clipboard(self):
        if pyperclip is None:
            logger.warning("pyperclip unavailable - cannot backup clipboard")
            return None
        try:
            return pyperclip.paste()
        except Exception as e:
            logger.debug(f"Could not read clipboard: {e}")
            return None

    def _set_clipboard(self, text: str):
        if pyperclip is None:
            logger.warning("pyperclip unavailable - cannot set clipboard")
            return False
        try:
            pyperclip.copy(text)
            return True
        except Exception as e:
            logger.error(f"Could not set clipboard: {e}")
            return False

    def _restore_clipboard(self, original_text: str):
        if pyperclip is None:
            return
        try:
            pyperclip.copy(original_text)
        except Exception as e:
            logger.debug(f"Could not restore clipboard: {e}")

    # ------------------------------------------------------------------
    # X11 active-window detection
    # ------------------------------------------------------------------
    def _get_active_window_class_x11(self):
        try:
            win_id = subprocess.check_output(["xdotool", "getactivewindow"])
            win_id = win_id.decode().strip()
            xprop_output = subprocess.check_output(["xprop", "-id", win_id, "WM_CLASS"])
            return re.findall(r'"([^"]+)"', xprop_output.decode())
        except Exception as e:
            logger.debug(f"X11 active window detection failed: {e}")
            return []

    def _is_terminal_window_x11(self, classes: list):
        for cls in classes:
            if any(t in cls.lower() for t in TERMINAL_IDENTIFIERS_X11):
                return True
        return False

    # ------------------------------------------------------------------
    # Wayland active-window detection (Sway)
    # ------------------------------------------------------------------
    def _get_focused_container_wayland(self):
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

    def _is_terminal_window_wayland(self, container):
        if not container:
            return False
        name = (container.get("app_id", "") + container.get("name", "")).lower()
        return any(t in name for t in TERMINAL_IDENTIFIERS_WAYLAND)

    # ------------------------------------------------------------------
    # Paste helpers
    # ------------------------------------------------------------------
    def _paste_x11(self, is_terminal: bool):
        """Send the paste shortcut on X11."""
        time.sleep(0.05)  # give clipboard time to settle
        if is_terminal:
            # Ctrl+Shift+V
            self.keyboard_controller.press(keyboard.Key.ctrl_l)
            self.keyboard_controller.press(keyboard.Key.shift)
            self.keyboard_controller.press("v")
            time.sleep(0.01)
            self.keyboard_controller.release("v")
            self.keyboard_controller.release(keyboard.Key.shift)
            self.keyboard_controller.release(keyboard.Key.ctrl_l)
        else:
            # Ctrl+V
            self.keyboard_controller.press(keyboard.Key.ctrl_l)
            self.keyboard_controller.press("v")
            time.sleep(0.01)
            self.keyboard_controller.release("v")
            self.keyboard_controller.release(keyboard.Key.ctrl_l)

    def _send_key_wayland(self, combo: str):
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

    def _paste_wayland(self, is_terminal: bool):
        combo = "ctrl+shift+v" if is_terminal else "ctrl+v"
        success = self._send_key_wayland(combo)
        if not success:
            logger.warning(
                "Auto-paste failed on Wayland; please paste manually (Ctrl+Shift+V)."
            )

    def _paste_to_active_window(self):
        """Detect the focused window and issue the appropriate paste shortcut."""
        if os.getenv("WAYLAND_DISPLAY"):
            container = self._get_focused_container_wayland()
            is_terminal = self._is_terminal_window_wayland(container)
            self._paste_wayland(is_terminal)
        else:
            classes = self._get_active_window_class_x11()
            is_terminal = self._is_terminal_window_x11(classes)
            self._paste_x11(is_terminal)

    # ------------------------------------------------------------------
    # Transcription and sending
    # ------------------------------------------------------------------
    def transcribe_and_send(self, audio_data):
        try:
            # ---------- run the requested model ----------
            if self.settings.model_type == "whisper":
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
                with torch.inference_mode():
                    out = self.model.transcribe([audio_data])
                transcribed_text = out[0].text if out else ""

            elif self.settings.model_type == "canary":
                lang_parts = self.settings.language.split("-")
                if len(lang_parts) != 2:
                    source_lang, target_lang = "en", "en"
                else:
                    source_lang, target_lang = lang_parts
                with torch.inference_mode():
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".wav", delete=False
                        ) as f:
                            temp_path = f.name
                        sf.write(temp_path, audio_data, self.sample_rate)
                        out = self.model.transcribe(
                            audio=[temp_path],
                            source_lang=source_lang,
                            target_lang=target_lang,
                        )
                        transcribed_text = (
                            out[0].text.strip() if out and len(out) > 0 else ""
                        )
                    finally:
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)

            elif self.settings.model_type == "voxtral":
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp_audio:
                    sf.write(tmp_audio.name, audio_data, self.sample_rate)
                    audio_path = tmp_audio.name

                class FileWrapper:
                    def __init__(self, file_obj):
                        self.file = file_obj

                try:
                    with open(audio_path, "rb") as f:
                        wrapped_file = FileWrapper(f)

                        openai_req = {
                            "model": self.settings.model_name,
                            "file": wrapped_file,
                        }
                        if self.settings.language != "auto":
                            openai_req["language"] = self.settings.language

                        tr = self.TranscriptionRequest.from_openai(openai_req)

                        tok = self.processor.tokenizer.tokenizer.encode_transcription(tr)

                        audio_feats = self.processor.feature_extractor(
                            audio_data,
                            sampling_rate=self.sample_rate,
                            return_tensors="pt",
                        ).input_features.to(self.model.device)

                        with torch.no_grad():
                            ids = self.model.generate(
                                input_features=audio_feats,
                                input_ids=torch.tensor(
                                    [tok.tokens], device=self.model.device
                                ),
                                max_new_tokens=500,
                                num_beams=1,
                            )

                        transcribed_text = self.processor.batch_decode(
                            ids, skip_special_tokens=True
                        )[0]

                finally:
                    os.unlink(audio_path)
            else:
                raise ValueError(f"Unknown model type: {self.settings.model_type}")

            # ---------- send the text ----------
            if transcribed_text.strip():
                if pyperclip is None:
                    # fallback typing - preserves case / punctuation
                    for char in transcribed_text:
                        self.keyboard_controller.press(char)
                        self.keyboard_controller.release(char)
                        time.sleep(0.001)
                else:
                    original_clip = self._backup_clipboard()
                    if not self._set_clipboard(transcribed_text):
                        logger.error("Could not set clipboard - falling back to typing")
                        for char in transcribed_text:
                            self.keyboard_controller.press(char)
                            self.keyboard_controller.release(char)
                            time.sleep(0.001)
                    else:
                        time.sleep(0.01)  # give clipboard time to settle
                        self._paste_to_active_window()
                        if original_clip is not None:
                            time.sleep(0.05)
                            self._restore_clipboard(original_clip)

                logger.info(f"Transcribed text: {transcribed_text}")

        except Exception as e:
            logger.error(f"Transcription error: {e}")
        finally:
            self.is_transcribing = False
            self.last_transcription_end_time = time.time()
            self.process_next_transcription()

    def process_next_transcription(self):
        if self.transcription_queue and not self.is_transcribing:
            audio_data = self.transcription_queue.pop(0)
            self.is_transcribing = True
            threading.Thread(
                target=self.transcribe_and_send, args=(audio_data,), daemon=True
            ).start()

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------
    def start_recording(self):
        if not self.is_recording:
            logger.info("Starting recording...")
            self.stop_event.clear()
            self.is_recording = True
            self.timer = threading.Timer(40, self.stop_recording_and_transcribe)
            self.timer.start()
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=4000,
                device="default",
            )
            self.stream.start()

    def stop_recording_and_transcribe(self):
        if hasattr(self, "timer"):
            self.timer.cancel()
        if self.is_recording:
            logger.info("Stopping recording and starting transcription...")
            self.stop_event.set()
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            if self.buffer_index > 0:
                audio_data = self.audio_buffer[: self.buffer_index]
                self.audio_buffer = np.zeros(self.max_buffer_length, dtype=np.float32)
                self.buffer_index = 0
                self.transcription_queue.append(audio_data)
                self.process_next_transcription()
            else:
                self.buffer_index = 0
                self.is_transcribing = False
                self.last_transcription_end_time = time.time()
                self.process_next_transcription()

    # ------------------------------------------------------------------
    # Hotkey handlers
    # ------------------------------------------------------------------
    def on_press(self, key):
        try:
            current_time = time.time()
            if self.is_recording or (current_time - self.last_transcription_end_time < 0.1):
                return True
            if key == self.hotkey_key and not self.is_recording:
                self.start_recording()
                return True
        except AttributeError:
            pass
        return True

    def on_release(self, key):
        try:
            current_time = time.time()
            if self.is_recording and (
                self.is_transcribing or (current_time - self.last_transcription_end_time < 0.1)
            ):
                return True
            if key == self.hotkey_key and self.is_recording:
                self.stop_recording_and_transcribe()
                return True
        except AttributeError:
            pass
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        self.set_default_audio_source()
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


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
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
                with pulsectl.Pulse() as pulse:
                    sources = pulse.source_list()
                    source_names = [src.name for src in sources]
                    device_name = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "", source_names)
                    )
                    if not device_name:
                        continue

                model_type_options = ["Whisper", "Parakeet", "Canary", "Voxtral"]
                model_type = curses.wrapper(
                    lambda stdscr: curses_menu(
                        stdscr, "Select Model Type", model_type_options
                    )
                )
                if not model_type:
                    continue

                if model_type == "Whisper":
                    original_models = config.get("accepted_models_whisper", [])
                    display_models = []
                    for model in original_models:
                        display_models.append(model)
                    selected_model = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "", display_models)
                    )
                    if not selected_model:
                        continue

                    model_name = selected_model

                    english_only = model_name in english_only_models_whisper

                    device = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "", accepted_devices)
                    )
                    if not device:
                        continue

                    if device == "cpu":
                        available_compute_types = ["int8"]
                        info_message = (
                            "float16 is not supported on CPU: only int8 is available"
                        )
                    else:
                        available_compute_types = accepted_compute_types
                        info_message = ""

                    if english_only:
                        info_message += "\n\nLanguage selection skipped for this English-only model."

                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "",
                            available_compute_types,
                            message=info_message,
                        )
                    )
                    if not compute_type:
                        continue

                    if english_only:
                        language = "en"
                    else:
                        language = curses.wrapper(
                            lambda stdscr: curses_menu(
                                stdscr, "", accepted_languages_whisper
                            )
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
                elif model_type == "Canary":
                    canary_message = (
                        "Canary can only process up to 40 seconds of audio."
                    )
                    curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Info", ["Continue"], message=canary_message
                        )
                    )

                    model_name = "nvidia/canary-1b-flash"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Select Device", accepted_devices
                        )
                    )
                    if not device:
                        continue

                    available_compute_types = accepted_compute_types
                    info_message = ""

                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "", available_compute_types, message=info_message
                        )
                    )
                    if not compute_type:
                        continue

                    source_language = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Select Source Language",
                            config.get(
                                "canary_source_target_languages",
                                ["en", "fr", "de", "es"],
                            ),
                            "Source language",
                        )
                    )
                    if not source_language:
                        continue

                    allowed_pairs = config.get("canary_allowed_language_pairs", [])
                    allowed_targets = set()
                    for pair in allowed_pairs:
                        src, tgt = pair.split("-")
                        if src == source_language:
                            allowed_targets.add(tgt)
                    target_options = sorted(allowed_targets)

                    target_language = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Select Target Language",
                            target_options,
                            "Target language (pick the same as source for transcription)",
                        )
                    )
                    if not target_language:
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

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "canary",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": f"{source_language}-{target_language}",
                            "hotkey": hotkey,
                        }
                    )
                    settings = Settings(
                        device_name=device_name,
                        model_type="canary",
                        model_name=model_name,
                        compute_type=compute_type,
                        device=device,
                        language=f"{source_language}-{target_language}",
                        hotkey=hotkey,
                    )
                elif model_type == "Parakeet":
                    model_name = "nvidia/parakeet-tdt-0.6b-v2"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Select Device", accepted_devices
                        )
                    )
                    if not device:
                        continue

                    available_compute_types = accepted_compute_types
                    info_message = "Language selection skipped for this English-only model. Select compute type."

                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "",
                            available_compute_types,
                            message=info_message,
                        )
                    )
                    if not compute_type:
                        continue

                    language = "en"

                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Select Hotkey", hotkey_options
                        )
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

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
                elif model_type == "Voxtral":
                    model_name = "mistralai/Voxtral-Mini-3B-2507"
                    device = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr,
                            "Select Device",
                            accepted_device_voxtral,
                            message="Only GPU for now.",
                        )
                    )
                    if not device:
                        continue

                    available_compute_types = ["float16", "int8", "int4"]
                    compute_type = curses.wrapper(
                        lambda stdscr: curses_menu(stdscr, "", available_compute_types)
                    )
                    if not compute_type:
                        continue

                    info_message_voxtral = (
                        "Voxtral supports automatic language detection among English, Spanish, French, "
                        "Portuguese, Hindi, German, Dutch, and Italian."
                    )
                    curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Info", ["Continue"], message=info_message_voxtral
                        )
                    )

                    language = "auto"

                    hotkey_options = ["Pause", "F4", "F8", "INSERT"]
                    selected_hotkey = curses.wrapper(
                        lambda stdscr: curses_menu(
                            stdscr, "Select Hotkey", hotkey_options
                        )
                    )
                    if not selected_hotkey:
                        continue
                    hotkey = selected_hotkey.lower()

                    save_settings(
                        {
                            "device_name": device_name,
                            "model_type": "voxtral",
                            "model_name": model_name,
                            "compute_type": compute_type,
                            "device": device,
                            "language": language,
                            "hotkey": hotkey,
                        }
                    )
                    settings = Settings(
                        device_name=device_name,
                        model_type="voxtral",
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