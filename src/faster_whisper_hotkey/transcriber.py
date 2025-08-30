import time
import threading
import logging
import numpy as np
import sounddevice as sd

from pynput import keyboard

from .settings import Settings
from .models import ModelWrapper
from .clipboard import backup_clipboard, set_clipboard, restore_clipboard
from .paste import paste_to_active_window
import pulsectl

logger = logging.getLogger(__name__)

accepted_compute_types = ["float16", "int8"]
accepted_devices = ["cuda", "cpu"]
accepted_device_voxtral = ["cuda"]

class MicrophoneTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sample_rate = 16000
        self.max_buffer_length = 10 * 60 * self.sample_rate
        self.audio_buffer = np.zeros(self.max_buffer_length, dtype=np.float32)
        self.buffer_index = 0

        # Load the requested model wrapper
        self.model_wrapper = ModelWrapper(self.settings)

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
        try:
            with pulsectl.Pulse("set-default-source") as pulse:
                for source in pulse.source_list():
                    if source.name == self.device_name:
                        pulse.source_default_set(source)
                        logger.info(f"Default source set to: {source.name}")
                        return
                logger.warning(f"Source '{self.device_name}' not found")
        except Exception as e:
            logger.debug(f"Failed to set default source: {e}")

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
    # Transcription and sending
    # ------------------------------------------------------------------
    def transcribe_and_send(self, audio_data):
        try:
            self.is_transcribing = True
            transcribed_text = self.model_wrapper.transcribe(
                audio_data, sample_rate=self.sample_rate, language=self.settings.language
            )

            # ---------- send the text ----------
            if transcribed_text.strip():
                if not set_clipboard:
                    # fallback typing - preserves case / punctuation
                    for char in transcribed_text:
                        self.keyboard_controller.press(char)
                        self.keyboard_controller.release(char)
                        time.sleep(0.001)
                else:
                    original_clip = backup_clipboard()
                    if not set_clipboard(transcribed_text):
                        logger.error("Could not set clipboard - falling back to typing")
                        for char in transcribed_text:
                            self.keyboard_controller.press(char)
                            self.keyboard_controller.release(char)
                            time.sleep(0.001)
                    else:
                        time.sleep(0.01)  # give clipboard time to settle
                        paste_to_active_window()
                        if original_clip is not None:
                            time.sleep(0.05)
                            restore_clipboard(original_clip)

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
            
            # Only apply 40-second time limit for Canary model
            if self.model_wrapper.model_type == "canary":
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
        if hasattr(self, "timer") and self.timer:
            self.timer.cancel()
        if self.is_recording:
            logger.info("Stopping recording and starting transcription...")
            self.stop_event.set()
            self.is_recording = False
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
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
        # set default audio device (best-effort)
        try:
            self.set_default_audio_source()
        except Exception:
            pass

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
