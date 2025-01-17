import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue
from pynput import keyboard
import logging
import pulsectl
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicrophoneTranscriber:
    def __init__(self, device_name):
        self.model_size = "large-v2"
        self.model = WhisperModel(
            self.model_size,
            device="cuda",
            compute_type="int8"
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

    def set_default_audio_source(self):
        with pulsectl.Pulse('set-default-source') as pulse:
            for source in pulse.source_list():
                if source.name == self.device_name:
                    pulse.source_default_set(source)
                    logger.info(f"Set default source to: {source.name}")
                    return
            logger.warning(f"Source '{self.device_name}' not found")

    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Status: {status}")

        # Convert stereo to mono by averaging the two channels
        if indata.ndim > 1 and indata.shape[1] == 2:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()

        audio_data = audio_data.astype(np.float32)
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max()

        self.audio_buffer.extend(audio_data)
        if len(self.audio_buffer) > self.max_buffer_samples:
            excess_samples = len(self.audio_buffer) - self.max_buffer_samples
            self.audio_buffer = self.audio_buffer[excess_samples:]

    def transcribe_and_send(self, audio_data):
        try:
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=5,
                language="en",
                condition_on_previous_text=False
            )
            transcribed_text = ""
            for segment in segments:
                if segment.text.strip():
                    transcribed_text += segment.text + " "
            if transcribed_text.strip():
                # Directly type the text at the current cursor position with a small delay
                for char in transcribed_text:
                    self.keyboard_controller.press(char)
                    self.keyboard_controller.release(char)
                    time.sleep(0.01)  # Small delay between characters
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
                device='default'
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
                self.transcription_thread = threading.Thread(
                    target=self.transcribe_and_send,
                    args=(audio_data,)
                )
                self.transcription_thread.daemon = True
                self.transcription_thread.start()
            self.audio_buffer = []

    def on_press(self, key):
        # logger.info(f"Key pressed: {key}")  # Uncomment for debugging key presses
        try:
            if key == keyboard.Key.ctrl_l:
                self.ctrl_pressed = True
            elif key == keyboard.Key.shift_r:
                self.shift_pressed = True

            if self.ctrl_pressed and self.shift_pressed:
                if self.is_recording:
                    self.stop_recording_and_transcribe()
                else:
                    self.start_recording()
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.ctrl_l:
                self.ctrl_pressed = False
            elif key == keyboard.Key.shift_r:
                self.shift_pressed = False
        except AttributeError:
            pass

    def run(self):
        self.set_default_audio_source()
        self.ctrl_pressed = False
        self.shift_pressed = False
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            logger.info("Press left CTRL + right SHIFT to start/stop recording. Press Ctrl+C to exit.")
            try:
                listener.join()
            except KeyboardInterrupt:
                if self.is_recording:
                    self.stop_recording_and_transcribe()
                logger.info("Program terminated by user")

if __name__ == "__main__":
    try:
        device_name = "Broo"
        transcriber = MicrophoneTranscriber(device_name)
        transcriber.run()
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if hasattr(transcriber, 'keyboard_controller'):
            del transcriber.keyboard_controller
