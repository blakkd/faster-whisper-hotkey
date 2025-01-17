import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyperclip
import threading
import queue
from pynput import keyboard
import logging
import pulsectl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, device_name):
        self.model_size = "distil-small.en"
        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8"
        )
        self.text_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_recording = False
        self.sample_rate = 16000
        self.device_name = device_name
        self.audio_buffer = []
        self.buffer_time = 30
        self.max_buffer_samples = self.sample_rate * self.buffer_time
        self.segment_number = 0

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
        audio_data = indata.flatten() if indata.ndim > 1 else indata
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
                pyperclip.copy(transcribed_text)
                logger.info(f"Transcribed: {transcribed_text}")
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
        try:
            if key == keyboard.Key.ctrl_l:
                if self.is_recording:
                    self.stop_recording_and_transcribe()
                else:
                    self.start_recording()
        except AttributeError:
            pass

    def run(self):
        self.set_default_audio_source()
        with keyboard.Listener(on_press=self.on_press) as listener:
            logger.info("Press left CTRL to start/stop recording. Press Ctrl+C to exit.")
            try:
                listener.join()
            except KeyboardInterrupt:
                if self.is_recording:
                    self.stop_recording_and_transcribe()
                logger.info("Program terminated by user")

if __name__ == "__main__":
    try:
        device_name = "Broo"
        transcriber = AudioTranscriber(device_name)
        transcriber.run()
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
