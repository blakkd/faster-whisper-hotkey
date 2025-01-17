import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyperclip
import threading
import queue
from pynput import keyboard
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, device_index):
        self.model_size = "medium"  # Using tiny model to reduce memory usage
        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type="int8"
        )
        self.text_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_transcribing = False
        self.sample_rate = 16000
        self.device_index = device_index
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Status: {status}")
            
        # Convert to mono if necessary and ensure correct shape
        audio_data = indata.flatten() if indata.ndim > 1 else indata
        audio_data = audio_data.astype(np.float32)
        
        # Normalize audio
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max()
            
        try:
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=1,  # Reduced beam size for faster processing
                language="en",
                condition_on_previous_text=False
            )
            
            for segment in segments:
                if segment.text.strip():  # Only process non-empty segments
                    self.text_queue.put(segment.text)
                    
        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def transcribe_and_send(self):
        while not self.stop_event.is_set():
            try:
                # Get text with a timeout to prevent blocking forever
                text = self.text_queue.get(timeout=0.1)
                if text.strip():
                    print(f"Debug Transcription: {text}")  # Debug print
                    pyperclip.copy(text)
                    logger.info(f"Transcribed: {text}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing transcription: {e}")

    def start_transcription(self):
        if not self.is_transcribing:
            logger.info("Starting transcription...")
            self.stop_event.clear()
            self.is_transcribing = True
            
            # Start the transcription thread
            self.transcription_thread = threading.Thread(
                target=self.transcribe_and_send
            )
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
            # Start audio stream
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=4000,  # Smaller blocksize for more frequent processing
                device=self.device_index
            )
            self.stream.start()

    def stop_transcription(self):
        if self.is_transcribing:
            logger.info("Stopping transcription...")
            self.stop_event.set()
            self.is_transcribing = False
            self.stream.stop()
            self.stream.close()
            self.transcription_thread.join(timeout=1.0)

    def on_press(self, key):
        try:
            if key == keyboard.Key.ctrl_l:  # Left Control key
                if self.is_transcribing:
                    self.stop_transcription()
                else:
                    self.start_transcription()
        except AttributeError:
            pass

    def run(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            logger.info("Press left CTRL to start/stop transcription. Press Ctrl+C to exit.")
            listener.join()

def list_audio_devices():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device['name']}")
    return int(input("Enter the number of the audio device you want to use: "))

if __name__ == "__main__":
    device_index = list_audio_devices()
    transcriber = AudioTranscriber(device_index)
    try:
        transcriber.run()
    except KeyboardInterrupt:
        if transcriber.is_transcribing:
            transcriber.stop_transcription()
        logger.info("Program terminated by user")
