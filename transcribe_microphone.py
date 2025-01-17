import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyperclip
import threading
import queue

# Initialize the WhisperModel with the correct model size and settings
model_size = "distil-small.en"
model = WhisperModel(model_size, device="cpu", compute_type="int8")  # Adjust device and compute type as needed

# Define a function to capture audio in real-time
def record_audio(callback):
    duration = 10  # Duration of recording in seconds (adjust as needed)
    with sd.InputStream(callback=callback):
        sd.sleep(duration * 1000)

# Define a function to transcribe the captured audio and send it to the cursor position
def transcribe_and_send(text_queue):
    while True:
        if not text_queue.empty():
            text = text_queue.get()
            pyperclip.copy(text)  # Send text to clipboard, which can be pasted at cursor position

# Define the callback function for audio input
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_data = indata.copy()
    segments, _ = model.transcribe(audio_data, beam_size=5, language="en", condition_on_previous_text=False)
    for segment in segments:
        text_queue.put(segment.text)

# Main function to capture and transcribe
if __name__ == "__main__":
    text_queue = queue.Queue()

    # Start transcription thread
    transcription_thread = threading.Thread(target=transcribe_and_send, args=(text_queue,))
    transcription_thread.daemon = True
    transcription_thread.start()

    # Start audio recording
    record_audio(audio_callback)
