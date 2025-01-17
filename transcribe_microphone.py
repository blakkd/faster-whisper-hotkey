import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyperclip
import threading
import queue
from pynput import keyboard

# Initialize the WhisperModel with the correct model size and settings
model_size = "distil-small.en"
model = WhisperModel(model_size, device="cpu", compute_type="int8")  # Adjust device and compute type as needed

# Define a function to capture audio in real-time
def record_audio(callback):
    duration = 10  # Duration of recording in seconds (adjust as needed)
    with sd.InputStream(callback=callback):
        sd.sleep(duration * 1000)

# Define a function to transcribe the captured audio and send it to the cursor position
def transcribe_and_send(text_queue, stop_event):
    while not stop_event.is_set():
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

# Function to start transcription
def start_transcription(text_queue, stop_event):
    if not stop_event.is_set():
        print("Starting transcription...")
        transcription_thread = threading.Thread(target=transcribe_and_send, args=(text_queue, stop_event))
        transcription_thread.daemon = True
        transcription_thread.start()
        record_audio(audio_callback)

# Function to stop transcription
def stop_transcription(stop_event):
    if not stop_event.is_set():
        print("Stopping transcription...")
        stop_event.set()

# Hotkey listener function
def on_press(key):
    try:
        if key.char == 't':  # Change 't' to any other key you prefer
            global is_transcribing
            if is_transcribing:
                stop_transcription(stop_event)
                is_transcribing = False
            else:
                start_transcription(text_queue, stop_event)
                is_transcribing = True
    except AttributeError:
        pass

if __name__ == "__main__":
    text_queue = queue.Queue()
    stop_event = threading.Event()
    is_transcribing = False

    # Start the hotkey listener in a separate thread
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    print("Press 't' to start/stop transcription. Press Ctrl+C to exit.")
