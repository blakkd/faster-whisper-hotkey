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
import logging

logging.basicConfig(level=logging.DEBUG)

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
import logging

logging.basicConfig(level=logging.DEBUG)

def record_audio(callback):
    duration = 5  # Reduced duration to 5 seconds
    with sd.InputStream(callback=callback, blocksize=1024):  # Set blocksize to 1024 for smaller chunks
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
    logging.debug(f"Audio data shape: {audio_data.shape}")
    # Process audio data in smaller chunks
    blocksize = 1024  # Ensure chunk_length is a multiple of blocksize
    chunk_length = blocksize * 3  # Adjust chunk length as needed, e.g., 3 blocks
    for i in range(0, len(audio_data), chunk_length):
        chunk = audio_data[i:i + chunk_length]
        segments, _ = model.transcribe(chunk, beam_size=5, language="en", condition_on_previous_text=False)
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
        if key == keyboard.Key.ctrl_l:  # Use left control key
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
    keyboard_listener.join()  # Keep the script running until interrupted
