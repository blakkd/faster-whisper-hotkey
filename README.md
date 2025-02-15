# Microphone Transcription Project

This repository contains a Python script for a push-to-talk style transcription experience using the `faster-whisper` library. The project aims to provide an easy-to-use tool that transcribes audio input from your microphone and simulates typing the transcribed text.

## Features

- **Automatic Download**: The missing models will automatically be retrieved from huggingface, faster-whisper handles this.
- **Push-to-talk Transcription**: Just hold the PAUSE key, speak and release when you're done.
- **Efficient Performance**: Utilizes `faster-whisper` for efficient and fast transcription.
- **User-Friendly Interface**: Simple interactive menu for configuration, with fast "last config" reuse.
- **Configurable Settings**: Allows users to configure the transcription model, compute type, device, and language through the menu.

## Installation

Install the required dependencies:

    uv pip install sounddevice numpy faster-whisper pynput pulsectl curses logging

*see https://docs.astral.sh/uv/ for more info on the advantages of using uv, or try by yourself with this example ;)*

*or just use `pip install` instead*
    

## Usage

1. Run the script:
    ```sh
    python transcribe_microphone.py
    ```

2. Choose between using last settings or choosing new ones.

3. If you choose "Choose New Settings", you will be presented with a series of menus to select your desired configuration:

    - **Device Name**: Select the audio input device.
    - **Model Size**: Choose the transcription model size.
    - **Compute Type**: Specify the compute type (`float16` or `int8`).
    - **Device**: Select the computing device (`cpu` or `cuda`).
    - **Language**: Choose the language for transcription.

4. Press `PAUSE` on your keyboard to start/stop recording.

5. The transcribed text will be simulated as if typed by a user.

## Configuration

The script loads configuration from `available_models_languages.json`, which includes accepted models and languages. You can modify this file to add or remove supported models and languages.

### Settings File

The script saves and loads settings to/from `transcriber_settings.json`. This allows the application to remember your previous configurations for future use.

## Logging

Logs are written to `transcriber.log` for debugging purposes. The log level is set to `INFO`, so you will see informative messages about the script's operation.

## Dependencies

- **sounddevice**: For capturing audio from the microphone.
- **numpy**: For numerical operations on audio data.
- **faster_whisper**: For efficient transcription using Whisper models.
- **pynput**: For keyboard simulation to type out transcribed text.
- **pulsectl**: For managing PulseAudio sources.
- **curses**: For creating the user interface menu.

## License

See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgements

Special thanks to the developers of `faster-whisper` for providing an efficient transcription library, and to all contributors of the used libraries <3

---

Feel free to contribute, report issues, or suggest improvements!