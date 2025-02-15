This repository contains a Python script for a push-to-talk style transcription experience using the `faster-whisper` library.

**Hold the hotkey - Speak -> Baamm in the currently focused text field**

In the terminal, in a text editor, or even in the chat of an online game, there is no limit!

## Motivations

I know there are many other projects trying to achieve the same goal. But unfortunately, few were about a simple push-to-talk approach.
Also, I wanted a solution convenient enough for me, that would be no pain to launch - no pain to use!
So the goal was to provide a simple tool that **can be used anywhere** and can run in the background without using any resources apart from RAM.

## Features

- **Automatic Download**: The missing models will automatically be retrieved from Hugging Face; `faster-whisper` handles this.
- **Push-to-talk Transcription**: Just hold the PAUSE key, speak and release when you're done.
- **Efficient Performance**: Utilizes `faster-whisper` for efficient and fast transcription, with blazing-fast model loading.
- **User-Friendly Interface**: Simple interactive menu for configuration, with quick "last config" reuse.
- **Configurable Settings**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.

## Installation

First, clone the repository:

    git clone https://github.com/blakkd/faster-whisper-hotkey
    cd faster-whisper-hotkey

Install the required dependencies:

    uv pip install -r requirements.txt

*see https://docs.astral.sh/uv/ for more information on the advantages of using uv, or try it by yourself with this example ;)*

*or just use `pip install -r requirements.txt` instead*

## Usage

1. Run the script:
    ```sh
    python transcribe_microphone.py
    ```
2. Go through the menu steps.
3. When the model is loaded, be sure to be in a text field.
4. Simply press the hotkey (PAUSE by default) while you speak, then release, and see the magic happening!

Once the script is running, you can forget it, the model will remain loaded, and it's ready to transcribe at any time.

## Configuration

The script loads configuration from `available_models_languages.json`, which includes all the accepted models and languages.

- GPU (cuda): instant transcription, even on large models.
- On CPU: even medium model size, INT8, can still be acceptable (less than 5sec for ~1-2 sentences on my setup)
- While you will always face the initial delay, transcribing longer sequences compared to just few words won't lead to significant added delay

### Settings File

The script saves and loads settings to/from `transcriber_settings.json`. This allows remembering your previous settings for ease.

## Logging

Logs are written to `transcriber.log` for debugging purposes. The log level is set to `INFO`, so you will see informative messages about the script's operation.

## Dependencies

- **sounddevice**: For capturing audio from the microphone.
- **numpy**: For numerical operations on audio data.
- **faster_whisper**: For efficient transcription using Whisper models.
- **pynput**: For keyboard simulation to type out transcribed text.
- **pulsectl**: For managing PulseAudio sources.
- **curses**: For creating the user interface menu.

## LIMITATIONS

Currently, the script doesn't propose translating, but only transcription. That said, if you select `en` as language while talking in another language it will be translated to English.

## License

See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgements

Special thanks to the developers of `faster-whisper` for providing an efficient transcription library, and to all contributors of the used libraries <3