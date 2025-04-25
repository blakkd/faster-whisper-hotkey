# *faster-whisper Hotkey*

a minimalist push-to-talk style transcription tool built upon `faster-whisper`.

**Hold the hotkey, Speak, Release ==> And baamm in your text field!**

In the terminal, in a text editor, or even in the text chat of your online video game, anywhere!

## Motivations

Many projects are revolving around `faster-whisper`. But unfortunately, I coudln't find any about a simple push-to-talk approach.
Also, I wanted a solution convenient enough for me, that would be no pain to launch - no pain to use!
So the goal was to provide a simple tool that **works everywhere**, with **zero impact on resources** apart from RAM (cause we want the load to stay loaded to be always ready-to-use).

## Features

- **Automatic Download**: The missing models will automatically be retrieved from Hugging Face; `faster-whisper` handles this.
- **No clipboard usage**: Uses `pynput` to directly simulate keypresses instead.
- **Efficient Performance**: Utilizes `faster-whisper` for fast and reliable transcription.
- **User-Friendly Interface**: Simple interactive menu for configuration, with quick "last config" reuse.
- **Configurable Settings**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.

## Installation

*see https://docs.astral.sh/uv/ for more information on uv. uv is fast :\)*

### From PyPi

- As a pip package:
    ```
    uv pip install faster-whisper-hotkey
    ```

- or as an tool, so that you can run faster-whisper-hotkey from any venv:

    ```
    uv tool install faster-whisper-hotkey
    ```

### From source

1. Clone the repository:

    ```
    git clone https://github.com/blakkd/faster-whisper-hotkey
    cd faster-whisper-hotkey
    ```

2. Install the package and dependencies:

- as a pip package:

    ```
    uv pip install .
    ```

- or as an uv tool:

    ```
    uv tool install .
    ```

### For Nvidia GPU

You need to install cudnn https://developer.nvidia.com/cudnn-downloads

## Usage

1. Whether you installed from PyPi or from source, just run `faster-whisper-hotkey`
2. Go through the menu steps.
3. Once the model is loaded, focus on any text field.
4. Then, simply press the hotkey (PAUSE, F4 or F8) while you speak, release it when you're done, and see the magic happening!

When the script is running, you can forget it, the model will remain loaded, and it's ready to transcribe at any time.

## Configuration File

The script automatically saves your settings to `~/.config/faster_whisper_hotkey/transcriber_settings.json`.

## Performances

- **GPU (cuda)**: instant transcription using any models, even with auto language detection.
- **CPU**: Time-to-first-word can be longer, but transcribing longer sequences compared to just few words won't lead to significant added delay. For large model, time to first word should still be acceptable without language detection.

**Consideration**

It seems distilled model are lacking precision for non-native English speakers. I personally don't really like them, finding them a bit "rigid".

## Dependencies

- **faster_whisper**: For efficient transcription using Whisper models.
- **pulsectl**: For managing PulseAudio sources.
- **sounddevice**: For capturing audio from the microphone.
- **numpy**: For numerical operations on audio data.
- **pynput**: For keyboard simulation to type out transcribed text.
- **curses**: For creating the user interface menu.

## Limitations

- The script doesn't propose translating, only transcription. But if you pick a multilingual model and select `en` as language while talking in another language it will be natively translated to English, provided you speak for long enough.
- Almost all text fields are supported. But there can be some rare exception such as the cinnamon start menu search bar.

## Acknowledgements

Many thanks to the developers of `faster-whisper` for providing such an efficient transcription library, and to all contributors of the libraries I used.

Also a special mention to @siddhpant for their useful [broo](https://github.com/siddhpant/broo) script which gave me a mic <3