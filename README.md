# *faster-whisper Hotkey*

This repository contains a Python script for a push-to-talk style transcription experience using the `faster-whisper` library.

**Hold the hotkey, Speak, Release ==> And baamm in the currently focused text field!**

In the terminal, in a text editor, or even in the chat of a fullscreen video game, there is no limit!

## Motivations

Many projects are revolving around `faster-whisper`. But unfortunately, I coudln't find any about a simple push-to-talk approach.
Also, I wanted a solution convenient enough for me, that would be no pain to launch - no pain to use!
So the goal was to provide a simple tool that **works everywhere**, with **zero impact on resources** apart from RAM (cause we want the load to stay loaded to be always ready-to-use).

## Features

- **Automatic Download**: The missing models will automatically be retrieved from Hugging Face; `faster-whisper` handles this.
- **Push-to-talk Transcription**: Just hold the PAUSE key, speak and release when you're done.
- **No clipboard usage**: The script uses `pynput` to directly simulate keypresses instead.
- **Efficient Performance**: Utilizes `faster-whisper` for efficient and fast transcription, with blazing-fast model loading.
- **User-Friendly Interface**: Simple interactive menu for configuration, with quick "last config" reuse.
- **Configurable Settings**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.

## Installation

*see https://docs.astral.sh/uv/ for more information on uv. uv is fast :\)*

### From PyPi

```
uv pip install faster-whisper-hotkey # or pip install faster-whisper-hotkey
```

or install as a tool, so that you can run faster-whisper-hotkey from any env:

```
uv tool install faster-whisper-hotkey
```

### From source

1. Clone the repository:

    ```
    git clone https://github.com/blakkd/faster-whisper-hotkey
    cd faster-whisper-hotkey
    ```

2. Install the required dependencies:

    ```
    uv pip install -r requirements.txt # or pip install -r requirements.txt
    ```

    or as tool:

    ```
    uv tool install .
    ```

### For Nvidia GPU

You need to install cudnn https://developer.nvidia.com/cudnn-downloads

## Usage

1. If you installed from PyPi, simply run:

    ```
    faster-whisper-hotkey
    ```
   
   Or if installed from source:

    ```
    python -m faster_whisper_hotkey
    ```
   
2. Go through the menu steps.
3. Once the model is loaded, just focus on any text field.
4. Then, simply press the hotkey (PAUSE by default) while you speak, release it when you're done, and see the magic happening!

When the script is running, you can forget it, the model will remain loaded, and it's ready to transcribe at any time.

## Configuration File

The script automatically saves your settings to `~/.faster_whisper_hotkey/transcriber_settings.json`.

## Performances

- **GPU (cuda)**: instant transcription, on any models, even with auto language detection.
- **CPU**: Time-to-first-word can be longer, but transcribing longer sequences compared to just few words won't lead to significant added delay. That said, even for large model, time to first word should still be acceptable without language detection.

**Consideration**

It seems distilled model are lacking precision for non-native English speakers. I personally don't really like them, I also find them a bit "rigid".

Another thing: I personnaly always had the feeling of getting better accuracy with large-v2 compared to large-v3 which seems broken to me.

## Logging

Logs are written to `~/.faster_whisper_hotkey/transcriber.log` for debugging purposes (the default log level is set to `INFO`).

## Dependencies

- **faster_whisper**: For efficient transcription using Whisper models.
- **pulsectl**: For managing PulseAudio sources.
- **sounddevice**: For capturing audio from the microphone.
- **numpy**: For numerical operations on audio data.
- **pynput**: For keyboard simulation to type out transcribed text.
- **curses**: For creating the user interface menu.

## Limitations

- Currently, the script doesn't propose translating, only transcription. Nevertheless, if you select `en` as language while talking in another language it will be translated to English.
- Almost all text fields are supported. But there can be some rare exception such as the cinnamon start menu search bar for example.

## License

See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgements

Many thanks to the developers of `faster-whisper` for providing an efficient transcription library, and to all contributors of the libraries I used.

Also a special mention to @siddhpant for their useful [broo](https://github.com/siddhpant/broo) script which gaveaway me a virtual mic <3