# *faster-whisper Hotkey*

This repository contains a Python script for a push-to-talk style transcription experience using the `faster-whisper` library.

**Hold the hotkey, Speak, Release ==> And baamm in the currently focused text field!**

In the terminal, in a text editor, or even in the chat of a fullscreen video game, there is no limit!

## Motivations

I know there are many other projects trying to achieve the same goal. But unfortunately, few were about a simple push-to-talk approach.
Also, I wanted a solution convenient enough for me, that would be no pain to launch - no pain to use!
So the goal was to provide a simple tool that **can be used anywhere** and can run in the background without using any resources apart from RAM.

## Features

- **Automatic Download**: The missing models will automatically be retrieved from Hugging Face; `faster-whisper` handles this.
- **Push-to-talk Transcription**: Just hold the PAUSE key, speak and release when you're done.
- **No clipboard usage**: The script uses `pynput` to directly simulate keypresses instead.
- **Efficient Performance**: Utilizes `faster-whisper` for efficient and fast transcription, with blazing-fast model loading.
- **User-Friendly Interface**: Simple interactive menu for configuration, with quick "last config" reuse.
- **Configurable Settings**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.

## Installation

First, clone the repository:

    git clone https://github.com/blakkd/faster-whisper-hotkey
    cd faster-whisper-hotkey

Install the required dependencies:

    uv pip install -r requirements.txt

*see https://docs.astral.sh/uv/ for more information on uv, uv is fast :)

*or just use `pip install -r requirements.txt` instead*

### Nvidia GPU

You need to install cudnn https://developer.nvidia.com/cudnn-downloads

## Usage

1. Run the script:
    ```sh
    python transcribe_microphone.py
    ```
2. Go through the menu steps.
3. When the model is loaded, just click in any text field.
4. Just press the hotkey (PAUSE by default) while you speak, release it when you're done, and see the magic happening!

Once the script is running, you can forget it, the model will remain loaded, and it's ready to transcribe at any time.

## Configuration Files

The script loads configuration from `available_models_languages.json`, which includes all the accepted models and languages.

The script saves and loads settings to/from `transcriber_settings.json`. This allows remembering your previous settings for ease.

## Performances

- **initial delay**: While you will always face the initial delay, transcribing longer sequences compared to just few words won't lead to significant added delay
- **GPU (cuda)**: instant transcription, even on large models.
- **CPU**: even for large model sizes, time to first word is still be acceptable when language is set (language detection doubles the time): ~ 8sec for ~15s length audio on my setup

**Consideration**

It seems distilled model are lacking precision for non-native English speakers. I personally don't really like them, I also find them a bit "rigid".

Another thing: I personnaly always had the feeling of getting better accuracy with large-v2 compared to large-v3 which seems broken to me.

## Logging

Logs are written to `transcriber.log` for debugging purposes (the default log level is set to `INFO`).

## Dependencies

- **sounddevice**: For capturing audio from the microphone.
- **numpy**: For numerical operations on audio data.
- **faster_whisper**: For efficient transcription using Whisper models.
- **pynput**: For keyboard simulation to type out transcribed text.
- **pulsectl**: For managing PulseAudio sources.
- **curses**: For creating the user interface menu.

## Limitations

- Currently, the script doesn't propose translating, but only transcription. That said, if you select `en` as language while talking in another language it will be translated to English.
- Almost all text fields are supported. But there can be some rare exception such as the cinnamon start menu search bar for example.

## License

See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgements

Many thanks to the developers of `faster-whisper` for providing an efficient transcription library, and to all contributors of the used libraries

Also a special mention to @siddhpant for their useful [broo](https://github.com/siddhpant/broo) script which gaveaway me a mic <3
