# _faster-whisper Hotkey_

a minimalist push-to-talk style transcription tool built upon **[cutting-edge ASR models](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)** such as Canary, Parakeet or Whisper.

**Hold the hotkey, Speak, Release ==> And baamm in your text field!**

In the terminal, in a text editor, or even in the text chat of your online video game, anywhere!

## Current models

- (NEW) **[canary-1b-flash](https://huggingface.co/nvidia/canary-1b-flash)**: 4 languages: en, fr, de, es
- (NEW)**[parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)**: en-only model
- **[whisper](https://github.com/SYSTRAN/faster-whisper)**: many languages

## Features

- **Automatic Download**: The missing models are automatically retrieved from Hugging Face.
- **No clipboard usage**: Uses `pynput` to directly simulate keypresses instead.
- **Zero impact on resources** apart from RAM/VRAM (cause we want the load to stay loaded to be always ready-to-use).
- **User-Friendly Interface**: Simple interactive menu for configuration, with quick "last config" reuse.
- **Configurable Settings**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.

## Performances

- **openai/whisper** (multilanguage):
  - **GPU (cuda)**: instant transcription using any models, even with auto language detection.
  - **CPU**: Time-to-first-word can be longer, but transcribing longer sequences compared to just few words won't lead to significant added delay. For large model, time to first word should still be acceptable without language detection.

_Personnal note:
I feel distilled whisper models are lacking precision for non-native English speakers. I personally don't really like them, finding them a bit "rigid"._

- (New) **nvidia/parakeet-tdt-0.6b-v2** (english only):

  - **~20% lower error rate** than whisper-large-v3 despite being **~20x faster!**
  - **CPU**: instant transcription, even in F16!
  - **GPU**: really not necessary

- (New) **nvidia/canary-1b-flash** (4 languages):
  - **~20% lower error rate** than whisper-large-v3 despite being **~10x faster!**
  - **CPU**: almost instant transcription, even in F16!
  - **GPU**: really not necessary

See https://huggingface.co/spaces/hf-audio/open_asr_leaderboard for details.

## Installation

_see https://docs.astral.sh/uv/ for more information on uv. uv is fast :\)_

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

## Limitations

- Canary is limited to 40s of audio only (because we don't use the batching script provided by Nvidia for now, maybe later, but this may be out of scope)
- Canary is really good at transcribing, but not so much at translating.
- Almost all text fields are supported. But there can be some rare exception such as the cinnamon start menu search bar.

## Tricks

- If you you pick a multilingual **whisper** model, and select `en` as source while speaking another language it will be translated to English, provided you speak for at least few seconds.

## Acknowledgements

Many thanks to:

- **the developers of `faster-whisper`** for providing such an efficient transcription library, **NVIDIA** for their awesome **parakeet-tdt-0.6b-v2 model**, and to all contributors of the libraries I used.

- @wgabrys88 for their [parakeet HF space demo example](https://huggingface.co/spaces/WJ88/NVIDIA-Parakeet-TDT-0.6B-v2-INT8-Real-Time-Mic-Transcription) that has been helpful!

And to finish, a special mention to @siddhpant for their useful [broo](https://github.com/siddhpant/broo) script which gave me a mic <3
