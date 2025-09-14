# _faster-whisper Hotkey_

a minimalist push-to-talk style transcription tool built upon **[cutting-edge ASR models](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)**.

**Hold the hotkey, Speak, Release ==> And baamm in your text field!**

In the terminal, in a text editor, or even in the text chat of your online video game, anywhere!

## Current models

- (NEW) **[nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)**:

  - 25 languages supported
  - Transcription and translation
  - No automatic language recognition
  - Crazy fast even on CPU in F16

- (NEW) **[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)**:

  - 25 languages supported
  - Transcription only
  - Automatic language recognition
  - Crazy fast even on CPU in F16

- **[mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)**:

  - English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian
  - Transcription only
  - Automatic language recognition
  - Smart (it even guesses when to put some quotes, etc.) and less error-prone for non English native speakers
  - GPU only

- **[Systran/faster-whisper](https://github.com/SYSTRAN/faster-whisper)**:

  - Many languages
  - Transcription only

***What I personally use currently?***

*- parakeet-tdt-0.6b-v3, on CPU, when I need all my VRAM to run my LMs*

*- Voxtral-Mini-3B-2507, on GPU, when I run smaller models and can fit it along them*

## Features

- **Models downloading**: Missing models are automatically downloaded from Hugging Face.
- **User-Friendly Interface**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.
- **Fast**: Almost instant transcription, even on CPU when picking parakeet or canary.

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

- **voxtral**: because of some limitations, and to keep the automatic language recognition capabilities, we are splitting the audio by chunks of 30s. So even if we can still transcribe long speech, best results are when audio is shorter than this.
In the current state it seems impossible to concile long audio as 1 chunk and automatic language detection. We may need to patch upstream https://huggingface.co/docs/transformers/v4.56.1/en/model_doc/voxtral#transformers.VoxtralProcessor.apply_transcription_request

## Tricks

- If you you pick a multilingual **faster-whisper** model, and select `en` as source while speaking another language it will be translated to English, provided you speak for at least few seconds.
- If you pick parakeet-tdt-0.6b-v3, you can even use multiple languages during your recording!

## Acknowledgements

Many thanks to:

- **the developers of faster-whisper** for providing such an efficient transcription inference engine
- **NVIDIA** for their blazing fast parakeet and canary models
- **Mistral** for their impressively accurate model Voxtral-Mini-3B model
- and to **all the contributors** of the libraries I used


Also thanks to [wgabrys88](https://huggingface.co/spaces/WJ88/NVIDIA-Parakeet-TDT-0.6B-v2-INT8-Real-Time-Mic-Transcription) and [MohamedRashadthat](https://huggingface.co/spaces/MohamedRashad/Voxtral) for their huggingface spaces that have been helpful!

And to finish, a special mention to **@siddhpant** for their useful [broo](https://github.com/siddhpant/broo) tool, who gave me a mic <3
