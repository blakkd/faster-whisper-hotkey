# _faster-whisper Hotkey_

a minimalist push-to-talk style transcription tool built upon **[cutting-edge ASR models](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)**.

**Hold the hotkey, Speak, Release ==> And baamm in your text field!**

In the terminal, in a text editor, or even in the text chat of your online video game, anywhere!

## Current models

- (NEW) **[mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)**:

  - English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian
  - Transcription only
  - GPU only for now

- **[nvidia/canary-1b-flash](https://huggingface.co/nvidia/canary-1b-flash)**:

  - English, French, German, Spanish
  - Transcription and translation

- **[nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)**: English only

- **[Systran/faster-whisper](https://github.com/SYSTRAN/faster-whisper)**:

  - Many languages
  - Transcription only

## Features

- **Models downloading**: Missing models are automatically retrieved from Hugging Face.
- **Zero impact on resources** (apart from RAM/VRAM cause we want the load to stay loaded to be always ready-to-use).
- **User-Friendly Interface**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.

## Performances (audio < 30s)

- **mistralai/Voxtral-Mini-3B-2507** (8 languages):

  - **Really accurate, takes into account the entire context**, it even use "quotes" and Uppercases where it should, crazy!
  - **GPU**:
    - ~3s for 30s audio on a RTX 3090 in F16 or INT4
    - ~9s for 30s in INT8, not really recommended, but still usable for shorter audio, as it seems the performances are degrading exponentially, i don't know why!

- **nvidia/canary-1b-flash** (4 languages):

  - **~20% lower error rate** than whisper-large-v3 despite being **~10x faster!**
  - **CPU**: almost instant transcription, even in F16!
  - **GPU**: really not necessary

- **nvidia/parakeet-tdt-0.6b-v2** (english only):

  - **~20% lower error rate** than whisper-large-v3 despite being **~20x faster!**
  - **CPU**: instant transcription, even in F16!
  - **GPU**: really not necessary

- **Systran/faster-whisper** (multilanguage):

  - **CPU**: time-to-first-word can be longer, but transcribing longer sequences compared to just few words won't lead to significant added delay. For large model, time to first word should still be acceptable without language detection.
  - **GPU (cuda)**: instant transcription using any models, even with auto language detection.
    _Personnal note:
    I feel distilled whisper models are lacking precision for non-native English speakers. I personally don't really like them, finding them a bit "rigid"._

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

- **canary**: limited to 40s of audio only (because we don't use the batching script provided by Nvidia for now, maybe later, but this may be out of scope).
- **voxtral**: because of some limitations, and to keep the automatic language recognition capabilities, we are splitting the audio by chunks of 30s. So even if we can still transcribe long speech, best results are when audio is shorter than this.
In the current state it seems impossible to concile long audio as 1 chunk and automatic language detection. We may need to patch upstream https://huggingface.co/docs/transformers/v4.56.1/en/model_doc/voxtral#transformers.VoxtralProcessor.apply_transcription_request

## Tricks

- If you you pick a multilingual **faster-whisper** model, and select `en` as source while speaking another language it will be translated to English, provided you speak for at least few seconds.

## Acknowledgements

Many thanks to:

- **the developers of faster-whisper** for providing such an efficient transcription inference engine
- **NVIDIA** for their awesome parakeet-tdt-0.6b-v2 and canary-1b-flash models
- and to **all the contributors** of the libraries I used

Also thanks to [wgabrys88](https://huggingface.co/spaces/WJ88/NVIDIA-Parakeet-TDT-0.6B-v2-INT8-Real-Time-Mic-Transcription) and [MohamedRashadthat](https://huggingface.co/spaces/MohamedRashad/Voxtral) for their huggingface spaces that have been helpful!

And to finish, a special mention to **@siddhpant** for their useful [broo](https://github.com/siddhpant/broo) tool, who gave me a mic <3
