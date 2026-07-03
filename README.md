# _faster-whisper Hotkey_

a minimalist push-to-talk style transcription tool built upon **[cutting-edge ASR models](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)**.

**Hold the hotkey, Speak, Release ==> And baamm in your text field!**

In the terminal, in a text editor, or even in the text chat of your online video game, anywhere!

## Features

- **User-Friendly Interface**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.
- **Fast**: Almost instant transcription, even on CPU when picking parakeet or canary.
- **LLM correction** _(experimental)_: Optionally try to repair broken transcriptions text via any OpenAI-compatible API endpoint.

## Current models

_To help with choosing your model, you can see their [AA-AgentTalk score](https://artificialanalysis.ai/speech-to-text/non-streaming#error-rate-by-dataset-tabs) which is particularly relevant for our use case._

- **[ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)**:
  - 6 source languages (en, de, es, fr, ja, pt)
  - Transcription and translation (bidirectional to/from English, plus English→Italian and English→Mandarin)
  - No automatic language recognition
  - Autoregressive with punctuation and capitalization
  - CPU/GPU

- **[ibm-granite/granite-speech-4.1-2b-nar](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)**:
  - 5 languages
  - Transcription only
  - No automatic language recognition
  - Non-autoregressive — very fast inference
  - No capitalization in output
  - CPU/GPU (requires FlashAttention on GPU)

- **[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)**:
  - 14 languages
  - Transcription only
  - No automatic language recognition
  - Runs well on CPU
  - Quite smart, deals well with hesitation and stutters
  - **Limitation**: model supports 30s audio max, longer recordings are handled by chunking

- **[nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)**:
  - 25 languages
  - Transcription and translation (bidirectional to/from English)
  - No automatic language recognition
  - Crazy fast even on CPU in F16

- **[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)**:
  - 25 languages
  - Transcription only
  - Automatic language recognition
  - Crazy fast even on CPU in F16

- **[mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)**:
  - 8 languages
  - Transcription only
  - Automatic language recognition
  - Smart (it even guesses when to put some quotes, etc.) and seems less error-prone for non English native speakers
  - GPU only
  - **Limitation**: auto language recognition requires <30s chunks

- **[Systran/faster-whisper](https://github.com/SYSTRAN/faster-whisper)**:
  - Many languages
  - Transcription only

**_What I personally use currently?_**

_- granite-speech-4.1-2b-nar sometimes with, but often without LLM correction_

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

   **Note:** Currently tested with Python 3.13.12.

- as a pip package:

  ```
  uv pip install .
  ```

- or as an uv tool:

  ```
  uv tool install .
  ```

### For Nvidia GPU

**FlashAttention** is required for GPU inference with the **granite-nar** model. Building it from source can take a while, so you can grab pre-built wheels from https://mjunya.com/flash-attention-prebuild-wheels/

## Usage

1. Run the tool using one of these methods:
   - If installed from PyPi or from source as a package/tool:

     ```
     faster-whisper-hotkey
     ```

   - Or using uv run (handles the venv and dependencies automatically):

     ```
     cd faster-whisper-hotkey
     uv run faster-whisper-hotkey
     ```

2. Go through the menu steps.
3. Once the model is loaded, focus on any text field.
4. Then, simply press the hotkey (PAUSE, F4, F8 or INSERT) while you speak, release it when you're done, and see the magic happening!

When the script is running, you can forget it, the model will remain loaded, and it's ready to transcribe at any time.

## Configuration File

The script automatically saves your settings to `~/.config/faster_whisper_hotkey/transcriber_settings.json`.

## Limitations

- **voxtral**: because of some limitations, and to keep the automatic language recognition capabilities, we are splitting the audio by chunks of 30s. So even if we can still transcribe long speech, best results are when audio is shorter than this.
  In the current state it seems impossible to concile long audio as 1 chunk and automatic language detection. We may need to patch upstream https://huggingface.co/docs/transformers/v4.56.1/en/model_doc/voxtral#transformers.VoxtralProcessor.apply_transcription_request

- **cohere**: the model's feature extractor has a `max_duration` of 30s. Audio longer than 30s is automatically split into chunks for processing. Best results when audio is shorter than this.

- **granite-nar**: requires FlashAttention on GPU. No punctuation or capitalization in output (by design of the non-autoregressive architecture). Use the autoregressive `granite` variant if you need punctuation.

- Due to window type detection to send appropriate key stroke, unfortunately the VSCodium/VSCode terminal isn't supported for now. No clue if we can workaround this.

- Windows supported is not planned. That said, you can use [eutychius](https://github.com/eutychius/faster-whisper-hotkey/tree/feature/supportWindows)'s branch which seems working fine. See [this comment](https://github.com/blakkd/faster-whisper-hotkey/issues/8#issuecomment-3412700777) for instructions.

## Tips

- If you you pick a multilingual **faster-whisper** model, and select `en` as source while speaking another language it will be translated to English, provided you speak for at least few seconds.
- If you pick parakeet-tdt-0.6b-v3, you can even use multiple languages during your recording!
- If you pick a granite model with punctuation enabled, the output will include proper punctuation and capitalization (including German noun capitalization).

## Acknowledgements

Many thanks to:

- **the developers of faster-whisper** for providing such an efficient transcription inference engine
- **NVIDIA** for their blazing fast parakeet and canary models
- **Mistral** for their impressively accurate model Voxtral-Mini-3B model
- **Cohere** for their cohere-transcribe-03-2026 model
- **IBM** for their granite-speech-4.1 models
- and to **all the contributors** of the libraries I used

Also thanks to [wgabrys88](https://huggingface.co/spaces/WJ88/NVIDIA-Parakeet-TDT-0.6B-v2-INT8-Real-Time-Mic-Transcription) and [MohamedRashadthat](https://huggingface.co/spaces/MohamedRashad/Voxtral) for their huggingface spaces that have been helpful!

And to finish, a special mention to **@siddhpant** for their useful [broo](https://github.com/siddhpant/broo) tool, who gave me a mic <3
