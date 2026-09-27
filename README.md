# _faster-whisper Hotkey_

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/blakkd/faster-whisper-hotkey)

a minimalist push-to-talk style transcription tool built upon **[cutting-edge ASR models](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)**.

**Hold the hotkey, Speak, Release ==> And baamm in your text field!**

In the terminal, in a text editor, or even in the text chat of your online video game, anywhere!

## Features

- **User-Friendly Interface**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.
- **CPU-friendly**: [parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) gives instant transcription, on any CPU.
- **LLM correction**: Optionally try to repair broken transcriptions text via any OpenAI-compatible API endpoint.

## Current models

_To help with choosing your model, you can look at [AA-AgentTalk score](https://artificialanalysis.ai/speech-to-text/non-streaming)._

**CPU speed** for a 40 s clip (best compute type per model, see _Tips_):

| Model                                           | Engine       | Compute |     Time | Speed |
| ----------------------------------------------- | ------------ | ------- | -------: | ----: |
| moondream/parakeet-ultra                        | Photon       | float32 |    2.6 s | 15.7× |
| Systran/faster-whisper (small)                  | CTranslate2  | int8    |    4.1 s |  9.7× |
| ibm-granite/granite-speech-5.0-470m-turboctc-nc | Transformers | float32 |    4.7 s |  8.4× |
| CohereLabs/cohere-transcribe-03-2026            | Transformers | float32 |   33.2 s |  1.2× |
| ibm-granite/granite-speech-4.1-2b-nar           | Transformers | float32 |   35.7 s |  1.1× |
| Qwen/Qwen3-ASR-1.7B-hf                          | Transformers | float32 |   48.3 s |  0.8× |
| ibm-granite/granite-speech-4.1-2b               | Transformers | float32 |   55.2 s |  0.7× |
| nvidia/canary-1b-v2                             | NeMo         | float32 |   55.7 s |  0.7× |
| mistralai/Voxtral-Mini-3B-2507                  | Transformers | —       | GPU only |     — |

On GPU every model is super fast (all < 3.5 s for the same clip), so that's not particularly useful to benchmark that. But anyway, detailed evaluation results can be found in [`test_audio_data/transcription_results.md`](test_audio_data/transcription_results.md).

- **(NEW) [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra)**:
  - 25 languages
  - Transcription only
  - Post-trained version of nvidia/parakeet-tdt-0.6b-v3: same 0.6B architecture, but:
    - **lower word error rate** on every benchmark
    - **3x faster** than nvidia/parakeet-tdt-0.6b-v3, thanks to Moondream's [Photon](https://moondream.ai/photon) engine
  - CPU/GPU (GPU is left as option, but there is no point using it... Anyway!)

- **[Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)**:
  - 30 languages
  - Transcription only
  - Automatic language recognition
  - CPU/GPU

- **[ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)**:
  - 6 source languages (en, de, es, fr, ja, pt)
  - Transcription and translation (bidirectional to/from English, plus English→Italian and English→Mandarin)
  - No automatic language recognition
  - Punctuation and capitalization
  - CPU/GPU

- **[ibm-granite/granite-speech-4.1-2b-nar](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)**:
  - 5 languages (en, de, es, fr, pt)
  - Transcription only
  - No automatic language recognition
  - No capitalization for proper nouns
  - CPU/GPU

- **[ibm-granite/granite-speech-5.0-470m-turboctc-nc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc)**:
  - English only
  - Transcription only
  - No capitalization or punctuation in output --> LLM-correction can be helpful
  - CPU/GPU

- **[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)**:
  - 14 languages
  - Transcription only
  - Officially, no automatic language recognition, but it still works pretty great
  - CPU/GPU
  - Quite smart, deals well with hesitation and stutters

- **[nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)**:
  - 25 languages
  - Transcription and translation (bidirectional to/from English)
  - No automatic language recognition
  - CPU/GPU

- **[mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)**:
  - 8 languages
  - Transcription only
  - Automatic language recognition
  - Smart (it even guesses when to put some quotes, etc.) and seems less error-prone for non English native speakers
  - GPU only

- **[Systran/faster-whisper](https://github.com/SYSTRAN/faster-whisper)**:
  - Many languages
  - Transcription only
  - CPU/GPU

## Installation

_see https://docs.astral.sh/uv/ for more information on uv. uv is fast :\)_

**Requires Python `3.11.14`**

### From PyPi

- As a pip package:

  ```
  uv venv --python 3.11.14
  source .venv/bin/activate
  uv pip install faster-whisper-hotkey
  ```

- or as an tool, so that you can run faster-whisper-hotkey from any venv:

  ```
  uv tool install faster-whisper-hotkey --python=3.11.14
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
  uv venv --python 3.11.14
  source .venv/bin/activate
  uv pip install .
  ```

- or as an uv tool:

  ```
  uv tool install .
  ```

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

- **granite-nar**: punctuation comes from the model, but capitalization is a post-processing pass (sentence starts + English "I"), so proper nouns like names stay lowercase.
- **granite-turboctc**: no punctuation or capitalization in output (by design of the non-autoregressive architecture).

  Use the autoregressive `granite` variant if you need full capitalization, or the LLM correction option to fix proper nouns.

- Using window type detection to send appropriate key strokes, we unfortunately can't see "sub windows". So for example, the VSCodium/VSCode terminal isn't supported for now. No clue if we can workaround this.

- Windows supported is not planned. That said, you can use [eutychius](https://github.com/eutychius/faster-whisper-hotkey/tree/feature/supportWindows)'s branch which seems working fine. See [this comment](https://github.com/blakkd/faster-whisper-hotkey/issues/8#issuecomment-3412700777) for instructions.

## Tips

- If you you pick a multilingual **faster-whisper** model, and select `en` as source while speaking another language it will be translated to English, provided you speak for at least few seconds.
- If you pick parakeet-ultra, you can even use multiple languages during your recording!
- For models whose weights are natively bfloat16 (the precision menu marks the native one), you can still pick float32: it upcasts the weights, which costs extra memory, but is far faster on CPUs without good native bf16 support. If your transcriptions are slow, try float32.

## Acknowledgements

Many thanks to:

- **the developers of faster-whisper** for providing such an efficient transcription inference engine
- **NVIDIA** for their blazing fast parakeet and canary models
- **Moondream** for parakeet-ultra and their Photon local inference engine
- **Mistral** for their impressively accurate model Voxtral-Mini-3B model
- **Cohere** for their cohere-transcribe-03-2026 model
- **IBM** for their granite-speech models
- **Qwen** for their Qwen3-ASR-1.7B model
- and to **all the contributors** of the libraries I used

Also thanks to [wgabrys88](https://huggingface.co/spaces/WJ88/NVIDIA-Parakeet-TDT-0.6B-v2-INT8-Real-Time-Mic-Transcription) and [MohamedRashadthat](https://huggingface.co/spaces/MohamedRashad/Voxtral) for their huggingface spaces that have been helpful!

And to finish, a special mention to **@siddhpant** for their useful [broo](https://github.com/siddhpant/broo) tool, who gave me a mic <3
