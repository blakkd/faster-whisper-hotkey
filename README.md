# _faster-whisper Hotkey_

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/blakkd/faster-whisper-hotkey)

a minimalist push-to-talk style transcription tool built upon **[cutting-edge ASR models](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)**.

**Hold the hotkey, Speak, Release ==> And baamm in your text field!**

In the terminal, in a text editor, or even in the text chat of your online video game, anywhere!

## Features

- **User-Friendly Interface**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.
- **CPU-friendly**: [parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) gives instant transcription, on any CPU.
- **One engine for everything**: all models are GGUF checkpoints running on [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (ggml) — CPU, Vulkan (any AMD/Intel/NVIDIA GPU) or CUDA backend, no per-model dependency zoo.
- **Instant first recording**: models are warmed up at startup with a 2 s audio clip, so your very first recording is as fast as your hundredth. (The clip is Never Gonna Give You Up. Run with `--debug` to see what the model thinks it heard.)
- **LLM correction**: Optionally try to repair broken transcriptions text via any OpenAI-compatible API endpoint.

## Current models

_The model menu lists **every GGUF checkpoint the engine publishes** (downloaded live from the `handy-computer` hub — currently 70+ models, with each model's languages, translation targets and quantizations read from the model itself), so you are not limited to this list. These are the current highlights:_

_To help with choosing your model, you can also take a look at [AA-AgentTalk score](https://artificialanalysis.ai/speech-to-text/non-streaming)._

Speed of every model × quantization × backend (cpu/cuda/vulkan), with VRAM
footprints and load times, measured on the 40 s clip `test_audio_data/test_16k.wav`:
**[`test_audio_data/transcription_results.md`](test_audio_data/transcription_results.md)**.
Reproduce a subset with `uv run python dev/bench_models.py`.

- **(NEW) [parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra)**:
  - 25 languages
  - Automatic language recognition
  - Transcription only
  - Post-trained version of nvidia/parakeet-tdt-0.6b-v3: same 0.6B architecture, but with a **lower word error rate** on every benchmark
  - CPU/GPU

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
  - The model itself has no language detection: with `auto` a small VoxLingua107 language-ID model detects the language before transcribing
  - CPU/GPU
  - Quite smart, deals well with hesitation and stutters

- **[nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)**:
  - 25 languages
  - Transcription and translation (bidirectional to/from English)
  - No automatic language recognition
  - CPU/GPU
  - Recordings are limited to ~6 minutes (model architecture)

- **[mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)**:
  - 8 languages
  - Transcription (and translation, on the engine side — not wired into the menu yet)
  - Automatic language recognition
  - Smart (it even guesses when to put some quotes, etc.) and seems less error-prone for non English native speakers
  - CPU/GPU

- **faster-whisper models (tiny → large-v3, large-v3-turbo)**:
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

- **Recording length**: whisper, the parakeet models, qwen3-asr and voxtral handle recordings of any practical length. canary and cohere reject anything above ~6.7 minutes (positional table in the checkpoint), and the granite models top out around 30 s of dense speech — the app logs a clear message in those cases instead of returning junk.
- **granite-nar**: punctuation comes from the model, but capitalization is a post-processing pass (sentence starts + English "I"), so proper nouns like names stay lowercase.
- **granite-turboctc**: no punctuation or capitalization in output (by design of the non-autoregressive architecture).

  Use the autoregressive `granite` variant if you need full capitalization, or the LLM correction option to fix proper nouns.

- Using window type detection to send appropriate key strokes, we unfortunately can't see "sub windows". So for example, the VSCodium/VSCode terminal isn't supported for now. No clue if we can workaround this.

- Windows supported is not planned. That said, you can use [eutychius](https://github.com/eutychius/faster-whisper-hotkey/tree/feature/supportWindows)'s branch which seems working fine. See [this comment](https://github.com/blakkd/faster-whisper-hotkey/issues/8#issuecomment-3412700777) for instructions.

## Tips

- If you you pick a multilingual **whisper** model, and select `en` as source while speaking another language it will be translated to English, provided you speak for at least few seconds.
- If you pick parakeet-ultra, you can even use multiple languages during your recording!
- Precisions are the quantizations each model actually publishes — the menu lists them per model, with their download size (labelled `int8` = Q8_0, `int4` = Q4_K_M, plus `float32`/`float16`/`bfloat16` and the Q6_K/Q5_K_M mid-points when present). On a recent CPU, `int8` (Q8_0) is virtually lossless and usually the best pick.
- Models are downloaded once from Hugging Face (public `handy-computer` repositories) to `~/.cache/faster_whisper_hotkey/models/` and work offline from then on.

## Bugs & Feature request

If you encounter any bug, please report it. If you use an agent to write it, please keep it concise and error/symtpom focused.
I accept feature proposal, put that in Discussions. Though keep in mind I'd like to keep this tool minimal. So that should be more about QoL than exanding it. For example, you can propose to add the possibility to set different hotkeys to use LLM correction with custom, various prompts that would fit best the current context (agent, chat, mail, ...). That could could be a good feature to request.

## Developers/Agents

Read `dev/onboarding.md` first: it has the setup, the architecture, and the gotchas you won't guess from the code. Only open PRs you can explain.

## Acknowledgements

Many thanks to:

- **the transcribe.cpp team** ([handy-computer](https://github.com/handy-computer/transcribe.cpp)) for the single ggml-based engine that runs every model here, and their verified GGUF conversions
- **the ggml project** for the runtime underneath it
- **NVIDIA** for their blazing fast parakeet and canary models
- **Moondream** for parakeet-ultra
- **Mistral** for their impressively accurate model Voxtral-Mini-3B model
- **Cohere** for their cohere-transcribe-03-2026 model
- **IBM** for their granite-speech models
- **Qwen** for their Qwen3-ASR-1.7B model
- and to **all the contributors** of the libraries I used

Also thanks to [wgabrys88](https://huggingface.co/spaces/WJ88/NVIDIA-Parakeet-TDT-0.6B-v2-INT8-Real-Time-Mic-Transcription) and [MohamedRashadthat](https://huggingface.co/spaces/MohamedRashad/Voxtral) for their huggingface spaces that have been helpful!

And to finish, a special mention to **@siddhpant** for their useful [broo](https://github.com/siddhpant/broo) tool, who gave me a mic <3
