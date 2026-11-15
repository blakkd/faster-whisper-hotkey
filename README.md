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

_To help with choosing your model, you can look at [AA-AgentTalk score](https://artificialanalysis.ai/speech-to-text/non-streaming)._

Benchmarked on the 40 s clip `test_audio_data/test_16k.wav` (best-of-3 per config).
Full 180-config matrix — 10 models × 6 published quants × cpu/cuda/vulkan — run
2026-11-15 on a 16-core Ryzen 9 5950X with an NVIDIA RTX PRO 4500 Blackwell (32 GB).
Reproduce: `uv run pytest tests/test_model_all_configs.py --runs 3` (or a subset via
`dev/bench_models.py`). The complete grid with VRAM footprints per config:
[`test_audio_data/transcription_results.md`](test_audio_data/transcription_results.md).

**CPU** (fastest quantization per model):

| Model                                           | Compute |     Time | Speed |
| ----------------------------------------------- | ------- | -------: | ----: |
| ibm-granite/granite-speech-5.0-470m-turboctc-nc | Q4_K_M  |    0.9 s | 46.4× |
| nvidia/parakeet-tdt-0.6b-v3                     | F16     |    1.2 s | 32.2× |
| moondream/parakeet-ultra                        | F16     |    1.4 s | 28.9× |
| nvidia/canary-1b-v2                             | Q4_K_M  |    2.3 s | 17.6× |
| openai/whisper-small                            | Q4_K_M  |    2.6 s | 15.7× |
| CohereLabs/cohere-transcribe-03-2026            | Q4_K_M  |    4.0 s | 10.1× |
| ibm-granite/granite-speech-4.1-2b-nar           | F16     |    6.1 s |  6.6× |
| Qwen/Qwen3-ASR-1.7B-hf                          | Q4_K_M  |    7.9 s |  5.0× |
| ibm-granite/granite-speech-4.1-2b               | Q4_K_M  |    9.9 s |  4.1× |
| mistralai/Voxtral-Mini-3B-2507                  | Q4_K_M  |   22.4 s |  1.8× |

(Small quants are not automatically fastest on CPU — the parakeet/NAR decoders
outrun their Q4_K_M builds with plain F16, dequantization costs more than the
memory saved. The full grid shows where each quant wins.)

**CUDA** (fastest quant, with its VRAM working set — weights + buffers + runtime):

| Model                                           | Compute |     Time |   Speed |    VRAM |
| ----------------------------------------------- | ------- | -------: | ------: | ------: |
| ibm-granite/granite-speech-5.0-470m-turboctc-nc | Q5_K_M  |    0.1 s |  374.3× | 5165 MiB |
| openai/whisper-small                            | F16     |    0.2 s |  181.8× | 1562 MiB |
| nvidia/parakeet-tdt-0.6b-v3                     | Q8_0    |    0.3 s |  136.0× | 1995 MiB |
| ibm-granite/granite-speech-4.1-2b-nar           | Q8_0    |    0.3 s |  127.3× | 5045 MiB |
| moondream/parakeet-ultra                        | Q4_K_M  |    0.3 s |  119.8× | 1917 MiB |
| Qwen/Qwen3-ASR-1.7B-hf                          | Q4_K_M  |    0.4 s |  112.8× | 2493 MiB |
| nvidia/canary-1b-v2                             | Q5_K_M  |    0.4 s |  102.1× | 3429 MiB |
| CohereLabs/cohere-transcribe-03-2026            | F16     |    0.5 s |   75.4× | 6619 MiB |
| mistralai/Voxtral-Mini-3B-2507                  | Q4_K_M  |    0.6 s |   64.7× | 5945 MiB |
| ibm-granite/granite-speech-4.1-2b               | Q4_K_M  |    0.7 s |   61.4× | 6407 MiB |

**Vulkan** (same GPU via the portable ggml backend — fastest quant):

| Model                                           | Compute |     Time |   Speed |    VRAM |
| ----------------------------------------------- | ------- | -------: | ------: | ------: |
| ibm-granite/granite-speech-5.0-470m-turboctc-nc | BF16    |    0.1 s |  433.7× | 6054 MiB |
| nvidia/parakeet-tdt-0.6b-v3                     | Q8_0    |    0.2 s |  243.4× | 2302 MiB |
| moondream/parakeet-ultra                        | Q8_0    |    0.2 s |  206.6× | 2462 MiB |
| CohereLabs/cohere-transcribe-03-2026            | Q8_0    |    0.3 s |  160.8× | 5302 MiB |
| nvidia/canary-1b-v2                             | Q4_K_M  |    0.3 s |  158.1× | 3638 MiB |
| openai/whisper-small                            | Q4_K_M  |    0.3 s |  151.4× | 1842 MiB |
| ibm-granite/granite-speech-4.1-2b-nar           | Q4_K_M  |    0.4 s |  113.9× | 5577 MiB |
| Qwen/Qwen3-ASR-1.7B-hf                          | Q4_K_M  |    0.6 s |   65.9× | 2603 MiB |
| mistralai/Voxtral-Mini-3B-2507                  | Q4_K_M  |    0.8 s |   48.0× | 6270 MiB |
| ibm-granite/granite-speech-4.1-2b               | Q4_K_M  |    0.9 s |   46.6× | 6718 MiB |

On a modern GPU, every model transcribes a 40 s clip in under a second — push-to-talk
feels instant everywhere. Even the biggest model (Voxtral) needs < 7 GB VRAM, so any
decent card works.

For comparison with the old four-engine setup: parakeet-ultra 2.6 s → 1.4 s (Photon → GGUF), whisper small 4.1 s → 2.6 s, cohere 33.2 s → 4.0 s, granite 55.2 s → 9.9 s, canary 55.7 s → 2.3 s — and voxtral, previously GPU-only, now runs on CPU too.

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
- Precisions map to GGUF quantizations: `float32/float16/bfloat16` → full-precision files, `int8` → Q8_0, `int4` → Q4_K_M. Q8_0 is virtually lossless and roughly half the memory; on a recent CPU, `int8` is usually the best pick (and always works — every precision runs on every device).
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
