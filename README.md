# _faster-whisper Hotkey_

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/blakkd/faster-whisper-hotkey)

a minimalist push-to-talk style transcription tool built upon **[cutting-edge ASR models](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)**.

**Hold the hotkey, Speak, Release ==> And baamm in your text field!**

In the terminal, in a text editor, or even in the text chat of your online video game, anywhere!

## Features

- **User-Friendly Interface**: Allows users to set the input device, transcription model, compute type, device, and language directly through the menu.
- **CPU-friendly**: That's the main motivation.
- **One engine for everything**: all models are GGUF checkpoints running on [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (ggml) — CPU, Vulkan (any AMD/Intel/NVIDIA GPU) or CUDA backend.
- **LLM correction**: Optionally try to repair broken transcriptions text via any OpenAI-compatible API endpoint.

## Current models

The model menu lists **every GGUF checkpoint the engine publishes** (downloaded live from the `handy-computer` hub — currently 70+ models). See https://huggingface.co/handy-computer/models for the list.

_To help with choosing your model, you can also take a look at [AA-AgentTalk score](https://artificialanalysis.ai/speech-to-text/non-streaming)._

## Installation

_see https://docs.astral.sh/uv/ for more information on uv. uv is fast :\)_

**Requires Python `3.10` or newer** (tested through 3.15)

### From PyPi

- As a pip package:

  ```
  uv venv --python 3.14
  source .venv/bin/activate
  uv pip install faster-whisper-hotkey
  ```

- or as an tool, so that you can run faster-whisper-hotkey from any venv:

  ```
  uv tool install faster-whisper-hotkey
  ```

- want CUDA (NVIDIA GPUs)? Same commands plus the `cu12` provider. Its wheel is too large for PyPI, so it comes from the engine's own index (only a CUDA 12-capable driver is needed, no CUDA Toolkit):

  ```
  uv pip install faster-whisper-hotkey "transcribe-cpp[cu12]" --extra-index-url https://handy-computer.github.io/transcribe.cpp/whl/cu12
  ```

  ```
  uv tool install faster-whisper-hotkey --with "transcribe-cpp[cu12]" --extra-index-url https://handy-computer.github.io/transcribe.cpp/whl/cu12
  ```

  `cuda` then shows up in the Compute Device menu.

### From source

1. Clone the repository:

   ```
   git clone https://github.com/blakkd/faster-whisper-hotkey
   cd faster-whisper-hotkey
   ```

2. Install the package and dependencies:

- as a pip package:

  ```
  uv venv --python 3.14
  source .venv/bin/activate
  uv pip install .
  ```

- or as an uv tool:

  ```
  uv tool install .
  ```

- CUDA variant: same as above, just append `--with "transcribe-cpp[cu12]" --extra-index-url https://handy-computer.github.io/transcribe.cpp/whl/cu12` (same flag for `uv pip install .`).

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

## Bugs & Feature request

If you encounter any bug, please report it. If you use an agent to write it, please keep it concise and error/symtpom focused.

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

And to finish, a special mention to **@siddhpant** for their useful [broo](https://github.com/siddhpant/broo) tool, who gave me a mic <3
