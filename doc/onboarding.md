# faster-whisper-hotkey - Codebase Onboarding

## Project Overview

**faster-whisper-hotkey** (v0.5.0) is a push-to-talk speech-to-text (STT) transcription tool for Linux. The core workflow: **hold a hotkey while speaking, release it, and the transcribed text is automatically pasted into whatever text field is currently focused.**

- **License:** WTFPL v2 (public domain equivalent)
- **Platform:** Linux only (X11 and Wayland)
- **Python:** >= 3.10 (tested with 3.13)
- **Entry point command:** `faster-whisper-hotkey`

---

## Project Structure

```
/mnt/storage/workspaces/faster-whisper-hotkey/
|-- README.md                          # Documentation
|-- TODO                               # Planned improvements
|-- pyproject.toml                     # Build config + dependencies
|-- pytest.ini                         # Test runner config
|-- LICENSE.txt                        # WTFPL
|-- .gitignore                         # Build/venv/cache exclusions
|-- uv.lock                            # Dependency lock file
|
+-- doc/
|   +-- onboarding.md                  # This file
|
+-- test_audio_data/                   # Local data
|   |-- test.flac                      # Test audio fixture
|   |-- transcription_results.txt      # Benchmark results
|
+-- src/faster_whisper_hotkey/         # Main package (12 source files)
|   |-- __init__.py                    # Package metadata, version
|   |-- __main__.py                    # CLI entry point
|   |-- config.py                      # Loads available_languages.json
|   |-- settings.py                    # Settings persistence (JSON file)
|   |-- models.py                      # ModelWrapper - loads/runs all ASR models
|   |-- transcribe.py                  # Orchestrator: UI -> Transcriber
|   |-- transcriber.py                 # MicrophoneTranscriber - recording + hotkey loop
|   |-- ui.py                          # Curses TUI configuration screens
|   |-- terminal.py                    # Window detection (X11/Wayland)
|   |-- clipboard.py                   # Clipboard backup/restore via pyperclip
|   |-- paste.py                       # Paste shortcut (Ctrl+V or Ctrl+Shift+V)
|   |-- llm_corrector.py               # Optional LLM post-correction
|   |-- available_languages.json       # Language/model configuration data
|
+-- tests/                             # 14 test files, comprehensive coverage
    |-- test_transcribe.py             # Main flow + logging tests
    |-- test_models.py                 # ModelWrapper init + transcribe tests
    |-- test_models_extended.py        # Edge cases for all models
    |-- test_model_all_configs.py      # Integration: transcribe with all model combos
    |-- test_config.py                 # Config loading tests
    |-- test_config_extended.py        # Extended config tests
    |-- test_settings.py               # Settings save/load/roundtrip
    |-- test_headless_config.py        # --headless and --config CLI options
    |-- test_ui.py                     # Text input + menu tests
    |-- test_ui_edge_cases.py          # Extreme terminal size edge cases
    |-- test_terminal.py               # X11/Wayland window detection
    |-- test_clipboard.py              # Clipboard backup/set/restore
    |-- test_paste.py                  # Paste shortcut routing
    |-- test_llm_corrector.py          # LLM correction API + error handling
```

---

## What This Tool Does (Detailed)

### Core Workflow

1. **Startup:** Runs a curses-based TUI configuration wizard
2. **Configuration:** User selects audio device, model type, device (CPU/GPU), precision, language, hotkey, and optional LLM correction settings
3. **Model Loading:** The selected ASR model is downloaded/loaded into memory
4. **Recording Loop:** A global keyboard listener waits for the configured hotkey (Pause, F4, F8, or Insert)
5. **On Hotkey Press:** Starts recording audio from the default microphone
6. **On Hotkey Release:** Stops recording, transcribes the audio, and pastes the result into the focused text field
7. **Persistent:** The model stays loaded; you can transcribe repeatedly without reloading

### Output Delivery

- **Primary:** Clipboard-based paste (backup original -> set transcribed text -> send Ctrl+V -> restore original)
- **Fallback:** Character-by-character typing via pynput if clipboard fails
- **Smart Paste Detection:** Detects whether the active window is a terminal (uses Ctrl+Shift+V) or GUI application (uses Ctrl+V)
- **Wayland Support:** Uses `wtype` for key simulation, falls back to X11 methods

---

## Component Architecture

### Entry Point: `__main__.py`

- Parses CLI arguments: `--debug`, `--headless`, `--config`
- `--debug`: sets `FASTER_WHISPER_HOTKEY_DEBUG` environment variable; enables verbose logging with full module paths (third-party library output included)
- `--headless`: skips config UI, uses saved settings to start transcribing immediately
- `--config <path>`: custom settings file path (default: `~/.config/faster_whisper_hotkey/transcriber_settings.json`)
- Delegates to `transcribe.main(headless=..., settings_file=...)`

### Orchestrator: `transcribe.py`

- Sets up logging: non-debug mode silences third-party output (root logger at WARNING, package logger at INFO); debug mode shows everything (root logger at DEBUG)
- **Normal mode:** runs the curses configuration UI via `curses.wrapper(config_screen_main)`
- **Headless mode:** loads settings from file (default or `--config` path), starts transcriber directly
- On successful config, creates `MicrophoneTranscriber(settings)` and calls `run()`

### TUI: `ui.py`

- **State machine** with 26 configuration steps (`ConfigStep` enum)
- Steps include: Initial choice, Device selection, Model type selection, then model-specific screens (Whisper/Parakeet/Canary/Voxtral/Cohere/Granite), Hotkey, LLM settings
- Key functions:
  - `curses_menu()` - scrollable option list with scrollbar
  - `get_text_input()` - text entry with cursor support
  - `_handle_key_transition()` - step routing dispatcher
  - ESC returns to initial screen; Enter proceeds

### Settings Persistence: `settings.py`

- Settings saved to `~/.config/faster_whisper_hotkey/transcriber_settings.json` (or custom path via `--config`)
- `save_settings()` and `load_settings()` accept optional `settings_file` parameter for custom paths
- `Settings` dataclass with fields: device_name, model_type, model_name, compute_type, device, language, hotkey, llm_correction_enabled, llm_endpoint, llm_model_name
- Settings auto-restore on next run

### Configuration Data: `config.py` + `available_languages.json`

- Loads JSON at module import time
- Exports: whisper models/languages, canary language pairs, cohere languages, granite languages
- English-only whisper models are tracked to skip language selection

### Model Wrapper: `models.py`

- `suppress_output()` context manager: redirects stdout/stderr to /dev/null at fd level (used during NeMo imports)
- `suppress_nemo()` context manager: silences NeMo's OneLogger during parakeet/canary model loading by redirecting fds and patching module-level logging functions; is a no-op when `--debug` is set
- **`ModelWrapper`** class handles all 7 model types:
  - **whisper:** Uses `faster_whisper.WhisperModel` directly
  - **parakeet:** Uses NeMo's `ASRModel.from_pretrained()`
  - **canary:** Uses NeMo's `EncDecMultiTaskModel.from_pretrained()` (with SentencePiece EOS patch)
  - **voxtral:** Uses transformers `VoxtralForConditionalGeneration` with native chunking via `apply_transcription_request()` (up to ~30min, 30s feature chunks processed together)
  - **cohere:** Uses `CohereAsrForConditionalGeneration` with chunking (30s max)
  - **granite (AR):** Uses `AutoModelForSpeechSeq2Seq` with chat template prompts
  - **granite-nar:** Uses `AutoModel` with FlashAttention on GPU, SDPA on CPU
- Supports multiple precision modes (float16, bfloat16, int8, int4 depending on model)
- Audio chunking for models with 30s feature extractor limits (voxtral uses native chunking with cross-chunk context; cohere uses manual chunking)
- Canary tokenizer patch: fixes EOS token ID detection

### Transcriber: `transcriber.py`

- **`MicrophoneTranscriber`** class:
  - Audio buffer (10 minutes max at 16kHz)
  - Audio normalization and mono conversion
  - Recording via `sounddevice.InputStream`
  - Keyboard listener via `pynput.keyboard.Listener`
  - Transcription queue for handling rapid successive recordings
  - Clipboard send with typing fallback
  - Minimum recording duration filter (1 second)
  - SIGINT handler for clean shutdown

### Text Output: `paste.py` + `clipboard.py` + `terminal.py`

- **clipboard.py:** Backup, set, restore using pyperclip
- **paste.py:** Route paste shortcut based on display server (X11/Wayland) and window type
  - X11: `xdotool` + `xprop` for window class detection
  - Wayland: `swaymsg` for focused container, `wtype` for key simulation
- **terminal.py:** Window class matching against known terminal identifiers

### LLM Correction: `llm_corrector.py`

- Sends raw transcription to an OpenAI-compatible API
- Prompt instructs the LLM to fix grammar, normalize numbers/symbols, correct homophones
- Falls back to original text on any error
- Strips matching wrapper quotes from LLM output

---

## Supported Models

| Model                          | Type        | Languages | Auto-lang | CPU        | GPU                     | Notes                              |
| ------------------------------ | ----------- | --------- | --------- | ---------- | ----------------------- | ---------------------------------- |
| faster-whisper (tiny-large-v3) | whisper     | 99+       | Yes       | Yes (int8) | Yes (float16/int8)      | SYSTRAN engine                     |
| parakeet-tdt-0.6b-v3           | parakeet    | 25        | Yes       | Yes        | Yes                     | NeMo, fast on CPU                  |
| canary-1b-v2                   | canary      | 25        | No        | Yes        | Yes                     | Transcribe + translate             |
| Voxtral-Mini-3B-2507           | voxtral     | 8         | Yes       | No         | Yes (float16/int8/int4) | GPU only, native chunking (~30min) |
| cohere-transcribe-03-2026      | cohere      | 14        | No        | Yes        | Yes                     | 30s chunks                         |
| granite-speech-4.1-2b          | granite     | 6+        | No        | Yes        | Yes                     | Autoregressive, punctuation        |
| granite-speech-4.1-2b-nar      | granite-nar | 5         | No        | Yes        | Yes (FlashAttention)    | Non-AR, very fast, no punctuation  |

---

## Key Dependencies

From `pyproject.toml`:

- **Audio:** sounddevice 0.5.5, soundfile 0.14.0, librosa 0.11.0, numpy 2.2.6
- **ASR Engines:** faster-whisper 1.2.1, nemo-toolkit[asr] 2.1.0, transformers 5.12.1, mistral-common[audio] 1.11.3
- **GPU/ML:** torch 2.12.1, torchaudio 2.11.0, bitsandbytes 0.45.3, accelerate 1.14.0
- **Input/UI:** pynput 1.8.2, pulsectl 24.12.0, pyperclip 1.11.0
- **Other:** requests 2.34.2, sentencepiece 0.2.1, protobuf 5.29.6, onnx 1.22.0, importlib_resources 7.1.0

---

## Known Issues, TODOs, and Notes

### From README limitations:

- **voxtral:** uses `apply_transcription_request()` with native chunking — audio split into 30s feature chunks stacked along batch dimension for single forward pass; supports up to ~30min total
- **cohere:** Same 30s max_duration limit, handled by chunking
- **granite-nar:** Requires FlashAttention on GPU; no punctuation/capitalization in output; use autoregressive variant if needed
- **VSCode/VSCodium terminal:** Not supported due to window type detection limitations
- **Windows:** Not planned; fork by @eutychius has Windows support

### Code-level observations:

- **SentencePieceTokenizer EOS patch** in `models.py` (lines 79-102): Workaround for Canary model where `<s>` (token 3) is EOS but not flagged as special token
- **Cohere audio normalization** in `models.py`: The Cohere ASR model generates tokens with missing spaces (e.g., `Minister.he's`) when input audio amplitude is below [-1, 1]. Normalization is applied automatically in the cohere transcribe path.
- The UI handles extreme terminal sizes gracefully (tested down to 1x1 character terminals)
- Settings file path: `~/.config/faster_whisper_hotkey/transcriber_settings.json` (overridable with `--config`)

---

## Test Coverage Summary

The test suite is very thorough with approximately 15 test files covering:

- **Model initialization** for all 7 model types with edge cases (uppercase, old transformers version, trust_remote_code)
- **Transcription** for each model type with mocked outputs
- **Chunking behavior** for cohere (native chunking); voxtral tests cover native chunking via `apply_transcription_request` (short/long audio, error handling)
- **Language handling** for whisper (auto/empty), canary (invalid format/empty/None), granite (source/target pairs)
- **UI edge cases** including extreme terminal sizes (1x1, 3x5, narrow widths 1-50, various heights)
- **X11/Wayland** window detection workflow
- **Clipboard** backup/set/restore with error handling
- **Paste routing** between X11/Wayland and terminal/GUI windows
- **LLM corrector** with comprehensive error handling (timeouts, HTTP errors, invalid JSON, missing keys, empty content)
- **Settings** save/load/roundtrip with corruption handling
- **Config** loading with missing file and invalid JSON
- **CLI options** --headless and --config (with/without saved settings, custom paths, combined with --debug)
