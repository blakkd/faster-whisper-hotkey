# Onboarding Guide

## Project Overview

**faster-whisper-hotkey** is a push-to-talk transcription tool for Linux. Hold a hotkey (PAUSE, F4, F8, or INSERT), speak, release — and the transcribed text is pasted into the focused text field, anywhere on screen (terminal, text editor, game chat, etc.).

- **Version**: 0.5.2
- **License**: WTFPL v2
- **Python**: 3.11.14 (strict requirement)
- **OS**: Linux only
- **Package Manager**: [uv](https://docs.astral.sh/uv/)

## Quick Start

### Install from source

```bash
git clone https://github.com/blakkd/faster-whisper-hotkey
cd faster-whisper-hotkey
uv pip install .
faster-whisper-hotkey
```

### Install as a uv tool (run from any directory)

```bash
uv tool install .
faster-whisper-hotkey
```

### Run without installing

```bash
cd faster-whisper-hotkey
uv run faster-whisper-hotkey
```

### Run tests

```bash
uv run pytest
```

## Architecture

### High-Level Data Flow

```
[Hotkey Pressed]
  → Audio capture via sounddevice callback (16kHz mono float32)
[Hotkey Released]
  → Audio buffer queued for transcription
  → ModelWrapper.transcribe() runs selected ASR model
  → (Optional) LLMCorrector sends text to OpenAI-compatible API for cleanup
  → Clipboard backup → Set clipboard → Send paste shortcut → Restore clipboard
```

### Runtime Loop

```
[__main__.main()]
  → [transcribe.main()]
      → Logging setup
      → curses.wrapper(config_screen_main)  ← Interactive TUI config
      → MicrophoneTranscriber(settings).run()
          → pynput keyboard listener (hotkey detection)
          → sounddevice InputStream (audio capture)
          → Transcription queue + worker thread
```

### Module Overview

| File               | Responsibility                                                             |
| ------------------ | -------------------------------------------------------------------------- |
| `__main__.py`      | CLI entry point; `--debug`, `--headless`, `--config` flags                 |
| `transcribe.py`    | Orchestrator: logging, curses TUI → transcriber launch                     |
| `settings.py`      | `Settings` dataclass + JSON save/load (`~/.config/faster_whisper_hotkey/`) |
| `models.py`        | `ModelWrapper` — loads/runs 7 model types with output suppression          |
| `transcriber.py`   | `MicrophoneTranscriber` — audio capture, hotkey detection, paste           |
| `ui.py`            | Curses TUI — 29-step config flow (`ConfigStep` enum)                       |
| `clipboard.py`     | pyperclip wrapper: backup, set, restore                                    |
| `paste.py`         | X11/Wayland detection; sends correct paste shortcut                        |
| `terminal.py`      | Window detection via xdotool/xprop (X11) or swaymsg (Wayland)              |
| `llm_corrector.py` | `LLMCorrector` — sends transcription to LLM API for cleanup                |
| `config.py`        | Loads `available_languages.json`; exposes language/model lists             |

### Supported Models (7)

| Model Type                | Internal Key  | HuggingFace Repo                      | Languages | Device   | Precision                                            | Notes                              |
| ------------------------- | ------------- | ------------------------------------- | --------- | -------- | ---------------------------------------------------- | ---------------------------------- |
| faster-whisper            | `whisper`     | Systran/faster-whisper                | 100+      | CPU/GPU  | User-selectable (int8/f16 CPU, f16/f32/int8 GPU)     | 16 sizes                           |
| parakeet-tdt-0.6b-v3      | `parakeet`    | nvidia/parakeet-tdt-0.6b-v3           | 25 (auto) | CPU/GPU  | User-selectable (f32/bf16/f16 CPU, +int8/int4 GPU)   | Very fast on CPU (float32)         |
| canary-1b-v2              | `canary`      | nvidia/canary-1b-v2                   | 25        | CPU/GPU  | User-selectable (f32/bf16/f16 CPU, +int8/int4 GPU)   | Transcription + translation        |
| Voxtral-Mini-3B-2507      | `voxtral`     | mistralai/Voxtral-Mini-3B-2507        | 8 (auto)  | GPU only | User-selectable (f16/int8/int4)                      | Native chunking                    |
| cohere-transcribe-03-2026 | `cohere`      | CohereLabs/cohere-transcribe-03-2026  | 14        | CPU/GPU  | User-selectable (f32/bf16/f16 CPU, +bf16/f32/int8/4) | Native chunking                    |
| granite-speech-4.1-2b-nar | `granite-nar` | ibm-granite/granite-speech-4.1-2b-nar | 9         | CPU/GPU  | User-selectable (f32/bf16/f16 CPU, +bf16/f32/int8/4) | Non-AR, very fast, no punctuation  |
| granite-speech-4.1-2b     | `granite`     | ibm-granite/granite-speech-4.1-2b     | 6→8       | CPU/GPU  | User-selectable (f32/bf16/f16 CPU, +bf16/f32/int8/4) | AR with punctuation/capitalization |

### Model Native Precisions

Precision each model was released and trained in:

| Model                     | Native Precision |
| ------------------------- | ---------------- |
| faster-whisper            | float16          |
| parakeet-tdt-0.6b-v3      | float32          |
| canary-1b-v2              | float32          |
| Voxtral-Mini-3B-2507      | float32          |
| cohere-transcribe-03-2026 | bfloat16         |
| granite-speech-4.1-2b     | bfloat16         |
| granite-speech-4.1-2b-nar | bfloat16         |

## Development Setup

### Prerequisites

- **Python 3.11.14** — exact version required (`requires-python = "==3.11.14"`)
- **uv** — [install from docs](https://docs.astral.sh/uv/)
- **PulseAudio** — required for audio device detection (pulsectl)
- **X11 or Wayland (Sway)** — required for paste shortcut detection
- **Optional (X11)**: `xdotool`, `xprop` — for window type detection
- **Optional (Wayland)**: `wtype` — for paste shortcut on Wayland
- **Optional (GPU)**: NVIDIA drivers + CUDA for GPU inference

### For Nvidia GPU

FlashAttention is required for GPU inference with `granite-nar`. Pre-built wheels are available at https://mjunya.com/flash-attention-prebuild-wheels/

### Install for development

```bash
git clone https://github.com/blakkd/faster-whisper-hotkey
cd faster-whisper-hotkey
uv pip install .
```

This installs the package and all dependencies from `pyproject.toml`. No separate dev dependencies are configured — tests run against the installed package.

### Run from source

```bash
uv run faster-whisper-hotkey              # Interactive config UI → transcribe
uv run faster-whisper-hotkey --headless   # Skip UI, use saved settings
uv run faster-whisper-hotkey --debug      # Verbose logging with module paths
uv run faster-whisper-hotkey --config /path/to/settings.json  # Custom config
```

### Run tests

```bash
uv run pytest                          # All tests (verbose, short traceback)
uv run pytest tests/test_models.py     # Specific test file
uv run pytest tests/test_models.py::TestModelWrapperInitialization -v  # Specific class
```

Pytest config is in `pytest.ini`: DeprecationWarning, UserWarning, and FutureWarning are filtered out.

## Code Conventions

### Style

- Follows standard Python conventions (PEP 8)
- Ruff is configured in `pyproject.toml` with no ignored rules
- No explicit formatter is enforced; code uses 4-space indentation
- Line length is 120 characters (configured in `pyproject.toml`)
- Docstrings use triple double quotes (`"""`)

### Naming

- **Files**: snake_case (`llm_corrector.py`, `audio_callback.py`)
- **Classes**: PascalCase (`ModelWrapper`, `MicrophoneTranscriber`, `LLMCorrector`)
- **Functions/methods**: snake_case (`transcribe_and_send`, `_normalize_audio`)
- **Private methods**: leading underscore (`_parse_hotkey`, `_send_via_clipboard`)
- **Constants**: UPPER_SNAKE_CASE (`MIN_RECORDING_DURATION`, `TERMINAL_IDENTIFIERS`)
- **Type hints**: used throughout; Python 3.11+ syntax (`str | None`)

### Logging

- Every module has `logger = logging.getLogger(__name__)` at the top
- In debug mode (`--debug`): format is `LEVEL:module.path:message`, all levels shown
- In normal mode: format is `LEVEL:message` (no module path), only INFO+ from `faster_whisper_hotkey` package, WARNING+ from third-party
- Use `logger.info()` for user-facing events, `logger.debug()` for internals, `logger.warning()` for recoverable issues

### Error Handling

- Heavy dependencies (torch, nemo) are imported inside `with suppress_output():` context managers to silence their noisy output
- Transcription errors return `""` (empty string) rather than raising — the transcriber logs the error and continues
- LLM correction failures silently fall back to original text
- Clipboard failures fall back to character-by-character typing via pynput

### Testing Patterns

See `tests/` directory (15 files, ~2500+ assertions). Key patterns:

```python
# Mock settings with a simple class
class MockSettings:
    def __init__(self, model_type, model_name, device, compute_type=None, language="auto"):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language

# Patch heavy dependencies at the module level
@patch("faster_whisper_hotkey.models.WhisperModel")
def test_init_whisper_model(self, mock_whisper):
    from faster_whisper_hotkey.models import ModelWrapper
    settings = MockSettings(model_type="whisper", model_name="small", device="cpu", compute_type="int8")
    wrapper = ModelWrapper(settings)
    mock_whisper.assert_called_once_with(...)

# Patch settings file path for save/load tests
@patch("faster_whisper_hotkey.settings.SETTINGS_FILE", "/tmp/mock_settings.json")
def test_save_settings_success(self, tmp_path):
    ...
```

- Use `unittest.mock.MagicMock` and `patch` extensively
- Tests import from installed package (`from faster_whisper_hotkey.models import ModelWrapper`)
- No external services required — all network/audio/ML calls are mocked
- Use `pytest` classes for organization (`Test*` classes)

## How to Add a New Model

Adding a new model type involves changes across 4 files:

### 1. `models.py` — Model loading and transcription

Add a new branch in `_load_model()`:

```python
elif mt == "my_model":
    repo_id = self.settings.model_name
    self.processor = AutoProcessor.from_pretrained(repo_id)
    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        repo_id,
        device_map={"": self.settings.device},
        torch_dtype=torch.bfloat16 if self.settings.device == "cuda" else torch.float32,
    ).eval()
```

Add a new branch in `transcribe()`:

```python
elif mt == "my_model":
    waveform = torch.from_numpy(audio_data).to(self.settings.device)
    inputs = self.processor(waveform, return_tensors="pt").to(self.settings.device)
    outputs = self.model.generate(**inputs, max_new_tokens=400)
    return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

If the model library produces noisy output, wrap imports and loading in `with suppress_nemo():` or `with suppress_output():`.

### 2. `ui.py` — Configuration screens

Add new `ConfigStep` enum values:

```python
class ConfigStep(Enum):
    # ... existing steps ...
    MY_MODEL_DEVICE = auto()
    MY_MODEL_LANGUAGE = auto()
```

Add screen handlers:

```python
def _screen_my_model_device(stdscr, config: ConfigData):
    options = ["cuda", "cpu"]
    selected = curses_menu(stdscr, "Compute Device", options)
    if selected is None:
        return _back_to_initial(config)
    config.device = selected
    return (ConfigStep.MY_MODEL_LANGUAGE, config)

def _screen_my_model_language(stdscr, config: ConfigData):
    from .config import accepted_languages_my_model
    selected = curses_menu(stdscr, "Language", accepted_languages_my_model)
    if selected is None:
        return _back_to_initial(config)
    config.model_name = "org/my-model"
    config.compute_type = "float16"
    config.language = selected
    return (ConfigStep.HOTKEY, config)  # Route to common hotkey step
```

Register the step in `_handle_key_transition()`:

```python
elif current_step == ConfigStep.MY_MODEL_DEVICE:
    return _screen_my_model_device(stdscr, config)
elif current_step == ConfigStep.MY_MODEL_LANGUAGE:
    return _screen_my_model_language(stdscr, config)
```

Route from model type selection in `_screen_model_type()`:

```python
type_map = {
    # ... existing ...
    "my-model-name": "my_model",
}
# After setting config.model_type:
elif config.model_type == "my_model":
    return (ConfigStep.MY_MODEL_DEVICE, config)
```

### 3. `available_languages.json` — Language definitions

Add language lists for your model:

```json
{
  "accepted_languages_my_model": ["en", "fr", "de", "es"],
  "my_model_source_target_languages": ["en", "fr"],
  "my_model_allowed_language_pairs": ["en-en", "en-fr", "fr-en", "fr-fr"]
}
```

### 4. `config.py` — Export language lists

```python
accepted_languages_my_model = _CONFIG["accepted_languages_my_model"]
my_model_source_target_languages = _CONFIG["my_model_source_target_languages"]
my_model_allowed_language_pairs = _CONFIG["my_model_allowed_language_pairs"]
```

### 5. Tests

Add test cases to `tests/test_models.py`:

```python
@patch("faster_whisper_hotkey.models.AutoProcessor")
@patch("faster_whisper_hotkey.models.AutoModelForSpeechSeq2Seq")
def test_init_my_model_cuda(self, mock_model, mock_processor):
    from faster_whisper_hotkey.models import ModelWrapper
    mock_model.from_pretrained.return_value = MagicMock()
    settings = MockSettings(model_type="my_model", model_name="org/my-model", device="cuda")
    wrapper = ModelWrapper(settings)
    assert wrapper.model_type == "my_model"
```

## How to Add a New TUI Configuration Screen

The TUI follows a state machine pattern driven by the `ConfigStep` enum.

### Navigation rules

- **ENTER**: proceed to the next step
- **ESC**: return to the initial screen (not exit, unless at initial screen where ESC exits)
- At the initial screen: "Use Last Settings" skips all screens and starts immediately

### Screen handler signature

Every screen handler takes `(stdscr, config: ConfigData)` and returns one of:

- `None` — ESC at initial screen, exit without saving
- `tuple(ConfigStep, ConfigData)` — proceed to next step with updated config
- `Settings` — configuration complete, save and return

### Example: adding a "sample rate" screen

```python
# 1. Add to ConfigStep enum
SAMPLE_RATE = auto()

# 2. Add screen handler
def _screen_sample_rate(stdscr, config: ConfigData):
    options = ["8000", "16000", "44100"]
    selected = curses_menu(stdscr, "Sample Rate", options)
    if selected is None:
        return _back_to_initial(config)
    config.sample_rate = int(selected)  # Add field to ConfigData
    return (ConfigStep.HOTKEY, config)

# 3. Register in _handle_key_transition()
elif current_step == ConfigStep.SAMPLE_RATE:
    return _screen_sample_rate(stdscr, config)

# 4. Route from a previous step to SAMPLE_RATE
```

### Text input screens

For free-form input (like LLM endpoint), use `get_text_input()`:

```python
def _screen_my_text_input(stdscr, config: ConfigData):
    default = config.my_field or "default value"
    result = get_text_input(stdscr, "Enter value: ", default)
    if result is None:
        return _back_to_initial(config)
    config.my_field = result
    return (ConfigStep.NEXT_STEP, config)
```

## Key Implementation Details

### Audio Pipeline

- Sample rate: 16000 Hz (fixed)
- Format: mono float32, normalized to [-1, 1]
- Buffer: circular buffer of 10 minutes max (`10 * 60 * 16000` samples)
- Minimum recording: 1 second (`MIN_RECORDING_DURATION`) — shorter recordings are silently discarded
- Block size: 4000 samples per callback

### Paste Mechanics

The paste system must detect whether the focused window is a terminal or GUI app, and whether the compositor is X11 or Wayland:

| Environment | GUI Window      | Terminal              |
| ----------- | --------------- | --------------------- |
| X11         | Ctrl+V (pynput) | Ctrl+Shift+V (pynput) |
| Wayland     | `wtype ctrl+v`  | `wtype ctrl+shift+v`  |

Detection:

- X11: `xdotool getactivewindow` + `xprop -id <win> WM_CLASS` → check class against `TERMINAL_IDENTIFIERS`
- Wayland: `swaymsg -t get_tree` → find focused node → check `app_id` + `name`

**Limitation**: Cannot detect "sub-windows" — VSCode/VSCodium integrated terminals are seen as the editor, not a terminal, so paste uses Ctrl+V which doesn't work inside the embedded terminal.

### Transcription Queue

When a transcription is in progress and the user releases the hotkey, the new audio is queued (`self.transcription_queue`). After the current transcription completes, `process_next_transcription()` checks the queue and starts the next one in a daemon thread.

### LLM Correction Prompt

The LLM prompt (in `llm_corrector.py`) asks the model to:

1. Fix disjointed phrasing and sentence structure
2. Convert spelled-out numbers/symbols to standard form (e.g., "five percent" → "5%")
3. Correct homophones based on context
4. Preserve original language (no translation)
5. Output ONLY the corrected text

Temperature: 0.3, max_tokens: 1024, timeout: 120s.

### Settings File

Path: `~/.config/faster_whisper_hotkey/transcriber_settings.json`

```json
{
  "device_name": "alsa_input.pci-0000_00_1f.3.analog-stereo",
  "model_type": "granite-nar",
  "model_name": "ibm-granite/granite-speech-4.1-2b-nar",
  "compute_type": "float32",
  "device": "cpu",
  "language": "en",
  "hotkey": "pause",
  "llm_correction_enabled": false,
  "llm_endpoint": "",
  "llm_model_name": ""
}
```

The `language` field uses `source-target` format (e.g., `"en-de"`) for translation-capable models.

## Debugging Tips

### Enable debug logging

```bash
uv run faster-whisper-hotkey --debug
```

This shows full module paths and all log levels, including third-party library output.

### Test with specific model

Run without the TUI using `--headless` with a custom config:

```bash
uv run faster-whisper-hotkey --headless --config /path/to/settings.json
```

### Common issues

| Issue                         | Cause                             | Fix                                                              |
| ----------------------------- | --------------------------------- | ---------------------------------------------------------------- |
| "Source not found"            | PulseAudio device name changed    | Re-run without `--headless` to reselect device                   |
| Paste not working on terminal | VSCode/VSCodium embedded terminal | Use standalone terminal emulator                                 |
| "wtype not found" on Wayland  | wtype not installed               | `sudo apt install wtype`                                         |
| FlashAttention error          | Missing FlashAttention on GPU     | Install from https://mjunya.com/flash-attention-prebuild-wheels/ |
| NeMo model noisy output       | NeMo's OneLogger                  | Already suppressed; use `--debug` to see if needed               |

## Testing Strategy

The test suite covers all modules with mocked dependencies:

| Test File                   | Coverage                                                   |
| --------------------------- | ---------------------------------------------------------- |
| `test_models.py`            | Model loading + transcription for all 7 types              |
| `test_models_extended.py`   | Additional edge cases for models                           |
| `test_transcribe.py`        | Main entry point flow                                      |
| `test_settings.py`          | Settings save/load/roundtrip/corruption                    |
| `test_ui.py`                | TUI menu rendering and navigation                          |
| `test_ui_edge_cases.py`     | Terminal size edge cases (1x1 to 300px width)              |
| `test_config.py`            | Config loading and resource path resolution                |
| `test_config_extended.py`   | Extended config validation                                 |
| `test_headless_config.py`   | Headless mode with saved/missing settings                  |
| `test_llm_corrector.py`     | LLM correction: success, failure, edge cases               |
| `test_clipboard.py`         | Clipboard backup/set/restore                               |
| `test_paste.py`             | X11/Wayland paste shortcut selection                       |
| `test_terminal.py`          | Terminal window detection (X11 + Wayland)                  |
| `test_model_all_configs.py` | Integration test: per-model transcription (7 test classes) |

## Release Process

1. Update version in `pyproject.toml` and `src/faster_whisper_hotkey/__init__.py`
2. Regenerate lockfile: `uv lock`
3. Commit with message: `bump version X.Y.Z`

## File Reference

```
faster-whisper-hotkey/
├── pyproject.toml          # Build config, dependencies, version
├── pytest.ini              # Test runner configuration
├── uv.lock                 # Locked dependency versions
├── README.md               # User-facing documentation
├── LICENSE.txt             # WTFPL v2
├── TODO                    # Open issues to address
├── docs/
│   └── onboarding.md       # ← This file
├── test_audio_data/
│   └── test.mp3            # Audio fixture for integration tests
├── src/faster_whisper_hotkey/
│   ├── __init__.py         # Package version
│   ├── __main__.py         # CLI entry point
│   ├── config.py           # Language/model list loader
│   ├── available_languages.json  # Static config data
│   ├── settings.py         # Settings dataclass + JSON persistence
│   ├── models.py           # ModelWrapper (7 model types)
│   ├── transcriber.py      # MicrophoneTranscriber (hotkey + audio + paste)
│   ├── ui.py               # Curses TUI (25 config steps)
│   ├── clipboard.py        # Clipboard backup/set/restore
│   ├── paste.py            # X11/Wayland paste shortcut
│   ├── terminal.py         # Window type detection
│   └── llm_corrector.py    # LLM transcription correction
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_models_extended.py
    ├── test_transcribe.py
    ├── test_settings.py
    ├── test_ui.py
    ├── test_ui_edge_cases.py
    ├── test_config.py
    ├── test_config_extended.py
    ├── test_headless_config.py
    ├── test_llm_corrector.py
    ├── test_clipboard.py
    ├── test_paste.py
    ├── test_terminal.py
    ├── conftest.py
    └── test_model_all_configs.py
```
