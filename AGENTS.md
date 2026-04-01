# Faster Whisper Hotkey - Agent Guide

## Overview
Push-to-talk transcription tool for Linux. Press and hold a hotkey to record speech, release to transcribe and type output into the active window.

## Architecture

### Entry Points
- `__main__.py`: Parses CLI args, calls `transcribe.main()`
- CLI commands:
  - `faster-whisper-hotkey` - Normal mode (simplified logs)
  - `faster-whisper-hotkey --debug` - Debug mode (full module paths in logs)

### Core Modules

| File | Purpose |
|------|---------|
| `transcribe.py` | Main flow: curses UI wizard → settings → launches transcriber |
| `transcriber.py` | `MicrophoneTranscriber` class: audio capture, hotkey handling, transcription queue |
| `models.py` | `ModelWrapper`: unified interface for 5 STT backends |
| `llm_corrector.py` | `LLMCorrector`: HTTP requests to OpenAI-compatible endpoints for text correction |
| `config.py` | Loads available models/languages from JSON |
| `settings.py` | Settings persistence at `~/.config/faster_whisper_hotkey/transcriber_settings.json` |
| `clipboard.py` | Clipboard backup/restore using `pyperclip` |
| `paste.py` | OS-specific paste (X11 vs Wayland, terminal vs GUI) |
| `terminal.py` | Detects focused window type (uses `xdotool`, `swaymsg`) |
| `ui.py` | Curses-based scrollable menu for configuration |

## Supported Models

### 1. faster-whisper (`model_type="whisper"`)
- **Models**: tiny, base, small, medium, large (v1-v3), distil variants
- **Device**: cuda or cpu
- **Precision**: int8 (cpu), float16/int8 (cuda)
- **Languages**: 96+ languages + auto-detection
- **Note**: `.en` suffix models are English-only

### 2. NVIDIA Parakeet (`model_type="parakeet"`)
- **Model**: `nvidia/parakeet-tdt-0.6b-v3`
- **Device**: cuda or cpu
- **Language**: Language-agnostic (auto-detection, no explicit language required)
- **Precision**: Uses `compute_type="float16"` in settings (placeholder; NeMo doesn't use it for precision control)

### 3. NVIDIA Canary (`model_type="canary"`)
- **Model**: `nvidia/canary-1b-v2`
- **Device**: CUDA only
- **Mode**: Speech-to-text or speech-to-speech translation
- **Language**: Format `"source-target"` (e.g., `"en-de"`, `"fr-en"`)
- **Pairs**: 40+ supported pairs (see `available_models_languages.json`)

### 4. Mistral Voxtral (`model_type="voxtral"`)
- **Model**: `mistralai/Voxtral-Mini-3B-2507`
- **Device**: CUDA only
- **Precision**: int4, int8, float16
- **Language**: `"auto"` (auto-detection)
- **Constraint**: Audio <30s recommended; long audio is chunked automatically

### 5. Cohere (`model_type="cohere"`)
- **Model**: `CohereLabs/cohere-transcribe-03-2026`
- **Device**: cuda or cpu
- **Language**: Explicit selection required (no auto-detection)
- **Supported**: en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko
- **Note**: Loads with `dtype=torch.float32` internally despite `compute_type="float16"` in settings

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `MIN_RECORDING_DURATION` | 1.0s | `transcriber.py:23` |
| Sample Rate | 16000 Hz | `transcriber.py:29` |
| Buffer Max | 10 minutes (~960,000 samples) | `transcriber.py:30` |
| Block Size | 4000 samples | `transcriber.py:163` |

## Hotkeys

Configurable via UI; options: `pause`, `f4`, `f8`, `insert`. Press to start recording, release to stop and transcribe. Default: `pause`.

## Text Output Methods

1. **Clipboard + Paste** (preferred): Uses `pyperclip` → OS-specific paste shortcut
   - X11: Ctrl+V (GUI), Ctrl+Shift+V (terminal)
   - Wayland: Via `wtype` command
2. **Fallback Typing**: Sends key-by-key via `pynput.keyboard.Controller()`

## Settings Schema

```python
{
  "device_name": str,       # PulseAudio source name
  "model_type": str,        # whisper|parakeet|canary|voxtral|cohere
  "model_name": str,        # Model path/identifier
  "compute_type": str,      # float16|int8|int4 (model-dependent)
  "device": str,            # cuda|cpu
  "language": str,          # ISO lang code or "source-target" for canary
  "hotkey": str,            # pause|f4|f8|insert (lowercase)
  "llm_correction_enabled": bool,  # Enable/disable LLM correction
  "llm_endpoint": str,      # OpenAI-compatible endpoint URL
  "llm_model_name": str     # Model name for LLM correction
}
```

## Dependencies (Critical)

- **Audio capture**: `sounddevice`, `pulsectl` (PulseAudio control)
- **Hotkey detection**: `pynput`
- **Whisper**: `faster-whisper`, `torch`
- **NVIDIA models**: `nemo_toolkit[asr]`
- **Transformers**: `transformers`, `mistral-common[audio]`, `bitsandbytes` (quantization)
- **LLM correction**: `requests` (HTTP client for OpenAI-compatible endpoints)
- **Audio processing**: `librosa`, `soundfile`, `numpy`

## Common Tasks

### Add new model backend
1. Extend UI in `transcribe.py` with configuration menu
2. Add loading logic in `models.py::_load_model()`
3. Implement transcription in `models.py::transcribe()`

### Change default hotkey
- Modify dataclass default in `settings.py:22`: `hotkey: str = "pause"`
- Also update fallback in `settings.py:37`: `data.setdefault("hotkey", "pause")`

### Adjust audio chunking limit (Voxtral)
- Edit `MAX_DURATION_SECONDS = 30` in `models.py:255`

### Support new terminal emulator  
- Add identifier to `TERMINAL_IDENTIFIERS_X11` in `terminal.py:10-20`
- WAYLAND uses same list (`terminal.py:23`)

## Debugging Tips

- **Debug mode**: Run with `--debug` flag for full module paths in log output
- Default mode uses simplified logs (e.g., `INFO:Recording...`)
- Debug mode shows module names (e.g., `INFO:faster_whisper_hotkey.transcriber:...`)
- Voxtral/NeMo warnings suppressed via `_setup_logging()` in `models.py:39`
- Audio device issues: Use `arecord -l` to list PulseAudio sources
- CUDA errors: Verify model supports requested precision/device

## Testing

### Test Suite Overview
Pytest-based test suite with **189 tests** across 8 test files:

| Test File | Tests | Target Module | Description |
|-----------|-------|---------------|-------------|
| `tests/test_settings.py` | 9 | `settings.py` | Settings dataclass, save/load JSON |
| `tests/test_config.py` | 22 | `config.py` | Model/language config constants, JSON loading |
| `tests/test_clipboard.py` | 14 | `clipboard.py` | Clipboard backup/restore with pyperclip mocking |
| `tests/test_terminal.py` | 24 | `terminal.py` | X11/Wayland window detection |
| `tests/test_paste.py` | 11 | `paste.py` | Paste shortcuts, terminal vs GUI routing |
| `tests/test_models.py` | 17 | `models.py` | ModelWrapper initialization and transcription |
| `tests/test_models_extended.py` | 23 | `models.py` | Extended edge cases for all model types, chunking, error handling |
| `tests/test_config_extended.py` | 22 | `config.py` | Extended tests for model/language configuration |

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_models.py -v

# Run with coverage (install pytest-cov first)
python -m pytest tests/ --cov=src/faster_whisper_hotkey --cov-report=term-missing
```

### Test Configuration
Configuration stored in `pytest.ini` at project root. Uses unittest.mock for all external dependencies.

### Testing Patterns

#### Mocking External Dependencies
```python
from unittest.mock import MagicMock, patch

@patch("faster_whisper_hotkey.models.WhisperModel")
def test_whisper_transcription(mock_whisper):
    mock_model = MagicMock()
    mock_segment = MagicMock(text="hello world")
    mock_model.transcribe.return_value = ([mock_segment], None)
    mock_whisper.return_value = mock_model
    # ... assertions
```

#### Mocking Subprocess Calls
```python
@patch("faster_whisper_hotkey.terminal.subprocess.check_output")
def test_x11_window_detection(mock_check_output):
    mock_check_output.return_value = b'_NET_ACTIVE_WINDOW 8675309'
    # ... assertions
```

#### Mocking Environment Variables
```python
import os
@patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"})
def test_wayland_paste():
    # ... test Wayland path
```

### CI Integration
Tests are designed for CI/CD pipelines. All external dependencies (OS APIs, ML models, audio devices) are mocked.

### Known Test Limitations
- UI (`ui.py`) excluded from automation due to curses complexity
- Actual STT model inference not tested (all mocked via MagicMock)
- Audio capture paths in `transcriber.py` not yet covered

### Mock Guidelines
1. **ML Models**: Always mock `WhisperModel`, `ASRModel`, `VoxtralForConditionalGeneration` via `@patch("faster_whisper_hotkey.models...")`
2. **Subprocess**: Use `@patch("subprocess.check_output")` or `@patch("subprocess.run")`
3. **Environment**: Use `@patch.dict(os.environ, {...})` for environment-dependent paths (X11 vs Wayland)
4. **Temp Files**: Mock `tempfile.NamedTemporaryFile` context manager with `__enter__` returning mock with `.name` attribute

### Test Troubleshooting
- If tests import real modules: verify patch targets use `"faster_whisper_hotkey.models..."`, not `"models..."`
- For Voxtral tests: use `patch.object(wrapper, "_transcribe_single_chunk_voxtral")` instead of full mock chain
- Pyperclip not installed? Tests still pass—clipboard.py has graceful fallback checks
