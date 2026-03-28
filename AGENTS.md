# Faster Whisper Hotkey - Agent Guide

## Overview
Push-to-talk transcription tool for Linux. Press and hold a hotkey to record speech, release to transcribe and type output into the active window.

## Architecture

### Entry Points
- `__main__.py` (line 3): Calls `transcribe.main()`
- CLI command: `faster-whisper-hotkey`

### Core Modules

| File | Purpose |
|------|---------|
| `transcribe.py` | Main flow: curses UI wizard → settings → launches transcriber |
| `transcriber.py` | `MicrophoneTranscriber` class: audio capture, hotkey handling, transcription queue |
| `models.py` | `ModelWrapper`: unified interface for 5 STT backends |
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
- **Language**: Language-agnostic (none required)
- **Precision**: float16 only

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

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `MIN_RECORDING_DURATION` | 1.0s | `transcriber.py:23` |
| Sample Rate | 16000 Hz | `transcriber.py:29` |
| Buffer Max | 10 minutes | `transcriber.py:28` |
| Block Size | 4000 samples | `transcriber.py:173` |

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
  "hotkey": str             # pause|f4|f8|insert (lowercase)
}
```

## Dependencies (Critical)

- **Audio capture**: `sounddevice`, `pulsectl` (PulseAudio control)
- **Hotkey detection**: `pynput`
- **Whisper**: `faster-whisper`, `torch`
- **NVIDIA models**: `nemo_toolkit[asr]`
- **Transformers**: `transformers`, `mistral-common[audio]`, `bitsandbytes` (quantization)
- **Audio processing**: `librosa`, `soundfile`, `numpy`

## Common Tasks

### Add new model backend
1. Extend UI in `transcribe.py` with configuration menu
2. Add loading logic in `models.py::_load_model()`
3. Implement transcription in `models.py::transcribe()`

### Change default hotkey
- Modify defaults in `settings.py:40`

### Adjust audio chunking limit (Voxtral)
- Edit `MAX_DURATION_SECONDS = 30` in `models.py:256`

### Support new terminal emulator  
- Add identifier to `TERMINAL_IDENTIFIERS_X11/WAYLAND` in `terminal.py:9-23`

## Debugging Tips

- Set logging level: `logging.basicConfig(level=logging.DEBUG)` before other imports
- Voxtral/NeMo warnings suppressed via `_setup_logging()` in `models.py:39`
- Audio device issues: Use `arecord -l` to list PulseAudio sources
- CUDA errors: Verify model supports requested precision/device

## Testing

No test suite exists. Manual testing flow:
1. Run `python -m faster_whisper_hotkey`
2. Configure model settings via curses menu
3. Press hotkey, speak, release
4. Verify text appears in focused application
