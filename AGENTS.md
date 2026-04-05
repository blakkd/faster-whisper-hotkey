## 📋 Project Overview

**faster-whisper-hotkey** is a minimalist push-to-talk transcription tool for Linux that leverages cutting-edge ASR models. Hold a hotkey, speak, release → text appears instantly in your focused field.

### Key Value Proposition
- **Speed**: Near-instant transcription even on CPU with smaller models
- **Flexibility**: Multiple model backends (faster-whisper, parakeet, canary, voxtral, cohere)
- **Integration**: Works in terminals, editors, chat apps anywhere you can type

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interaction Layer                  │
│  (hotkey → audio capture → transcription → paste)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Core Application                       │
│  transcribe.py ──→ settings.py ──→ ui.py (curses menu)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Transcription Pipeline                   │
│  transcriber.py ──→ models.py ──→ model backends            │
│  (recording)     (model wrapper) (whisper/parakeet/etc)     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     Output Delivery                         │
│  clipboard.py ──→ paste.py ──→ terminal.py                  │
│  (backup/restore) (X11/Wayland) (window detection)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Key Files & Responsibilities

| File | Purpose | Owner to Contact |
|------|---------|------------------|
| `__main__.py` | CLI entry point, debug flag setup | Anyone |
| `transcribe.py` | Main app logic, model config flow | Core team |
| `models.py` | Model abstraction layer, ASR backends | Core team |
| `transcriber.py` | Recording, hotkey handling, text delivery | Core team |
| `settings.py` | Settings persistence (`~/.config/faster_whisper_hotkey/`) | Anyone |
| `ui.py` | Curses-based menus (model selection, config) | UI specialists |
| `terminal.py` | X11/Wayland window detection for paste targeting | Platform experts |
| `paste.py` | Clipboard & keyboard paste logic | Platform experts |
| `clipboard.py` | Clipboard backup/restore utilities | Anyone |
| `llm_corrector.py` | Optional LLM-based text correction | Core team |
| `config.py` | Loads model/language config from JSON | Anyone |

---

## 🧠 Model Backends (CRITICAL KNOWLEDGE)

### Supported Models & Their Constraints

| Model | Type | Device | Key Notes |
|-------|------|--------|-----------|
| **cohere-transcribe-03-2026** | cohere | CPU/GPU | 15 languages, NO auto-detection, smart about hesitations |
| **parakeet-tdt-0.6b-v3** | parakeet | CPU/GPU | 25 langs, auto-detection, multilingual recording supported |
| **canary-1b-v2** | canary | CPU (F16) | 25 langs, transcription + translation possible |
| **Voxtral-Mini-3B-2507** | voxtral | GPU only | English + 7 langs, smart formatting, max 30s chunks |
| **faster-whisper** | whisper | CPU/GPU | Many langs, translation when source ≠ target |

### Important: Model Loading in `models.py`

Each model type has specific initialization logic. When adding a new backend:

1. Add to `_load_model()` method with proper device handling
2. Implement `transcribe()` with chunking if needed (see Voxtral's 30s limit)
3. Handle temporary file creation for models requiring it (canary, voxtral)
4. Add to config in `available_languages.json`

**⚠️ Critical**: The model wrapper uses `suppress_output()` context manager to hide OneLogger/NeMo initialization spam at import time. Keep this!

---

## 🔧 Configuration Flow

### Settings File Location
```
~/.config/faster_whisper_hotkey/transcriber_settings.json
```

### Setting Fields (Settings dataclass)
- `device_name`: PulseAudio input device name
- `model_type`: whisper/parakeet/canary/voxtral/cohere
- `model_name`: Hugging Face model identifier
- `compute_type`: int8/float16/int4 (model-dependent)
- `device`: cpu/cuda
- `language`: Language code or "auto"
- `hotkey`: pause/f4/f8/insert
- `llm_correction_enabled`: Boolean for LLM cleanup
- `llm_endpoint`: OpenAI-compatible endpoint URL
- `llm_model_name`: Model to use for correction

### Debug Logging
Set environment variable to enable detailed logging:
```bash
export FASTER_WHISPER_HOTKEY_DEBUG=1
# or run with --debug flag
faster-whisper-hotkey --debug
```

---

## 🚨 Known Limitations & Workarounds

| Issue | Status | Workaround |
|-------|--------|------------|
| Voxtral 30s audio limit | **Hardcoded** | Chunked processing in `_transcribe_single_chunk_voxtral()` |
| VSCode/VSCodium terminal paste | **Unsupported** | No workaround currently |
| Windows support | **Separate branch** | Use eutychius's feature/supportWindows branch |
| Uppercase chars on Wayland | **pyperclip fallback** | Typing mode may fail with special symbols |

---

## 📦 Development Setup

### Prerequisites
- Python 3.10+
- `uv` package manager (recommended for speed)
- PulseAudio/PipeWire backend for audio capture
- CUDA/cuDNN for GPU models (optional)

### Install Dependencies
```bash
# As a tool (recommended)
uv tool install .

# Or as editable package for development
uv pip install -e .
```

### Run Development Server
```bash
# Normal mode
faster-whisper-hotkey

# Debug mode
faster-whisper-hotkey --debug
```

### Test Pipeline
```bash
pytest
```

---

## 🎯 Core Technical Patterns

### 1. Audio Buffering Strategy (transcriber.py)
- Fixed buffer: `max_buffer_length = 10 * 60 * sample_rate` samples
- Circular overwrite: New audio replaces oldest when full
- Minimum recording duration: 1.0 seconds to avoid noise transcriptions

**Key Code Pattern**:
```python
def audio_callback(self, indata, frames, time_, status):
    audio_data = self._to_mono(indata)
    # ... normalize ...
    # ... append to buffer with bounds checking ...
```

### 2. Hotkey & Recording Control
- **Press** hotkey → `start_recording()` → Begin streaming audio
- **Release** hotkey → `stop_recording_and_transcribe()` → Queue for processing
- **Transcription queue**: Ensures multiple recordings are processed sequentially

**Critical**: Release stops recording even if transcription is in progress. Audio gets queued, text sent when done.

### 3. Paste Delivery (Platform-Specific)
- **X11**: Uses `xprop` + `pynput` for keyboard simulation
- **Wayland**: Uses `wtype` command (if available) with fallback to X11
- **Terminal detection**: Checks window class/name against `TERMINAL_IDENTIFIERS` list

**Paste Shortcut Mapping**:
| Window Type | Shortcut |
|-------------|----------|
| Terminal | Ctrl+Shift+V |
| Regular app | Ctrl+V |

### 4. Clipboard Backup/Restore Pattern
```python
original = backup_clipboard()
set_clipboard(transcribed_text)
paste_to_active_window()
restore_clipboard(original)
```

---

## 🛠 Common Tasks & Guidelines

### TUI Configuration Screen Architecture (ui.py)

The configuration flow uses a unified state machine pattern:

- **Single curses session**: All screens run within one `curses.wrapper()` call
- **State machine**: `ConfigStep` enum tracks current step
- **Data persistence**: `ConfigData` dataclass stores partial settings between steps
- **ESC behavior**: At any sub-screen → returns to initial screen; at initial screen → exits without saving
- **"Use Last Settings"**: Skips all screens immediately and starts transcriber

**Key functions**:
- `config_screen_main(stdscr)`: Main entry point, loads last settings into `ConfigData`
- `_handle_key_transition()`: Routes to appropriate screen handler based on current step
- `_screen_*()` handlers: Individual screens return `(next_step, config)` or `None` (ESC at initial)
- `_back_to_initial()`: Helper for ESC navigation back to initial screen
- `_final_save()`: Shows confirmation screen and saves settings
- `_create_settings_from_config()`: Creates Settings object from ConfigData

### Adding a New Model Backend

1. **Update `available_languages.json`** with supported languages/models
2. **Add config screens in `ui.py`**:
   - Add steps to `ConfigStep` enum (e.g., `FOO_DEVICE = auto()`)
   - Implement `_screen_foo_device()` following existing patterns
   - Route from `_screen_model_type()` based on selected model type
3. **Implement in `models.py._load_model()`**:
   - Device mapping logic
   - Quantization support if applicable
4. **Implement `transcribe()` method**:
   - Handle chunking if input size limited
   - Use temp files if model requires it
5. **Add to UI options** in `_screen_model_type()` (`model_options` list)

### Modifying Hotkey Behavior

- **Map keys** in `transcriber.py._parse_hotkey()` 
- Update `HOTKEY_OPTIONS` list in `transcribe.py`
- Test with `keyboard.Key` enum values from pynput

### LLM Correction Flow

1. User enables in config menu → `llm_correction_enabled = True`
2. After transcription, `llm_corrector.correct(text)` is called
3. Falls back to original text on any error
4. Uses OpenAI-compatible endpoint (vLLM, Ollama, etc.)

**Prompt**: Hardcoded professional cleanup instructions in `llm_corrector.py`. Modify there if needed.

---

## 🔍 Troubleshooting Checklist

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No audio captured | Wrong PulseAudio device selected | Check device name with `pactl list sources` |
| Transcription empty | Recording < 1s or no speech | Wait longer or speak louder |
| Paste not working | Wayland wtype missing | Install xdotool/wtype or use X11 fallback |
| Model load fails | Missing CUDA for GPU models | Switch to CPU device or install cuDNN |
| Curses menu broken | Terminal too small | Resize terminal > 40 chars wide |

---

## 📚 Important External Dependencies

| Library | Purpose | Key Notes |
|---------|---------|-----------|
| `faster-whisper` | Whisper inference engine | CPU/GPU optimized |
| `nemo_toolkit[asr]` | Parakeet/Canary models | NVIDIA ASR toolkit |
| `transformers` | Voxtral/Cohere models | Hugging Face backend |
| `sounddevice` | Audio capture | Cross-platform audio API |
| `pulsectl` | PulseAudio device selection | Linux audio server control |
| `pynput` | Keyboard simulation | Paste & hotkey handling |
| `pyperclip` | Clipboard operations | Optional, fallback to typing if missing |
