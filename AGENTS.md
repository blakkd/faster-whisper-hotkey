# Faster Whisper Hotkey - Agent Documentation

## Project Overview

A minimalist push-to-talk transcription tool for Linux using multiple ASR models (faster-whisper, NVIDIA Parakeet/Canary, Mistral Voxtral, Cohere). Users press a hotkey to capture speech and release to transcribe directly into the focused text field.

**Key Flow**: Hold hotkey → speak → release → text appears in active window

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Entry Point                               │
│   __main__.py → transcribe.py (main entry)                  │
├─────────────────────────────────────────────────────────────┤
│                    Setup Phase                               │
│   ui.py (curses menu) → settings.py (save/load)             │
├─────────────────────────────────────────────────────────────┤
│                    Runtime                                   │
│   transcriber.py:                                            │
│     • pynput.KeyboardListener (hotkey detection)            │
│     • sounddevice.InputStream (audio capture)               │
│     • pulsectl (PulseAudio source selection)                │
├─────────────────────────────────────────────────────────────┤
│                    Models                                    │
│   models.py: ModelWrapper                                   │
│     • faster-whisper (WhisperModel)                         │
│     • parakeet (nemo ASRModel)                              │
│     • canary (nemo EncDecMultiTaskModel)                    │
│     • voxtral (transformers VoxtralForConditionalGeneration)│
│     • cohere (AutoModelForSpeechSeq2Seq)                    │
├─────────────────────────────────────────────────────────────┤
│                    Output                                    │
│   clipboard.py (pyperclip backup/set/restore)               │
│   paste.py (detect window type → send Ctrl+V or Ctrl+Shift+V)│
│   terminal.py (X11/Wayland window detection)                │
└─────────────────────────────────────────────────────────────┘
```

## Supported Models & Capabilities

| Model | Device | Languages | Translation | Auto Lang Detect | Notes |
|-------|--------|-----------|-------------|------------------|-------|
| faster-whisper (various) | CPU/CUDA | Many | ✓ (if multilingual) | Partial | int8/f16 |
| parakeet-tdt-0.6b-v3 | CPU/CUDA | 25 | ✗ | ✓ | Fastest, f16 only |
| canary-1b-v2 | CUDA only | 25 | ✓ | ✗ | f16 only |
| Voxtral-Mini-3B-2507 | CUDA only | 8 | ✗ | ✓ | <30s chunks, int4/8/f16 |
| cohere-transcribe-03-2026 | CPU/CUDA | 15 | ✗ | ✗ | f16 only |

## Key Files & Responsibilities

**Entry point**: `transcribe.py:main()` - Initializes UI, collects settings, launches transcriber

**Settings**: `settings.py`
- Location: `~/.config/faster_whisper_hotkey/transcriber_settings.json`
- Structure: device_name, model_type, model_name, compute_type, device, language, hotkey
- Model types: "whisper", "parakeet", "canary", "voxtral", "cohere"

**Transcriber**: `transcriber.py:MicrophoneTranscriber`
- Hotkeys: Pause, F4, F8, INSERT (default: Pause)
- Audio buffer: 10min circular, 16kHz sample rate, mono float32
- MIN_RECORDING_DURATION: 1.0s minimum to avoid noise

**Model wrapper**: `models.py:ModelWrapper`
- Loads model based on settings.model_type
- `.transcribe(audio_data, sample_rate, language)` returns string

**Output**: 
- `clipboard.py`: Backup/transcript/restore clipboard flow
- `paste.py`: Detects terminal vs non-terminal window → sends appropriate paste shortcut
- `terminal.py`: X11 (xdotool/xprop) or Wayland (swaymsg) window detection

## Configuration

**Audio devices**: Listed via `pulsectl.Pulse().source_list()` at startup

**Languages per model** (`config.py` from `available_models_languages.json`):
- faster-whisper: 90+ languages in `accepted_languages_whisper`
- canary: Pairs like "en-fr", "de-en" (see `canary_allowed_language_pairs`)
- Voxtral/cohere: Language codes specified directly

## Build & Run

**Install**: 
```bash
uv pip install .        # or
uv tool install .       # executable as 'faster-whisper-hotkey'
```

**Dependencies**: sounddevice, pynput, pulsectl, torch, nemo_toolkit[asr], faster-whisper, transformers, bitsandbytes, mistral-common[audio], pyperclip, soundfile, librosa

**Lint/Check**: `ruff check .` (configured in pyproject.toml)

## Known Limitations

1. **Voxtral**: <30s audio limit due to encoder constraints; chunks longer audio automatically
2. **VSCodium/VSCode terminal**: Not supported (can't reliably detect as terminal window)
3. **Canary/Voxtral**: CUDA-only
4. **Wayland paste**: Requires `wtype` tool for keyboard simulation

## Common Tasks

**Add new model type**:
1. Extend `transcribe.py:main()` with config UI section
2. Add loading logic in `models.py:_load_model()` under new model_type
3. Implement transcribe logic in `models.py:transcribe()` 
4. Update Settings if new required fields exist

**Change hotkey options**: Modify `hotkey_options` list in `transcribe.py` + `_parse_hotkey()` in `transcriber.py`

**Adjust audio handling**: Sample rate/buffer settings in `transcriber.py:MicrophoneTranscriber.__init__()`

**Terminal detection**: Add identifiers to `TERMINAL_IDENTIFIERS_X11/WAYLAND` lists in `terminal.py`

## Settings Format Example

```json
{
  "device_name": "alsa_input.usb-microphone-00.analog-stereo",
  "model_type": "parakeet",
  "model_name": "nvidia/parakeet-tdt-0.6b-v3",
  "compute_type": "float16",
  "device": "cuda",
  "language": "",
  "hotkey": "pause"
}
```

**Note**: `language` format for canary is `"source-target"` (e.g., "de-en"), empty string for parakeet, single code for others.

## File Structure Summary

```
src/faster_whisper_hotkey/
├── __main__.py           # Entry point: calls transcribe.main()
├── config.py             # Loads language/model configs from JSON
├── models.py             # ModelWrapper class for all model types
├── paste.py              # Window detection → paste shortcuts
├── settings.py           # Settings dataclass & file I/O
├── terminal.py           # X11/Wayland terminal detection
├── transcribe.py         # Main logic: curses UI, config flow
├── transcriber.py        # MicrophoneTranscriber class
└── ui.py                 # Curses menu helper functions
```

## Key Class Signatures

```python
# settings.py
@dataclass
class Settings:
    device_name: str
    model_type: str   # "whisper", "parakeet", "canary", "voxtral", "cohere"
    model_name: str
    compute_type: str # "float16", "int8", "int4"
    device: str       # "cuda" or "cpu"
    language: str
    hotkey: str = "pause"

# transcriber.py
class MicrophoneTranscriber:
    def __init__(self, settings: Settings)
    def run(self)                     # Main loop with keyboard listener
    def start_recording(self)         # Begin audio capture
    def stop_recording_and_transcribe(self)  # Process and queue
    def transcribe_and_send(self, audio_data)

# models.py
class ModelWrapper:
    def __init__(self, settings: Settings)   # Loads model into self.model
    def transcribe(self, audio_data, sample_rate=16000, language=None) -> str
```

## Dependency Requirements by Feature

| Feature | Dependencies |
|---------|--------------|
| Audio capture | sounddevice, numpy, pulsectl |
| Keyboard hotkey | pynput |
| fast-whisper models | faster-whisper |
| parakeet/canary | nemo_toolkit[asr], torch |
| voxtral | transformers, mistral-common[audio], bitsandbytes |
| cohere | transformers |
| Clipboard ops | pyperclip |
| Audio file I/O | soundfile, librosa |
