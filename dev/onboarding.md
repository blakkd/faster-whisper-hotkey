# onboarding.md

Context document for collaborators and agents working in this repo.

## What this is

**faster-whisper-hotkey** — minimalist push-to-talk STT for Linux. Hold a hotkey (Pause/F4/F8/Insert), speak, release → transcribed text is pasted into the currently focused text field (terminal, editor, chat, anywhere). Model stays loaded in the background for instant use.

## Setup & commands

- Python **==3.11.14** (strict pin), managed with **uv**.
- Run the tool: `uv run faster-whisper-hotkey` (flags: `--debug`, `--headless`, `--config <path>`).
- Fast unit suite: `uv run pytest tests/ --ignore=tests/test_model_all_configs.py`
- Slow integration suite (real downloads, full device×precision matrix: 14 CPU + 32 CUDA configs): `uv run pytest tests/test_model_all_configs.py -v --tb=long`. CUDA configs auto-skip without a GPU (`--force-cuda` to override); `--cuda-only` skips the CPU configs (for local GPU runs); `--cpu-only` skips the CUDA configs (for local CPU runs); whisper `cuda/int8` is skipped on Blackwell.
- CI split: the CPU matrix (14 configs) runs as a manual-dispatch GitHub Action (`.github/workflows/cpu-inference-matrix.yml` — "Run workflow" button on the Actions tab) as 8 parallel per-model jobs (one `ubuntu-latest` runner each, `fail-fast: false`, so a recycled runner only loses that model's ~5 min). Requires the `HF_TOKEN` secret on the `actions` environment (verified up front: whoami + gated-model check; the gated cohere model is pre-downloaded in a shell step). Tested HF models are deleted between test classes (14 GB runner disk) with per-class disk usage logged; output streams live (`-s`) and each job publishes its transcription to its own step summary. The repo's GitHub policy only allows actions owned by `blakkd`, so the workflow uses no external actions at all (plain shell: `git clone` checkout, official uv installer); the side effect is that the ~30 GB of model downloads happen on every run. The CUDA matrix (32 configs) is intended to run locally on the GPU machine with `--cuda-only`; GitHub's hosted GPU runners are not Blackwell, so they can't reproduce the sm_12x-specific skips.
- **When real inference needs testing, offer the parallel split**: propose running the CUDA matrix locally on the GPU machine and the CPU matrix as the GitHub Action **at the same time** (`uv run pytest tests/test_model_all_configs.py --cuda-only` locally, dispatch `cpu-inference-matrix.yml` for CPU). Disjoint configs, so running them in parallel covers the full 46 in less wall-clock time than running them one after another. Run only one half if the user cares about just that half.
- Lint `uv run ruff check .`, type-check `uv run pyright` (covers all of `src/`, including models.py).
- **flash-attn is not required by any model** — granite-nar uses SDPA (which dispatches to PyTorch's flash kernel on CUDA). It is not in `pyproject.toml`.

## Architecture

```
__main__.py        CLI: --debug / --headless / --config
  └─ transcribe.py   logging setup, curses TUI (or headless load), then MicrophoneTranscriber
       ├─ ui.py        curses config TUI (state machine)
       ├─ settings.py  Settings dataclass + JSON persistence
       └─ transcriber.py  the long-running engine
               ├─ models.py       ModelWrapper: load + run 9 model families
               │    └─ photon_no_telemetry.py  no-op Photon telemetry reporter (no official opt-out)
               ├─ hf_offline.py   HF hub reachability probe → offline mode
              ├─ capitalization.py  sentence-case post-processing for granite-nar
              ├─ llm_corrector.py optional LLM post-correction
              ├─ clipboard.py    pyperclip backup / set / restore
              ├─ paste.py        Ctrl+V / Ctrl+Shift+V (pynput or wtype)
              └─ terminal.py     focused-window detection (X11/Wayland)
```

- **Flow**: hotkey press → `sd.InputStream` (16 kHz mono) into a 10-min ring buffer; release → skip if <1 s or no speech (Silero VAD, if enabled) → a daemon thread runs `ModelWrapper.transcribe()` → optional LLM correction → clipboard set + paste (fallback: char-by-char typing).
- **TUI** (ui.py): `ConfigStep` state machine; per-model Device → Precision → Language; ESC returns to the initial screen. The model's native precision is tagged ` (native)` for display only. Text inputs disambiguate a bare ESC (cancel) from Alt+key via a short follow-up window (Alt+Backspace deletes a word).
- **Models** (models.py): `_load_model()` dispatches on `model_type` (whisper, parakeet, canary, voxtral, cohere, granite-nar, granite, granite-turboctc, qwen3-asr); `transcribe()` has a per-model inference path. Per-model device/precision/language options live in the TUI — the non-obvious bits are below.

## Gotchas & design notes

Not obvious from the code:

- **whisper CUDA int8 is broken on Blackwell (sm_120/121)** — CTranslate2 int8 GEMM fails (`CUBLAS_STATUS_NOT_SUPPORTED`). `_cuda_int8_supported()` detects it; the TUI blocks the pick and `ModelWrapper` raises (covers stale settings / `--headless`).
- **parakeet/canary int8/int4 are a no-op** (known bug): the chosen quantization is never applied — `from_pretrained` is called without a `quantization_config`, and the `.to(dtype)` step is skipped for int8/int4 — quantization is silently ignored.
- **canary's bundled timestamps submodel**: NeMo `restore_from` defaults a missing `map_location` to CUDA, so a CPU selection would still load that submodel on the GPU (and OOM). `_nemo_restore_on(device)` forces the requested device; the submodel is required, so it can't be disabled.
- **granite-turboctc bnb int8 needs two guards**: (1) `_patch_bnb_int8_noncontiguous()` — bnb's 8-bit GEMM reads strided (non-contiguous) inputs as row-major → garbage (all-pad output); the patch contiguifies inputs. (2) `llm_int8_skip_modules` keeps `input_linear`/`ctc_head`/`encoder.out` in float (a quantized `input_linear` crashes the bias cast).
- **CPU transformers loads** use `low_cpu_mem_usage=False` + `_materialize_weights()` to break safetensors mmap (mmap'd weights read from disk during inference = severe slowdown). fp32 is often faster than bf16 on CPUs without native bf16 support.
- **NeMo** (parakeet/canary): `suppress_nemo()` silences OneLogger at load; canary needs a patched `SentencePieceTokenizer.eos_id` (its EOS token isn't flagged by SentencePiece). CUDA uses CUDA-graphs decoding (driver ≥ 12.6).
- **The canary `ERROR:hydra.utils: ...get_nemo_transformer...` line at load is cosmetic**: the shipped `model_config.yaml` targets a factory *function* for `transf_decoder`; NeMo 3.0's target allow-list probe (class-only `get_class`) logs the failure, then falls back to `get_object` and instantiation succeeds. Only visible with `--debug`.
- **transformers version floors**: granite ≥5.17, granite-nar ≥5.5.3, qwen3-asr ≥5.13, granite-turboctc ≥5.16 (checked in `_check_transformers_version`; pinned at 5.17.0). Only granite-nar needs `trust_remote_code`.
- **CUDA warmup**: one silent 2 s transcription at load pays the one-time CUDA costs so the first real recording isn't slow.
- **HF offline fallback** (hf_offline.py): probes the hub once per load; if unreachable, sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` so all loaders use the local cache instead of hanging. `tests/conftest.py` stubs the probe.
- **Test audio is native 16 kHz** (`test_audio_data/test_16k.wav`, mono) because production records 16 kHz with no resample — and cohere-transcribe is hypersensitive to the 44.1k→16k resampler: the old 44.1 kHz mp3 fixture came out spacing-less on the first ~20 s (soxr at any mid quality; scipy `resample_poly` and soxr LQ/QQ survived). The fixture only resamples if a non-16 kHz file is used, so keep the fixture native.
- **canary/voxtral** write a temp .wav (file-based APIs). **granite-nar** is all-lowercase → `capitalization.py` restores sentence case (English `i` → `I`); **granite-turboctc** emits no punctuation and is intentionally not post-processed.

### LLM correction

Optional OpenAI-compatible endpoint. The settings API key may be a literal or an `env:VAR` reference — the reference is stored verbatim (the secret never lands on disk) and resolved at startup (unset var → warn + send unauthenticated). Falls back to the original text on any error.

### Platform input

- Terminal detection **never uses window titles** (a VSCode window titled `term-calc` must not be misdetected): X11 matches WM_CLASS, Wayland matches the container `app_id`; XWayland containers carry the X11 window ID in their `window` field → resolved to a real WM_CLASS via xprop.
- Two-tier matching: `TERMINAL_IDENTIFIERS` (substrings) + `TERMINAL_EXACT_IDENTIFIERS` (`st`, `foot`, `tabby`, `hyper`, `rio` as whole words). Paste is `Ctrl+V`, or `Ctrl+Shift+V` in terminals (e.g. Ghostty's default binding).
- Pre-TUI keystrokes are captured and replayed (transcribe.py `_read_pending_input`/`_parse_key_sequence`/`_ReplayWindow`): keys pressed during the heavy startup imports are buffered by the shell's cooked-mode tty; without the replay a stale ESC misreads as "cancel" and a stale Enter picks "Use Last Settings".
- Silero VAD loads at `transcriber.py` import (before the TUI); `on_press` has a 0.1 s guard after the previous transcription ended. Sub-windows aren't visible (VSCode terminal unsupported); Windows not supported.
