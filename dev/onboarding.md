# onboarding.md

Context document for collaborators and agents working in this repo.

## What this is

**faster-whisper-hotkey** — minimalist push-to-talk STT for Linux. Hold a hotkey (Pause/F4/F8/Insert), speak, release → transcribed text is pasted into the currently focused text field (terminal, editor, chat, anywhere). Model stays loaded in the background for instant use.

## Setup & commands

- Python **>=3.10** (`requires-python`; CI/dev environment on **3.11.14** via `.python-version`), managed with **uv**. Dependency floors, exact versions in `uv.lock`. The full suite is matrix-tested on CPython 3.10-3.15 (uv-provided): per version `UV_PROJECT_ENVIRONMENT=/tmp/venv-py$V uv sync --python $V` then that venv's `python -m pytest tests/`.
- Run the tool: `uv run faster-whisper-hotkey` (flags: `--debug`, `--headless`, `--config <path>`).
- Fast unit suite: `uv run pytest tests/ --ignore=tests/test_model_all_configs.py`
- Engine smoke test: `uv run pytest tests/test_model_all_configs.py -v --tb=long` — ONE tiny model (granite-turboctc Q4_K_M, ~280 MB, addressed by hub repo + GGUF file) through the full app stack on every backend (cpu/cuda/vulkan; GPU configs auto-skip where the backend is absent). Everything above the engine boundary is our code and unit-tested; per-model engine behavior is transcribe.cpp's job. Metrics flags are backend-generic: `--runs N`, lanes `--cpu-only`/`--cuda-only`/`--vulkan-only`/`--gpu-only`, `--force-cuda`/`--force-vulkan`, `--configs device:QUANT` filter, `--skip-download`, `--results-file PATH`. A GPU row must run on the requested backend (strict, no substitution) and empty transcriptions fail loudly.
- Benchmarks: `uv run python dev/bench_models.py parakeet-ultra canary-1b-v2 --device cpu --quant int8` (loads once, times 3 runs on the 40 s fixture; model names are catalog display names).
- **Known engine bug (transcribe.cpp, qwen3_asr)**: device memory is never returned on model free — VRAM climbs monotonically across models in one process until the card OOMs; process exit can abort with `double free or corruption`. Repro: two qwen3 GPU loads in one process; confirmed clean when each load runs in its own process. The app is unaffected (one model per process). Upstream issue pending.
- CI: manual-dispatch GitHub Action (`.github/workflows/engine-smoke.yml`), one job running the smoke test (GPU lanes auto-skip: runners have neither GPU nor Vulkan driver). No secrets needed: every GGUF is a public `handy-computer` HF repo. The workflow is plain shell (git clone + official uv installer) to satisfy the repo's actions-owner policy.
- Lint `uv run ruff check .`, type-check `uv run pyright` (covers all of `src/`).

## Architecture

```
__main__.py        CLI: --debug / --headless / --config
  └─ transcribe.py   logging setup, curses TUI (or headless load), then MicrophoneTranscriber
       ├─ ui.py         curses config TUI (state machine): model -> precision -> device -> LOAD -> language
       ├─ settings.py   Settings dataclass (repo+file+language+target) + JSON persistence + legacy migration
       └─ transcriber.py  the long-running engine (mic, ring buffer, VAD gate, paste); reuses the loaded model
             ├─ models.py       ModelWrapper: load one GGUF by repo+file, transcribe/translate via capabilities
             │    ├─ catalog.py       live model list from the HF hub (+ disk cache + bundled offline snapshot)
             │    ├─ downloader.py    single-file GGUF download + cache
             │    └─ (transcribe-cpp)  pure-Python ctypes binding to the C library
             ├─ capitalization.py  sentence-case post-processing for the *-nar checkpoints
             ├─ llm_corrector.py optional LLM post-correction
             ├─ clipboard.py    pyperclip backup / set / restore
             ├─ paste.py        Ctrl+V / Ctrl+Shift+V (pynput or wtype)
             └─ terminal.py     focused-window detection (X11/Wayland)
```

- **One engine, no per-model tables**: everything is a GGUF checkpoint run through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) via `transcribe-cpp` (ctypes). Backends `cpu`, `vulkan`, `cuda`. **The app keeps no hardcoded model catalog and no language lists** (they were removed with the `gguf_models.py` / `available_languages.json` files): which models exist and which files/quants they publish come from the hub (`catalog.py`); what each model can do comes from the loaded GGUF's `capabilities`.
- **Model discovery is hub-driven, not engine-driven**: transcribe.cpp is a pure GGUF *runner* with no discovery API. `catalog.list_models()` queries the HF hub for `handy-computer/*-gguf` repos (one call), caches to `~/.cache/faster_whisper_hotkey/catalog.json` on a 3-day TTL, and falls back to the bundled snapshot `assets/model_catalog.json` (committed; refresh with `dev/refresh_catalog.py`) when offline. Support/diarization repos (silero-vad, lang-id, sortformer) are filtered out of the picker. `catalog.files_for(repo)` lists a repo's real GGUF files (with sizes) for the precision menu.
- **Capabilities, not families**: `ModelWrapper` reads `capabilities` from the loaded model: `languages` (transcription languages), `supports_language_detect` (whether `auto` is offered), `supports_translate` + `translate_target_languages` (translation), `max_audio_ms` (input cap). `_run_kwargs(language, target)` maps the chosen language/target to `{task, language, target_language}` — translation is simply a non-empty target; there is no pair table. There is no per-model branching in the run path.
- **Flow**: hotkey press → `sd.InputStream` (16 kHz mono) into a 10-min ring buffer; release → skip if <1 s or no speech (Silero VAD v6.2 GGUF, if enabled) → a daemon thread runs `ModelWrapper.transcribe()` → optional LLM correction → clipboard set + paste (fallback: char-by-char typing).
- **TUI (ui.py)**: `ConfigStep` state machine. `INITIAL → AUDIO_DEVICE → VAD_ENABLE → MODEL (hub list, type-to-fuzzy-search) → QUANT (real files + sizes) → COMPUTE_DEVICE → LOAD → LANGUAGE (+ optional TRANSLATE_SOURCE/TARGET) → HOTKEY → LLM_*`. **The model is downloaded+loaded at the LOAD step** (a background thread with a live status screen) because the language menu needs its `capabilities`; the loaded `ModelWrapper` is then returned and **reused by the transcriber** (no second load). ESC returns to `INITIAL` from any screen (in a searchable menu, ESC first clears the filter). There is no manual "refresh model list" entry — the hub listing refreshes itself on the cache TTL.
- **Settings (settings.py)**: `device_name, model_repo, model_file, device, language, translate_target, hotkey, vad_enabled, llm_*`. A model is addressed by hub `model_repo` + `model_file` (the exact GGUF, quant included); `translate_target` empty means transcribe. Files written before this migration addressed models by `(model_type, model_name, compute_type)`; `_migrate_legacy` rewrites those to `(model_repo, model_file)` (and splits a legacy `"src-tgt"` language into `language`/`translate_target`) so old configs keep booting — the legacy table is migration-only and the runtime never reads it.

## Gotchas & design notes

- **Strict backend, no silent substitution** (`resolve_backend`): the device you pick is the device you get. A saved device the provider can't serve raises with a clear remedy; a load that hits `OutOfMemory` raises with an actionable message. The app never quietly runs elsewhere. The TUI device menu only lists available backends (`_available_devices`).
- **CUDA is not on PyPI** — the `transcribe-cpp-native` wheel ships CPU+Vulkan. CUDA is a source-built provider (recipe in git history / project notes); `transcribe_cpp.backend_available("cuda")` must return True for it to be offered.
- **Warmup (every device, language-independent)**: at load, one pass of the bundled 2 s clip (`assets/never_gonna_give_you_up_2s.wav`) through the production path, with `language=None`. It's real audio on purpose; the result is debug-logged (`--debug`); a failure warns but never blocks startup.
- **VAD loads lazily** on the first speech gate, on the main model's backend, cached per backend (`models._load_support_model`), using the `catalog.VAD_REPO`/`VAD_FILE` constants; falls open if the GGUF can't load. (The old VoxLingua107 lang-id support model was dropped: models without language detection simply aren't offered `auto`.)
- **Output quirks the engine doesn't expose** — keep an eye on: `*-nar` checkpoints (granite non-autoregressive) emit punctuation but lowercase, so `_needs_sentence_case(filename)` (a `nar` filename token) triggers `capitalization.py` sentence-casing; `*-turboctc` emit neither punctuation nor case and are intentionally not post-processed. Length caps (`max_audio_ms`) are read from capabilities and checked before the run; `InputTooLong`/`OutputTruncated` (partial kept) are surfaced as log lines, never crashes.
- **GGUF metadata is post-load only**: `capabilities`, `variant`, `arch` are readable only after `transcribe_cpp.Model(path)`. The model *list* therefore comes from the hub, the per-model *capabilities* from the loaded file.
- **Test audio is native 16 kHz** (`test_audio_data/test_16k.wav`) — production records 16 kHz with no resample, and cohere is hypersensitive to resampling.

### LLM correction

Optional OpenAI-compatible endpoint. The settings API key may be a literal or an `env:VAR` reference — the reference is stored verbatim (the secret never lands on disk) and resolved at startup. Falls back to the original text on any error.

### Platform input

- Terminal detection **never uses window titles**: X11 matches WM_CLASS, Wayland matches the container `app_id`; XWayland containers resolve their real WM_CLASS via xprop.
- Two-tier matching: `TERMINAL_IDENTIFIERS` (substrings) + `TERMINAL_EXACT_IDENTIFIERS` (whole words). Paste is `Ctrl+V`, or `Ctrl+Shift+V` in terminals.
- Pre-TUI keystrokes are captured and replayed (transcribe.py `_read_pending_input`/`_parse_key_sequence`/`_ReplayWindow`).
- `on_press` has a 0.1 s guard after the previous transcription ended. Sub-windows aren't visible (VSCode terminal unsupported); Windows not supported.
