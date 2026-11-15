# onboarding.md

Context document for collaborators and agents working in this repo.

## What this is

**faster-whisper-hotkey** — minimalist push-to-talk STT for Linux. Hold a hotkey (Pause/F4/F8/Insert), speak, release → transcribed text is pasted into the currently focused text field (terminal, editor, chat, anywhere). Model stays loaded in the background for instant use.

## Setup & commands

- Python **==3.11.14** (strict pin), managed with **uv**.
- Run the tool: `uv run faster-whisper-hotkey` (flags: `--debug`, `--headless`, `--config <path>`).
- Fast unit suite: `uv run pytest tests/ --ignore=tests/test_model_all_configs.py`
- Engine smoke test: `uv run pytest tests/test_model_all_configs.py -v --tb=long` — ONE tiny model (granite-turboctc Q4_K_M, 280 MB) through the full app stack on every backend (cpu/cuda/vulkan; GPU configs auto-skip where the backend is absent). Everything above the engine boundary is our code and unit-tested; per-model engine behavior is transcribe.cpp's job. Metrics flags remain backend-generic: `--runs N` (default 1; best/median/mean + speed factor + VRAM snapshot on GPU rows), lanes `--cpu-only`/`--cuda-only`/`--vulkan-only`/`--gpu-only`, `--force-cuda`/`--force-vulkan`, `--configs dev:quant` pair filter, `--skip-download`, `--results-file PATH` (custom paths append). A GPU row must run on the requested backend (strict, no substitution) and empty transcriptions fail loudly. The retired 180-config benchmark matrix lives in git history at `3e4a025`; its permanent results: README tables + `test_audio_data/transcription_results.md`.
- Benchmarks: `uv run python dev/bench_models.py parakeet-ultra canary --device cpu --quant int8` (loads once, times 3 runs on the 40 s fixture).
- **Known engine bug (transcribe.cpp, qwen3_asr)**: device memory is never returned on model free — VRAM climbs monotonically across models in one process until the card OOMs; process exit can abort with `double free or corruption`. Repro: two qwen3 GPU loads in one process; confirmed clean when each load runs in its own process. Suspects: the aborted-decode path (qwen3 hits `OutputRepetition` on the 40 s fixture) and/or gate+up aliasing on GPU. The app is unaffected (one model per process). Upstream issue pending.
- CI: manual-dispatch GitHub Action (`.github/workflows/engine-smoke.yml` — "Run workflow" on the Actions tab), one job running the smoke test (GPU lanes auto-skip: runners have neither GPU nor Vulkan driver). No secrets needed: every GGUF is a public `handy-computer` HF repo. The repo's GitHub policy only allows actions owned by `blakkd`, so the workflow is plain shell (git clone checkout + official uv installer). Full GPU verification happens locally on the GPU box with `--gpu-only` when VRAM is free.
- Lint `uv run ruff check .`, type-check `uv run pyright` (covers all of `src/`, including models.py).

## Architecture

```
__main__.py        CLI: --debug / --headless / --config
  └─ transcribe.py   logging setup, curses TUI (or headless load), then MicrophoneTranscriber
       ├─ ui.py        curses config TUI (state machine)
       ├─ settings.py  Settings dataclass + JSON persistence (+ model_name migration)
       └─ transcriber.py  the long-running engine (mic, ring buffer, VAD gate, paste)
             ├─ models.py       ModelWrapper: load + run all families via transcribe.cpp
             │    ├─ gguf_models.py    static GGUF catalog (variant → HF repo, files, quants)
             │    ├─ downloader.py     single-file GGUF download + cache
             │    └─ (transcribe-cpp)  pure-Python ctypes binding to the C library
             ├─ capitalization.py  sentence-case post-processing for granite-nar
             ├─ llm_corrector.py optional LLM post-correction
             ├─ clipboard.py    pyperclip backup / set / restore
             ├─ paste.py        Ctrl+V / Ctrl+Shift+V (pynput or wtype)
             └─ terminal.py     focused-window detection (X11/Wayland)
```

- **One engine**: everything is a GGUF checkpoint run through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) via its `transcribe-cpp` Python binding (ctypes). Backends: `cpu`, `vulkan`, `cuda` — the PyPI `transcribe-cpp-native` wheel ships CPU+Vulkan; **CUDA is not on PyPI yet** (the cu12 provider wheel is a placeholder).
- **Enabling CUDA locally** (verified on this box, CUDA 13.2 toolkit + RTX PRO 4500): the clean one-shot build `uv pip install --no-binary transcribe-cpp-native ...` fails unless `glslc` is installed (needed for the Vulkan shaders; `apt install glslang-tools` or the Vulkan SDK — no sudo here, and the glslang GitHub releases do NOT ship glslc). The working recipe: build the sdist of the **same version as the installed wheel** with CUDA only, then graft the module onto the official CPU+Vulkan provider:

  ```sh
  pip download transcribe-cpp-native==<installed-ver> --no-binary :all: --no-deps -d tc && cd tc && tar xzf *.tar.gz && cd transcribe_cpp_native-*/
  CMAKE_ARGS="-DTRANSCRIBE_CUDA=ON -DTRANSCRIBE_GGML_BACKEND_DL=ON -DTRANSCRIBE_VULKAN=OFF \
    -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=native" uv build --wheel --out-dir dist
  unzip -j dist/*.whl "*/_native/libggml-cuda.so"
  cp libggml-cuda.so <venv>/lib/python3.11/site-packages/transcribe_cpp_native/_native/
  # then add "cuda" to "backends" in that _native/contract.json (provider metadata only;
  # ggml itself just dlopens every libggml-*.so in that directory)
  ```

  `uv sync`/any native reinstall reverts to the CPU+Vulkan wheel — with a saved `cuda` device the app then refuses to start (clear error, see strict-backend note below): redo the graft or re-pick the device in the settings screen. `transcribe_cpp.backend_available("cuda")` should return True; Blackwell compiles as `sm_120a` automatically.
- **Flow**: hotkey press → `sd.InputStream` (16 kHz mono) into a 10-min ring buffer; release → skip if <1 s or no speech (Silero VAD v6.2 GGUF via the engine's VAD role, if enabled) → a daemon thread runs `ModelWrapper.transcribe()` → optional LLM correction → clipboard set + paste (fallback: char-by-char typing).
- **TUI** (ui.py): `ConfigStep` state machine; per-model Device → Precision → Language; ESC returns to the initial screen. Device is `cuda|vulkan|cpu` for every model; all precisions work on every device (GGUF files exist per quant). ESC handling and keystroke replay as before.
- **Models** (models.py): one load path (`gguf_models.repo_and_filename()` → `downloader.ensure_gguf()` → `transcribe_cpp.Model`), plus one run path where `_run_kwargs()` maps the settings language to `task/language/target_language`. Support models on the same engine: Silero VAD (speech gate) and VoxLingua107 lang-id (cohere's `auto`).
- **Settings migration** (settings.py): pre-migration files stored upstream HF repo ids (`nvidia/canary-1b-v2`, `moondream/parakeet-ultra`, distil whispers…); `MODEL_NAME_MIGRATIONS` rewrites them to GGUF variant ids on load, so old configs keep booting.
- **Model files**: `~/.cache/faster_whisper_hotkey/models/<file>.gguf`, downloaded on first use (public `handy-computer` repos, honoring `HF_ENDPOINT`), `.part` + rename so torn downloads never load. A cached file works fully offline.

## Gotchas & design notes

Not obvious from the code:

- **GGUF "hard caps"**: canary and cohere reject input above ~400 s (5000-frame encoder positional table baked into the checkpoint), the granite AR family has a 4096-token output budget, whisper/parakeet/voxtral/qwen3 are effectively unbounded. `capabilities.max_audio_ms` is checked before the run; the engine error (`InputTooLong`) and `OutputTruncated` (partial text kept) are both surfaced as log lines, never a crash.
- **cohere `auto` language**: the model itself has no LID — the TUI offers `auto`, which runs the small VoxLingua107 lang-id GGUF (allowed = cohere's 14 languages) and falls back to `en` if detection is unusable/too short.
- **whisper `large-v1`** is published upstream as plain `whisper-large` — the catalog carries an explicit stem override; don't "fix" the filename.
- **Quant mapping fallbacks** (gguf_models.py): the NeMo ports (parakeet/canary/whisper) don't publish F32 (whisper) or BF16 files — picking `bfloat16` there silently maps to F16, `float32` on whisper maps to F16. This is intended; the menu keeps one uniform list.
- **granite-nar** is all-lowercase → `capitalization.py` restores sentence case (English `i` → `I`); **granite-turboctc** emits no punctuation and is intentionally not post-processed. Both are properties of the checkpoint, identical to the old HF path.
- **canary translate pairs**: the TUI stores `src-tgt`; same-same is transcription, different triggers `task="translate"`. granite works the same way (engine-validated pair list).
- **Strict backend, no silent substitution** (`resolve_backend` in models.py): the device you pick is the device you get. A saved device the installed provider can't serve (e.g. `"cuda"` after reverting to the CPU+Vulkan wheel) raises with a clear remedy, and `main()` turns it into one clean error line + exit — the app never quietly runs somewhere else. The TUI device menu only lists available backends (`ui._available_devices`), so a fresh pick is always serviceable; only a stale settings file can trigger this, and the fix is to re-pick in the settings screen. Likewise a model load that fails with `OutOfMemory` (GPU full of another app's work — e.g. the LLM-corrector server — or RAM exhausted) raises with an actionable message (free VRAM / smaller quant / switch device) and stops; it never falls back to another backend.
- **Warmup (every device)**: at model load, one pass of a bundled 2 s audio clip (`assets/never_gonna_give_you_up_2s.wav` — yes, the first two seconds of Never Gonna Give You Up) through the production transcription path. GPU pays kernel-module/context init, CPU pages in the mmap'd weights (~0.1–0.25 s first-run penalty measured on 40 s audio). It's REAL audio on purpose: silence would warm the encoder but barely feed the decoder. The clip's first 2 s (instrumental intro) carry no speech; the transcription result is ALWAYS debug-logged (`--debug`): `Warmup heard: '<hallucination>'` for loose models (whisper: expect "Thanks for watching!"), `heard nothing` for clean ones (parakeet/qwen3 CTC-LLM models return empty). End users see only the neutral `Warmup finished in X s` INFO line. A warmup failure logs a warning and never blocks startup.
- **VAD model loads lazily** on the first speech gate (not at import like the old torch version) and falls open (`no speech gating`) if the GGUF can't be loaded.
- **Test audio is native 16 kHz** (`test_audio_data/test_16k.wav`, mono) because production records 16 kHz with no resample — and cohere-transcribe is hypersensitive to the 44.1k→16k resampler: the old 44.1 kHz mp3 fixture came out spacing-less on the first ~20 s (soxr at any mid quality; scipy `resample_poly` and soxr LQ/QQ survived). Keep the fixture native.

### LLM correction

Optional OpenAI-compatible endpoint. The settings API key may be a literal or an `env:VAR` reference — the reference is stored verbatim (the secret never lands on disk) and resolved at startup (unset var → warn + send unauthenticated). Falls back to the original text on any error.

### Platform input

- Terminal detection **never uses window titles** (a VSCode window titled `term-calc` must not be misdetected): X11 matches WM_CLASS, Wayland matches the container `app_id`; XWayland containers carry the X11 window ID in their `window` field → resolved to a real WM_CLASS via xprop.
- Two-tier matching: `TERMINAL_IDENTIFIERS` (substrings) + `TERMINAL_EXACT_IDENTIFIERS` (`st`, `foot`, `tabby`, `hyper`, `rio` as whole words). Paste is `Ctrl+V`, or `Ctrl+Shift+V` in terminals (e.g. Ghostty's default binding).
- Pre-TUI keystrokes are captured and replayed (transcribe.py `_read_pending_input`/`_parse_key_sequence`/`_ReplayWindow`): keys pressed during startup are buffered by the shell's cooked-mode tty; without the replay a stale ESC misreads as "cancel" and a stale Enter picks "Use Last Settings". Startup is now much lighter (no torch), but the replay stays.
- `on_press` has a 0.1 s guard after the previous transcription ended. Sub-windows aren't visible (VSCode terminal unsupported); Windows not supported.
