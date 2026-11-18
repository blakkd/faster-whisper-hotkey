"""
Engine smoke test: one tiny model through the full app stack, every backend.

WHY ONE MODEL: everything runs on transcribe.cpp now — checkpoint loading,
quantization, decode and backends are the engine's job, exercised upstream by
their own CI. This test verifies OUR remaining stack end-to-end: GGUF
downloader, settings->engine plumbing, warmup, the production transcription
path, and that each ggml backend (cpu/cuda/vulkan) resolves and transcribes. If
granite-turboctc Q4_K_M (the smallest published checkpoint, ~280 MB) works
here, the engine works; per-model behavior is unit-tested plumbing only.

The model is addressed by hub repo + GGUF file (a stable catalog entry), the
same way the running app addresses every model after the hub-catalog migration.

Run: pytest tests/test_model_all_configs.py -v --tb=long

Backend lanes / flags still work (they're backend-generic): --cpu-only,
--cuda-only, --vulkan-only, --gpu-only, --force-cuda/--force-vulkan,
--configs cpu:Q4_K_M, --runs N, --skip-download, --results-file PATH.

"cuda" means the ggml CUDA backend (source-built provider; the PyPI wheel
ships CPU+Vulkan). CI runners have no GPU, so the GPU lanes auto-skip there.
"""

import gc
import statistics
import time
from typing import cast

import numpy as np
import pytest
import soundfile as sf
import transcribe_cpp

from faster_whisper_hotkey.downloader import ensure_gguf, model_path
from faster_whisper_hotkey.models import ModelWrapper
from faster_whisper_hotkey.settings import Settings

# ---------------------------------------------------------------------------
# Audio fixture
# ---------------------------------------------------------------------------

# 16kHz mono to match production (mic records native 16kHz, no resample).
AUDIO_PATH = "test_audio_data/test_16k.wav"
RESULTS_FILE = "test_audio_data/transcription_results.txt"
TARGET_SR = 16000


@pytest.fixture(scope="module")
def audio():
    """Load the native-16kHz test audio once for the whole module."""
    data, sr = sf.read(AUDIO_PATH, dtype="float32", always_2d=True)
    if sr != TARGET_SR:
        raise RuntimeError(f"{AUDIO_PATH} must be native {TARGET_SR} Hz (got {sr})")
    return data[:, 0].astype(np.float32), sr


@pytest.fixture(scope="module")
def results_file(request):
    """The results path; each test appends its block.

    The default path is truncated once per run. A custom --results-file is
    never truncated, so per-config re-runs (one pytest process each, e.g. to
    isolate a leaking model) accumulate blocks in a single file.
    """
    path = request.config.getoption("--results-file")
    if path == RESULTS_FILE:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"faster-whisper-hotkey transcription results — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"audio: {AUDIO_PATH}\n")
            f.write("=" * 80 + "\n")
            f.write("\n")
    return path


# ---------------------------------------------------------------------------
# Smoke config: the smallest published checkpoint, on every backend.
# (Language "auto" — granite-turboctc is English-only and ignores the hint.)
# ---------------------------------------------------------------------------

SMOKE_REPO = "handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf"
SMOKE_FILE = "granite-speech-5.0-470m-turboctc-nc-Q4_K_M.gguf"
SMOKE_QUANT = "Q4_K_M"
SMOKE_CONFIGS = [(SMOKE_REPO, SMOKE_FILE, dev, SMOKE_QUANT) for dev in ("cpu", "cuda", "vulkan")]


def _backend_ok(device: str) -> bool:
    if device == "cpu":
        return True
    from transcribe_cpp import Backend  # typing-only import

    return transcribe_cpp.backend_available(cast("Backend", device))


def _prefetch(configs, request) -> None:
    """Download every config's GGUF BEFORE any timer runs.

    ModelWrapper's first act is ensure_gguf(), so a cold cache would fold
    multi-GB network fetches into the load+warm metric. Prefetching warms the
    cache, making that call a stat(); load time then means load time. With
    --skip-download nothing is fetched: a missing file makes the timed config
    fail loudly instead of producing download-poisoned metrics.
    """
    for repo, filename in sorted({(repo, fn) for repo, fn, _dev, _quant in configs}):
        if request.config.getoption("--skip-download"):
            if not model_path(repo, filename).exists():
                raise RuntimeError(f"--skip-download: {filename} is not in the cache — prefetch it first")
            continue
        path = ensure_gguf(repo, filename)  # raises on network failure: fail fast, before timing
        print(f"[prefetch] {filename}: {path.stat().st_size / 1e9:.2f} GB ready", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_configs(configs, audio, request):
    """Run transcription for a list of configs. Returns (results, skipped, errors)."""
    force = {"cuda": request.config.getoption("--force-cuda"), "vulkan": request.config.getoption("--force-vulkan")}
    lanes = {lane: request.config.getoption(f"--{lane}-only") for lane in ("cpu", "cuda", "vulkan")}
    gpu_only = request.config.getoption("--gpu-only")
    runs = max(1, request.config.getoption("--runs"))
    wanted = {w.strip() for w in request.config.getoption("--configs").split(",") if w.strip()}

    _prefetch(configs, request)  # warm the cache OUTSIDE the timed region

    audio_data, sr = audio
    audio_sec = len(audio_data) / sr

    results = []
    skipped = []
    errors = []

    for repo, filename, device, quant in configs:
        label = f"{filename} ({device}/{quant})"
        if wanted and f"{device}:{quant}" not in wanted:
            continue
        allowed = (
            lanes["cpu"] and device == "cpu"
            or lanes["cuda"] and device == "cuda"
            or lanes["vulkan"] and device == "vulkan"
            or gpu_only and device != "cpu"
            or not (lanes["cpu"] or lanes["cuda"] or lanes["vulkan"] or gpu_only)
        )
        if not allowed:
            lanes_on = [lane for lane, on in lanes.items() if on] + (["gpu"] if gpu_only else [])
            skipped.append((repo, filename, device, quant, f"excluded by lane {','.join(lanes_on)}"))
            continue
        if device != "cpu" and not (_backend_ok(device) or force[device]):
            skipped.append((repo, filename, device, quant, f"{device} backend not available"))
            continue

        print(f"\n>>> Testing: {label}")
        try:
            settings = Settings(
                device_name="",
                model_repo=repo,
                model_file=filename,
                device=device,
                language="auto",
            )

            t0 = time.monotonic()
            wrapper = ModelWrapper(settings)
            load_time = round(time.monotonic() - t0, 2)

            # Invariant: strict backend resolution means the model runs on
            # exactly the requested device (a bad backend raises before this).
            if wrapper.backend != device:
                raise RuntimeError(f"requested {device!r}, but model resolved to {wrapper.backend!r}")

            # VRAM footprint from the engine's live snapshot, taken after
            # load+warmup so context buffers are allocated; weights + buffers
            # + runtime overhead. None for CPU (or runtimes without device info).
            vram_used_mib = None
            try:
                dev = wrapper.model.device
                if dev.device_type != "cpu" and dev.memory_total:
                    vram_used_mib = round((dev.memory_total - dev.memory_free) / (1024 * 1024))
            except Exception:  # noqa: BLE001, S110 — backend without resolved device info; vram stays unreported
                pass

            times = []
            texts = []
            for _ in range(runs):
                t1 = time.monotonic()
                text = wrapper.transcribe(audio_data, sample_rate=sr, language="auto")
                times.append(time.monotonic() - t1)
                texts.append(text.strip() if text else "")

            # Explicit native teardown (close() frees sessions first, then the
            # model — the C ordering contract) instead of leaning on gc order.
            wrapper.close()
            del wrapper
            gc.collect()

            # A model whose transcribe() swallowed an engine error returns ""
            # (the app degrades gracefully); here that must be an error row,
            # never a green one with a nonsense speed factor.
            if not any(texts):
                raise RuntimeError("empty transcription (engine likely errored mid-run)")

            results.append(
                (
                    repo,
                    filename,
                    device,
                    quant,
                    load_time,
                    round(min(times), 2),
                    round(statistics.median(times), 2),
                    round(statistics.fmean(times), 2),
                    round(audio_sec / min(times), 1),
                    vram_used_mib,
                    next(t for t in texts if t),  # first non-empty transcript for the quality check
                )
            )

        except Exception as exc:  # noqa: BLE001
            errors.append((repo, filename, device, quant, str(exc)))

    return results, skipped, errors


def _assert_ok(results, errors):
    """Fail if any config raised or returned an empty transcription."""
    assert errors == [], f"{len(errors)} config(s) failed:\n" + "\n".join(
        f"  - {e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
    )
    empty = [f"{r[1]} ({r[2]}/{r[3]})" for r in results if not r[10].strip()]
    assert not empty, "empty transcription for:\n" + "\n".join(f"  - {c}" for c in empty)


def _format_results(model_label, results, skipped, errors):
    """Format results for a single model test."""
    lines = [
        f"{model_label}",
        f"{'=' * 80}",
    ]

    for row in results:
        _repo, filename, device, quant, load_time, best, median, mean, speed, vram_mib, transcription = row
        lines.append(f"model: {filename}")
        lines.append(f"device: {device}  quant: {quant}")
        line = f"load+warm: {load_time}s  best: {best}s  median: {median}s  mean: {mean}s  speed: {speed}x realtime"
        if vram_mib is not None:
            line += f"  vram: {vram_mib} MiB"
        lines.append(line)
        lines.append("transcription:")
        for line in transcription.split("\n"):
            lines.append(f"  {line}")
        lines.append("")

    if skipped:
        lines.append("SKIPPED")
        for s in skipped:
            lines.append(f"  {s[1]} ({s[2]}/{s[3]}): {s[4]}")
        lines.append("")

    if errors:
        lines.append("ERRORS")
        for e in errors:
            lines.append(f"  {e[1]} ({e[2]}/{e[3]}): {e[4]}")
        lines.append("")

    lines.append(f"Summary: {len(results)} OK, {len(skipped)} skipped, {len(errors)} errors")
    lines.append("")

    return "\n".join(lines)


def _model_test(configs, model_label, audio, request, results_file):
    results, skipped, errors = _run_configs(configs, audio, request)

    block = _format_results(model_label, results, skipped, errors)
    print(f"\n{block}")
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(block + "\n")

    _assert_ok(results, errors)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestEngineSmoke:
    """The smoke model through every available backend, via the full app stack."""

    def test_engine_smoke(self, audio, request, results_file):
        _model_test(SMOKE_CONFIGS, "engine smoke — granite-turboctc Q4_K_M", audio, request, results_file)
