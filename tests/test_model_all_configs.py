"""
Integration test: transcribe a fixed audio sample with every supported model config.

Output: test_audio_data/transcription_results.txt, one block per config, with
metrics per line: load+warm time, then best/median/mean of --runs (default 1)
transcriptions, the real-time speed factor (audio seconds / best run) and, for
GPU rows, the VRAM footprint the engine reports after load+warmup
(weights + context buffers + runtime overhead).

Run: pytest tests/test_model_all_configs.py -v --tb=long

Download timing: each family's GGUFs are fetched by a prefetch pass BEFORE
the timed loop, so load+warm never includes network time (a cold-cache first
run downloads once, subsequent runs are cache hits). --skip-download refuses
to fetch: a cache miss then fails the config loudly rather than poisoning its
load metric.

Coverage: 10 families x every quant the repos publish (F32/BF16, F16, Q8_0,
Q6_K, Q5_K_M, Q4_K_M — "q6_k"/"q5_k_m" are off-menu settings labels) x three
backends (cpu, cuda, vulkan) = 180 configs.

Skips GPU configs automatically when the backend is unavailable. Force with:
    pytest tests/test_model_all_configs.py --force-cuda / --force-vulkan

Device lanes (default is everything available):
    pytest ... --cpu-only      # CI lane
    pytest ... --cuda-only     # CUDA configs only
    pytest ... --vulkan-only   # Vulkan configs only
    pytest ... --gpu-only      # every GPU backend (the full VRAM-free box run)

Run count: --runs N (default 1; 3+ to steady a noisy box, 1 keeps 180 configs feasible).

Sub-setting & re-runs: --configs cuda:float16,vulkan:int4 runs only those pairs
(one process per config isolates engines that leak device memory on free — see
dev/rerun_qwen3_gpu.sh); --results-file PATH redirects output, and a custom
path APPENDS instead of truncating, so many single-config runs stack up in one
file. An empty transcription (the app's swallow-engine-errors path) is a
config ERROR here, never a green row.

"cuda" here means the ggml CUDA backend (needs a source-built provider, the
PyPI wheel ships CPU+Vulkan); "vulkan" works with the default wheel. CI runners
have no GPU device at all, so all GPU configs auto-skip there.
"""

import gc
import statistics
import time
from typing import cast

import numpy as np
import pytest
import soundfile as sf
import transcribe_cpp

from faster_whisper_hotkey.downloader import ensure_gguf
from faster_whisper_hotkey.gguf_models import repo_and_filename
from faster_whisper_hotkey.models import ModelWrapper
from faster_whisper_hotkey.settings import Settings

# ---------------------------------------------------------------------------
# Audio fixture
# ---------------------------------------------------------------------------

# 16kHz mono to match production (mic records native 16kHz, no resample).
# cohere-transcribe is hypersensitive to the 44.1k->16k resampler, so the
# fixture uses a native-16kHz file to avoid resampling.
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
# Config matrix
# ---------------------------------------------------------------------------
# Language the config TUI saves for each model type (mirrors the ui.py screens):
# parakeet has no language step ("" = auto), voxtral is always "auto" (LID),
# canary/granite are src-tgt pairs, the rest are plain codes.
UI_LANGUAGE_BY_MODEL = {
    "whisper": "en",
    "parakeet": "",
    "parakeet-ultra": "",
    "canary": "en-en",
    "voxtral": "auto",
    "cohere": "en",
    "granite-nar": "en",
    "granite": "en-en",
    "granite-turboctc": "en",
    "qwen3-asr": "en",
}
# Each tuple: (model_type, model_name, device, compute_type)
# CPU always; CUDA and Vulkan run whenever their backend resolves (CI runners
# have neither device, so the GPU lanes auto-skip there — that is what the
# cpu-inference-matrix.yml jobs rely on).

GPU_DEVS = ("cpu", "cuda", "vulkan")
WHISPER_MODELS = ["small"]

# Every quant the HF repos publish, as a settings label. The NeMo/whisper
# families get F32 floats, the LLM-head families BF16 (float32 on those maps
# to BF16 — the identical file — so the LLM lane names it directly and every
# row below tests a distinct GGUF).
FULL_FLOAT_INT = ("float32", "float16", "int8", "q6_k", "q5_k_m", "int4")
FULL_BFLOAT_INT = ("bfloat16", "float16", "int8", "q6_k", "q5_k_m", "int4")


def _family(model_type: str, model_name: str, precs: tuple) -> list:
    """All (device, precision) configs for one model — every backend, every
    published quant. Quant files are backend-agnostic, so the same download
    serves all three lanes."""
    return [(model_type, model_name, dev, prec) for dev in GPU_DEVS for prec in precs]


WHISPER = [cfg for m in WHISPER_MODELS for cfg in _family("whisper", m, FULL_FLOAT_INT)]

PARAKEET = _family("parakeet", "parakeet-tdt-0.6b-v3", FULL_FLOAT_INT)

PARAKEET_ULTRA = _family("parakeet-ultra", "parakeet-ultra", FULL_FLOAT_INT)

CANARY = _family("canary", "canary-1b-v2", FULL_FLOAT_INT)

VOXTRAL = _family("voxtral", "voxtral-mini-3b-2507", FULL_BFLOAT_INT)

COHERE = _family("cohere", "cohere-transcribe-03-2026", FULL_BFLOAT_INT)

GRANITE_NAR = _family("granite-nar", "granite-speech-4.1-2b-nar", FULL_BFLOAT_INT)

GRANITE = _family("granite", "granite-speech-4.1-2b", FULL_BFLOAT_INT)

GRANITE_TURBOCTC = _family("granite-turboctc", "granite-speech-5.0-470m-turboctc-nc", FULL_BFLOAT_INT)

QWEN3_ASR = _family("qwen3-asr", "qwen3-asr-1.7b", FULL_BFLOAT_INT)


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
    unique = {(repo_and_filename(mt, nm, ct)[:2]) for mt, nm, _dev, ct in configs}
    for repo, filename in sorted(unique):
        if request.config.getoption("--skip-download"):
            from faster_whisper_hotkey.downloader import model_path

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

    for model_type, model_name, device, compute_type in configs:
        if wanted and f"{device}:{compute_type}" not in wanted:
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
            skipped.append((model_type, model_name, device, compute_type, f"excluded by lane {','.join(lanes_on)}"))
            continue
        if device != "cpu" and not (_backend_ok(device) or force[device]):
            skipped.append((model_type, model_name, device, compute_type, f"{device} backend not available"))
            continue

        print(f"\n>>> Testing: {model_type}/{model_name} ({device}/{compute_type})")
        try:
            settings = Settings(
                device_name="",
                model_type=model_type,
                model_name=model_name,
                compute_type=compute_type,
                device=device,
                language=UI_LANGUAGE_BY_MODEL[model_type],
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
                text = wrapper.transcribe(audio_data, sample_rate=sr, language=settings.language)
                times.append(time.monotonic() - t1)
                texts.append(text.strip() if text else "")

            # Explicit native teardown (close() frees sessions first, then the
            # model — the C ordering contract) instead of leaning on gc order.
            wrapper.model.close()
            del wrapper
            gc.collect()

            # A model whose transcribe() swallowed an engine error returns ""
            # (the app degrades gracefully); here that must be an error row,
            # never a green one with a nonsense speed factor.
            if not any(texts):
                raise RuntimeError("empty transcription (engine likely errored mid-run)")

            results.append(
                (
                    model_type,
                    model_name,
                    device,
                    compute_type,
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
            errors.append(
                (
                    model_type,
                    model_name,
                    device,
                    compute_type,
                    str(exc),
                )
            )

    return results, skipped, errors


def _assert_ok(results, errors):
    """Fail if any config raised or returned an empty transcription."""
    assert errors == [], f"{len(errors)} config(s) failed:\n" + "\n".join(
        f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
    )
    empty = [f"{r[0]}/{r[1]} ({r[2]}/{r[3]})" for r in results if not r[10].strip()]
    assert not empty, "empty transcription for:\n" + "\n".join(f"  - {c}" for c in empty)


def _format_results(model_label, results, skipped, errors):
    """Format results for a single model test."""
    lines = [
        f"{model_label}",
        f"{'=' * 80}",
    ]

    for row in results:
        model_type, model_name, device, compute_type, load_time, best, median, mean, speed, vram_mib, transcription = row
        lines.append(f"model: {model_type}/{model_name}")
        lines.append(f"device: {device}  compute_type: {compute_type}")
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
            lines.append(f"  {s[0]}/{s[1]} ({s[2]}/{s[3]}): {s[4]}")
        lines.append("")

    if errors:
        lines.append("ERRORS")
        for e in errors:
            lines.append(f"  {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}")
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
# Tests — one per model type
# ---------------------------------------------------------------------------


class TestTranscribeWhisper:
    """Transcribe test audio with whisper configs."""

    def test_whisper(self, audio, request, results_file):
        _model_test(WHISPER, "whisper", audio, request, results_file)


class TestTranscribeParakeet:
    """Transcribe test audio with parakeet configs."""

    def test_parakeet(self, audio, request, results_file):
        _model_test(PARAKEET, "parakeet", audio, request, results_file)


class TestTranscribeParakeetUltra:
    """Transcribe test audio with parakeet-ultra configs."""

    def test_parakeet_ultra(self, audio, request, results_file):
        _model_test(PARAKEET_ULTRA, "parakeet-ultra", audio, request, results_file)


class TestTranscribeCanary:
    """Transcribe test audio with canary configs."""

    def test_canary(self, audio, request, results_file):
        _model_test(CANARY, "canary", audio, request, results_file)


class TestTranscribeVoxtral:
    """Transcribe test audio with voxtral configs."""

    def test_voxtral(self, audio, request, results_file):
        _model_test(VOXTRAL, "voxtral", audio, request, results_file)


class TestTranscribeCohere:
    """Transcribe test audio with cohere configs."""

    def test_cohere(self, audio, request, results_file):
        _model_test(COHERE, "cohere", audio, request, results_file)


class TestTranscribeGraniteNAR:
    """Transcribe test audio with granite-nar configs."""

    def test_granite_nar(self, audio, request, results_file):
        _model_test(GRANITE_NAR, "granite-nar", audio, request, results_file)


class TestTranscribeGranite:
    """Transcribe test audio with granite (AR) configs."""

    def test_granite(self, audio, request, results_file):
        _model_test(GRANITE, "granite", audio, request, results_file)


class TestTranscribeGraniteTurboCTC:
    """Transcribe test audio with granite-turboctc configs."""

    def test_granite_turboctc(self, audio, request, results_file):
        _model_test(GRANITE_TURBOCTC, "granite-turboctc", audio, request, results_file)


class TestTranscribeQwen3ASR:
    """Transcribe test audio with qwen3-asr configs."""

    def test_qwen3_asr(self, audio, request, results_file):
        _model_test(QWEN3_ASR, "qwen3-asr", audio, request, results_file)
