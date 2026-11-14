"""
Integration test: transcribe a fixed audio sample with every supported model config.

Output: test_audio_data/transcription_results.txt, one block per config.
Run: pytest tests/test_model_all_configs.py -v --tb=long

Skips GPU configs automatically when the backend is unavailable. Force with:
    pytest tests/test_model_all_configs.py --force-cuda / --force-vulkan

Device lanes (default is everything available):
    pytest ... --cpu-only      # CI lane
    pytest ... --cuda-only     # CUDA configs only
    pytest ... --vulkan-only   # Vulkan configs only
    pytest ... --gpu-only      # every GPU backend (the full VRAM-free box run)

"cuda" here means the ggml CUDA backend (needs a source-built provider, the
PyPI wheel ships CPU+Vulkan); "vulkan" works with the default wheel. CI runners
have no GPU device at all, so all GPU configs auto-skip there.
"""

import gc
import time

import numpy as np
import pytest
import soundfile as sf
import transcribe_cpp

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
def results_file():
    """Truncate the shared results file once per run; each test appends its block."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"faster-whisper-hotkey transcription results — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"audio: {AUDIO_PATH}\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
    return RESULTS_FILE


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

GPU_DEVS = ("cuda", "vulkan")
WHISPER_MODELS = ["small"]
GPU_PRECS = ("float16", "int8", "int4")
CPU_PRECS = ("int8", "int4")


def _family(model_type: str, model_name: str, cpu_precs: tuple, gpu_precs: tuple) -> list:
    """All (device, precision) configs for one model.

    GGUF quantizations are backend-agnostic, but CPU stays on the two quants
    that matter for CI cost while GPU lanes additionally cover the float path.
    """
    return [(model_type, model_name, "cpu", prec) for prec in cpu_precs] + [
        (model_type, model_name, dev, prec) for dev in GPU_DEVS for prec in gpu_precs
    ]


FLOAT32_INT8 = ("float32", "int8")
BFLOAT16_INT = ("bfloat16", "int8", "int4")

WHISPER = [cfg for m in WHISPER_MODELS for cfg in _family("whisper", m, CPU_PRECS, GPU_PRECS)]

PARAKEET = _family("parakeet", "parakeet-tdt-0.6b-v3", FLOAT32_INT8, FLOAT32_INT8)

PARAKEET_ULTRA = _family("parakeet-ultra", "parakeet-ultra", ("int8", "int4"), ("int8", "int4"))

CANARY = _family("canary", "canary-1b-v2", ("float32", "int8", "int4"), ("float32", "int8", "int4"))

VOXTRAL = _family("voxtral", "voxtral-mini-3b-2507", BFLOAT16_INT, BFLOAT16_INT)

COHERE = _family("cohere", "cohere-transcribe-03-2026", BFLOAT16_INT, BFLOAT16_INT)

GRANITE_NAR = _family("granite-nar", "granite-speech-4.1-2b-nar", BFLOAT16_INT, BFLOAT16_INT)

GRANITE = _family("granite", "granite-speech-4.1-2b", BFLOAT16_INT, BFLOAT16_INT)

GRANITE_TURBOCTC = _family("granite-turboctc", "granite-speech-5.0-470m-turboctc-nc", BFLOAT16_INT, BFLOAT16_INT)

QWEN3_ASR = _family("qwen3-asr", "qwen3-asr-1.7b", BFLOAT16_INT, BFLOAT16_INT)


def _backend_ok(device: str) -> bool:
    return device == "cpu" or transcribe_cpp.backend_available(device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_configs(configs, audio, request):
    """Run transcription for a list of configs. Returns (results, skipped, errors)."""
    force = {"cuda": request.config.getoption("--force-cuda"), "vulkan": request.config.getoption("--force-vulkan")}
    lanes = {lane: request.config.getoption(f"--{lane}-only") for lane in ("cpu", "cuda", "vulkan")}
    gpu_only = request.config.getoption("--gpu-only")

    audio_data, sr = audio

    results = []
    skipped = []
    errors = []

    for model_type, model_name, device, compute_type in configs:
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

            # ModelWrapper retries an OOM'd GPU load on CPU (production safety
            # net); in this matrix that must surface as a failure, never as a
            # green row labelled with a device the model never ran on.
            if wrapper.backend != device:
                raise RuntimeError(f"requested {device!r}, but model resolved to {wrapper.backend!r} (VRAM full?)")

            t1 = time.monotonic()
            text = wrapper.transcribe(audio_data, sample_rate=sr, language=settings.language)
            transcribe_time = round(time.monotonic() - t1, 2)

            del wrapper
            gc.collect()

            results.append(
                (
                    model_type,
                    model_name,
                    device,
                    compute_type,
                    load_time,
                    transcribe_time,
                    "OK",
                    text.strip() if text else "",
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

    return results, skipped, errors, audio_data, sr


def _assert_ok(results, errors):
    """Fail if any config raised or returned an empty transcription."""
    assert errors == [], f"{len(errors)} config(s) failed:\n" + "\n".join(
        f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
    )
    empty = [f"{r[0]}/{r[1]} ({r[2]}/{r[3]})" for r in results if not r[7].strip()]
    assert not empty, "empty transcription for:\n" + "\n".join(f"  - {c}" for c in empty)


def _format_results(model_label, results, skipped, errors, audio_data, sr):
    """Format results for a single model test."""
    lines = [
        f"{model_label}",
        f"{'=' * 80}",
    ]

    for row in results:
        model_type, model_name, device, compute_type, load_time, transcribe_time, status, transcription = row
        lines.append(f"model: {model_type}/{model_name}")
        lines.append(f"device: {device}  compute_type: {compute_type}")
        lines.append(f"load_time: {load_time}s  transcribe_time: {transcribe_time}s  status: {status}")
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
    results, skipped, errors, audio_data, sr = _run_configs(configs, audio, request)

    block = _format_results(model_label, results, skipped, errors, audio_data, sr)
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
