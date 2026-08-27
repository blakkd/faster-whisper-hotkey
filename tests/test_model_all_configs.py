"""
Integration test: transcribe a fixed audio sample with every supported model config.

Output: test_audio_data/transcription_results.txt, one block per config.
Run: pytest tests/test_model_all_configs.py -v --tb=long

Skip CUDA configs automatically when no GPU is available. Override with:
    pytest tests/test_model_all_configs.py -v --tb=long --force-cuda
"""

import gc
import time

import numpy as np
import pytest
import torch

from faster_whisper_hotkey.models import ModelWrapper, _cuda_int8_supported
from faster_whisper_hotkey.settings import Settings

# ---------------------------------------------------------------------------
# Audio fixture
# ---------------------------------------------------------------------------

AUDIO_PATH = "test_audio_data/test.mp3"
RESULTS_FILE = "test_audio_data/transcription_results.txt"
TARGET_SR = 16000


@pytest.fixture(scope="module")
def audio():
    """Load and resample the test audio once for the whole module."""
    import librosa

    data, sr = librosa.load(AUDIO_PATH, sr=None, dtype="float32")
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return data.astype(np.float32), sr


@pytest.fixture(scope="module")
def results_file():
    """Truncate the shared results file once per run; each test appends its block."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"faster-whisper-hotkey transcription results — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"audio: {AUDIO_PATH}\n")
        f.write(f"cuda_available: {_cuda_available()}\n")
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
    "canary": "en-en",
    "voxtral": "auto",
    "cohere": "en",
    "granite-nar": "en",
    "granite": "en-en",
    "qwen3-asr": "en",
}
# Each tuple: (model_type, model_name, device, compute_type)

WHISPER_MODELS = ["small"]

# whisper: CPU (int8) + CUDA (float16, float32, int8)
# float16 unsupported on CPU (CTranslate2 hw requirement); int4 not supported
_configs = [(m, "cpu", prec) for m in WHISPER_MODELS for prec in ("int8",)] + [
    (m, "cuda", prec) for m in WHISPER_MODELS for prec in ("float16", "float32", "int8")
]

WHISPER = [("whisper", m, dev, prec) for m, dev, prec in _configs]

# parakeet: native=float32, CPU (f32/bf16) + CUDA (f32/bf16/int8/int4)
PARAKEET = [("parakeet", "nvidia/parakeet-tdt-0.6b-v3", "cpu", prec) for prec in ("float32", "bfloat16")] + [
    ("parakeet", "nvidia/parakeet-tdt-0.6b-v3", "cuda", prec) for prec in ("float32", "bfloat16", "int8", "int4")
]

# canary: native=float32, CPU (f32/bf16) + CUDA (f32/bf16/int8/int4)
CANARY = [("canary", "nvidia/canary-1b-v2", "cpu", prec) for prec in ("float32", "bfloat16")] + [
    ("canary", "nvidia/canary-1b-v2", "cuda", prec) for prec in ("float32", "bfloat16", "int8", "int4")
]

# voxtral: native=float32, CUDA (f32/bf16/int8/int4)
VOXTRAL = [
    ("voxtral", "mistralai/Voxtral-Mini-3B-2507", "cuda", prec) for prec in ("float32", "bfloat16", "int8", "int4")
]

# cohere: native=bfloat16, CPU (f32/bf16) + CUDA (bf16/f32/int8/int4)
COHERE = [("cohere", "CohereLabs/cohere-transcribe-03-2026", "cpu", prec) for prec in ("float32", "bfloat16")] + [
    ("cohere", "CohereLabs/cohere-transcribe-03-2026", "cuda", prec) for prec in ("bfloat16", "float32", "int8", "int4")
]

# granite-nar: native=bfloat16, CPU (f32/bf16) + CUDA (bf16/f32/int8/int4)
GRANITE_NAR = [
    ("granite-nar", "ibm-granite/granite-speech-4.1-2b-nar", "cpu", prec) for prec in ("float32", "bfloat16")
] + [
    ("granite-nar", "ibm-granite/granite-speech-4.1-2b-nar", "cuda", prec)
    for prec in ("bfloat16", "float32", "int8", "int4")
]

# granite (AR): native=bfloat16, CPU (f32/bf16) + CUDA (bf16/f32/int8/int4)
GRANITE = [("granite", "ibm-granite/granite-speech-4.1-2b", "cpu", prec) for prec in ("float32", "bfloat16")] + [
    ("granite", "ibm-granite/granite-speech-4.1-2b", "cuda", prec) for prec in ("bfloat16", "float32", "int8", "int4")
]

# qwen3-asr: weights natively bf16, CPU (bf16) + CUDA (bf16/int8/int4)
QWEN3_ASR = [("qwen3-asr", "Qwen/Qwen3-ASR-1.7B-hf", "cpu", "bfloat16")] + [
    ("qwen3-asr", "Qwen/Qwen3-ASR-1.7B-hf", "cuda", prec) for prec in ("bfloat16", "int8", "int4")
]


def _cuda_available():
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_configs(configs, audio, request):
    """Run transcription for a list of configs. Returns (results, skipped, errors)."""
    force_cuda = request.config.getoption("--force-cuda")
    cuda_ok = _cuda_available() or force_cuda

    audio_data, sr = audio

    results = []
    skipped = []
    errors = []

    for model_type, model_name, device, compute_type in configs:
        if device == "cuda" and not cuda_ok:
            skipped.append((model_type, model_name, device, compute_type, "CUDA not available"))
            continue
        if model_type == "whisper" and device == "cuda" and compute_type == "int8" and not _cuda_int8_supported():
            skipped.append(
                (model_type, model_name, device, compute_type, "int8 not supported on this CUDA GPU (Blackwell sm_12x)")
            )
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

            t1 = time.monotonic()
            text = wrapper.transcribe(audio_data, sample_rate=sr, language=settings.language)
            transcribe_time = round(time.monotonic() - t1, 2)

            del wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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


# ---------------------------------------------------------------------------
# Tests — one per model type
# ---------------------------------------------------------------------------


class TestTranscribeWhisper:
    """Transcribe test audio with faster-whisper configs."""

    def test_whisper(self, audio, request, results_file):
        results, skipped, errors, audio_data, sr = _run_configs(WHISPER, audio, request)

        block = _format_results("faster-whisper", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(block + "\n")

        _assert_ok(results, errors)


class TestTranscribeParakeet:
    """Transcribe test audio with parakeet configs."""

    def test_parakeet(self, audio, request, results_file):
        results, skipped, errors, audio_data, sr = _run_configs(PARAKEET, audio, request)

        block = _format_results("parakeet", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(block + "\n")

        _assert_ok(results, errors)


class TestTranscribeCanary:
    """Transcribe test audio with canary configs."""

    def test_canary(self, audio, request, results_file):
        results, skipped, errors, audio_data, sr = _run_configs(CANARY, audio, request)

        block = _format_results("canary", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(block + "\n")

        _assert_ok(results, errors)


class TestTranscribeVoxtral:
    """Transcribe test audio with voxtral configs."""

    def test_voxtral(self, audio, request, results_file):
        results, skipped, errors, audio_data, sr = _run_configs(VOXTRAL, audio, request)

        block = _format_results("voxtral", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(block + "\n")

        _assert_ok(results, errors)


class TestTranscribeCohere:
    """Transcribe test audio with cohere configs."""

    def test_cohere(self, audio, request, results_file):
        results, skipped, errors, audio_data, sr = _run_configs(COHERE, audio, request)

        block = _format_results("cohere", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(block + "\n")

        _assert_ok(results, errors)


class TestTranscribeGraniteNAR:
    """Transcribe test audio with granite-nar configs."""

    def test_granite_nar(self, audio, request, results_file):
        results, skipped, errors, audio_data, sr = _run_configs(GRANITE_NAR, audio, request)

        block = _format_results("granite-nar", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(block + "\n")

        _assert_ok(results, errors)


class TestTranscribeGranite:
    """Transcribe test audio with granite (AR) configs."""

    def test_granite(self, audio, request, results_file):
        results, skipped, errors, audio_data, sr = _run_configs(GRANITE, audio, request)

        block = _format_results("granite", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(block + "\n")

        _assert_ok(results, errors)


class TestTranscribeQwen3ASR:
    """Transcribe test audio with qwen3-asr configs."""

    def test_qwen3_asr(self, audio, request, results_file):
        results, skipped, errors, audio_data, sr = _run_configs(QWEN3_ASR, audio, request)

        block = _format_results("qwen3-asr", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(block + "\n")

        _assert_ok(results, errors)
