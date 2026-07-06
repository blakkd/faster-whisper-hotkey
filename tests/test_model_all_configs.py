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

from faster_whisper_hotkey.models import ModelWrapper
from faster_whisper_hotkey.settings import Settings

# ---------------------------------------------------------------------------
# Audio fixture
# ---------------------------------------------------------------------------

AUDIO_PATH = "test_audio_data/test.mp3"
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


# ---------------------------------------------------------------------------
# Config matrix  (language = "en" for every config)
# ---------------------------------------------------------------------------
# Each tuple: (model_type, model_name, device, compute_type)

WHISPER_MODELS = ["small"]

# whisper: CPU (int8) + CUDA (float16, float32, int8)
# float16 unsupported on CPU (CTranslate2 hw requirement); int4 not supported
_configs = [
    (m, "cpu", prec)
    for m in WHISPER_MODELS
    for prec in ("int8",)
] + [
    (m, "cuda", prec)
    for m in WHISPER_MODELS
    for prec in ("float16", "float32", "int8")
]

WHISPER = [
    ("whisper", m, dev, prec) for m, dev, prec in _configs
]

# parakeet: native=float32, CPU (f32/bf16) + CUDA (f32/bf16/f16/int8/int4)
PARAKEET = [
    ("parakeet", "nvidia/parakeet-tdt-0.6b-v3", "cpu", prec)
    for prec in ("float32", "bfloat16")
] + [
    ("parakeet", "nvidia/parakeet-tdt-0.6b-v3", "cuda", prec)
    for prec in ("float32", "bfloat16", "float16", "int8", "int4")
]

# canary: native=float32, CPU (f32/bf16) + CUDA (f32/bf16/f16/int8/int4)
CANARY = [
    ("canary", "nvidia/canary-1b-v2", "cpu", prec)
    for prec in ("float32", "bfloat16")
] + [
    ("canary", "nvidia/canary-1b-v2", "cuda", prec)
    for prec in ("float32", "bfloat16", "float16", "int8", "int4")
]

# voxtral: native=float32, CPU (no int4, hangs) + CUDA (all precisions)
VOXTRAL = [
    ("voxtral", "mistralai/Voxtral-Mini-3B-2507", "cpu", prec)
    for prec in ("float32", "float16", "bfloat16", "int8")
] + [
    ("voxtral", "mistralai/Voxtral-Mini-3B-2507", "cuda", prec)
    for prec in ("float32", "float16", "bfloat16", "int8", "int4")
]

# cohere: native=bfloat16, CPU (f32/bf16) + CUDA (bf16/f32/f16/int8/int4)
COHERE = [
    ("cohere", "CohereLabs/cohere-transcribe-03-2026", "cpu", prec)
    for prec in ("float32", "bfloat16")
] + [
    ("cohere", "CohereLabs/cohere-transcribe-03-2026", "cuda", prec)
    for prec in ("bfloat16", "float32", "float16", "int8", "int4")
]

# granite-nar: native=bfloat16, CPU (f32/bf16) + CUDA (bf16/f32/f16/int8/int4)
GRANITE_NAR = [
    ("granite-nar", "ibm-granite/granite-speech-4.1-2b-nar", "cpu", prec)
    for prec in ("float32", "bfloat16")
] + [
    ("granite-nar", "ibm-granite/granite-speech-4.1-2b-nar", "cuda", prec)
    for prec in ("bfloat16", "float32", "float16", "int8", "int4")
]

# granite (AR): native=bfloat16, CPU (f32/bf16) + CUDA (bf16/f32/f16/int8/int4)
GRANITE = [
    ("granite", "ibm-granite/granite-speech-4.1-2b", "cpu", prec)
    for prec in ("float32", "bfloat16")
] + [
    ("granite", "ibm-granite/granite-speech-4.1-2b", "cuda", prec)
    for prec in ("bfloat16", "float32", "float16", "int8", "int4")
]

def _cuda_available():
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_configs(configs, audio, request, results_file):
    """Run transcription for a list of configs. Returns (results, skipped, errors)."""
    force_cuda = request.config.getoption("--force-cuda")
    cuda_ok = _cuda_available() or force_cuda

    audio_data, sr = audio

    results = []
    skipped = []
    errors = []

    for model_type, model_name, device, compute_type in configs:
        if device == "cuda" and not cuda_ok:
            skipped.append((model_type, model_name, device, compute_type))
            continue

        print(f"\n>>> Testing: {model_type}/{model_name} ({device}/{compute_type})")
        try:
            settings = Settings(
                device_name="",
                model_type=model_type,
                model_name=model_name,
                compute_type=compute_type,
                device=device,
                language="en",
            )

            t0 = time.monotonic()
            wrapper = ModelWrapper(settings)
            load_time = round(time.monotonic() - t0, 2)

            t1 = time.monotonic()
            text = wrapper.transcribe(audio_data, sample_rate=sr)
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

        except Exception as exc:
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
        lines.append("SKIPPED (CUDA not available)")
        for s in skipped:
            lines.append(f"  {s[0]}/{s[1]} ({s[2]}/{s[3]})")
        lines.append("")

    if errors:
        lines.append("ERRORS")
        for e in errors:
            lines.append(f"  {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}")
        lines.append("")

    lines.append(
        f"Summary: {len(results)} OK, {len(skipped)} skipped, {len(errors)} errors"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tests — one per model type
# ---------------------------------------------------------------------------


class TestTranscribeWhisper:
    """Transcribe test audio with faster-whisper configs."""

    def test_whisper(self, audio, request, tmp_path):
        results, skipped, errors, audio_data, sr = _run_configs(
            WHISPER, audio, request, None
        )

        block = _format_results("faster-whisper", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")

        assert errors == [], (
            f"{len(errors)} config(s) failed:\n"
            + "\n".join(
                f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
            )
        )


class TestTranscribeParakeet:
    """Transcribe test audio with parakeet configs."""

    def test_parakeet(self, audio, request, tmp_path):
        results, skipped, errors, audio_data, sr = _run_configs(
            PARAKEET, audio, request, None
        )

        block = _format_results("parakeet", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")

        assert errors == [], (
            f"{len(errors)} config(s) failed:\n"
            + "\n".join(
                f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
            )
        )


class TestTranscribeCanary:
    """Transcribe test audio with canary configs."""

    def test_canary(self, audio, request, tmp_path):
        results, skipped, errors, audio_data, sr = _run_configs(
            CANARY, audio, request, None
        )

        block = _format_results("canary", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")

        assert errors == [], (
            f"{len(errors)} config(s) failed:\n"
            + "\n".join(
                f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
            )
        )


class TestTranscribeVoxtral:
    """Transcribe test audio with voxtral configs."""

    def test_voxtral(self, audio, request, tmp_path):
        results, skipped, errors, audio_data, sr = _run_configs(
            VOXTRAL, audio, request, None
        )

        block = _format_results("voxtral", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")

        assert errors == [], (
            f"{len(errors)} config(s) failed:\n"
            + "\n".join(
                f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
            )
        )


class TestTranscribeCohere:
    """Transcribe test audio with cohere configs."""

    def test_cohere(self, audio, request, tmp_path):
        results, skipped, errors, audio_data, sr = _run_configs(
            COHERE, audio, request, None
        )

        block = _format_results("cohere", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")

        assert errors == [], (
            f"{len(errors)} config(s) failed:\n"
            + "\n".join(
                f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
            )
        )


class TestTranscribeGraniteNAR:
    """Transcribe test audio with granite-nar configs."""

    def test_granite_nar(self, audio, request, tmp_path):
        results, skipped, errors, audio_data, sr = _run_configs(
            GRANITE_NAR, audio, request, None
        )

        block = _format_results("granite-nar", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")

        assert errors == [], (
            f"{len(errors)} config(s) failed:\n"
            + "\n".join(
                f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
            )
        )


class TestTranscribeGranite:
    """Transcribe test audio with granite (AR) configs."""

    def test_granite(self, audio, request, tmp_path):
        results, skipped, errors, audio_data, sr = _run_configs(
            GRANITE, audio, request, None
        )

        block = _format_results("granite", results, skipped, errors, audio_data, sr)
        print(f"\n{block}")

        assert errors == [], (
            f"{len(errors)} config(s) failed:\n"
            + "\n".join(
                f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
            )
        )



