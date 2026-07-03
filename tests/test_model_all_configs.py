"""
Integration test: transcribe a fixed audio sample with every supported model config.

Output: test_audio_data/transcription_results.txt, one row per config.
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

# whisper: cpu/int8, cuda/float16, cuda/int8
_configs = [
    (m, "cpu", "int8")
    for m in WHISPER_MODELS
] + [
    (m, "cuda", "float16")
    for m in WHISPER_MODELS
] + [
    (m, "cuda", "int8")
    for m in WHISPER_MODELS
]

WHISPER = [
    ("whisper", m, dev, prec) for m, dev, prec in _configs
]

# parakeet: cpu, cuda
PARAKEET = [
    ("parakeet", "nvidia/parakeet-tdt-0.6b-v3", dev, "float16")
    for dev in ("cpu", "cuda")
]

# canary: cpu only
CANARY = [
    ("canary", "nvidia/canary-1b-v2", "cpu", "float16"),
]

# voxtral: cuda only, 3 precisions
VOXTRAL = [
    ("voxtral", "mistralai/Voxtral-Mini-3B-2507", "cuda", prec)
    for prec in ("float16", "int8", "int4")
]

# cohere: cpu, cuda
COHERE = [
    ("cohere", "CohereLabs/cohere-transcribe-03-2026", dev, "float16")
    for dev in ("cpu", "cuda")
]

# granite-nar: cpu, cuda
GRANITE_NAR = [
    ("granite-nar", "ibm-granite/granite-speech-4.1-2b-nar", dev, "float16")
    for dev in ("cpu", "cuda")
]

# granite (AR): cpu, cuda
GRANITE = [
    ("granite", "ibm-granite/granite-speech-4.1-2b", dev, "float16")
    for dev in ("cpu", "cuda")
]

ALL_CONFIGS = (
    WHISPER + PARAKEET + CANARY + VOXTRAL + COHERE + GRANITE_NAR + GRANITE
)


def _cuda_available():
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestTranscribeAllConfigs:
    """Transcribe test audio with every model/device/precision combo."""

    def test_all_configs(self, audio, request):
        force_cuda = request.config.getoption("--force-cuda", default=False)
        cuda_ok = _cuda_available() or force_cuda

        audio_data, sr = audio

        results = []
        skipped = []
        errors = []

        for model_type, model_name, device, compute_type in ALL_CONFIGS:
            if device == "cuda" and not cuda_ok:
                skipped.append((model_type, model_name, device, compute_type))
                continue

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

                # Unload model before testing next config
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

        # ---- write results file ----
        sep = "|"
        header_cols = [
            "model_type",
            "model_name",
            "device",
            "compute_type",
            "load_time_s",
            "transcribe_time_s",
            "status",
            "transcription",
        ]

        lines = [
            "faster-whisper-hotkey transcription results",
            f"audio: {AUDIO_PATH}  ({round(len(audio_data) / sr, 2)}s @ {sr}Hz)",
            f"cuda_available: {_cuda_available()}",
            f"{'=' * 80}",
            "",
            f"{'RESULTS' if results else ''}",
            sep.join(header_cols),
        ]

        for row in results:
            lines.append(sep.join(str(c) for c in row))

        if skipped:
            lines.append("")
            lines.append("SKIPPED (CUDA not available)")
            lines.append(sep.join(["model_type", "model_name", "device", "compute_type"]))
            for s in skipped:
                lines.append(sep.join(str(c) for c in s))

        if errors:
            lines.append("")
            lines.append("ERRORS")
            err_cols = ["model_type", "model_name", "device", "compute_type", "error"]
            lines.append(sep.join(err_cols))
            for e in errors:
                lines.append(sep.join(str(c) for c in e))

        lines.append("")
        lines.append(
            f"Summary: {len(results)} OK, {len(skipped)} skipped, {len(errors)} errors"
        )

        output_path = "test_audio_data/transcription_results.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Print summary to test output
        print(f"\nResults written to {output_path}")
        print(f"  OK: {len(results)}, Skipped: {len(skipped)}, Errors: {len(errors)}")

        # Fail the test if there were errors
        assert errors == [], (
            f"{len(errors)} config(s) failed:\n"
            + "\n".join(
                f"  - {e[0]}/{e[1]} ({e[2]}/{e[3]}): {e[4]}" for e in errors
            )
        )
