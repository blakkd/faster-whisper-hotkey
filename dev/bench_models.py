"""Benchmark GGUF models on the fixed 16 kHz fixture.

Usage:
    uv run python dev/bench_models.py [parakeet-ultra canary ...] [--device cuda|vulkan|cpu] [--quant int8]

Loads each model once (warm), then transcribes test_audio_data/test_16k.wav
three times, printing load time and best/median transcribe time.
"""

import argparse
import statistics
import time

import numpy as np
import soundfile as sf

from faster_whisper_hotkey.models import ModelWrapper
from faster_whisper_hotkey.settings import Settings

AUDIO_PATH = "test_audio_data/test_16k.wav"
LANGUAGE_BY_MODEL = {
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
DEFAULT_MODEL_NAME = {
    "whisper": "large-v3",
    "parakeet": "parakeet-tdt-0.6b-v3",
    "parakeet-ultra": "parakeet-ultra",
    "canary": "canary-1b-v2",
    "voxtral": "voxtral-mini-3b-2507",
    "cohere": "cohere-transcribe-03-2026",
    "granite-nar": "granite-speech-4.1-2b-nar",
    "granite": "granite-speech-4.1-2b",
    "granite-turboctc": "granite-speech-5.0-470m-turboctc-nc",
    "qwen3-asr": "qwen3-asr-1.7b",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["parakeet-ultra"],
                        help="model types, or type:name (e.g. whisper:small)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quant", default="int8")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    audio, sr = sf.read(AUDIO_PATH, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
    duration = len(audio) / sr
    print(f"audio: {AUDIO_PATH}  {duration:.1f} s @ {sr} Hz  device={args.device} quant={args.quant}")

    for spec in args.models:
        model_type, _, name_override = spec.partition(":")
        model_name = name_override or DEFAULT_MODEL_NAME[model_type]
        settings = Settings(
            device_name="",
            model_type=model_type,
            model_name=model_name,
            compute_type=args.quant,
            device=args.device,
            language=LANGUAGE_BY_MODEL[model_type],
        )
        t0 = time.perf_counter()
        wrapper = ModelWrapper(settings)
        load = time.perf_counter() - t0

        times = []
        for _ in range(args.runs):
            t1 = time.perf_counter()
            text = wrapper.transcribe(audio, sample_rate=sr, language=settings.language)
            times.append(time.perf_counter() - t1)

        best, med = min(times), statistics.median(times)
        print(
            f"{model_type:18s} {wrapper.quant:6s} {args.device:6s} load={load:5.1f}s  best={best:5.2f}s "
            f"median={med:5.2f}s  speed={duration / best:5.1f}x  text={text.strip()[:60]!r}"
        )


if __name__ == "__main__":
    main()
