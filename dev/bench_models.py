"""Benchmark GGUF models on the fixed 16 kHz fixture.

Usage:
    uv run python dev/bench_models.py [parakeet-ultra canary-1b-v2 ...] [--device cuda|vulkan|cpu] [--quant int8]

Model names are the catalog display names (see the app's model menu, or
``catalog.list_models()``). Each model is loaded once (warm), then transcribes
test_audio_data/test_16k.wav three times, printing load time and best/median
transcribe time.
"""

import argparse
import statistics
import time

import numpy as np
import soundfile as sf

from faster_whisper_hotkey import catalog
from faster_whisper_hotkey.models import ModelWrapper
from faster_whisper_hotkey.settings import Settings

AUDIO_PATH = "test_audio_data/test_16k.wav"

# Friendly --quant labels -> GGUF quant labels (mirrors catalog._QUANT_LABELS).
_QUANT_BY_LABEL = {v: k for k, v in catalog._QUANT_LABELS.items()}


def resolve_model(token: str):
    entry = next((e for e in catalog.list_models() if e.name.lower() == token.lower()), None)
    if entry is None:
        raise SystemExit(f"Unknown model {token!r}. Names come from catalog.list_models() (e.g. parakeet-ultra).")
    return entry


def pick_file(entry, quant: str):
    want = _QUANT_BY_LABEL.get(quant, quant.upper())
    for f in entry.files:
        if f.quant == want:
            return f
    if not entry.files:
        raise SystemExit(f"{entry.name} publishes no GGUF files.")
    print(f"  note: no {quant} file for {entry.name}; using {entry.files[0].quant}")
    return entry.files[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["parakeet-ultra"], help="catalog model names")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quant", default="int8")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    audio, sr = sf.read(AUDIO_PATH, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
    duration = len(audio) / sr
    print(f"audio: {AUDIO_PATH}  {duration:.1f} s @ {sr} Hz  device={args.device} quant={args.quant}")

    for token in args.models:
        entry = resolve_model(token)
        chosen = pick_file(entry, args.quant)
        settings = Settings(
            device_name="",
            model_repo=entry.repo_id,
            model_file=chosen.filename,
            device=args.device,
            language="auto",
        )
        t0 = time.perf_counter()
        wrapper = ModelWrapper(settings)
        load = time.perf_counter() - t0

        times = []
        for _ in range(args.runs):
            t1 = time.perf_counter()
            text = wrapper.transcribe(audio, sample_rate=sr, language="auto")
            times.append(time.perf_counter() - t1)
        wrapper.close()

        best, med = min(times), statistics.median(times)
        print(
            f"{entry.name:24s} {chosen.quant:6s} {args.device:6s} load={load:5.1f}s  best={best:5.2f}s "
            f"median={med:5.2f}s  speed={duration / best:5.1f}x  text={text.strip()[:60]!r}"
        )


if __name__ == "__main__":
    main()
