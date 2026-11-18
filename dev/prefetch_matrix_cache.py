"""Warm the GGUF cache for chosen catalog models (one quantization each).

Usage:
    uv run python dev/prefetch_matrix_cache.py [parakeet-ultra whisper-small ...] [--quant int8]

Downloads run outside any timing and existing files are skipped, so re-run freely.
Model names are catalog display names; a model with no file in --quant is skipped.
"""

import argparse
import time

from faster_whisper_hotkey import catalog
from faster_whisper_hotkey.downloader import ensure_gguf

# A sensible default set: the headline models from the README.
DEFAULT_MODELS = [
    "parakeet-ultra",
    "canary-1b-v2",
    "voxtral-mini-3b-2507",
    "cohere-transcribe-03-2026",
    "granite-speech-4.1-2b",
    "granite-speech-4.1-2b-nar",
    "granite-speech-5.0-470m-turboctc-nc",
    "qwen3-asr-1.7b",
    "whisper-large-v3",
    "whisper-small",
]

_QUANT_BY_LABEL = {v: k for k, v in catalog._QUANT_LABELS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS, help="catalog model names")
    parser.add_argument("--quant", default="int8", help="friendly precision label (int8, int4, float16, ...)")
    args = parser.parse_args()

    want = _QUANT_BY_LABEL.get(args.quant, args.quant.upper())
    by_name = {e.name.lower(): e for e in catalog.list_models()}

    jobs = []
    for token in args.models:
        entry = by_name.get(token.lower())
        if entry is None:
            print(f"SKIP unknown model {token!r}")
            continue
        chosen = next((f for f in entry.files if f.quant == want), None)
        if chosen is None:
            print(f"SKIP {entry.name}: no {want} file (has {', '.join(f.quant for f in entry.files)})")
            continue
        jobs.append((entry.repo_id, chosen.filename))

    for attempt in range(1, 4):
        pending = []
        for repo, filename in jobs:
            try:
                path = ensure_gguf(repo, filename)
                print(f"[warm {attempt}] {filename}: {path.stat().st_size / 1e9:.2f} GB", flush=True)
            except Exception as exc:  # noqa: BLE001 — any download failure is retried, not fatal here
                print(f"[warm {attempt}] FAILED {repo}/{filename}: {exc}", flush=True)
                pending.append((repo, filename))
        jobs = pending
        if not jobs:
            print("PREFETCH-COMPLETE")
            break
        time.sleep(10)

    if jobs:
        print(f"PREFETCH-INCOMPLETE: {len(jobs)} files failed after 3 passes")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
