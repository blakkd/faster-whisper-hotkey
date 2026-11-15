"""Warm the GGUF cache with every file the benchmark matrix needs (60 quants).

Downloads run OUTSIDE any timing; the matrix's own prefetch only verifies the
cache afterwards. Re-run any time — existing files are skipped.
"""

import time

from faster_whisper_hotkey.downloader import ensure_gguf
from faster_whisper_hotkey.gguf_models import repo_and_filename

SPECS = [
    ("whisper", "small"),
    ("parakeet", "parakeet-tdt-0.6b-v3"),
    ("parakeet-ultra", "parakeet-ultra"),
    ("canary", "canary-1b-v2"),
    ("voxtral", "voxtral-mini-3b-2507"),
    ("cohere", "cohere-transcribe-03-2026"),
    ("granite-nar", "granite-speech-4.1-2b-nar"),
    ("granite", "granite-speech-4.1-2b"),
    ("granite-turboctc", "granite-speech-5.0-470m-turboctc-nc"),
    ("qwen3-asr", "qwen3-asr-1.7b"),
]
QUANTS = ["float32", "float16", "int8", "q6_k", "q5_k_m", "int4"]

jobs = [(mt, nm, ct) for mt, nm in SPECS for ct in QUANTS]
for attempt in range(1, 4):
    pending = []
    for mt, nm, ct in jobs:
        try:
            repo, fn, _, _ = repo_and_filename(mt, nm, ct)
            path = ensure_gguf(repo, fn)
            print(f"[warm {attempt}] {fn}: {path.stat().st_size / 1e9:.2f} GB", flush=True)
        except Exception as exc:  # noqa: BLE001 — any download failure is retried, not fatal here
            print(f"[warm {attempt}] FAILED {mt}/{nm}/{ct}: {exc}", flush=True)
            pending.append((mt, nm, ct))
    jobs = pending
    if not jobs:
        print("PREFETCH-COMPLETE")
        break
    time.sleep(10)

if jobs:
    print(f"PREFETCH-INCOMPLETE: {len(jobs)} files failed after 3 passes")
    raise SystemExit(1)
