#!/usr/bin/env bash
# Isolated re-run for qwen3-asr's GPU lane.
#
# WHY: the transcribe.cpp qwen3_asr family does not return device memory to
# the driver on model free (VRAM climbs monotonically across configs until the
# card OOMs; upstream engine bug). Running every other family in one process is
# fine — their free() works. Here: one pytest process per config, so every model
# starts on a fully empty VRAM. Each process appends its block to the results
# file (custom --results-file paths are never truncated).
#
# Run with the LLM server stopped. Output: /tmp/opencode/qwen3-gpu.txt
set -u
cd "$(dirname "$0")/.." || exit 1
OUT="${1:-/tmp/opencode/qwen3-gpu.txt}"
rm -f "$OUT"
for cfg in cuda:bfloat16 cuda:float16 cuda:int8 cuda:q6_k cuda:q5_k_m cuda:int4 \
           vulkan:bfloat16 vulkan:float16 vulkan:int8 vulkan:q6_k vulkan:q5_k_m vulkan:int4; do
  echo "=== $cfg -> $OUT"
  # || true: a native teardown abort at interpreter exit (exit != 0) must not
  # stop the loop; the pytest block was already written by then.
  uv run pytest tests/test_model_all_configs.py::TestTranscribeQwen3ASR \
    --configs "$cfg" --results-file "$OUT" -v --tb=short || echo "!!! nonzero exit for $cfg (check block above in $OUT)"
done
echo "DONE. Paste or send: $OUT"
