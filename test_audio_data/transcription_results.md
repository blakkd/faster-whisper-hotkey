# Transcription speed by configuration

Audio: `test.mp3` — 40.05 s
Hardware: NVIDIA RTX PRO 4500 Blackwell (32 GB) · AMD Ryzen 9 5950X (16 cores / 32 threads) · DDR4-3000 MT/s RAM
Speed = `40.05 / transcribe_time` (× realtime).

## faster-whisper — `whisper/small`

| Device | Compute | Load (s) | Transcribe (s) |     Speed | Status                                           |
| ------ | ------- | -------: | -------------: | --------: | ------------------------------------------------ |
| cpu    | int8    |     0.85 |       **4.14** |  **9.7×** | OK                                               |
| cuda   | float16 |    16.22 |           2.40 |     16.7× | OK                                               |
| cuda   | float32 |     7.71 |       **0.64** | **62.6×** | OK                                               |
| cuda   | int8    |        — |              — |         — | ⚠ skipped (int8 unsupported on Blackwell sm_12x) |

## parakeet-ultra — `moondream/parakeet-ultra` (Moondream Photon)

| Device | Compute | Load (s) | Transcribe (s) |     Speed | Status |
| ------ | ------- | -------: | -------------: | --------: | ------ |
| cpu    | float32 |     1.00 |       **2.55** | **15.7×** | OK     |

Compute is auto-selected by Photon (no user choice): bf16 only when the CPU has native BF16 support (AVX512-BF16/AMX), float32 otherwise — this machine (Zen 3) ran float32. On CUDA the fast path requires bf16, which is the default.

No CUDA row: not benchmarked — GPU inference for this model class is well below 1 s and not relevant for our use case (the GPU was also fully occupied during the run).

## canary — `canary-1b-v2`

| Device | Compute  | Load (s) | Transcribe (s) |    Speed | Status |
| ------ | -------- | -------: | -------------: | -------: | ------ |
| cpu    | float32  |    21.28 |      **55.72** | **0.7×** | OK     |
| cpu    | bfloat16 |    16.05 |         430.71 |     0.1× | OK     |
| cuda   | float32  |    17.80 |           0.91 |      44× | OK     |
| cuda   | bfloat16 |    15.84 |           0.86 |      47× | OK     |
| cuda   | int8     |    15.81 |           0.85 |      47× | OK     |
| cuda   | int4     |    15.78 |       **0.84** |  **48×** | OK     |

## voxtral — `Voxtral-Mini-3B-2507` (CUDA only)

| Device | Compute  | Load (s) | Transcribe (s) |     Speed | Status |
| ------ | -------- | -------: | -------------: | --------: | ------ |
| cuda   | float32  |    11.16 |           2.96 |     13.5× | OK     |
| cuda   | bfloat16 |     5.68 |       **2.28** | **17.6×** | OK     |
| cuda   | int8     |    13.19 |          11.37 |      3.5× | OK     |
| cuda   | int4     |     5.86 |           4.24 |      9.4× | OK     |

## cohere — `cohere-transcribe-03-2026`

| Device | Compute  | Load (s) | Transcribe (s) |    Speed | Status |
| ------ | -------- | -------: | -------------: | -------: | ------ |
| cpu    | float32  |     7.02 |      **33.15** | **1.2×** | OK     |
| cpu    | bfloat16 |     3.77 |         324.07 |     0.1× | OK     |
| cuda   | bfloat16 |     3.83 |       **0.60** |  **67×** | OK     |
| cuda   | float32  |     3.74 |           0.68 |      59× | OK     |
| cuda   | int8     |     5.62 |           3.10 |      13× | OK     |
| cuda   | int4     |     4.00 |           1.58 |      25× | OK     |

## granite-nar — `granite-speech-4.1-2b-nar`

| Device | Compute  | Load (s) | Transcribe (s) |     Speed | Status |
| ------ | -------- | -------: | -------------: | --------: | ------ |
| cpu    | float32  |     8.12 |      **35.68** |  **1.1×** | OK     |
| cpu    | bfloat16 |     4.75 |         357.89 |      0.1× | OK     |
| cuda   | bfloat16 |     4.69 |       **0.06** | **~667×** | OK     |
| cuda   | float32  |     5.43 |           0.20 |      200× | OK     |
| cuda   | int8     |     6.29 |           0.20 |      200× | OK     |
| cuda   | int4     |     4.72 |           0.19 |      211× | OK     |

## granite — `granite-speech-4.1-2b`

| Device | Compute  | Load (s) | Transcribe (s) |     Speed | Status |
| ------ | -------- | -------: | -------------: | --------: | ------ |
| cpu    | float32  |     7.10 |      **55.17** |  **0.7×** | OK     |
| cpu    | bfloat16 |     4.09 |         322.03 |      0.1× | OK     |
| cuda   | bfloat16 |     3.96 |       **3.17** | **12.6×** | OK     |
| cuda   | float32  |     4.91 |           3.31 |     12.1× | OK     |
| cuda   | int8     |     5.53 |          15.65 |      2.6× | OK     |
| cuda   | int4     |     4.26 |           5.03 |      8.0× | OK     |

## granite-turboctc — `granite-speech-5.0-470m-turboctc-nc`

| Device | Compute  | Load (s) | Transcribe (s) |      Speed | Status |
| ------ | -------- | -------: | -------------: | ---------: | ------ |
| cpu    | float32  |     3.67 |       **4.74** |   **8.4×** | OK     |
| cpu    | bfloat16 |     3.10 |          44.65 |       0.9× | OK     |
| cuda   | bfloat16 |     3.12 |       **0.02** | **~2000×** | OK     |
| cuda   | float32  |     3.29 |           0.03 |     ~1333× | OK     |
| cuda   | int8     |     3.51 |           0.07 |       571× | OK     |
| cuda   | int4     |     3.14 |           0.03 |     ~1333× | OK     |

## qwen3-asr — `Qwen3-ASR-1.7B-hf`

| Device | Compute  | Load (s) | Transcribe (s) |     Speed | Status |
| ------ | -------- | -------: | -------------: | --------: | ------ |
| cpu    | float32  |     5.26 |      **48.34** |  **0.8×** | OK     |
| cpu    | bfloat16 |     6.92 |         204.69 |      0.2× | OK     |
| cuda   | bfloat16 |     4.14 |       **2.65** | **15.1×** | OK     |
| cuda   | float32  |     4.98 |           2.84 |     14.1× | OK     |
| cuda   | int8     |     5.80 |          11.57 |      3.5× | OK     |
| cuda   | int4     |     4.27 |           4.17 |      9.6× | OK     |
