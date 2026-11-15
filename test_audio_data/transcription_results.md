# Transcription speed by configuration — transcribe.cpp (GGUF)

Audio: `test_audio_data/test_16k.wav` — 40.05 s, native 16 kHz mono
Hardware: NVIDIA RTX PRO 4500 Blackwell (32 GB) · AMD Ryzen 9 5950X (16 cores / 32 threads)
Speed = `40.05 / best run` (× realtime); `load+warm` = model init + warmup clip, downloads excluded (prefetch).
Method: `tests/test_model_all_configs.py` 180-config benchmark matrix (10 models × 6 published quants × cpu/cuda/vulkan), 2026-11-15; the matrix was retired after this run and lives in git history at commit 3e4a025.
qwen3-asr GPU rows re-run one config per process (transcribe.cpp qwen3_asr VRAM free bug; see dev/onboarding.md).

## whisper

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| F32     | cpu    |  0.93 s |  4.81 s |    8.3x |         — |
| F16     | cpu    |  0.81 s |  3.31 s |   12.1x |         — |
| Q8_0    | cpu    |  0.86 s |  2.89 s |   13.8x |         — |
| Q6_K    | cpu    |  0.82 s |  2.65 s |   15.1x |         — |
| Q5_K_M  | cpu    |  0.78 s |  2.58 s |   15.5x |         — |
| Q4_K_M  | cpu    |  0.80 s |  2.55 s |   15.7x |         — |
| F32     | cuda   |  0.37 s |  0.28 s |  141.3x |  2078 MiB |
| F16     | cuda   |  0.13 s |  0.22 s |  181.8x |  1562 MiB |
| Q8_0    | cuda   |  0.10 s |  0.24 s |  163.7x |  1348 MiB |
| Q6_K    | cuda   |  0.08 s |  0.25 s |  162.5x |  1296 MiB |
| Q5_K_M  | cuda   |  0.08 s |  0.24 s |  165.9x |  1278 MiB |
| Q4_K_M  | cuda   |  0.08 s |  0.24 s |  165.5x |  1256 MiB |
| F32     | vulkan |  1.38 s |  0.34 s |  116.1x |  2492 MiB |
| F16     | vulkan |  2.77 s |  0.28 s |  143.2x |  1970 MiB |
| Q8_0    | vulkan |  0.60 s |  0.27 s |  150.9x |  1757 MiB |
| Q6_K    | vulkan |  5.07 s |  0.28 s |  143.6x |  1705 MiB |
| Q5_K_M  | vulkan |  6.27 s |  0.30 s |  131.4x |  1858 MiB |
| Q4_K_M  | vulkan |  0.82 s |  0.26 s |  151.4x |  1842 MiB |

## parakeet

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| F32     | cpu    |  1.57 s |  1.36 s |   29.5x |         — |
| F16     | cpu    |  0.94 s |  1.24 s |   32.2x |         — |
| Q8_0    | cpu    |  0.70 s |  1.41 s |   28.4x |         — |
| Q6_K    | cpu    |  0.74 s |  1.30 s |   30.8x |         — |
| Q5_K_M  | cpu    |  0.44 s |  1.28 s |   31.2x |         — |
| Q4_K_M  | cpu    |  0.51 s |  1.27 s |   31.4x |         — |
| F32     | cuda   |  0.43 s |  0.30 s |  131.9x |  3677 MiB |
| F16     | cuda   |  0.21 s |  0.30 s |  133.0x |  2487 MiB |
| Q8_0    | cuda   |  0.16 s |  0.29 s |  136.0x |  1995 MiB |
| Q6_K    | cuda   |  0.14 s |  0.30 s |  134.1x |  1871 MiB |
| Q5_K_M  | cuda   |  0.13 s |  0.30 s |  134.4x |  1815 MiB |
| Q4_K_M  | cuda   |  0.12 s |  0.30 s |  135.0x |  1755 MiB |
| F32     | vulkan |  1.05 s |  1.47 s |   27.1x |  4001 MiB |
| F16     | vulkan |  0.71 s |  1.52 s |   26.4x |  2795 MiB |
| Q8_0    | vulkan |  0.18 s |  0.16 s |  243.4x |  2302 MiB |
| Q6_K    | vulkan |  0.44 s |  0.17 s |  229.2x |  2191 MiB |
| Q5_K_M  | vulkan |  0.44 s |  0.17 s |  235.6x |  2133 MiB |
| Q4_K_M  | vulkan |  0.13 s |  0.17 s |  235.8x |  2072 MiB |

## parakeet-ultra

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| F32     | cpu    |  1.34 s |  1.49 s |   26.9x |         — |
| F16     | cpu    |  0.72 s |  1.39 s |   28.9x |         — |
| Q8_0    | cpu    |  0.70 s |  1.49 s |   26.8x |         — |
| Q6_K    | cpu    |  0.45 s |  1.46 s |   27.4x |         — |
| Q5_K_M  | cpu    |  0.43 s |  1.42 s |   28.2x |         — |
| Q4_K_M  | cpu    |  0.51 s |  1.43 s |   28.0x |         — |
| F32     | cuda   |  0.35 s |  0.35 s |  114.1x |  3839 MiB |
| F16     | cuda   |  0.21 s |  0.34 s |  117.5x |  2653 MiB |
| Q8_0    | cuda   |  0.14 s |  0.33 s |  119.6x |  2159 MiB |
| Q6_K    | cuda   |  0.13 s |  0.34 s |  117.1x |  2037 MiB |
| Q5_K_M  | cuda   |  0.12 s |  0.34 s |  117.1x |  1979 MiB |
| Q4_K_M  | cuda   |  0.11 s |  0.33 s |  119.8x |  1917 MiB |
| F32     | vulkan |  0.49 s |  0.21 s |  186.5x |  4160 MiB |
| F16     | vulkan |  0.26 s |  2.09 s |   19.1x |  2954 MiB |
| Q8_0    | vulkan |  0.18 s |  0.19 s |  206.6x |  2462 MiB |
| Q6_K    | vulkan |  0.15 s |  0.19 s |  206.5x |  2339 MiB |
| Q5_K_M  | vulkan |  0.14 s |  0.20 s |  204.1x |  2280 MiB |
| Q4_K_M  | vulkan |  0.13 s |  0.20 s |  205.1x |  2220 MiB |

## canary

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| F32     | cpu    |  2.28 s |  4.88 s |    8.2x |         — |
| F16     | cpu    |  1.03 s |  3.35 s |   11.9x |         — |
| Q8_0    | cpu    |  0.99 s |  2.87 s |   14.0x |         — |
| Q6_K    | cpu    |  0.63 s |  2.56 s |   15.6x |         — |
| Q5_K_M  | cpu    |  0.60 s |  2.45 s |   16.3x |         — |
| Q4_K_M  | cpu    |  0.71 s |  2.28 s |   17.6x |         — |
| F32     | cuda   |  0.59 s |  0.47 s |   86.0x |  6367 MiB |
| F16     | cuda   |  0.29 s |  0.40 s |   99.9x |  4507 MiB |
| Q8_0    | cuda   |  0.21 s |  0.40 s |  100.3x |  3723 MiB |
| Q6_K    | cuda   |  0.18 s |  0.39 s |  102.0x |  3521 MiB |
| Q5_K_M  | cuda   |  0.17 s |  0.39 s |  102.1x |  3429 MiB |
| Q4_K_M  | cuda   |  0.16 s |  0.40 s |  101.2x |  3333 MiB |
| F32     | vulkan |  1.34 s |  0.38 s |  105.2x |  6707 MiB |
| F16     | vulkan |  0.38 s |  0.30 s |  134.9x |  4812 MiB |
| Q8_0    | vulkan |  0.24 s |  0.28 s |  141.6x |  4028 MiB |
| Q6_K    | vulkan |  0.20 s |  0.27 s |  149.5x |  3825 MiB |
| Q5_K_M  | vulkan |  0.19 s |  0.27 s |  150.9x |  3734 MiB |
| Q4_K_M  | vulkan |  0.17 s |  0.25 s |  158.1x |  3638 MiB |

## voxtral

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16    | cpu    | 12.83 s | 45.60 s |    0.9x |         — |
| F16     | cpu    | 10.00 s | 40.71 s |    1.0x |         — |
| Q8_0    | cpu    |  9.39 s | 32.03 s |    1.2x |         — |
| Q6_K    | cpu    |  7.96 s | 25.59 s |    1.6x |         — |
| Q5_K_M  | cpu    |  7.80 s | 23.94 s |    1.7x |         — |
| Q4_K_M  | cpu    |  7.28 s | 22.37 s |    1.8x |         — |
| BF16    | cuda   |  1.75 s |  1.29 s |   31.1x | 12059 MiB |
| F16     | cuda   |  1.51 s |  1.34 s |   29.9x | 12219 MiB |
| Q8_0    | cuda   |  0.88 s |  0.83 s |   48.0x |  7867 MiB |
| Q6_K    | cuda   |  0.73 s |  0.76 s |   53.0x |  6789 MiB |
| Q5_K_M  | cuda   |  0.67 s |  0.68 s |   58.7x |  6403 MiB |
| Q4_K_M  | cuda   |  0.62 s |  0.62 s |   64.7x |  5945 MiB |
| BF16    | vulkan |  4.75 s |  1.52 s |   26.3x | 12345 MiB |
| F16     | vulkan |  1.69 s |  1.53 s |   26.1x | 12366 MiB |
| Q8_0    | vulkan |  1.90 s |  1.07 s |   37.5x |  8243 MiB |
| Q6_K    | vulkan |  0.81 s |  0.97 s |   41.2x |  7114 MiB |
| Q5_K_M  | vulkan |  0.87 s |  0.90 s |   44.3x |  6728 MiB |
| Q4_K_M  | vulkan |  0.75 s |  0.83 s |   48.0x |  6270 MiB |

## cohere

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16    | cpu    |  3.02 s |  6.33 s |    6.3x |         — |
| F16     | cpu    |  2.52 s |  4.66 s |    8.6x |         — |
| Q8_0    | cpu    |  2.03 s |  4.67 s |    8.6x |         — |
| Q6_K    | cpu    |  2.34 s |  4.09 s |    9.8x |         — |
| Q5_K_M  | cpu    |  2.58 s |  4.19 s |    9.6x |         — |
| Q4_K_M  | cpu    |  1.52 s |  3.98 s |   10.1x |         — |
| BF16    | cuda   |  0.55 s |  0.55 s |   73.2x |  6619 MiB |
| F16     | cuda   |  0.54 s |  0.53 s |   75.4x |  6619 MiB |
| Q8_0    | cuda   |  0.39 s |  0.53 s |   74.8x |  5003 MiB |
| Q6_K    | cuda   |  0.33 s |  0.54 s |   74.0x |  4585 MiB |
| Q5_K_M  | cuda   |  0.30 s |  0.53 s |   75.0x |  4391 MiB |
| Q4_K_M  | cuda   |  0.29 s |  0.53 s |   75.3x |  4189 MiB |
| BF16    | vulkan |  1.23 s |  2.31 s |   17.3x |  6914 MiB |
| F16     | vulkan |  0.77 s |  0.26 s |  152.1x |  6919 MiB |
| Q8_0    | vulkan |  0.50 s |  0.25 s |  160.8x |  5302 MiB |
| Q6_K    | vulkan |  0.41 s |  0.26 s |  155.9x |  4884 MiB |
| Q5_K_M  | vulkan |  0.37 s |  0.25 s |  160.2x |  4691 MiB |
| Q4_K_M  | vulkan |  0.35 s |  0.25 s |  159.6x |  4489 MiB |

## granite-nar

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16    | cpu    |  2.51 s |  8.04 s |    5.0x |         — |
| F16     | cpu    |  2.73 s |  6.05 s |    6.6x |         — |
| Q8_0    | cpu    |  1.78 s |  7.07 s |    5.7x |         — |
| Q6_K    | cpu    |  2.25 s |  6.37 s |    6.3x |         — |
| Q5_K_M  | cpu    |  1.42 s |  6.18 s |    6.5x |         — |
| Q4_K_M  | cpu    |  1.11 s |  6.08 s |    6.6x |         — |
| BF16    | cuda   |  0.62 s |  0.33 s |  121.1x |  6965 MiB |
| F16     | cuda   |  0.60 s |  0.32 s |  124.4x |  6969 MiB |
| Q8_0    | cuda   |  0.38 s |  0.31 s |  127.3x |  5045 MiB |
| Q6_K    | cuda   |  0.31 s |  0.33 s |  122.1x |  4549 MiB |
| Q5_K_M  | cuda   |  0.28 s |  0.32 s |  123.2x |  4363 MiB |
| Q4_K_M  | cuda   |  0.26 s |  0.32 s |  125.3x |  4151 MiB |
| BF16    | vulkan |  8.71 s |  3.31 s |   12.1x |  8395 MiB |
| F16     | vulkan |  4.53 s |  0.35 s |  113.6x |  8396 MiB |
| Q8_0    | vulkan |  2.85 s |  0.37 s |  107.4x |  6472 MiB |
| Q6_K    | vulkan |  1.00 s |  0.37 s |  108.9x |  5975 MiB |
| Q5_K_M  | vulkan |  0.66 s |  0.36 s |  112.2x |  5789 MiB |
| Q4_K_M  | vulkan |  0.56 s |  0.35 s |  113.9x |  5577 MiB |

## granite

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16    | cpu    |  2.90 s | 20.69 s |    1.9x |         — |
| F16     | cpu    |  5.23 s | 17.80 s |    2.2x |         — |
| Q8_0    | cpu    |  1.89 s | 13.21 s |    3.0x |         — |
| Q6_K    | cpu    |  2.44 s | 11.36 s |    3.5x |         — |
| Q5_K_M  | cpu    |  2.18 s | 10.58 s |    3.8x |         — |
| Q4_K_M  | cpu    |  1.38 s |  9.87 s |    4.1x |         — |
| BF16    | cuda   |  0.82 s |  0.90 s |   44.4x |  9289 MiB |
| F16     | cuda   |  0.87 s |  0.91 s |   43.8x |  9291 MiB |
| Q8_0    | cuda   |  0.47 s |  0.73 s |   55.1x |  7319 MiB |
| Q6_K    | cuda   |  0.43 s |  0.67 s |   59.4x |  6809 MiB |
| Q5_K_M  | cuda   |  0.41 s |  0.67 s |   59.9x |  6623 MiB |
| Q4_K_M  | cuda   |  0.35 s |  0.65 s |   61.4x |  6407 MiB |
| BF16    | vulkan |  1.20 s |  1.14 s |   35.1x |  9607 MiB |
| F16     | vulkan |  0.84 s |  1.17 s |   34.3x |  9608 MiB |
| Q8_0    | vulkan |  0.52 s |  0.94 s |   42.6x |  7631 MiB |
| Q6_K    | vulkan |  0.43 s |  0.90 s |   44.5x |  7134 MiB |
| Q5_K_M  | vulkan |  0.39 s |  0.88 s |   45.4x |  6935 MiB |
| Q4_K_M  | vulkan |  0.35 s |  0.86 s |   46.6x |  6718 MiB |

## granite-turboctc

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16    | cpu    |  0.64 s |  1.12 s |   35.7x |         — |
| F16     | cpu    |  0.64 s |  0.86 s |   46.3x |         — |
| Q8_0    | cpu    |  0.39 s |  0.98 s |   40.8x |         — |
| Q6_K    | cpu    |  0.33 s |  0.87 s |   45.9x |         — |
| Q5_K_M  | cpu    |  0.29 s |  0.87 s |   46.0x |         — |
| Q4_K_M  | cpu    |  0.26 s |  0.86 s |   46.4x |         — |
| BF16    | cuda   |  0.14 s |  0.11 s |  355.8x |  5745 MiB |
| F16     | cuda   |  0.14 s |  0.11 s |  371.1x |  5745 MiB |
| Q8_0    | cuda   |  0.09 s |  0.11 s |  355.7x |  5327 MiB |
| Q6_K    | cuda   |  0.08 s |  0.11 s |  360.6x |  5219 MiB |
| Q5_K_M  | cuda   |  0.07 s |  0.11 s |  374.3x |  5165 MiB |
| Q4_K_M  | cuda   |  0.06 s |  0.11 s |  369.1x |  5111 MiB |
| BF16    | vulkan |  0.18 s |  0.09 s |  433.7x |  6054 MiB |
| F16     | vulkan |  0.18 s |  0.10 s |  399.0x |  6055 MiB |
| Q8_0    | vulkan |  0.11 s |  0.10 s |  388.2x |  5633 MiB |
| Q6_K    | vulkan |  0.09 s |  0.11 s |  371.9x |  5524 MiB |
| Q5_K_M  | vulkan |  0.08 s |  0.11 s |  375.4x |  5471 MiB |
| Q4_K_M  | vulkan |  0.07 s |  0.10 s |  397.4x |  5417 MiB |

## qwen3-asr

| Compute | Device | Load+warm | Best | Speed | VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16    | cpu    |  8.28 s | 17.57 s |    2.3x |         — |
| F16     | cpu    | 10.23 s | 16.26 s |    2.5x |         — |
| Q8_0    | cpu    |  3.67 s | 11.22 s |    3.6x |         — |
| Q6_K    | cpu    |  4.35 s |  9.26 s |    4.3x |         — |
| Q5_K_M  | cpu    |  4.08 s |  8.64 s |    4.6x |         — |
| Q4_K_M  | cpu    |  1.08 s |  7.94 s |    5.0x |         — |
| BF16    | cuda   |  3.07 s |  0.66 s |   60.5x |  5129 MiB |
| F16     | cuda   |  3.64 s |  0.68 s |   58.9x |  5227 MiB |
| Q8_0    | cuda   |  0.72 s |  0.45 s |   88.4x |  3319 MiB |
| Q6_K    | cuda   |  1.69 s |  0.41 s |   98.3x |  2849 MiB |
| Q5_K_M  | cuda   |  1.54 s |  0.38 s |  106.2x |  2683 MiB |
| Q4_K_M  | cuda   |  1.00 s |  0.35 s |  112.8x |  2493 MiB |
| BF16    | vulkan |  1.24 s |  0.94 s |   42.4x |  5235 MiB |
| F16     | vulkan |  1.21 s |  0.95 s |   42.2x |  5246 MiB |
| Q8_0    | vulkan |  0.85 s |  0.71 s |   56.5x |  3428 MiB |
| Q6_K    | vulkan |  0.69 s |  0.66 s |   60.9x |  2958 MiB |
| Q5_K_M  | vulkan |  0.66 s |  0.64 s |   62.9x |  2791 MiB |
| Q4_K_M  | vulkan |  0.38 s |  0.61 s |   65.9x |  2603 MiB |

