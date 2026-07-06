# faster-whisper Benchmarks

Model: `Systran/faster-whisper` (size: `small`)
Audio: `test_audio_data/test.mp3` (16kHz mono float32)

## CPU

| Precision | Load Time | Transcribe Time | Verdict                          |
| --------- | --------- | --------------- | -------------------------------- |
| float32   | (N/A)     | (N/A)           | Not tested                       |
| float16   | (N/A)     | (N/A)           | Unsupported — CTranslate2 hw req |
| bfloat16  | (N/A)     | (N/A)           | Not tested                       |
| int8      | 8.94s     | 4.53s           | Only supported CPU precision     |
| int4      | (N/A)     | (N/A)           | Not tested                       |

## CUDA

| Precision | Load Time | Transcribe Time | Verdict                          |
| --------- | --------- | --------------- | -------------------------------- |
| float32   | 0.56s     | 0.82s           | Marginally slower inference      |
| float16   | 8.71s     | 0.74s           | Slight edge over float32         |
| bfloat16  | (N/A)     | (N/A)           | Not tested                       |
| int8      | 0.48s     | 0.69s           | Fastest load + fastest inference |
| int4      | (N/A)     | (N/A)           | Not supported by faster-whisper  |

## Recommendations

- **CUDA**: `int8` — fastest load (0.48s) and fastest inference (0.69s); `float16` if quality matters
- **CPU**: `int8` — only supported option
- **Avoid on CPU**: `float16` (not supported by CTranslate2)
- **Avoid on CUDA**: `int4` (not supported by faster-whisper)

---

# Voxtral-Mini-3B-2507 Benchmarks

Model: `mistralai/Voxtral-Mini-3B-2507`
Audio: `test_audio_data/test.mp3` (16kHz mono float32)

## CPU

| Precision | Load Time | Transcribe Time | Verdict                             |
| --------- | --------- | --------------- | ----------------------------------- |
| float32   | 28.72s    | 75.94s          | Usable but slow                     |
| float16   | 32.26s    | 72.38s          | ~30% faster inference than f32      |
| bfloat16  | 13.65s    | 79.03s          | Fastest load, but slowest inference |
| int8      | 30.54s    | 185.72s         | 2.5x slower than float — avoid      |
| int4      | (hung)    | (hung)          | Unusable — never completes          |

## CUDA

| Precision | Load Time | Transcribe Time | Verdict                                   |
| --------- | --------- | --------------- | ----------------------------------------- |
| float32   | 17.22s    | 4.22s           | Heaviest, slowest load                    |
| float16   | 7.19s     | 2.66s           | Best overall (fast load + fast inference) |
| bfloat16  | 6.41s     | 2.63s           | Fastest, nearly identical to float16      |
| int8      | 8.29s     | 11.25s          | 4x slower inference — avoid               |
| int4      | 15.06s    | 4.47s           | Decent, but no advantage over fp16        |

## Recommendations

- **CUDA**: `bfloat16` or `float16` — nearly identical, both fastest
- **CPU**: `float16` — best inference speed; `bfloat16` if load time matters
- **Avoid on both**: `int8` (bitsandbytes quantization is slower than float)
- **Avoid on CPU**: `int4` (hangs indefinitely)
- **Avoid on CUDA**: `int8`, `int4` (no advantage over float precisions)

---

# parakeet-tdt-0.6b-v3 Benchmarks

Model: `nvidia/parakeet-tdt-0.6b-v3`
Audio: `test_audio_data/test.mp3` (16kHz mono float32)

## CPU

| Precision | Load Time | Transcribe Time | Verdict                      |
| --------- | --------- | --------------- | ---------------------------- |
| float32   | 24.43s    | 1.76s           | Fastest inference by far     |
| float16   | 19.06s    | 35.52s          | 20x slower inference — avoid |
| bfloat16  | 22.61s    | 37.02s          | 21x slower inference — avoid |

## CUDA

| Precision | Load Time | Transcribe Time | Verdict                          |
| --------- | --------- | --------------- | -------------------------------- |
| float32   | 19.0s     | 0.89s           | Solid                            |
| float16   | 20.01s    | 0.84s           | Similar to float32               |
| bfloat16  | 19.25s    | 0.26s           | Fastest inference — best overall |

## Recommendations

- **CUDA**: `bfloat16` — fastest inference (0.26s)
- **CPU**: `float32` — only usable option (1.76s vs 35-37s for fp16/bf16)
- **Avoid on CPU**: `bfloat16`, `float16` (massive slowdown from dtype conversion)
- **Avoid on CUDA**: `float16`, `float32` (3-4x slower inference than bf16)

---

# canary-1b-v2 Benchmarks

Model: `nvidia/canary-1b-v2`
Audio: `test_audio_data/test.mp3` (16kHz mono float32)

## CPU

| Precision | Load Time | Transcribe Time | Verdict                       |
| --------- | --------- | --------------- | ----------------------------- |
| float32   | 29.97s    | 10.2s           | Fastest inference             |
| float16   | 31.45s    | 62.4s           | 6x slower inference — avoid   |
| bfloat16  | 29.64s    | 66.3s           | 6.5x slower inference — avoid |

## CUDA

| Precision | Load Time | Transcribe Time | Verdict                   |
| --------- | --------- | --------------- | ------------------------- |
| float32   | 30.35s    | 1.78s           | Solid                     |
| float16   | 26.56s    | 1.23s           | Fastest inference on CUDA |
| bfloat16  | 28.69s    | 1.97s           | Similar to float32        |

## Recommendations

- **CUDA**: `float16` — fastest inference (1.23s)
- **CPU**: `float32` — only usable option (10.2s vs 62-66s for fp16/bf16)
- **Avoid on CPU**: `bfloat16`, `float16` (massive slowdown from dtype conversion)
- **Avoid on CUDA**: `bfloat16`, `float32` (slower inference than float16)

---

# cohere-transcribe-03-2026 Benchmarks

Model: `CohereLabs/cohere-transcribe-03-2026`
Audio: `test_audio_data/test.mp3` (16kHz mono float32)

## CPU

| Precision | Load Time | Transcribe Time | Verdict                                 |
| --------- | --------- | --------------- | --------------------------------------- |
| float32   | (N/A)     | (N/A)           | Not tested (native default, ~8x faster) |
| float16   | 13.36s    | 85.88s          | Slightly faster inference               |
| bfloat16  | 26.82s    | 95.51s          | Slowest inference                       |
| int8      | (N/A)     | (N/A)           | Not tested                              |
| int4      | (N/A)     | (N/A)           | Not tested                              |

## CUDA

| Precision | Load Time | Transcribe Time | Verdict                                 |
| --------- | --------- | --------------- | --------------------------------------- |
| float32   | (N/A)     | (N/A)           | Not tested                              |
| float16   | 5.13s     | 1.03s           | Fastest load, slightly slower inference |
| bfloat16  | 13.69s    | 0.68s           | Fastest inference                       |
| int8      | (N/A)     | (N/A)           | Not tested                              |
| int4      | (N/A)     | (N/A)           | Not tested                              |

## Recommendations

- **CUDA**: `bfloat16` — fastest inference (0.68s); `float16` if load time matters
- **CPU**: `float32` (native default) — ~8x faster than tested bf16/fp16 overrides
- **Avoid on CPU**: `bfloat16`, `float16` (massive slowdown from dtype override)
- **Avoid on CUDA**: `float16` (slower inference than bf16)

---

# granite-speech-4.1-2b-nar Benchmarks

Model: `ibm-granite/granite-speech-4.1-2b-nar`
Audio: `test_audio_data/test.mp3` (16kHz mono float32)

## CPU

| Precision | Load Time | Transcribe Time | Verdict                         |
| --------- | --------- | --------------- | ------------------------------- |
| float32   | (N/A)     | (N/A)           | Not tested                      |
| float16   | 16.94s    | 130.97s         | ~8% faster inference            |
| bfloat16  | 8.2s      | 141.81s         | Fastest load, slowest inference |
| int8      | (N/A)     | (N/A)           | Not tested                      |
| int4      | (N/A)     | (N/A)           | Not tested                      |

## CUDA

| Precision | Load Time | Transcribe Time | Verdict                                  |
| --------- | --------- | --------------- | ---------------------------------------- |
| float32   | (N/A)     | (N/A)           | Not tested                               |
| float16   | 7.08s     | 0.26s           | Fastest load, nearly identical inference |
| bfloat16  | 15.55s    | 0.24s           | Fastest inference                        |
| int8      | (N/A)     | (N/A)           | Not tested                               |
| int4      | (N/A)     | (N/A)           | Not tested                               |

## Recommendations

- **CUDA**: `bfloat16` or `float16` — nearly identical (0.24s vs 0.26s)
- **CPU**: `float16` — marginally faster; both are very slow (130-142s)
- **Avoid on CPU**: `bfloat16` (slowest inference)
- **Note**: Non-AR architecture; no punctuation or capitalization in output

---

# granite-speech-4.1-2b Benchmarks

Model: `ibm-granite/granite-speech-4.1-2b`
Audio: `test_audio_data/test.mp3` (16kHz mono float32)

## CPU

| Precision | Load Time | Transcribe Time | Verdict                         |
| --------- | --------- | --------------- | ------------------------------- |
| float32   | (N/A)     | (N/A)           | Not tested                      |
| float16   | 16.01s    | 136.19s         | ~9% faster inference            |
| bfloat16  | 7.09s     | 149.85s         | Fastest load, slowest inference |
| int8      | (N/A)     | (N/A)           | Not tested                      |
| int4      | (N/A)     | (N/A)           | Not tested                      |

## CUDA

| Precision | Load Time | Transcribe Time | Verdict                           |
| --------- | --------- | --------------- | --------------------------------- |
| float32   | (N/A)     | (N/A)           | Not tested                        |
| float16   | 6.3s      | 3.03s           | Fastest load, identical inference |
| bfloat16  | 14.6s     | 3.03s           | Identical inference to float16    |
| int8      | (N/A)     | (N/A)           | Not tested                        |
| int4      | (N/A)     | (N/A)           | Not tested                        |

## Recommendations

- **CUDA**: `float16` — fastest load (6.3s); inference identical to bf16 at 3.03s
- **CPU**: `float16` — marginally faster; both are very slow (136-150s)
- **Avoid on CPU**: `bfloat16` (slowest inference)
- **Note**: AR variant with punctuation and capitalization (vs NAR without)
