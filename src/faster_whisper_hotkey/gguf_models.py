"""Static catalog of the GGUF models this app runs on the transcribe.cpp engine.

Every entry mirrors `catalog/<variant>.json` of the transcribe.cpp project
(https://github.com/handy-computer/transcribe.cpp): the published Hugging Face
repository and the per-quantization file name. The GGUF files carry all model
metadata (language tables, length limits), so nothing here is needed at
inference time beyond locating the right file.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GgufVariant:
    variant: str
    hf_repo: str
    # Settings compute_type label -> GGUF quantization label.
    quants: dict[str, str]
    default_quant: str
    # Filename stem inside the repository ("<stem>-<quant>.gguf"); defaults to
    # the variant id. Only the v1 checkpoint differs (variant "whisper-large-v1"
    # is published as plain "whisper-large").
    stem: str = ""

    @property
    def filename_stem(self) -> str:
        return self.stem or self.variant


_V = GgufVariant

# compute_type labels are the ones the settings file and the TUI precision
# screens use ("float32", "float16", "bfloat16", "int8", "int4", "" = default).
# Every label must map to a quant the family actually publishes:
#   whisper:  F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M (no float files beyond F16)
#   NeMo ports (parakeet/canary): F32, F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M (no BF16)
#   LLM-head ports (voxtral/cohere/granite*/qwen3): BF16, F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M (no F32)
_WHISPER = {"float32": "F16", "float16": "F16", "bfloat16": "F16", "int8": "Q8_0", "int4": "Q4_K_M"}
_NEMO = {"float32": "F32", "float16": "F16", "bfloat16": "F16", "int8": "Q8_0", "int4": "Q4_K_M"}
_LLM = {"float32": "BF16", "float16": "F16", "bfloat16": "BF16", "int8": "Q8_0", "int4": "Q4_K_M"}

VARIANTS: dict[str, GgufVariant] = {
    v.variant: v
    for v in [
        _V("whisper-tiny", "handy-computer/whisper-tiny-gguf", _WHISPER, "float16"),
        _V("whisper-tiny.en", "handy-computer/whisper-tiny.en-gguf", _WHISPER, "float16"),
        _V("whisper-base", "handy-computer/whisper-base-gguf", _WHISPER, "float16"),
        _V("whisper-base.en", "handy-computer/whisper-base.en-gguf", _WHISPER, "float16"),
        _V("whisper-small", "handy-computer/whisper-small-gguf", _WHISPER, "float16"),
        _V("whisper-small.en", "handy-computer/whisper-small.en-gguf", _WHISPER, "float16"),
        _V("whisper-medium", "handy-computer/whisper-medium-gguf", _WHISPER, "float16"),
        _V("whisper-medium.en", "handy-computer/whisper-medium.en-gguf", _WHISPER, "float16"),
        _V("whisper-large-v1", "handy-computer/whisper-large-gguf", _WHISPER, "float16", stem="whisper-large"),
        _V("whisper-large-v2", "handy-computer/whisper-large-v2-gguf", _WHISPER, "float16"),
        _V("whisper-large-v3", "handy-computer/whisper-large-v3-gguf", _WHISPER, "float16"),
        _V("whisper-large-v3-turbo", "handy-computer/whisper-large-v3-turbo-gguf", _WHISPER, "float16"),
        _V("parakeet-ultra", "handy-computer/parakeet-ultra-gguf", _NEMO, "int8"),
        _V("parakeet-tdt-0.6b-v3", "handy-computer/parakeet-tdt-0.6b-v3-gguf", _NEMO, "float32"),
        _V("canary-1b-v2", "handy-computer/canary-1b-v2-gguf", _NEMO, "float32"),
        _V("voxtral-mini-3b-2507", "handy-computer/Voxtral-Mini-3B-2507-gguf", _LLM, "bfloat16", stem="Voxtral-Mini-3B-2507"),
        _V("cohere-transcribe-03-2026", "handy-computer/cohere-transcribe-03-2026-gguf", _LLM, "bfloat16"),
        _V("granite-speech-4.1-2b", "handy-computer/granite-speech-4.1-2b-gguf", _LLM, "bfloat16"),
        _V("granite-speech-4.1-2b-nar", "handy-computer/granite-speech-4.1-2b-nar-gguf", _LLM, "bfloat16"),
        _V(
            "granite-speech-5.0-470m-turboctc-nc",
            "handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf",
            _LLM,
            "bfloat16",
        ),
        _V("qwen3-asr-1.7b", "handy-computer/Qwen3-ASR-1.7B-gguf", _LLM, "bfloat16", stem="Qwen3-ASR-1.7B"),
        # Support models (no precision choice; tiny single-file downloads).
        _V("silero-vad-v6.2", "handy-computer/silero-vad-v6.2-gguf", {"": "F32"}, ""),
        _V("lang-id-voxlingua107-ecapa", "handy-computer/lang-id-voxlingua107-ecapa-gguf", {"int8": "Q8_0"}, "int8"),
    ]
}

# ---------------------------------------------------------------------------
# model_type + model_name -> variant
# ---------------------------------------------------------------------------

# model_type is the stable family id from the settings file; it is unchanged
# from the previous (multi-engine) implementation.


def variant_for(model_type: str, model_name: str) -> str:
    """Resolve settings (model_type, model_name) to a VARIANTS key.

    Whisper keeps its historical short names ("large-v3", ...); every other
    family's model_name IS the variant id.
    """
    if model_type == "whisper":
        if model_name == "large-v1":
            # The GGUF catalog ships the v1 checkpoint as plain "whisper-large".
            return "whisper-large-v1"
        variant = f"whisper-{model_name}"
        if variant not in VARIANTS:
            raise ValueError(f"Unknown whisper model: {model_name!r}")
        return variant
    if model_name not in VARIANTS:
        raise ValueError(f"Unknown model: {model_type}/{model_name!r}")
    return model_name


# Settings files written before the transcribe.cpp migration stored upstream
# Hugging Face repository ids. Maps old model_name -> new model_name.
# (Whisper short names are unchanged; the distil-* variants are dropped — the
# GGUF catalog has no distil checkpoints.)
MODEL_NAME_MIGRATIONS = {
    "Systran/faster-whisper-large-v3": "large-v3",
    "distil-large-v3": "large-v3",
    "distil-large-v2": "large-v2",
    "distil-medium.en": "medium.en",
    "distil-small.en": "small.en",
    "moondream/parakeet-ultra": "parakeet-ultra",
    "nvidia/parakeet-tdt-0.6b-v3": "parakeet-tdt-0.6b-v3",
    "nvidia/canary-1b-v2": "canary-1b-v2",
    "mistralai/Voxtral-Mini-3B-2507": "voxtral-mini-3b-2507",
    "CohereLabs/cohere-transcribe-03-2026": "cohere-transcribe-03-2026",
    "ibm-granite/granite-speech-4.1-2b": "granite-speech-4.1-2b",
    "ibm-granite/granite-speech-4.1-2b-nar": "granite-speech-4.1-2b-nar",
    "ibm-granite/granite-speech-5.0-470m-turboctc-nc": "granite-speech-5.0-470m-turboctc-nc",
    "Qwen/Qwen3-ASR-1.7B-hf": "qwen3-asr-1.7b",
}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def resolve_quant(variant: GgufVariant, compute_type: str | None) -> str:
    """Settings compute_type -> GGUF quant label for this variant.

    Unknown or empty compute types fall back to the variant default; labels
    the variant does not publish (e.g. "bfloat16" for the NeMo ports, which
    ship F32/F16 instead) resolve to their closest available quant.
    """
    key = (compute_type or "").strip()
    if key == "":
        key = variant.default_quant
    return variant.quants.get(key, variant.quants[variant.default_quant])


def gguf_filename(variant: GgufVariant, quant: str) -> str:
    return f"{variant.filename_stem}-{quant}.gguf" if quant else f"{variant.filename_stem}.gguf"


def repo_and_filename(model_type: str, model_name: str, compute_type: str | None):
    """Resolve settings to (hf_repo, filename, variant, quant)."""
    variant_id = variant_for(model_type, model_name)
    variant = VARIANTS[variant_id]
    quant = resolve_quant(variant, compute_type)
    return variant.hf_repo, gguf_filename(variant, quant), variant, quant
