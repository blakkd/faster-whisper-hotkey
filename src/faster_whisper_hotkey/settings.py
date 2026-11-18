import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

conf_dir = os.path.expanduser("~/.config")
settings_dir = os.path.join(conf_dir, "faster_whisper_hotkey")
os.makedirs(settings_dir, exist_ok=True)
SETTINGS_FILE = os.path.join(settings_dir, "transcriber_settings.json")


@dataclass
class Settings:
    """A saved configuration.

    A model is addressed by hub ``model_repo`` + ``model_file`` (the exact GGUF,
    quant included), chosen from the live catalog. ``language`` is a transcription
    language code or ``"auto"``/``""``; ``translate_target`` is empty for
    transcription and a language code to translate ``language`` -> it.
    """

    device_name: str
    model_repo: str
    model_file: str
    device: str
    language: str = ""
    translate_target: str = ""
    hotkey: str = "pause"
    vad_enabled: bool = True
    llm_correction_enabled: bool = False
    llm_endpoint: str = ""
    llm_model_name: str = ""
    llm_api_key: str = ""


# ---------------------------------------------------------------------------
# Legacy settings migration
# ---------------------------------------------------------------------------
# Settings written before the hub-catalog migration addressed models by
# (model_type, model_name, compute_type). They are rewritten to (model_repo,
# model_file) on load so old configs keep booting. This table is migration-only —
# the running app never consults it; the model list comes from ``catalog``.
# Each entry maps a canonical legacy model_name -> (hub repo, a GGUF file that
# exists and is the family's former default, so the file is usually already
# cached). The former precision is not preserved (the menu re-derives it).
_LEGACY_MODEL_NAME_MIGRATIONS = {
    # Upstream Hugging Face ids used before the GGUF catalog existed.
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

_LEGACY_REPOS = {
    "tiny": ("handy-computer/whisper-tiny-gguf", "whisper-tiny-F16.gguf"),
    "tiny.en": ("handy-computer/whisper-tiny.en-gguf", "whisper-tiny.en-F16.gguf"),
    "base": ("handy-computer/whisper-base-gguf", "whisper-base-F16.gguf"),
    "base.en": ("handy-computer/whisper-base.en-gguf", "whisper-base.en-F16.gguf"),
    "small": ("handy-computer/whisper-small-gguf", "whisper-small-F16.gguf"),
    "small.en": ("handy-computer/whisper-small.en-gguf", "whisper-small.en-F16.gguf"),
    "medium": ("handy-computer/whisper-medium-gguf", "whisper-medium-F16.gguf"),
    "medium.en": ("handy-computer/whisper-medium.en-gguf", "whisper-medium.en-F16.gguf"),
    "large-v1": ("handy-computer/whisper-large-gguf", "whisper-large-F16.gguf"),
    "large-v2": ("handy-computer/whisper-large-v2-gguf", "whisper-large-v2-F16.gguf"),
    "large-v3": ("handy-computer/whisper-large-v3-gguf", "whisper-large-v3-F16.gguf"),
    "large-v3-turbo": ("handy-computer/whisper-large-v3-turbo-gguf", "whisper-large-v3-turbo-F16.gguf"),
    "parakeet-tdt-0.6b-v3": ("handy-computer/parakeet-tdt-0.6b-v3-gguf", "parakeet-tdt-0.6b-v3-F32.gguf"),
    "parakeet-ultra": ("handy-computer/parakeet-ultra-gguf", "parakeet-ultra-Q8_0.gguf"),
    "canary-1b-v2": ("handy-computer/canary-1b-v2-gguf", "canary-1b-v2-F32.gguf"),
    "voxtral-mini-3b-2507": ("handy-computer/Voxtral-Mini-3B-2507-gguf", "Voxtral-Mini-3B-2507-BF16.gguf"),
    "cohere-transcribe-03-2026": ("handy-computer/cohere-transcribe-03-2026-gguf", "cohere-transcribe-03-2026-BF16.gguf"),
    "granite-speech-4.1-2b": ("handy-computer/granite-speech-4.1-2b-gguf", "granite-speech-4.1-2b-BF16.gguf"),
    "granite-speech-4.1-2b-nar": ("handy-computer/granite-speech-4.1-2b-nar-gguf", "granite-speech-4.1-2b-nar-BF16.gguf"),
    "granite-speech-5.0-470m-turboctc-nc": (
        "handy-computer/granite-speech-5.0-470m-turboctc-nc-gguf",
        "granite-speech-5.0-470m-turboctc-nc-BF16.gguf",
    ),
    "qwen3-asr-1.7b": ("handy-computer/Qwen3-ASR-1.7B-gguf", "Qwen3-ASR-1.7B-BF16.gguf"),
}


def _split_legacy_language(language: str) -> tuple[str, str]:
    """Legacy pair "src-tgt" -> (language, translate_target); same-same is transcription."""
    if language and "-" in language:
        parts = language.split("-")
        if len(parts) == 2 and parts[0] != parts[1]:
            return parts[0], parts[1]
        if len(parts) == 2:
            return parts[0], ""
    return language, ""


def _migrate_legacy(data: dict) -> dict | None:
    """Rewrite a pre-catalog settings dict into the current schema, or None if unrecognizable."""
    name = str(data.get("model_name", "") or "")
    canonical = _LEGACY_MODEL_NAME_MIGRATIONS.get(name, name)
    target = _LEGACY_REPOS.get(canonical)
    if not target:
        logger.error(f"Saved model {name!r} has no known hub model; reconfigure to pick a model")
        return None
    repo, filename = target
    language, translate_target = _split_legacy_language(str(data.get("language", "") or ""))
    migrated = {
        "device_name": data.get("device_name", ""),
        "model_repo": repo,
        "model_file": filename,
        "device": data.get("device", "cpu"),
        "language": language,
        "translate_target": translate_target,
        "hotkey": data.get("hotkey", "pause"),
        "vad_enabled": data.get("vad_enabled", True),
        "llm_correction_enabled": data.get("llm_correction_enabled", False),
        "llm_endpoint": data.get("llm_endpoint", ""),
        "llm_model_name": data.get("llm_model_name", ""),
        "llm_api_key": data.get("llm_api_key", ""),
    }
    logger.info(f"Migrating saved model {name!r} to {repo}/{filename}")
    return migrated


_FIELDS = {f for f in Settings.__dataclass_fields__}


def _normalize(data: dict) -> dict | None:
    """Coerce any saved dict to current-schema keyword args (migrating legacy files)."""
    if "model_repo" not in data:
        if "model_name" not in data:
            logger.error("Settings file has no model information; reconfigure")
            return None
        migrated = _migrate_legacy(data)
        if migrated is None:
            return None
        data = migrated
    return {k: v for k, v in data.items() if k in _FIELDS}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_settings(settings: dict, settings_file: str | None = None):
    filepath = settings_file or SETTINGS_FILE
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(settings, f)
    except OSError as e:
        logger.error(f"Failed to save settings: {e}")


def load_settings(settings_file: str | None = None) -> Settings | None:
    filepath = settings_file or SETTINGS_FILE
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted settings file: {e}")
        return None

    fields = _normalize(data)
    if fields is None:
        return None
    try:
        return Settings(**fields)
    except TypeError as e:  # missing required field (e.g. empty new-schema file)
        logger.error(f"Settings file is missing required fields: {e}")
        return None
