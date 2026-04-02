import json
import logging
from importlib.resources import files

logger = logging.getLogger(__name__)


def get_resource_path(filename: str) -> str:
    """
    Return a filesystem path to a packaged resource inside this package.
    """
    return str(files("faster_whisper_hotkey").joinpath(filename))


try:
    config_path = get_resource_path("available_languages.json")
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(f"Configuration error while loading available_languages.json: {e}")
    raise

accepted_models_whisper = _CONFIG.get("accepted_models_whisper", [])
accepted_languages_whisper = _CONFIG.get("accepted_languages_whisper", [])
english_only_models_whisper = set(_CONFIG.get("english_only_models_whisper", []))

canary_source_target_languages = _CONFIG["canary_source_target_languages"]
canary_allowed_language_pairs = _CONFIG["canary_allowed_language_pairs"]
accepted_languages_cohere = _CONFIG["accepted_languages_cohere"]
