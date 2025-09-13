import json
import logging
from importlib.resources import files

logger = logging.getLogger(__name__)


def get_resource_path(filename: str) -> str:
    """
    Return a filesystem path to a packaged resource inside this package.
    """
    return files("faster_whisper_hotkey").joinpath(filename).as_posix()


try:
    config_path = get_resource_path("available_models_languages.json")
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.error(
        f"Configuration error while loading available_models_languages.json: {e}"
    )
    raise

accepted_models_whisper = _CONFIG.get("accepted_models_whisper", [])
accepted_languages_whisper = _CONFIG.get("accepted_languages_whisper", [])
english_only_models_whisper = set(_CONFIG.get("english_only_models_whisper", []))

canary_source_target_languages = _CONFIG.get(
    "canary_source_target_languages", ["en", "fr", "de", "es"]
)
canary_allowed_language_pairs = _CONFIG.get("canary_allowed_language_pairs", [])
