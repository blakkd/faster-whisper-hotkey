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
    device_name: str
    model_type: str
    model_name: str
    compute_type: str
    device: str
    language: str
    hotkey: str = "pause"
    llm_correction_enabled: bool = False
    llm_endpoint: str = ""
    llm_model_name: str = ""


def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f)
    except IOError as e:
        logger.error(f"Failed to save settings: {e}")


def load_settings() -> Settings | None:
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("hotkey", "pause")
            data.setdefault("model_type", "whisper")
            data.setdefault("model_name", "large-v3")
            data.setdefault("llm_correction_enabled", False)
            data.setdefault("llm_endpoint", "")
            data.setdefault("llm_model_name", "")
            return Settings(**data)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted settings file: {e}")
        return None
