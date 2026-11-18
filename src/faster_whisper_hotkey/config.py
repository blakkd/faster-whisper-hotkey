import logging
from importlib.resources import files

logger = logging.getLogger(__name__)


def get_resource_path(filename: str) -> str:
    """
    Return a filesystem path to a packaged resource inside this package.
    """
    return str(files("faster_whisper_hotkey").joinpath(filename))
