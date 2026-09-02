"""Fallback when the Hugging Face hub servers cannot be reached.

Before loading a model, the hub is probed once. If it cannot be reached,
huggingface_hub's offline mode is enabled so that every loader
(transformers, faster-whisper, NeMo) goes straight to the local cache
instead of retrying the network, which would otherwise hang the app.
"""

import logging
import os

import requests
from huggingface_hub import constants as hf_constants

logger = logging.getLogger(__name__)

PROBE_TIMEOUT = 5.0


def hf_hub_endpoint() -> str:
    """Base URL of the Hugging Face hub (honors the HF_ENDPOINT environment variable)."""
    return hf_constants.ENDPOINT.rstrip("/")


def hf_hub_reachable(timeout: float = PROBE_TIMEOUT) -> bool:
    """Check whether the Hugging Face hub answers (a 5xx server error counts as unreachable)."""
    endpoint = hf_hub_endpoint()
    try:
        response = requests.head(endpoint, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        logger.debug(f"Could not reach {endpoint}: {exc}")
        return False
    if response.status_code >= 500:
        logger.debug(f"{endpoint} replied {response.status_code}")
        return False
    return True


def enable_offline_if_unreachable(model_name: str | None = None, timeout: float = PROBE_TIMEOUT) -> bool:
    """Enable Hugging Face offline mode when the hub cannot be reached.

    Returns True if offline mode was enabled by this call, False if the hub is
    reachable or offline mode was already active. Local model directories are
    probed neither, since they need no hub access at all.
    """
    if hf_constants.is_offline_mode():
        logger.debug("Hugging Face offline mode is already enabled, skipping reachability probe")
        return False

    if model_name and os.path.isdir(model_name):
        return False

    if hf_hub_reachable(timeout):
        return False

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    hf_constants.HF_HUB_OFFLINE = True

    suffix = f" Loading '{model_name}'" if model_name else " Loading"
    logger.warning(
        f"Could not reach the Hugging Face hub at {hf_hub_endpoint()}.{suffix} models will be "
        "taken from the local cache only. Models never downloaded while online cannot be loaded."
    )
    return True
