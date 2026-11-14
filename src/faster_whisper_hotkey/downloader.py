"""Single-file GGUF downloads from the Hugging Face hub.

Models are cached under ``~/.cache/faster_whisper_hotkey/models/`` by file
name. A cached file is used without touching the network (the app works
offline once a model has been downloaded once). Downloads stream to a
``.part`` file, resume across runs when the server supports ranges, and are
renamed into place only after a complete transfer, so a torn download can
never masquerade as a usable model.
"""

import logging
import os
from pathlib import Path

import requests
from requests.exceptions import ContentDecodingError

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.path.expanduser("~/.cache/faster_whisper_hotkey/models"))

# HF endpoint override (same convention huggingface_hub uses), so mirrors and
# self-hosted instances work.
DEFAULT_ENDPOINT = "https://huggingface.co"

CHUNK = 1 << 20
TIMEOUT = (10, 60)  # (connect, read) seconds
MAX_ATTEMPTS = 3


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


def model_path(repo: str, filename: str) -> Path:
    return CACHE_DIR / filename


def _download(repo: str, filename: str, dest: Path) -> None:
    url = f"{hf_endpoint()}/{repo}/resolve/main/{filename}"
    part = dest.with_name(dest.name + ".part")
    resume = part.stat().st_size if part.exists() else 0

    headers = {"User-Agent": "faster-whisper-hotkey"}
    if resume:
        headers["Range"] = f"bytes={resume}-"

    with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers, allow_redirects=True) as response:
        if response.status_code == 416:
            # The .part file is at or past the server's file size: a corrupt/oversized
            # partial. Drop it and let the next attempt start from zero.
            part.unlink(missing_ok=True)
            raise ContentDecodingError(f"stale partial file for {filename} (HTTP 416)")

        response.raise_for_status()

        if response.status_code == 206 and part.exists():
            mode, done = "ab", part.stat().st_size
            logger.info(f"Resuming {filename} at {done / 1e6:.1f} MB")
        else:  # full body (server ignored Range or no partial existed)
            mode, done = "wb", 0

        length = response.headers.get("Content-Length")
        total_bytes = done + int(length) if length else None

        mb = 0
        with open(part, mode) as fh:
            for chunk in response.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if done // 20_971_520 > mb:  # every 20 MB
                    mb = done // 20_971_520
                    of = f"/{total_bytes / 1e6:.0f} MB" if total_bytes else ""
                    logger.info(f"Downloading {filename}: {done / 1e6:.0f} MB{of}")
                if total_bytes and done > total_bytes:
                    raise ContentDecodingError(f"server sent more than {total_bytes} bytes for {filename}")

    part.replace(dest)


def ensure_gguf(repo: str, filename: str) -> Path:
    """Return a local path for the file, downloading it if not cached.

    Raises requests.RequestException (after MAX_ATTEMPTS tries) when the file
    is missing locally and the hub cannot deliver it.
    """
    dest = model_path(repo, filename)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {filename} from {repo} to {CACHE_DIR} (first use only)")

    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _download(repo, filename, dest)
            logger.info(f"Downloaded {filename} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            logger.warning(f"Download attempt {attempt}/{MAX_ATTEMPTS} for {filename} failed: {exc}")
    raise RuntimeError(
        f"Could not download {repo}/{filename} ({last_exc}). The model is not in the local "
        f"cache ({CACHE_DIR}) and the Hugging Face hub could not deliver it."
    ) from last_exc
