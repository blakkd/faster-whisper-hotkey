"""Dynamic model catalog for the transcribe.cpp engine.

transcribe.cpp is a pure GGUF *runner*: it has no model-discovery API, so the
set of downloadable checkpoints lives entirely on the Hugging Face hub, where
every verified GGUF is a public ``handy-computer/*-gguf`` repository. This module
turns that hub into the app's model list:

* :func:`list_models` returns the ASR repositories (name + GGUF files) live from
  the hub, cached on disk and refreshed on a TTL, with a bundled snapshot as the
  offline fallback so the app still lists models with no network.
* :func:`files_for` returns one repository's real GGUF files (with sizes) for
  the precision step — so quantization choices are whatever the repo actually
  publishes, never a hardcoded table.

Per-model *capabilities* (supported languages, translation targets, length
limits) are not in the catalog: those are read from the loaded GGUF by
``models.ModelWrapper`` (see the engine's ``capabilities``). This module only
answers "which models exist and which files can I download".
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from importlib.resources import files as _res_files

import requests

logger = logging.getLogger(__name__)

AUTHOR = "handy-computer"
DEFAULT_ENDPOINT = "https://huggingface.co"

# Hub listings are cached here and reused until they age out; the bundled
# snapshot is only read when there is no usable cache (first run offline).
CACHE_DIR = os.path.expanduser("~/.cache/faster_whisper_hotkey")
CACHE_FILE = os.path.join(CACHE_DIR, "catalog.json")
SNAPSHOT_RESOURCE = "assets/model_catalog.json"
CACHE_TTL_S = 3 * 24 * 3600  # 3 days

HTTP_TIMEOUT = (10, 30)

# Repositories that are engine *support* / diarization models, not transcribable
# speech checkpoints. VAD and lang-id are loaded internally by constant; the
# diarization model has no plain ASR session. Excluded from the picker.
_EXCLUDED_PREFIXES = ("silero-vad", "lang-id-", "diar_")
_EXCLUDED_SUBSTRINGS = ("sortformer",)

_QUANT_LABELS = {
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "Q8_0": "int8",
    "Q4_K_M": "int4",
    # Q6_K / Q5_K_M have no friendly name; shown verbatim.
}


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GGUFFile:
    """One downloadable GGUF (a quantization of a model)."""

    filename: str
    quant: str  # quant label parsed from the file name, e.g. "Q8_0"
    size_bytes: int | None = None

    @property
    def label(self) -> str:
        """Friendly precision label for the menu ("int8" / "bfloat16" / "Q5_K_M")."""
        return _QUANT_LABELS.get(self.quant, self.quant)

    @property
    def size_mb(self) -> float | None:
        return None if self.size_bytes is None else self.size_bytes / 1e6


@dataclass(frozen=True)
class ModelEntry:
    """A model repository and the GGUF files it publishes."""

    repo_id: str  # e.g. "handy-computer/parakeet-ultra-gguf"
    name: str  # short display name, e.g. "parakeet-ultra"
    files: tuple[GGUFFile, ...]

    def pick(self, filename: str) -> GGUFFile | None:
        for f in self.files:
            if f.filename == filename:
                return f
        return None


def short_name(repo_id: str) -> str:
    """Repository id -> display name (``handy-computer/parakeet-ultra-gguf`` -> ``parakeet-ultra``)."""
    return repo_id.split("/")[-1].removesuffix("-gguf")


def quant_of(filename: str) -> str:
    """Parse the quant label from ``<stem>-<QUANT>.gguf`` (e.g. ``parakeet-ultra-Q8_0.gguf`` -> ``Q8_0``)."""
    stem = filename.removesuffix(".gguf")
    return stem.rsplit("-", 1)[-1] if "-" in stem else ""


def _is_asr_repo(repo_id: str) -> bool:
    name = short_name(repo_id)
    if name.startswith(_EXCLUDED_PREFIXES):
        return False
    return not any(sub in name for sub in _EXCLUDED_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Hub fetch
# ---------------------------------------------------------------------------


def _parse_repos(payload: list[dict]) -> list[dict]:
    """Normalize a HF ``/api/models`` listing into ``[{id, files:[name,...]}]`` (ASR only, sorted)."""
    repos: list[dict] = []
    for m in payload:
        rid = m.get("id") or m.get("modelId") or ""
        if not rid.startswith(f"{AUTHOR}/") or not rid.endswith("-gguf"):
            continue
        if not _is_asr_repo(rid):
            continue
        ggufs = sorted(s["rfilename"] for s in m.get("siblings", []) if s.get("rfilename", "").endswith(".gguf"))
        if not ggufs:
            continue
        repos.append({"id": rid, "files": ggufs})
    repos.sort(key=lambda r: short_name(r["id"]).lower())
    return repos


def _fetch_listing() -> list[dict]:
    """One call to the HF hub. Returns the normalized repos; raises on network/HTTP error."""
    url = f"{hf_endpoint()}/api/models"
    params = {"author": AUTHOR, "full": "true", "limit": "100"}
    resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT, headers={"User-Agent": "faster-whisper-hotkey"})
    resp.raise_for_status()
    return _parse_repos(resp.json())


# ---------------------------------------------------------------------------
# Cache + bundled snapshot
# ---------------------------------------------------------------------------


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("repos"), list) else None


def _load_cache() -> list[dict] | None:
    data = _read_json(CACHE_FILE)
    if data is None:
        return None
    if time.time() - float(data.get("fetched_at", 0)) > CACHE_TTL_S:
        return None
    return data["repos"]


def _load_snapshot() -> list[dict]:
    try:
        text = _res_files("faster_whisper_hotkey").joinpath(SNAPSHOT_RESOURCE).read_text(encoding="utf-8")
        data = json.loads(text)
        return data.get("repos", [])
    except (FileNotFoundError, ValueError) as exc:  # pragma: no cover - packaging error only
        logger.error(f"Bundled model snapshot unreadable: {exc}")
        return []


def _write_cache(repos: list[dict]) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE + ".tmp", "w", encoding="utf-8") as fh:
            json.dump({"fetched_at": time.time(), "repos": repos}, fh)
        os.replace(CACHE_FILE + ".tmp", CACHE_FILE)
    except OSError as exc:
        logger.debug(f"Could not write model catalog cache: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models(refresh: bool = False) -> list[ModelEntry]:
    """The listable ASR models: name + GGUF files (no sizes).

    Tries the hub when refreshing or when the cache has expired; otherwise uses
    the cache; if neither is usable (offline first run) falls back to the bundled
    snapshot. Never raises — a hub outage degrades to cache/snapshot.
    """
    repos: list[dict] | None = None
    if not refresh:
        repos = _load_cache()
    if repos is None:
        try:
            repos = _fetch_listing()
            _write_cache(repos)
        except (requests.RequestException, OSError, ValueError) as exc:
            logger.warning(f"Could not refresh the model catalog from the hub ({exc}); using a cached list")
            repos = _load_cache() or _load_snapshot()

    entries: list[ModelEntry] = []
    for r in repos:
        if not _is_asr_repo(r["id"]):  # snapshot/cache may carry support repos the hub parse already dropped
            continue
        files = tuple(GGUFFile(fn, quant_of(fn)) for fn in r["files"])
        entries.append(ModelEntry(repo_id=r["id"], name=short_name(r["id"]), files=files))
    return entries


def files_for(repo_id: str) -> list[GGUFFile]:
    """One repository's real GGUF files with sizes, for the precision step.

    Fetches fresh sizes from the hub when possible; on failure returns at least
    the file names from the cached/snapshot listing (sizes ``None``).
    """
    try:
        url = f"{hf_endpoint()}/api/models/{repo_id}"
        resp = requests.get(url, params={"blobs": "true"}, timeout=HTTP_TIMEOUT, headers={"User-Agent": "faster-whisper-hotkey"})
        resp.raise_for_status()
        files = [
            GGUFFile(s["rfilename"], quant_of(s["rfilename"]), s.get("size"))
            for s in resp.json().get("siblings", [])
            if s.get("rfilename", "").endswith(".gguf")
        ]
        if files:
            files.sort(key=lambda f: (f.size_bytes is None, f.size_bytes or 0))
            return files
    except (requests.RequestException, OSError, ValueError) as exc:
        logger.warning(f"Could not list files for {repo_id} from the hub ({exc}); using cached names")

    # Fallback: names from whatever listing we already have (sizes unknown).
    for entry in list_models():
        if entry.repo_id == repo_id:
            return list(entry.files)
    return []


def find_entry(repo_id: str) -> ModelEntry | None:
    for entry in list_models():
        if entry.repo_id == repo_id:
            return entry
    return None


# ---------------------------------------------------------------------------
# Engine support models (loaded internally, not offered in the model picker)
# ---------------------------------------------------------------------------

VAD_REPO = f"{AUTHOR}/silero-vad-v6.2-gguf"
VAD_FILE = "silero-vad-v6.2-F32.gguf"
