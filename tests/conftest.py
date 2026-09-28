"""Pytest configuration for integration tests."""

import os
import shutil
import subprocess
import time

import pytest


@pytest.fixture(autouse=True)
def _hf_hub_probe_reachable(monkeypatch):
    """Keep the Hugging Face reachability probe from making real network calls in tests."""
    monkeypatch.setattr("faster_whisper_hotkey.hf_offline.hf_hub_reachable", lambda *args, **kwargs: True)


# ---------------------------------------------------------------------------
# Per-class HF model cache cleanup (small-disk runners)
# ---------------------------------------------------------------------------
# GitHub-hosted runners have 14 GB of disk and the matrix downloads ~30 GB of
# models in total, so each tested model is deleted before the next one is
# downloaded. Off by default so local runs keep their cache; the CI workflow
# sets FWH_CLEANUP_HF_CACHE=1. Disk usage is logged to
# test_audio_data/disk_usage_log.txt at every class boundary.

HF_HUB_DIR = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
DISK_LOG_FILE = "test_audio_data/disk_usage_log.txt"


def _disk_state(label: str) -> None:
    try:
        df = subprocess.run(["df", "-h", "/"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[-1]
    except Exception:
        df = "df failed"
    sizes = []
    for path in (HF_HUB_DIR, os.path.expanduser("~/.cache/uv")):
        if os.path.isdir(path):
            du = subprocess.run(["du", "-sh", path], capture_output=True, text=True, timeout=60)
            if du.returncode == 0:
                sizes.append(f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}={du.stdout.split()[0]}")
    with open(DISK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {label} | {df} | " + " ".join(sizes) + "\n")


@pytest.fixture(autouse=True, scope="session")
def _disk_log_file():
    os.makedirs(os.path.dirname(DISK_LOG_FILE), exist_ok=True)
    with open(DISK_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("disk usage during the run (df -h /, du of caches)\n")
    yield


_last_disk_class = None


@pytest.fixture(autouse=True)
def _cleanup_hf_models_after_class(request):
    """Log disk usage at each test class boundary; with FWH_CLEANUP_HF_CACHE=1,
    delete the HF model cache after each test (each class of the matrix file is
    a single test that loads exactly one model) so the next model fits on disk."""
    global _last_disk_class
    cls = request.node.cls
    if cls is None:
        yield
        return
    if _last_disk_class is not cls:
        _disk_state(f"before {cls.__name__}")
    yield
    _last_disk_class = cls
    if os.environ.get("FWH_CLEANUP_HF_CACHE") != "1":
        return
    shutil.rmtree(HF_HUB_DIR, ignore_errors=True)
    _disk_state(f"after cleanup of {cls.__name__}")


def pytest_addoption(parser):
    parser.addoption(
        "--force-cuda",
        action="store_true",
        default=False,
        help="Force CUDA configs even when no GPU is available",
    )
    parser.addoption(
        "--cuda-only",
        action="store_true",
        default=False,
        help="Skip CPU configs (run CUDA configs only)",
    )
