"""Pytest configuration for integration tests."""

import os
import subprocess
import time

import pytest

# ---------------------------------------------------------------------------
# Disk telemetry
# ---------------------------------------------------------------------------
# The CPU matrix now runs one model per job (own runner), so disk is never a
# constraint (runners have ~145 GB). The [disk] line per test class is kept as
# cheap forensics: it streams into the CI log with -s and is written to
# test_audio_data/disk_usage_log.txt, so a future disk/OOM issue is visible.

GGUF_CACHE_DIR = os.path.expanduser("~/.cache/faster_whisper_hotkey/models")
DISK_LOG_FILE = "test_audio_data/disk_usage_log.txt"


def _disk_state(label: str) -> None:
    try:
        df = (
            subprocess.run(["df", "-h", "/"], capture_output=True, text=True, timeout=10, check=False)
            .stdout.strip()
            .splitlines()[-1]
        )
    except (OSError, subprocess.SubprocessError, IndexError):
        df = "df failed"
    try:
        free = (
            subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=10, check=False)
            .stdout.strip()
            .splitlines()[0]
        )
    except (OSError, subprocess.SubprocessError, IndexError):
        free = "free failed"
    sizes = []
    for path in (GGUF_CACHE_DIR, os.path.expanduser("~/.cache/uv")):
        if os.path.isdir(path):
            try:
                du = subprocess.run(["du", "-sh", path], capture_output=True, text=True, timeout=60, check=False)
            except (OSError, subprocess.SubprocessError):
                continue
            if du.returncode == 0:
                sizes.append(
                    f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}={du.stdout.split()[0]}"
                )
    line = f"{time.strftime('%H:%M:%S')} {label} | {df} | {free} | " + " ".join(sizes)
    with open(DISK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"[disk] {line}", flush=True)


@pytest.fixture(autouse=True, scope="session")
def _disk_log_file():
    os.makedirs(os.path.dirname(DISK_LOG_FILE), exist_ok=True)
    with open(DISK_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("disk usage during the run (df -h /, du of caches)\n")
    yield


@pytest.fixture(autouse=True)
def _log_disk_before_class(request):
    """Log disk usage before each test class (one [disk] line per class)."""
    cls = request.node.cls
    if cls is None:
        yield
        return
    _disk_state(f"before {cls.__name__}")
    yield


def pytest_addoption(parser):
    parser.addoption(
        "--force-cuda",
        action="store_true",
        default=False,
        help="Force CUDA configs even when no GPU is available",
    )
    parser.addoption(
        "--force-vulkan",
        action="store_true",
        default=False,
        help="Force Vulkan configs even when no GPU is available",
    )
    parser.addoption(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Skip GPU configs (run CPU configs only)",
    )
    parser.addoption(
        "--cuda-only",
        action="store_true",
        default=False,
        help="Skip CPU and Vulkan configs (run CUDA configs only)",
    )
    parser.addoption(
        "--vulkan-only",
        action="store_true",
        default=False,
        help="Skip CPU and CUDA configs (run Vulkan configs only)",
    )
    parser.addoption(
        "--gpu-only",
        action="store_true",
        default=False,
        help="Skip CPU configs (run all GPU backends that are available)",
    )
