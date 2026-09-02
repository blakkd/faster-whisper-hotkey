"""Pytest configuration for integration tests."""

import pytest


@pytest.fixture(autouse=True)
def _hf_hub_probe_reachable(monkeypatch):
    """Keep the Hugging Face reachability probe from making real network calls in tests."""
    monkeypatch.setattr("faster_whisper_hotkey.hf_offline.hf_hub_reachable", lambda *args, **kwargs: True)


def pytest_addoption(parser):
    parser.addoption(
        "--force-cuda",
        action="store_true",
        default=False,
        help="Force CUDA configs even when no GPU is available",
    )
