"""Tests for hf_offline.py (hub reachability probe and offline fallback)."""

import os
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import constants as hf_constants

from faster_whisper_hotkey import hf_offline

_real_hf_hub_reachable = hf_offline.hf_hub_reachable


@pytest.fixture(autouse=True)
def _clean_hf_offline_state():
    saved = {key: os.environ.pop(key, None) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    saved_flag = hf_constants.HF_HUB_OFFLINE
    hf_constants.HF_HUB_OFFLINE = False
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    hf_constants.HF_HUB_OFFLINE = saved_flag


def _head(status_code: int):
    return MagicMock(status_code=status_code)


class TestHfHubEndpoint:
    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setattr(hf_constants, "ENDPOINT", "https://hf.example.com/")
        assert hf_offline.hf_hub_endpoint() == "https://hf.example.com"

    def test_uses_hub_constants_endpoint(self, monkeypatch):
        monkeypatch.setattr(hf_constants, "ENDPOINT", "https://hf.example.com")
        assert hf_offline.hf_hub_endpoint() == "https://hf.example.com"


class TestHfHubReachable:
    def test_reachable_on_200(self):
        with patch.object(hf_offline.requests, "head", return_value=_head(200)) as mock_head:
            assert _real_hf_hub_reachable() is True
        mock_head.assert_called_once()

    def test_reachable_on_client_error_status(self):
        with patch.object(hf_offline.requests, "head", return_value=_head(404)):
            assert _real_hf_hub_reachable() is True

    def test_unreachable_on_server_error_status(self):
        with patch.object(hf_offline.requests, "head", return_value=_head(503)):
            assert _real_hf_hub_reachable() is False

    def test_unreachable_on_connection_error(self):
        with patch.object(hf_offline.requests, "head", side_effect=hf_offline.requests.ConnectionError()):
            assert _real_hf_hub_reachable() is False

    def test_unreachable_on_timeout(self):
        with patch.object(hf_offline.requests, "head", side_effect=hf_offline.requests.Timeout()):
            assert _real_hf_hub_reachable() is False

    def test_probe_targets_configured_endpoint(self, monkeypatch):
        monkeypatch.setattr(hf_constants, "ENDPOINT", "https://hf.example.com")
        with patch.object(hf_offline.requests, "head", return_value=_head(200)) as mock_head:
            _real_hf_hub_reachable(timeout=2.0)
        mock_head.assert_called_once_with("https://hf.example.com", timeout=2.0, allow_redirects=True)


class TestEnableOfflineIfUnreachable:
    def test_already_offline_skips_probe(self, monkeypatch):
        hf_constants.HF_HUB_OFFLINE = True
        probe = MagicMock(return_value=True)
        monkeypatch.setattr(hf_offline, "hf_hub_reachable", probe)
        assert hf_offline.enable_offline_if_unreachable("some/repo") is False
        probe.assert_not_called()

    def test_local_model_dir_skips_probe(self, monkeypatch, tmp_path):
        probe = MagicMock(return_value=True)
        monkeypatch.setattr(hf_offline, "hf_hub_reachable", probe)
        assert hf_offline.enable_offline_if_unreachable(str(tmp_path)) is False
        probe.assert_not_called()

    def test_reachable_hub_does_not_enable_offline(self, monkeypatch):
        monkeypatch.setattr(hf_offline, "hf_hub_reachable", lambda *args, **kwargs: True)
        assert hf_offline.enable_offline_if_unreachable("some/repo") is False
        assert "HF_HUB_OFFLINE" not in os.environ
        assert hf_constants.HF_HUB_OFFLINE is False

    def test_unreachable_hub_enables_offline(self, monkeypatch):
        monkeypatch.setattr(hf_offline, "hf_hub_reachable", lambda *args, **kwargs: False)
        assert hf_offline.enable_offline_if_unreachable("some/repo") is True
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert hf_constants.HF_HUB_OFFLINE is True

    def test_unreachable_end_to_end_via_requests(self, monkeypatch):
        monkeypatch.setattr(hf_offline, "hf_hub_reachable", _real_hf_hub_reachable)
        with patch.object(hf_offline.requests, "head", side_effect=hf_offline.requests.ConnectionError()):
            assert hf_offline.enable_offline_if_unreachable("some/repo") is True
        assert hf_constants.HF_HUB_OFFLINE is True
        assert os.environ["HF_HUB_OFFLINE"] == "1"

    def test_hub_down_5xx_enables_offline(self, monkeypatch):
        monkeypatch.setattr(hf_offline, "hf_hub_reachable", _real_hf_hub_reachable)
        with patch.object(hf_offline.requests, "head", return_value=_head(502)):
            assert hf_offline.enable_offline_if_unreachable() is True
        assert hf_constants.HF_HUB_OFFLINE is True


class TestModelWrapperOfflineFallback:
    def test_load_model_probes_hub_before_loading(self):
        from faster_whisper_hotkey.models import ModelWrapper

        settings = MagicMock()
        settings.model_type = "whisper"
        settings.model_name = "small"
        settings.device = "cpu"
        settings.compute_type = "int8"

        with (
            patch("faster_whisper_hotkey.models.enable_offline_if_unreachable", return_value=False) as probe,
            patch("faster_whisper_hotkey.models.WhisperModel"),
        ):
            ModelWrapper(settings)
        probe.assert_called_once_with("small")

    def test_load_model_still_loads_when_offline_enabled(self):
        from faster_whisper_hotkey.models import ModelWrapper

        settings = MagicMock()
        settings.model_type = "whisper"
        settings.model_name = "small"
        settings.device = "cpu"
        settings.compute_type = "int8"

        with (
            patch("faster_whisper_hotkey.models.enable_offline_if_unreachable", return_value=True),
            patch("faster_whisper_hotkey.models.WhisperModel") as mock_whisper,
        ):
            ModelWrapper(settings)
        mock_whisper.assert_called_once_with(model_size_or_path="small", device="cpu", compute_type="int8")
