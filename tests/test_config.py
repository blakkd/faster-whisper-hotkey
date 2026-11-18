"""Tests for config.py (packaged resources) and the bundled model snapshot."""

import json

from faster_whisper_hotkey.config import get_resource_path
from faster_whisper_hotkey.models import WARMUP_CLIP


class TestGetResourcePath:
    def test_returns_existing_path_for_warmup_clip(self):
        import os

        path = get_resource_path(WARMUP_CLIP)
        assert os.path.isfile(path)

    def test_snapshot_resource_exists(self):
        import os

        path = get_resource_path("assets/model_catalog.json")
        assert os.path.isfile(path)


class TestBundledSnapshot:
    """The offline snapshot the catalog falls back to must be well-formed."""

    def test_snapshot_is_valid_json_with_repos(self):
        with open(get_resource_path("assets/model_catalog.json"), encoding="utf-8") as f:
            data = json.load(f)
        assert data["schema"] == 1
        repos = data["repos"]
        assert len(repos) > 10  # the whole GGUF zoo, far more than the old curated list

    def test_snapshot_repos_have_gguf_files(self):
        with open(get_resource_path("assets/model_catalog.json"), encoding="utf-8") as f:
            repos = json.load(f)["repos"]
        for r in repos:
            assert r["id"].endswith("-gguf")
            assert all(fn.endswith(".gguf") for fn in r["files"])
            assert r["files"]
