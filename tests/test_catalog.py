"""Unit tests for catalog.py — hub parsing, caching, offline fallback. No network."""

from unittest.mock import patch

import requests

from faster_whisper_hotkey import catalog


def _listing():
    """A fake HF /api/models listing payload."""
    return [
        {"id": "handy-computer/parakeet-ultra-gguf", "siblings": [{"rfilename": "parakeet-ultra-Q8_0.gguf", "size": 740}]},
        {"id": "handy-computer/whisper-small-gguf", "siblings": [{"rfilename": "whisper-small-F16.gguf", "size": 200}]},
        # support / diarization repos must be filtered out of the picker:
        {"id": "handy-computer/silero-vad-v6.2-gguf", "siblings": [{"rfilename": "silero-vad-v6.2-F32.gguf", "size": 1}]},
        {"id": "handy-computer/lang-id-voxlingua107-ecapa-gguf", "siblings": [{"rfilename": "x-Q8_0.gguf", "size": 1}]},
        {"id": "handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf", "siblings": [{"rfilename": "d-Q4_K_M.gguf", "size": 1}]},
        # not a gguf repo:
        {"id": "handy-computer/not-an-asset", "siblings": [{"rfilename": "README.md"}]},
    ]


def test_short_name():
    assert catalog.short_name("handy-computer/parakeet-ultra-gguf") == "parakeet-ultra"
    assert catalog.short_name("handy-computer/whisper-large-v3-gguf") == "whisper-large-v3"


def test_quant_of():
    assert catalog.quant_of("parakeet-ultra-Q8_0.gguf") == "Q8_0"
    assert catalog.quant_of("whisper-base.en-F16.gguf") == "F16"
    assert catalog.quant_of("silero-vad-v6.2-F32.gguf") == "F32"
    assert catalog.quant_of("nohyphen.gguf") == ""


def test_gguf_file_label_and_size():
    f = catalog.GGUFFile("parakeet-ultra-Q8_0.gguf", "Q8_0", size_bytes=740_000_000)
    assert f.label == "int8"
    assert round(f.size_mb) == 740
    assert catalog.GGUFFile("x-Q5_K_M.gguf", "Q5_K_M").label == "Q5_K_M"
    assert catalog.GGUFFile("x-F32.gguf", "F32").label == "float32"
    assert catalog.GGUFFile("x-Q5_K_M.gguf", "Q5_K_M").size_mb is None


def test_parse_repos_filters_support_and_non_asr():
    repos = catalog._parse_repos(_listing())
    names = {r["id"] for r in repos}
    assert names == {"handy-computer/parakeet-ultra-gguf", "handy-computer/whisper-small-gguf"}


def test_list_models_live(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "CACHE_FILE", str(tmp_path / "catalog.json"))

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return _listing()

    with patch.object(catalog.requests, "get", return_value=Resp()):
        models = catalog.list_models(refresh=True)

    assert {m.name for m in models} == {"parakeet-ultra", "whisper-small"}
    assert (tmp_path / "catalog.json").exists()  # fetched listing was cached
    parakeet = catalog.find_entry("handy-computer/parakeet-ultra-gguf")
    assert parakeet.files[0].quant == "Q8_0"
    assert parakeet.pick("parakeet-ultra-Q8_0.gguf") is not None


def test_list_models_falls_back_to_snapshot_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "CACHE_FILE", str(tmp_path / "no-cache.json"))  # no cache file present

    def boom(*a, **k):
        raise requests.ConnectionError("no network")

    with patch.object(catalog.requests, "get", side_effect=boom):
        models = catalog.list_models()

    # Bundled snapshot: many models, ASR only (no silero/lang-id/sortformer), never empty.
    names = {m.name for m in models}
    assert "parakeet-ultra" in names and len(names) > 10
    assert not any(n.startswith(("silero-vad", "lang-id", "diar_")) for n in names)


def test_list_models_uses_fresh_cache_without_network(tmp_path, monkeypatch):
    import json
    import time

    cache = tmp_path / "catalog.json"
    cache.write_text(json.dumps({"fetched_at": time.time(), "repos": [{"id": "handy-computer/medasr-gguf", "files": ["medasr-Q8_0.gguf"]}]}))
    monkeypatch.setattr(catalog, "CACHE_FILE", str(cache))

    def explode(*a, **k):
        raise AssertionError("network must not be touched when the cache is fresh")

    with patch.object(catalog.requests, "get", side_effect=explode):
        models = catalog.list_models()

    assert [m.name for m in models] == ["medasr"]


def test_stale_cache_triggers_refresh(tmp_path, monkeypatch):
    import json

    cache = tmp_path / "catalog.json"
    cache.write_text(json.dumps({"fetched_at": 0, "repos": [{"id": "handy-computer/old-gguf", "files": ["old-F16.gguf"]}]}))
    monkeypatch.setattr(catalog, "CACHE_FILE", str(cache))

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return _listing()

    with patch.object(catalog.requests, "get", return_value=Resp()):
        models = catalog.list_models()

    assert {m.name for m in models} == {"parakeet-ultra", "whisper-small"}  # refreshed, not the stale cache


def test_files_for_parses_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "CACHE_FILE", str(tmp_path / "catalog.json"))

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"siblings": [
                {"rfilename": "parakeet-ultra-F16.gguf", "size": 1256},
                {"rfilename": "parakeet-ultra-Q4_K_M.gguf", "size": 486},
                {"rfilename": "README.md", "size": 10},
            ]}

    with patch.object(catalog.requests, "get", return_value=Resp()):
        files = catalog.files_for("handy-computer/parakeet-ultra-gguf")

    assert [f.filename for f in files] == ["parakeet-ultra-Q4_K_M.gguf", "parakeet-ultra-F16.gguf"]  # sorted by size asc
    assert all(f.size_bytes for f in files)


def test_files_for_falls_back_to_listing_names(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "CACHE_FILE", str(tmp_path / "catalog.json"))

    def boom(*a, **k):
        raise requests.ConnectionError("no network")

    # list_models() also needs the network for the snapshot fallback (fine);
    # patch the per-repo detail call to fail, and pre-seed a cache for list_models.
    import json
    cache = tmp_path / "catalog.json"
    cache.write_text(json.dumps({"fetched_at": 9999999999, "repos": [{"id": "handy-computer/parakeet-ultra-gguf", "files": ["parakeet-ultra-Q8_0.gguf"]}]}))

    with patch.object(catalog.requests, "get", side_effect=boom):
        files = catalog.files_for("handy-computer/parakeet-ultra-gguf")

    assert [f.filename for f in files] == ["parakeet-ultra-Q8_0.gguf"]
    assert files[0].size_bytes is None


def test_support_model_constants_exist():
    assert catalog.VAD_REPO.startswith("handy-computer/")
    assert catalog.VAD_FILE.endswith(".gguf")
