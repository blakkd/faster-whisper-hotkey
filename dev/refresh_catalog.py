"""Refresh the bundled offline model snapshot from the HF hub.

Regenerates ``src/faster_whisper_hotkey/assets/model_catalog.json`` — the
fallback the app lists models from when it is offline. Run this when new GGUFs
appear on the hub; the runtime already lists them live via the hub, so this only
refreshes the no-network fallback.

    uv run python dev/refresh_catalog.py
"""

import datetime
import json

import requests

RESOURCE = "src/faster_whisper_hotkey/assets/model_catalog.json"
AUTHOR = "handy-computer"
URL = f"https://huggingface.co/api/models?author={AUTHOR}&full=true&limit=100"


def main() -> None:
    resp = requests.get(URL, timeout=30, headers={"User-Agent": "faster-whisper-hotkey-catalog"})
    resp.raise_for_status()
    repos = []
    for m in resp.json():
        rid = m.get("id", "")
        if not rid.endswith("-gguf"):
            continue
        files = sorted(s["rfilename"] for s in m.get("siblings", []) if s["rfilename"].endswith(".gguf"))
        if files:
            repos.append({"id": rid, "files": files})
    repos.sort(key=lambda r: r["id"].lower())
    snapshot = {
        "schema": 1,
        "author": AUTHOR,
        "generated": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d"),
        "repos": repos,
    }
    with open(RESOURCE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(repos)} repos ({sum(len(r['files']) for r in repos)} GGUF files) -> {RESOURCE}")


if __name__ == "__main__":
    main()
