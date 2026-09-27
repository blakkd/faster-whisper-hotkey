"""Disable Kestrel/Photon usage telemetry.

Moondream's Photon runtime (the `kestrel` package) reports aggregate usage
telemetry (model, GPU, hostname, request/token counts) to
https://api.moondream.ai/v1/photon/telemetry on a 60s cadence. As of
moondream 2.5.0 / kestrel 0.8.2 there is no official opt-out: the
``PhotonReporter`` is created unconditionally in
``kestrel.engine.core.InferenceEngine._initialize()`` and there is no config
flag or environment variable to disable it (``MOONDREAM_API_BASE_URL`` is
allowlisted to Moondream's own hosts).

Until Moondream ships an official switch, we replace the reporter with a
no-op stub that satisfies the same interface the engine calls:

- ``PhotonReporter(runtime_cfg, runtime_device, api_key=..., api_base_url=...)``
- ``await reporter.validate_api_key() -> bool``
- ``reporter.start()``
- ``await reporter.shutdown()``
- ``reporter.record_success(finetune=..., input_tokens=..., output_tokens=...)``
- ``reporter.record_error(finetune=...)``

``InferenceEngine.create()`` also calls two helpers on the class itself
(``PhotonReporter._normalize_api_key`` / ``PhotonReporter._is_api_key_header_safe``);
the stub provides identical implementations so key handling is unchanged.

The engine resolves ``PhotonReporter`` from the ``kestrel.engine.core`` module
globals at call time, so swapping the attribute before the engine is created
is sufficient.

Note: ``validate_api_key`` doubles as the finetune-auth check. The stub
returns ``False``, which is exactly what the real reporter returns when
``MOONDREAM_API_KEY`` is unset — base-model inference (all we use) is
unaffected, but cloud finetune inference would be disabled.

If a future kestrel release moves or renames the reporter, this degrades
gracefully: telemetry simply stays on and a warning is logged.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


class NoTelemetryReporter:
    """Drop-in replacement for ``kestrel.photon.PhotonReporter``; sends nothing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def validate_api_key(self) -> bool:
        # Same result as the real reporter when MOONDREAM_API_KEY is unset.
        return False

    def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def record_success(self, **kwargs: object) -> None:
        pass

    def record_error(self, **kwargs: object) -> None:
        pass

    @staticmethod
    def _normalize_api_key(api_key: str | None) -> str | None:
        # Identical to the real implementation (pure string handling).
        normalized = api_key.strip() if api_key else ""
        return normalized or None

    @staticmethod
    def _is_api_key_header_safe(api_key: str) -> bool:
        # Identical to the real implementation (pure string handling).
        return api_key.isascii() and all(
            not c.isspace() and c.isprintable() for c in api_key
        )


_applied = False


def apply() -> None:
    """Install the no-op reporter. Safe to call multiple times."""
    global _applied
    if _applied:
        return
    try:
        import kestrel.engine.core as engine_core
    except Exception:  # noqa: BLE001
        _log.warning(
            "Could not import kestrel.engine.core; Photon telemetry patch not applied.",
            exc_info=True,
        )
        return
    if getattr(engine_core, "PhotonReporter", None) is None:
        _log.warning(
            "kestrel.engine.core.PhotonReporter not found (kestrel upgrade?); "
            "Photon telemetry may no longer be patchable."
        )
        return
    if engine_core.PhotonReporter is not NoTelemetryReporter:
        engine_core.PhotonReporter = NoTelemetryReporter
        _applied = True
        _log.info("Photon telemetry disabled (no-op reporter installed).")
