"""Lifecycle tests for lazy model loading and idle-based VRAM eviction."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from parakeet_transcribe.service import IDLE_UNLOAD_SECONDS, TranscriptionService


def _fake_backend() -> SimpleNamespace:
    backend = SimpleNamespace(unloaded=False)
    backend.unload = lambda: setattr(backend, "unloaded", True)
    return backend


def _fresh_service() -> TranscriptionService:
    service = TranscriptionService(cache_dir=Path("."))
    # Stop the background reaper so tests are deterministic; the reaper itself
    # is exercised by calling _maybe_idle_unload() explicitly.
    service._stop_reaper()
    service._reaper.join(timeout=5.0)
    return service


def _reload_service_module(monkeypatch) -> object:
    """Reload service.py so IDLE_UNLOAD_SECONDS re-reads the environment."""
    import parakeet_transcribe.service as service_module

    try:
        importlib.reload(service_module)
        yield service_module
    finally:
        monkeypatch.delenv("PARAKEET_IDLE_UNLOAD_SECONDS", raising=False)
        importlib.reload(service_module)


def test_idle_unload_default_is_180_seconds() -> None:
    assert IDLE_UNLOAD_SECONDS == 180.0


def test_idle_unload_constant_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_IDLE_UNLOAD_SECONDS", "42")
    for module in _reload_service_module(monkeypatch):
        assert module.IDLE_UNLOAD_SECONDS == 42.0


def test_idle_unload_constant_clamps_below_10(monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_IDLE_UNLOAD_SECONDS", "2")
    for module in _reload_service_module(monkeypatch):
        assert module.IDLE_UNLOAD_SECONDS == 10.0


def test_unload_when_nothing_loaded() -> None:
    service = _fresh_service()
    assert service.unload() == "No model is loaded."


def test_unload_releases_backend_and_clears_timer() -> None:
    service = _fresh_service()
    backend = _fake_backend()
    service._backend = backend
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    with patch("parakeet_transcribe.service.unload_punctuation_model") as unload_punct:
        message = service.unload()
    assert "unloaded" in message
    assert backend.unloaded is True
    assert service._backend is None
    assert service._model_key is None
    assert service._last_used is None
    unload_punct.assert_called_once()


def test_reaper_evicts_after_idle_timeout() -> None:
    service = _fresh_service()
    service._backend = _fake_backend()
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    with patch("parakeet_transcribe.service.time.monotonic", return_value=1.0 + IDLE_UNLOAD_SECONDS + 1.0):
        assert service._maybe_idle_unload() is True
    assert service._backend is None


def test_reaper_keeps_freshly_used_model() -> None:
    service = _fresh_service()
    backend = _fake_backend()
    service._backend = backend
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    with patch("parakeet_transcribe.service.time.monotonic", return_value=1.0 + IDLE_UNLOAD_SECONDS - 1.0):
        assert service._maybe_idle_unload() is False
    assert service._backend is backend


def test_reaper_never_evicts_during_job() -> None:
    service = _fresh_service()
    backend = _fake_backend()
    service._backend = backend
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    service._in_use = True
    with patch("parakeet_transcribe.service.time.monotonic", return_value=1.0 + IDLE_UNLOAD_SECONDS + 1.0):
        assert service._maybe_idle_unload() is False
    assert service._backend is backend


def test_get_backend_reuses_fresh_backend() -> None:
    service = _fresh_service()
    backend = _fake_backend()
    service._backend = backend
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    with patch("parakeet_transcribe.service.time.monotonic", return_value=1.0 + IDLE_UNLOAD_SECONDS - 1.0):
        resolved = service._get_backend("parakeet-v3")
    assert resolved is backend
    assert service._last_used == 1.0 + IDLE_UNLOAD_SECONDS - 1.0


def test_get_backend_reloads_stale_model() -> None:
    service = _fresh_service()
    stale = _fake_backend()
    service._backend = stale
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    with (
        patch("parakeet_transcribe.service.time.monotonic", return_value=1.0 + IDLE_UNLOAD_SECONDS + 1.0),
        patch("parakeet_transcribe.service.NeMoASRBackend") as backend_cls,
    ):
        backend_cls.return_value = _fake_backend()
        backend = service._get_backend("parakeet-v3")
    assert stale.unloaded is True
    assert backend is not stale
    assert service._model_key == "parakeet-v3"


def test_get_backend_swaps_model_key() -> None:
    service = _fresh_service()
    old_backend = _fake_backend()
    service._backend = old_backend
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    with patch("parakeet_transcribe.service.NeMoASRBackend") as backend_cls:
        backend_cls.return_value = _fake_backend()
        backend = service._get_backend("nemotron-3.5")
    assert old_backend.unloaded is True
    assert backend is service._backend
    assert service._model_key == "nemotron-3.5"


def test_in_use_guard_blocks_eviction_during_job() -> None:
    service = _fresh_service()
    backend = _fake_backend()
    service._backend = backend
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    service._in_use = True
    with patch("parakeet_transcribe.service.time.monotonic", return_value=1.0 + IDLE_UNLOAD_SECONDS + 1.0):
        assert service._maybe_idle_unload() is False
    assert service._backend is backend
