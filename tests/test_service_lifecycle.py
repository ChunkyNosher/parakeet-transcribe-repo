"""Lifecycle tests for lazy model loading and idle-based VRAM eviction."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from parakeet_transcribe.service import IDLE_UNLOAD_SECONDS, TranscriptionService


def _fake_backend() -> SimpleNamespace:
    backend = SimpleNamespace(unloaded=False, parked=False)
    backend.unload = lambda: setattr(backend, "unloaded", True)
    backend.park = lambda: setattr(backend, "parked", True)
    return backend


def _fresh_service() -> TranscriptionService:
    service = TranscriptionService(cache_dir=Path("."))
    # Stop the background reaper so tests are deterministic; the reaper itself
    # is exercised by calling _maybe_idle_unload() explicitly.
    service._stop_reaper()
    service._reaper.join(timeout=5.0)
    return service


def _reload_service_module(monkeypatch) -> object:
    """Reload service.py so module-level constants re-read the environment."""
    import parakeet_transcribe.service as service_module

    try:
        importlib.reload(service_module)
        yield service_module
    finally:
        monkeypatch.delenv("PARAKEET_IDLE_UNLOAD_SECONDS", raising=False)
        monkeypatch.delenv("PARAKEET_IDLE_PARK", raising=False)
        monkeypatch.delenv("PARAKEET_FORCE_INFERENCE_MODE", raising=False)
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


def test_idle_unload_zero_disables_eviction(monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_IDLE_UNLOAD_SECONDS", "0")
    for module in _reload_service_module(monkeypatch):
        assert module.IDLE_UNLOAD_SECONDS == 0.0
        service = module.TranscriptionService(cache_dir=Path("."))
        service._stop_reaper()
        service._reaper.join(timeout=5.0)
        backend = _fake_backend()
        service._backend = backend
        service._model_key = "parakeet-v3"
        service._last_used = 1.0
        with patch.object(module.time, "monotonic", return_value=1_000.0):
            assert service._maybe_idle_unload() is False
        assert backend.unloaded is False
        assert backend.parked is False


def test_idle_parking_enabled_by_default(monkeypatch) -> None:
    for module in _reload_service_module(monkeypatch):
        assert module.IDLE_PARK_ENABLED is True


def test_idle_parking_disabled_via_env(monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_IDLE_PARK", "0")
    for module in _reload_service_module(monkeypatch):
        assert module.IDLE_PARK_ENABLED is False


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


def test_reaper_parks_after_idle_timeout() -> None:
    service = _fresh_service()
    backend = _fake_backend()
    service._backend = backend
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    with (
        patch("parakeet_transcribe.service.unload_punctuation_model") as unload_punct,
        patch("parakeet_transcribe.service.time.monotonic", return_value=1.0 + IDLE_UNLOAD_SECONDS + 1.0),
    ):
        assert service._maybe_idle_unload() is True
    # Idle eviction parks the model in system RAM: the backend (and its
    # configured decoding fingerprint) is kept for a fast revive.
    assert backend.parked is True
    assert backend.unloaded is False
    assert service._backend is backend
    assert service._last_used is None
    unload_punct.assert_called_once()


def test_reaper_skips_already_parked_backend() -> None:
    service = _fresh_service()
    backend = _fake_backend()
    backend.parked = True
    service._backend = backend
    service._model_key = "parakeet-v3"
    service._last_used = None
    assert service._maybe_idle_unload() is False
    assert service._backend is backend


def test_reaper_drops_when_parking_disabled(monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_IDLE_PARK", "0")
    for module in _reload_service_module(monkeypatch):
        service = module.TranscriptionService(cache_dir=Path("."))
        service._stop_reaper()
        service._reaper.join(timeout=5.0)
        backend = _fake_backend()
        service._backend = backend
        service._model_key = "parakeet-v3"
        service._last_used = 1.0
        with patch.object(module.time, "monotonic", return_value=1.0 + module.IDLE_UNLOAD_SECONDS + 1.0):
            assert service._maybe_idle_unload() is True
        # PARAKEET_IDLE_PARK=0 restores the pre-parking full-drop behavior.
        assert backend.unloaded is True
        assert backend.parked is False
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
    assert backend.parked is False


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
    assert backend.parked is False


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


def test_get_backend_reuses_idle_same_key_backend() -> None:
    service = _fresh_service()
    resident = _fake_backend()
    service._backend = resident
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    with patch("parakeet_transcribe.service.time.monotonic", return_value=1.0 + IDLE_UNLOAD_SECONDS + 1.0):
        backend = service._get_backend("parakeet-v3")
    # A resident-but-idle backend is reused as-is: it is either still warm in
    # VRAM or parked, and backend.load() revives a parked one cheaply.
    assert backend is resident
    assert resident.unloaded is False
    assert resident.parked is False
    assert service._last_used == 1.0 + IDLE_UNLOAD_SECONDS + 1.0


def test_get_backend_reuses_parked_backend_and_restarts_timer() -> None:
    service = _fresh_service()
    backend = _fake_backend()
    backend.parked = True
    service._backend = backend
    service._model_key = "parakeet-v3"
    service._last_used = None
    with patch("parakeet_transcribe.service.time.monotonic", return_value=42.0):
        resolved = service._get_backend("parakeet-v3")
    assert resolved is backend
    assert service._last_used == 42.0


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


def test_warmup_loads_backend_and_marks_used() -> None:
    service = _fresh_service()
    loaded: list[bool] = []

    class WarmBackend:
        def __init__(self, spec) -> None:
            self.spec = spec

        def load(self) -> None:
            loaded.append(True)

        def unload(self) -> None:
            return None

    with (
        patch("parakeet_transcribe.service.NeMoASRBackend", WarmBackend),
        patch("parakeet_transcribe.service.time.monotonic", return_value=42.0),
    ):
        service.warmup("parakeet-v3")

    assert loaded == [True]
    assert service._model_key == "parakeet-v3"
    # The idle timer restarts from the completed warm-up.
    assert service._last_used == 42.0


def test_warmup_reuses_resident_backend() -> None:
    service = _fresh_service()
    backend = SimpleNamespace(spec=SimpleNamespace(key="parakeet-v3"), loaded=0)
    backend.load = lambda: setattr(backend, "loaded", backend.loaded + 1)
    backend.unload = lambda: None
    service._backend = backend
    service._model_key = "parakeet-v3"
    service._last_used = 1.0
    with patch("parakeet_transcribe.service.time.monotonic", return_value=1.0 + IDLE_UNLOAD_SECONDS - 1.0):
        service.warmup("parakeet-v3")
    assert backend.loaded == 1
    assert service._backend is backend


def test_force_inference_mode_defaults_to_auto(monkeypatch) -> None:
    monkeypatch.delenv("PARAKEET_FORCE_INFERENCE_MODE", raising=False)
    for module in _reload_service_module(monkeypatch):
        assert module.FORCE_INFERENCE_MODE == "auto"


def test_force_inference_mode_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_FORCE_INFERENCE_MODE", "streaming")
    for module in _reload_service_module(monkeypatch):
        assert module.FORCE_INFERENCE_MODE == "streaming"


def test_force_inference_mode_ignores_unknown_values(monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_FORCE_INFERENCE_MODE", "banana")
    for module in _reload_service_module(monkeypatch):
        assert module.FORCE_INFERENCE_MODE == "auto"


def _fake_prepared(duration_seconds: float) -> SimpleNamespace:
    return SimpleNamespace(
        duration_seconds=duration_seconds,
        source_path=Path("sample.wav"),
        samples=SimpleNamespace(),
        sample_rate=16000,
        canonical_path=Path("sample.wav"),
    )


def _fake_backend_for_routing() -> SimpleNamespace:
    backend = SimpleNamespace(
        spec=SimpleNamespace(
            model_id="nvidia/parakeet-tdt-0.6b-v3",
            capabilities=SimpleNamespace(timestamps=True, lowercase_vocab=False),
        ),
        word_confidence_fallback_used=False,
        used_paths=[],
        used_streaming=[],
    )

    def configure_decoding(key_phrases, boost_alpha) -> None:
        return None

    def transcribe_paths(paths, *, timestamps, batch_size=1):
        backend.used_paths.append(([str(path) for path in paths], timestamps, batch_size))
        from parakeet_transcribe.types import ChunkResult

        return [ChunkResult(text="offline text", words=[], detected_language=None, segments=[])]

    def transcribe_streaming_audio(samples, sample_rate, *, timestamps, progress=None, cancel=None):
        backend.used_streaming.append((timestamps, sample_rate))
        from parakeet_transcribe.types import ChunkResult

        return ChunkResult(text="streaming text", words=[], detected_language=None, segments=[])

    backend.configure_decoding = configure_decoding
    backend.transcribe_paths = transcribe_paths
    backend.transcribe_streaming_audio = transcribe_streaming_audio
    return backend


def _run_prepared(module, backend, prepared) -> object:
    service = module.TranscriptionService(cache_dir=Path("."))
    service._stop_reaper()
    service._reaper.join(timeout=5.0)
    return service._transcribe_prepared(
        backend,
        prepared,
        batch_size=1,
        language="auto",
        key_phrases=[],
        boost_alpha=1.0,
        progress=lambda fraction, description: None,
        cancel=lambda: False,
        progress_base=0.0,
        progress_span=1.0,
        work_dir=Path("."),
    )


def test_routing_uses_streaming_for_long_audio_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PARAKEET_FORCE_INFERENCE_MODE", raising=False)
    for module in _reload_service_module(monkeypatch):
        backend = _fake_backend_for_routing()
        result = _run_prepared(module, backend, _fake_prepared(duration_seconds=120.0))
        assert backend.used_streaming == [(True, 16000)]
        assert backend.used_paths == []
        assert result.text == "streaming text"
        assert result.runtime["inference_mode"] == "nemo_buffered_streaming"


def test_routing_uses_offline_for_short_audio_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PARAKEET_FORCE_INFERENCE_MODE", raising=False)
    for module in _reload_service_module(monkeypatch):
        backend = _fake_backend_for_routing()
        result = _run_prepared(module, backend, _fake_prepared(duration_seconds=10.0))
        assert backend.used_paths == [(["sample.wav"], True, 1)]
        assert backend.used_streaming == []
        assert result.text == "offline text"
        assert result.runtime["inference_mode"] == "nemo_offline"


def test_routing_force_offline_overrides_duration(monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_FORCE_INFERENCE_MODE", "offline")
    for module in _reload_service_module(monkeypatch):
        backend = _fake_backend_for_routing()
        result = _run_prepared(module, backend, _fake_prepared(duration_seconds=120.0))
        assert backend.used_paths == [(["sample.wav"], True, 1)]
        assert backend.used_streaming == []
        assert result.text == "offline text"
        assert result.runtime["inference_mode"] == "nemo_offline"


def test_routing_force_streaming_overrides_duration(monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_FORCE_INFERENCE_MODE", "streaming")
    for module in _reload_service_module(monkeypatch):
        backend = _fake_backend_for_routing()
        result = _run_prepared(module, backend, _fake_prepared(duration_seconds=10.0))
        assert backend.used_streaming == [(True, 16000)]
        assert backend.used_paths == []
        assert result.text == "streaming text"
        assert result.runtime["inference_mode"] == "nemo_buffered_streaming"
