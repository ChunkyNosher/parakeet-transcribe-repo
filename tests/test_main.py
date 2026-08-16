"""Entry-point tests: startup threads (import warm-up, cache pre-warm) are on by default."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

from parakeet_transcribe import __main__ as main_module


def _parse(command: str) -> SimpleNamespace:
    return SimpleNamespace(command=command)


class RecordingThread(threading.Thread):
    started_threads: list[threading.Thread] = []

    def start(self) -> None:
        RecordingThread.started_threads.append(self)


def _run_main(monkeypatch, *, env: dict[str, str] | None = None) -> list[threading.Thread]:
    monkeypatch.delenv("PARAKEET_PRELOAD_MODEL", raising=False)
    monkeypatch.delenv("PARAKEET_IMPORT_WARMUP", raising=False)
    monkeypatch.delenv("PARAKEET_CACHE_PREWARM", raising=False)
    monkeypatch.delenv("PARAKEET_PREWARM_MODELS", raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    RecordingThread.started_threads = []
    with (
        patch.object(main_module.argparse.ArgumentParser, "parse_args", return_value=_parse("run")),
        patch.object(main_module, "inference_runtime_supported", return_value=(True, "")),
        patch("parakeet_transcribe.app.build_app") as build_app,
        patch("threading.Thread", RecordingThread),
    ):
        app = build_app.return_value
        app.launch = lambda **kwargs: None
        main_module.main()
    build_app.assert_called_once()
    return list(RecordingThread.started_threads)


def test_run_path_spawns_import_warmup_and_cache_prewarm_by_default(monkeypatch) -> None:
    threads = _run_main(monkeypatch)
    assert [thread.name for thread in threads] == ["parakeet-import-warmup", "parakeet-cache-prewarm"]


def test_preload_model_env_spawns_preload_thread(monkeypatch) -> None:
    threads = _run_main(monkeypatch, env={"PARAKEET_PRELOAD_MODEL": "parakeet-v3"})
    assert [thread.name for thread in threads] == [
        "parakeet-import-warmup",
        "parakeet-cache-prewarm",
        "parakeet-preload",
    ]


def test_import_warmup_can_be_disabled(monkeypatch) -> None:
    threads = _run_main(monkeypatch, env={"PARAKEET_IMPORT_WARMUP": "0"})
    assert [thread.name for thread in threads] == ["parakeet-cache-prewarm"]


def test_cache_prewarm_can_be_disabled(monkeypatch) -> None:
    threads = _run_main(monkeypatch, env={"PARAKEET_CACHE_PREWARM": "0"})
    assert [thread.name for thread in threads] == ["parakeet-import-warmup"]


def test_preload_thread_calls_service_warmup() -> None:
    warmed: list[str] = []

    class FakeService:
        def warmup(self, model_key: str) -> None:
            warmed.append(model_key)

    main_module._preload_model(FakeService(), "parakeet-v3")
    assert warmed == ["parakeet-v3"]


def test_preload_thread_swallows_failures() -> None:
    class BrokenService:
        def warmup(self, model_key: str) -> None:
            raise RuntimeError("CUDA is unavailable")

    # Must not raise; the model simply loads on demand later.
    main_module._preload_model(BrokenService(), "parakeet-v3")


def test_cache_prewarm_defaults_to_default_model(monkeypatch, capsys) -> None:
    from parakeet_transcribe.models import DEFAULT_MODEL_KEY, get_model

    prewarmed: list = []
    with (
        monkeypatch.context() as ctx,
        patch("parakeet_transcribe.modelstore.prewarm_local_caches") as prewarm,
    ):
        ctx.setenv("PARAKEET_PREWARM_MODELS", "")
        prewarm.side_effect = lambda spec: prewarmed.append(spec.key) or True
        main_module._cache_prewarm()
    assert prewarmed == [DEFAULT_MODEL_KEY]
    assert "Pre-warmed local caches" in capsys.readouterr().out


def test_cache_prewarm_respects_model_list_and_skips_unknown_keys(monkeypatch) -> None:
    prewarmed: list = []
    with (
        monkeypatch.context() as ctx,
        patch("parakeet_transcribe.modelstore.prewarm_local_caches") as prewarm,
    ):
        ctx.setenv("PARAKEET_PREWARM_MODELS", "parakeet-v2, nonexistent-model")
        prewarm.side_effect = lambda spec: prewarmed.append(spec.key) or True
        main_module._cache_prewarm()
    # Unknown keys are skipped without aborting the remaining entries.
    assert prewarmed == ["parakeet-v2"]


def test_doctor_command_does_not_launch_app() -> None:
    with (
        patch.object(main_module.argparse.ArgumentParser, "parse_args", return_value=_parse("doctor")),
        patch.object(main_module, "doctor_report", return_value=(True, "OK")),
        patch("parakeet_transcribe.app.build_app") as build_app,
    ):
        main_module.main()
    build_app.assert_not_called()
