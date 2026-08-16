"""Entry-point tests: startup threads are opt-in except the import warm-up."""

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


def test_run_path_spawns_only_import_warmup_by_default(monkeypatch) -> None:
    threads = _run_main(monkeypatch)
    assert [thread.name for thread in threads] == ["parakeet-import-warmup"]


def test_preload_model_env_spawns_preload_thread(monkeypatch) -> None:
    threads = _run_main(monkeypatch, env={"PARAKEET_PRELOAD_MODEL": "parakeet-v3"})
    assert [thread.name for thread in threads] == ["parakeet-import-warmup", "parakeet-preload"]


def test_import_warmup_can_be_disabled(monkeypatch) -> None:
    threads = _run_main(monkeypatch, env={"PARAKEET_IMPORT_WARMUP": "0"})
    assert threads == []


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


def test_doctor_command_does_not_launch_app() -> None:
    with (
        patch.object(main_module.argparse.ArgumentParser, "parse_args", return_value=_parse("doctor")),
        patch.object(main_module, "doctor_report", return_value=(True, "OK")),
        patch("parakeet_transcribe.app.build_app") as build_app,
    ):
        main_module.main()
    build_app.assert_not_called()
