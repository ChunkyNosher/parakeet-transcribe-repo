"""Entry-point tests: no model warm-up thread is spawned at startup."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

from parakeet_transcribe import __main__ as main_module


def _parse(command: str) -> SimpleNamespace:
    return SimpleNamespace(command=command)


def test_run_path_does_not_spawn_warmup_thread() -> None:
    started: list[threading.Thread] = []

    class RecordingThread(threading.Thread):
        def start(self) -> None:
            started.append(self)

    with (
        patch.object(main_module.argparse.ArgumentParser, "parse_args", return_value=_parse("run")),
        patch.object(main_module, "inference_runtime_supported", return_value=(True, "")),
        patch("parakeet_transcribe.app.build_app") as build_app,
        patch("threading.Thread", RecordingThread),
    ):
        app = build_app.return_value
        app.launch = lambda **kwargs: None
        main_module.main()

    assert started == []
    build_app.assert_called_once()


def test_run_path_launches_app() -> None:
    with (
        patch.object(main_module.argparse.ArgumentParser, "parse_args", return_value=_parse("run")),
        patch.object(main_module, "inference_runtime_supported", return_value=(True, "")),
        patch("parakeet_transcribe.app.build_app") as build_app,
    ):
        app = build_app.return_value
        app.launch = lambda **kwargs: None
        main_module.main()
    build_app.assert_called_once()


def test_doctor_command_does_not_launch_app() -> None:
    with (
        patch.object(main_module.argparse.ArgumentParser, "parse_args", return_value=_parse("doctor")),
        patch.object(main_module, "doctor_report", return_value=(True, "OK")),
        patch("parakeet_transcribe.app.build_app") as build_app,
    ):
        main_module.main()
    build_app.assert_not_called()
