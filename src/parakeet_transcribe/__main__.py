from __future__ import annotations

import argparse
import os
import threading

from .diagnostics import doctor_report, inference_runtime_supported
from .exports import resolved_output_dir

_FALSY = {"0", "false", "no", "off"}


def _import_warmup() -> None:
    """Import the heavy runtime (torch/NeMo) ahead of the first request.

    Uses no VRAM and loads no model; it only moves the one-time import cost
    off the first transcription request by overlapping it with app startup.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
        import nemo.collections.asr  # noqa: F401
    except Exception as exc:  # pragma: no cover - installation error surfaces on first use
        print(f"Startup import warm-up skipped: {exc}", flush=True)


def _preload_model(service: object, model_key: str) -> None:
    try:
        service.warmup(model_key)
        print(f"Preloaded model '{model_key}' into VRAM.", flush=True)
    except Exception as exc:
        print(f"Model preload failed; it will load on demand: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local NVIDIA NeMo ASR transcription (Docker)")
    parser.add_argument("command", nargs="?", choices=("run", "doctor"), default="run")
    args = parser.parse_args()
    if args.command == "doctor":
        ready, report = doctor_report()
        print(report)
        if not ready:
            raise SystemExit(1)
        return

    supported, message = inference_runtime_supported()
    if not supported:
        text = message.removeprefix("- ACTION ").strip()
        print(text)
        print("See README.md Docker Compose section.")
        raise SystemExit(2)

    from .app import SERVICE, build_app

    # Models load lazily on the first transcription request and are evicted
    # after PARAKEET_IDLE_UNLOAD_SECONDS (default 180s) of inactivity.
    # Startup threads are optional: the import warm-up (no VRAM) runs unless
    # PARAKEET_IMPORT_WARMUP=0; a full model preload runs only when
    # PARAKEET_PRELOAD_MODEL names a model key (off by default).
    if os.environ.get("PARAKEET_IMPORT_WARMUP", "1").strip().lower() not in _FALSY:
        threading.Thread(target=_import_warmup, name="parakeet-import-warmup", daemon=True).start()
    preload_key = os.environ.get("PARAKEET_PRELOAD_MODEL", "").strip()
    if preload_key:
        threading.Thread(
            target=_preload_model,
            args=(SERVICE, preload_key),
            name="parakeet-preload",
            daemon=True,
        ).start()

    build_app().launch(
        server_name=os.environ.get("PARAKEET_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PARAKEET_SERVER_PORT", "7860")),
        inbrowser=os.environ.get("PARAKEET_INBROWSER", "true").lower() == "true",
        show_error=True,
        allowed_paths=[str(resolved_output_dir())],
    )


if __name__ == "__main__":
    main()
