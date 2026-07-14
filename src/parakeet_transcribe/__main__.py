from __future__ import annotations

import argparse
import os
import threading

from .diagnostics import doctor_report, inference_runtime_supported
from .exports import resolved_output_dir
from .models import DEFAULT_MODEL_KEY


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

    warm_thread = threading.Thread(
        target=SERVICE.warm_default_model,
        kwargs={"model_key": DEFAULT_MODEL_KEY},
        name="parakeet-model-warmup",
        daemon=True,
    )
    warm_thread.start()

    build_app().launch(
        server_name=os.environ.get("PARAKEET_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PARAKEET_SERVER_PORT", "7860")),
        inbrowser=os.environ.get("PARAKEET_INBROWSER", "true").lower() == "true",
        show_error=True,
        allowed_paths=[str(resolved_output_dir())],
    )


if __name__ == "__main__":
    main()
