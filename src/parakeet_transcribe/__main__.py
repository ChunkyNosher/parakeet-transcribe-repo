from __future__ import annotations

import argparse
import os

from .app import build_app
from .diagnostics import doctor_report
from .exports import resolved_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Local NVIDIA ASR transcription")
    parser.add_argument("command", nargs="?", choices=("run", "doctor"), default="run")
    args = parser.parse_args()
    if args.command == "doctor":
        ready, report = doctor_report()
        print(report)
        if not ready:
            raise SystemExit(1)
        return
    build_app().launch(
        server_name=os.environ.get("PARAKEET_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PARAKEET_SERVER_PORT", "7860")),
        inbrowser=os.environ.get("PARAKEET_INBROWSER", "true").lower() == "true",
        show_error=True,
        allowed_paths=[str(resolved_output_dir())],
    )


if __name__ == "__main__":
    main()
