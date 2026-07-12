from __future__ import annotations

import argparse

from .app import build_app
from .diagnostics import doctor_report


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
    build_app().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
