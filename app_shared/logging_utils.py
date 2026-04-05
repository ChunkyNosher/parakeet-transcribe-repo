import io
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .env_bootstrap import APP_ROOT


class TeeWriter:
    """Mirror writes to the original stream and an in-memory buffer."""

    def __init__(self, original: Any, buffer: io.StringIO) -> None:
        self.original = original
        self.buffer = buffer

    def write(self, text: str) -> int:
        written = self.original.write(text)
        self.buffer.write(text)
        return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
        self.original.flush()


class LogCapture:
    """Capture stdout, stderr, and logging output for download."""

    def __init__(self) -> None:
        self.log_buffer = io.StringIO()
        self.original_stdout: Optional[Any] = None
        self.original_stderr: Optional[Any] = None
        self.handler: Optional[logging.Handler] = None

    def start(self) -> None:
        self.log_buffer = io.StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        sys.stdout = TeeWriter(self.original_stdout, self.log_buffer)  # type: ignore[assignment]
        sys.stderr = TeeWriter(self.original_stderr, self.log_buffer)  # type: ignore[assignment]

        self.handler = logging.StreamHandler(self.log_buffer)
        self.handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self.handler)

    def stop(self) -> str:
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout
        if self.original_stderr is not None:
            sys.stderr = self.original_stderr
        if self.handler is not None:
            logging.getLogger().removeHandler(self.handler)
        return self.log_buffer.getvalue()

    def get_logs(self) -> str:
        return self.log_buffer.getvalue()


def save_logs(logs: str, prefix: str = "transcription", app_root: Optional[Path] = None) -> Optional[str]:
    """Persist captured logs under logs/transcription or logs/error."""

    if not logs:
        return None

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = app_root or APP_ROOT
        log_subdir = "error" if prefix == "error" else "transcription"
        log_dir = root / "logs" / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / f"{prefix}_log_{timestamp}.txt"
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(f"Transcription Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            handle.write("=" * 60 + "\n\n")
            handle.write(logs)

        return str(log_path)
    except Exception as exc:
        print(f"Failed to save log file: {exc}")
        return None


_save_logs = save_logs
log_capture = LogCapture()


__all__ = ["LogCapture", "TeeWriter", "_save_logs", "log_capture", "save_logs"]