from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .types import PreparedAudio, TranscriptionError


def require_ffmpeg() -> None:
    missing = [command for command in ("ffmpeg", "ffprobe") if shutil.which(command) is None]
    if missing:
        joined = ", ".join(missing)
        raise TranscriptionError(
            f"Missing required media tool(s): {joined}. Install FFmpeg and add it to PATH."
        )


def prepare_audio(source_path: str | Path, work_dir: Path) -> PreparedAudio:
    require_ffmpeg()
    source = Path(source_path).resolve()
    if not source.is_file():
        raise TranscriptionError(f"Input file does not exist: {source}")
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / f"{source.stem}.canonical.wav"
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_f32le",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or "FFmpeg could not decode this file."
        raise TranscriptionError(f"Could not normalize {source.name}: {detail}")
    samples, sample_rate = sf.read(target, dtype="float32", always_2d=False)
    if samples.ndim != 1:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    if len(samples) < int(sample_rate * 0.1):
        raise TranscriptionError(f"{source.name} is shorter than 0.1 seconds.")
    if not np.isfinite(samples).all():
        raise TranscriptionError(f"{source.name} contains invalid audio samples.")
    if float(np.max(np.abs(samples))) < 0.001:
        raise TranscriptionError(f"{source.name} appears to contain no audible speech.")
    return PreparedAudio(source, target, samples, int(sample_rate), len(samples) / sample_rate)
