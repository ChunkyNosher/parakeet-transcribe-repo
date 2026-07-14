import shutil
import subprocess

import pytest

from parakeet_transcribe.media import prepare_audio


def test_prepare_audio_accepts_aac_m4a(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is required for media normalization")
    source = tmp_path / "speech.m4a"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "0.2",
            "-c:a",
            "aac",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    prepared = prepare_audio(source, tmp_path / "work")
    assert prepared.sample_rate == 16_000
    assert prepared.duration_seconds == pytest.approx(0.2, abs=0.05)
    assert prepared.canonical_path.is_file()
