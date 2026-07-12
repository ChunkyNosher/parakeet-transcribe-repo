import json

from parakeet_transcribe.exports import create_run_directory, write_bundle, write_result
from parakeet_transcribe.types import Segment, TranscriptResult, WordTimestamp


def _result(with_timing: bool = True) -> TranscriptResult:
    words = [WordTimestamp("Hello", 0.0, 0.4), WordTimestamp("world.", 0.4, 0.9)] if with_timing else []
    segments = [Segment("Hello world.", 0.0, 0.9)] if with_timing else []
    return TranscriptResult(
        "1.0", "meeting.mp3", 0.9, "nvidia/test", "Hello world.", "en-US", words, segments
    )


def test_timed_result_writes_all_subtitle_formats(tmp_path) -> None:
    run_dir = create_run_directory(tmp_path)
    files = write_result(_result(), run_dir)
    assert {"txt", "json", "csv", "srt", "vtt"} == set(files)
    assert "00:00:00,000" in files["srt"].read_text(encoding="utf-8")
    assert files["vtt"].read_text(encoding="utf-8").startswith("WEBVTT")
    assert json.loads(files["json"].read_text(encoding="utf-8"))["detected_language"] == "en-US"


def test_untimed_result_never_writes_fake_subtitles(tmp_path) -> None:
    run_dir = create_run_directory(tmp_path)
    files = write_result(_result(with_timing=False), run_dir)
    assert {"txt", "json", "csv"} == set(files)
    bundle = write_bundle([_result(with_timing=False)], run_dir)
    assert bundle.is_file()
