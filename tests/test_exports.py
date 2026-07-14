import json
import zipfile

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


def test_bundle_excludes_scratch_work_audio(tmp_path) -> None:
    run_dir = create_run_directory(tmp_path)
    write_result(_result(), run_dir)
    work = run_dir / ".work"
    work.mkdir()
    fat = work / "canonical.wav"
    fat.write_bytes(b"0" * 1_000_000)
    youtube = run_dir / ".youtube"
    youtube.mkdir()
    (youtube / "clip.m4a").write_bytes(b"1" * 100_000)

    bundle = write_bundle([_result()], run_dir)
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
    assert "manifest.json" in names
    assert any(name.endswith(".json") for name in names)
    assert all(".work/" not in name and not name.startswith(".work/") for name in names)
    assert all(".youtube/" not in name and not name.startswith(".youtube/") for name in names)
    assert bundle.stat().st_size < 50_000


def test_speaker_labels_appear_in_srt(tmp_path) -> None:
    run_dir = create_run_directory(tmp_path)
    result = TranscriptResult(
        "1.1",
        "dialog.wav",
        1.0,
        "nvidia/test",
        "Hello there.",
        words=[
            WordTimestamp("Hello", 0.0, 0.4, "SPEAKER_00", 0.95),
            WordTimestamp("there.", 0.4, 0.9, "SPEAKER_01", 0.88),
        ],
        segments=[
            Segment("Hello", 0.0, 0.4, "SPEAKER_00"),
            Segment("there.", 0.4, 0.9, "SPEAKER_01"),
        ],
    )
    files = write_result(result, run_dir)
    assert "[SPEAKER_00] Hello" in files["srt"].read_text(encoding="utf-8")
    payload = json.loads(files["json"].read_text(encoding="utf-8"))
    assert payload["words"][0]["confidence"] == 0.95
    csv_text = files["csv"].read_text(encoding="utf-8")
    assert "confidence" in csv_text.splitlines()[0]
    assert "0.950000" in csv_text
