from pathlib import Path

from parakeet_transcribe.app import _caption_preview_html, _cue_table_rows
from parakeet_transcribe.types import Segment, TranscriptResult


def test_caption_preview_html_includes_audio_and_track(tmp_path: Path) -> None:
    audio = tmp_path / "clip.canonical.wav"
    vtt = tmp_path / "clip.vtt"
    audio.write_bytes(b"RIFF")
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
    html = _caption_preview_html(audio, vtt)
    assert "<audio" in html
    assert "kind=\"captions\"" in html
    assert audio.name in html or "file=" in html
    assert vtt.name in html or "file=" in html


def test_cue_table_rows_from_native_segments() -> None:
    result = TranscriptResult(
        schema_version="1.0",
        source_name="clip.wav",
        duration_seconds=2.0,
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        text="Hello there.",
        segments=[Segment("Hello there.", 0.12, 1.234)],
    )
    assert _cue_table_rows(result) == [[0.12, 1.234, "Hello there."]]
