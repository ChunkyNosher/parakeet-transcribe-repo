from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from parakeet_transcribe.diarization import (
    _parse_sortformer_segment,
    _speaker_for_segment,
    _speakers_from_rttm,
    diarize_transcript,
)
from parakeet_transcribe.types import Segment, TranscriptResult, WordTimestamp


def test_speaker_for_segment_majority_vote() -> None:
    segment = Segment("Hello there.", 0.0, 1.0)
    words = [
        WordTimestamp("Hello", 0.0, 0.4, speaker="SPEAKER_00"),
        WordTimestamp("there.", 0.4, 0.9, speaker="SPEAKER_00"),
        WordTimestamp("x", 0.5, 0.6, speaker="SPEAKER_01"),
    ]
    assert _speaker_for_segment(segment, words) == "SPEAKER_00"


def test_parse_sortformer_segment_formats() -> None:
    assert _parse_sortformer_segment((0.0, 1.5, 0)) == (0.0, 1.5, 0)
    assert _parse_sortformer_segment("0.10 0.80 1") == (0.1, 0.8, 1)
    assert _parse_sortformer_segment(SimpleNamespace(start=1.0, end=2.0, speaker=2)) == (1.0, 2.0, 2)


def test_speakers_from_rttm_overlap_align() -> None:
    rttm = [(0.0, 0.5, 0), (0.5, 1.0, 1)]
    speakers = _speakers_from_rttm(rttm, [0.0, 0.6], [0.4, 0.9])
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]


def test_diarize_keeps_native_segment_boundaries() -> None:
    result = TranscriptResult(
        schema_version="1.0",
        source_name="clip.wav",
        duration_seconds=2.0,
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        text="Hello there.",
        words=[
            WordTimestamp("Hello", 0.0, 0.4),
            WordTimestamp("there.", 0.5, 0.9),
        ],
        segments=[Segment("Hello there.", 0.0, 0.9)],
        runtime={"segment_source": "nemo_native"},
    )
    samples = np.random.default_rng(0).normal(0, 0.1, size=16000).astype(np.float32)

    with (
        patch(
            "parakeet_transcribe.diarization._frame_features",
            return_value=(
                np.asarray([0.1, 0.3, 0.6, 0.8]),
                np.ones((4, 2), dtype=np.float64),
            ),
        ),
        patch("parakeet_transcribe.diarization._kmeans", return_value=np.asarray([0, 0, 1, 1])),
        patch("parakeet_transcribe.diarization._choose_speaker_count", return_value=2),
    ):
        labeled = diarize_transcript(result, samples, 16000, prefer_sortformer=False)

    assert len(labeled.segments) == 1
    assert labeled.segments[0].text == "Hello there."
    assert labeled.segments[0].start == 0.0
    assert labeled.segments[0].end == 0.9
    assert labeled.segments[0].speaker is not None
    assert all(word.speaker for word in labeled.words)
    assert labeled.runtime["diarization"]["method"] == "mfcc-kmeans"


def test_sortformer_path_aligns_and_releases_vram(tmp_path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    result = TranscriptResult(
        schema_version="1.0",
        source_name="clip.wav",
        duration_seconds=1.0,
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        text="Hello there.",
        words=[
            WordTimestamp("Hello", 0.0, 0.4, confidence=0.9),
            WordTimestamp("there.", 0.5, 0.9, confidence=0.8),
        ],
        segments=[Segment("Hello there.", 0.0, 0.9)],
    )
    samples = np.zeros(16000, dtype=np.float32)
    released = MagicMock()

    with (
        patch(
            "parakeet_transcribe.diarization._sortformer_rttm",
            return_value=[(0.0, 0.45, 0), (0.45, 1.0, 1)],
        ) as sortformer,
        patch("parakeet_transcribe.diarization.unload_sortformer") as unload,
    ):
        labeled = diarize_transcript(
            result,
            samples,
            16000,
            audio_path=audio,
            release_vram=released,
        )

    released.assert_called_once()
    sortformer.assert_called_once()
    unload.assert_called()
    assert [word.speaker for word in labeled.words] == ["SPEAKER_00", "SPEAKER_01"]
    assert labeled.words[0].confidence == 0.9
    assert labeled.runtime["diarization"]["method"] == "sortformer"
    assert "Sortformer GPU" in labeled.warnings[-1]


def test_sortformer_failure_falls_back_to_mfcc(tmp_path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    result = TranscriptResult(
        schema_version="1.0",
        source_name="clip.wav",
        duration_seconds=1.0,
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        text="Hello.",
        words=[WordTimestamp("Hello", 0.0, 0.4)],
        segments=[Segment("Hello.", 0.0, 0.4)],
    )
    samples = np.zeros(16000, dtype=np.float32)

    with (
        patch(
            "parakeet_transcribe.diarization._sortformer_rttm",
            side_effect=RuntimeError("CUDA OOM"),
        ),
        patch("parakeet_transcribe.diarization.unload_sortformer"),
        patch(
            "parakeet_transcribe.diarization._frame_features",
            return_value=(np.asarray([0.1]), np.ones((1, 2))),
        ),
        patch("parakeet_transcribe.diarization._kmeans", return_value=np.asarray([0])),
        patch("parakeet_transcribe.diarization._choose_speaker_count", return_value=1),
    ):
        labeled = diarize_transcript(result, samples, 16000, audio_path=audio)

    assert labeled.runtime["diarization"]["method"] == "mfcc-kmeans"
    assert any("Sortformer unavailable" in warning for warning in labeled.warnings)
    assert labeled.words[0].speaker == "SPEAKER_00"
