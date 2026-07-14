from unittest.mock import patch

import numpy as np

from parakeet_transcribe.diarization import _speaker_for_segment, diarize_transcript
from parakeet_transcribe.types import Segment, TranscriptResult, WordTimestamp


def test_speaker_for_segment_majority_vote() -> None:
    segment = Segment("Hello there.", 0.0, 1.0)
    words = [
        WordTimestamp("Hello", 0.0, 0.4, speaker="SPEAKER_00"),
        WordTimestamp("there.", 0.4, 0.9, speaker="SPEAKER_00"),
        WordTimestamp("x", 0.5, 0.6, speaker="SPEAKER_01"),
    ]
    assert _speaker_for_segment(segment, words) == "SPEAKER_00"


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
        labeled = diarize_transcript(result, samples, 16000)

    assert len(labeled.segments) == 1
    assert labeled.segments[0].text == "Hello there."
    assert labeled.segments[0].start == 0.0
    assert labeled.segments[0].end == 0.9
    assert labeled.segments[0].speaker is not None
    assert all(word.speaker for word in labeled.words)
