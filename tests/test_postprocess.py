import numpy as np

from parakeet_transcribe.diarization import diarize_transcript
from parakeet_transcribe.postprocess import apply_postprocess, redact_pii_text
from parakeet_transcribe.types import Segment, TranscriptResult, WordTimestamp


def test_redact_pii_masks_common_patterns() -> None:
    text = "Email me at ada@example.com or call +1 (555) 123-4567 about 123-45-6789."
    redacted = redact_pii_text(text)
    assert "ada@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_SSN]" in redacted


def test_postprocess_summary_and_chapters() -> None:
    result = TranscriptResult(
        "1.0",
        "talk.wav",
        30.0,
        "nvidia/test",
        "Hello world. More words later.",
        segments=[
            Segment("Hello world.", 0.0, 2.0),
            Segment("More words later.", 12.0, 14.0),
        ],
    )
    updated = apply_postprocess(result, summarize=True)
    assert updated.chapters
    assert updated.summary
    assert updated.schema_version == "1.1"


def test_diarization_labels_words_without_changing_timings() -> None:
    # Two alternating tones to encourage two clusters.
    part_a = 0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, 16000, endpoint=False))
    part_b = 0.2 * np.sin(2 * np.pi * 880 * np.linspace(0, 1.0, 16000, endpoint=False))
    samples = np.concatenate([part_a, part_b]).astype(np.float32)
    result = TranscriptResult(
        "1.0",
        "dialog.wav",
        2.0,
        "nvidia/test",
        "Hello there.",
        words=[
            WordTimestamp("Hello", 0.2, 0.6),
            WordTimestamp("there.", 1.2, 1.6),
        ],
        segments=[Segment("Hello there.", 0.2, 1.6)],
    )
    labeled = diarize_transcript(result, samples, 16000, num_speakers=2)
    assert all(word.speaker for word in labeled.words)
    assert labeled.words[0].start == 0.2
    assert labeled.words[1].end == 1.6
    assert labeled.schema_version == "1.1"
    assert any("diarization" in warning.lower() or "speaker" in warning.lower() for warning in labeled.warnings)
