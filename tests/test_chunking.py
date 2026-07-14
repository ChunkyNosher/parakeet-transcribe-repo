import numpy as np

from parakeet_transcribe.chunking import (
    merge_text,
    merge_words,
    normalize_word_timing,
    segments_from_words,
    split_audio,
)
from parakeet_transcribe.types import AudioChunk, WordTimestamp


def test_split_audio_has_context_and_coverage() -> None:
    samples = np.ones(16000 * 245, dtype=np.float32)
    chunks = split_audio(samples, 16000, "recording.wav", chunk_seconds=120, overlap_seconds=2)
    assert len(chunks) == 3
    assert chunks[0].start == 0
    assert chunks[1].start < chunks[1].content_start
    assert chunks[-1].end == 245


def test_merge_text_removes_normalized_overlap() -> None:
    assert merge_text("Hello, world", "world this is new") == "Hello, world this is new"
    assert merge_text("one two", "three four") == "one two three four"


def test_merge_words_offsets_and_removes_overlap() -> None:
    first = AudioChunk(np.array([0]), 0, 4, 0, "a.wav")
    second = AudioChunk(np.array([0]), 2, 6, 4, "a.wav")
    merged = merge_words(
        [
            (first, [WordTimestamp("Hello", 0.0, 0.5), WordTimestamp("world", 1.0, 1.5)]),
            (second, [WordTimestamp("world", 0.0, 0.5), WordTimestamp("again", 2.0, 2.5)]),
        ]
    )
    assert [item.text for item in merged] == ["Hello", "world", "again"]
    assert merged[-1].start == 4.0


def test_segments_are_bounded_by_punctuation() -> None:
    words = [
        WordTimestamp("Hello", 0, 0.3),
        WordTimestamp("there.", 0.3, 0.7),
        WordTimestamp("Next", 0.8, 1.1),
    ]
    assert [segment.text for segment in segments_from_words(words)] == ["Hello there.", "Next"]


def test_segments_split_on_long_silence_before_next_sentence_start() -> None:
    """Mode A: post-pause first word must not stay on the silence-heavy prior cue."""

    words = [
        WordTimestamp("We", 30.0, 30.2),
        WordTimestamp("were", 30.3, 30.5),
        WordTimestamp("done", 30.6, 30.9),
        WordTimestamp("With", 54.5, 54.8),
        WordTimestamp("that", 54.9, 55.1),
        WordTimestamp("said,", 55.2, 55.5),
        WordTimestamp("we", 55.6, 55.7),
        WordTimestamp("continue.", 55.8, 56.2),
    ]
    segments = segments_from_words(words)
    assert segments[0].text == "We were done"
    assert segments[0].end <= 31.0
    assert segments[1].text.startswith("With")
    assert segments[1].start >= 54.0


def test_segments_clamp_stretched_word_start_across_silence() -> None:
    """Mode B: ASR glued start across silence; clamp then gap-split."""

    words = [
        WordTimestamp("We", 30.0, 30.2),
        WordTimestamp("were", 30.3, 30.5),
        WordTimestamp("done.", 30.6, 30.9),
        WordTimestamp("With", 30.9, 54.8),
        WordTimestamp("that", 54.9, 55.1),
        WordTimestamp("said,", 55.2, 55.5),
        WordTimestamp("we", 55.6, 55.7),
        WordTimestamp("continue.", 55.8, 56.2),
    ]
    segments = segments_from_words(words)
    assert segments[0].text == "We were done."
    assert segments[0].end <= 31.0
    with_segment = next(segment for segment in segments if segment.text.startswith("With"))
    # Clamped start (~54.8 - 2.0); must not keep the raw 30.9→54.8 silence span as one cue word.
    assert with_segment.start >= 52.0
    assert "done." not in with_segment.text


def test_short_pause_does_not_force_split() -> None:
    words = [
        WordTimestamp("Hello", 0.0, 0.3),
        WordTimestamp("there", 0.5, 0.8),
    ]
    segments = segments_from_words(words)
    assert len(segments) == 1
    assert segments[0].text == "Hello there"


def test_normalize_word_timing_clamps_long_span() -> None:
    start, end = normalize_word_timing(10.0, 40.0, max_word_duration=2.0)
    assert end == 40.0
    assert start == 38.0
