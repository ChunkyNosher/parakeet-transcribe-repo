import numpy as np

from parakeet_transcribe.chunking import merge_text, merge_words, segments_from_words, split_audio
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
