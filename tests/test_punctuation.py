from typing import Any

from parakeet_transcribe import punctuation
from parakeet_transcribe.types import (
    Segment,
    TranscriptionError,
    TranscriptResult,
    WordTimestamp,
)


def _result(words: list[WordTimestamp], *, text: str = "", segments: list[Segment] | None = None) -> TranscriptResult:
    return TranscriptResult(
        schema_version="1.0",
        source_name="clip.wav",
        duration_seconds=60.0,
        model_id="nvidia/parakeet-tdt-1.1b",
        text=text or " ".join(w.text for w in words),
        words=words,
        segments=segments or [],
    )


def _words(*forms: str) -> list[WordTimestamp]:
    return [WordTimestamp(form, i * 1.0, i * 1.0 + 0.5) for i, form in enumerate(forms)]


class _FakePunctuator:
    def __init__(self, sentences_per_input: list[list[str]]) -> None:
        self._sentences = sentences_per_input

    def infer(self, inputs: list[str]) -> list[list[str]]:
        return list(self._sentences)


def _patch(monkeypatch, sentences: list[str]) -> None:
    monkeypatch.setattr(punctuation, "_punctuator_model", _FakePunctuator([sentences]))


def test_strict_alignment_rewrites_words_and_builds_sentence_segments(monkeypatch) -> None:
    _patch(
        monkeypatch,
        ["Hey doing alright, hope you're having a good night.", "Hi Edgar."],
    )
    words = _words("hey", "doing", "alright", "hope", "you're", "having", "a", "good", "night", "hi", "edgar")
    result = _result(words)
    out = punctuation.restore_punctuation(result)

    assert out.text == "Hey doing alright, hope you're having a good night. Hi Edgar."
    assert out.runtime["punctuation_restored"] is True
    assert out.schema_version == "1.1"
    # Word surfaces carry restored casing + trailing punctuation.
    assert [w.text for w in out.words] == [
        "Hey",
        "doing",
        "alright,",
        "hope",
        "you're",
        "having",
        "a",
        "good",
        "night.",
        "Hi",
        "Edgar.",
    ]
    # Sentence-based segments span the underlying word times.
    assert len(out.segments) == 2
    assert out.segments[0].text == "Hey doing alright, hope you're having a good night."
    assert out.segments[0].start == 0.0
    assert out.segments[0].end == 8.5
    assert out.segments[1].text == "Hi Edgar."
    assert out.segments[1].start == 9.0
    assert out.segments[1].end == 10.5


def test_mismatch_falls_back_to_sentence_level_keep_words_lowercase(monkeypatch) -> None:
    # Model merges "u s" into one token "U.S.", breaking the 1:1 stem match.
    _patch(monkeypatch, ["I live in the U.S.", "It's cold here."])
    words = _words("i", "live", "in", "the", "u", "s", "it's", "cold", "here")
    result = _result(words)
    out = punctuation.restore_punctuation(result)

    assert out.text == "I live in the U.S. It's cold here."
    assert out.runtime["punctuation_restored"] is True
    # Words kept lowercase (their original surface) on merge mismatch.
    assert [w.text for w in out.words] == ["i", "live", "in", "the", "u", "s", "it's", "cold", "here"]
    assert any("sentence level" in w for w in out.warnings)
    # Sentence cues rebuilt from restored sentences; the merge point ("u s" -> "U.S.")
    # stops the first segment at "the" since the merged token stem no longer matches.
    assert [s.text for s in out.segments] == ["I live in the U.S.", "It's cold here."]
    assert out.segments[0].start == 0.0
    assert out.segments[0].end == 3.5
    assert out.segments[1].start == 6.0
    assert out.segments[1].end == 8.5


def test_empty_transcript_is_noop(monkeypatch) -> None:
    _patch(monkeypatch, [])
    result = _result([], text="")
    out = punctuation.restore_punctuation(result)
    assert out.text == ""
    assert out.runtime.get("punctuation_restored") is False
    assert out.words == []


def test_no_sentences_emits_warning(monkeypatch) -> None:
    _patch(monkeypatch, [])
    words = _words("hello", "world")
    result = _result(words)
    out = punctuation.restore_punctuation(result)
    assert out.runtime.get("punctuation_restored") is False
    assert any("returned no output" in w for w in out.warnings)
    # Original words/segments preserved.
    assert [w.text for w in out.words] == ["hello", "world"]


def test_missing_punctuators_raises_transcription_error(monkeypatch) -> None:
    monkeypatch.setattr(punctuation, "_punctuator_model", None)

    def _fail() -> Any:
        raise TranscriptionError("punctuators missing")

    monkeypatch.setattr(punctuation, "_load_punctuator", _fail)
    words = _words("hello", "world")
    result = _result(words)
    try:
        punctuation.restore_punctuation(result)
    except TranscriptionError as exc:
        assert "punctuators missing" in str(exc)
    else:
        raise AssertionError("expected TranscriptionError")


def test_unload_punctuation_model_clears_singleton(monkeypatch) -> None:
    monkeypatch.setattr(punctuation, "_punctuator_model", object())
    punctuation.unload_punctuation_model()
    assert punctuation._punctuator_model is None


def _words_with_speaker(*forms: str) -> list[WordTimestamp]:
    return [WordTimestamp(form, i * 1.0, i * 1.0 + 0.5, speaker="SPEAKER_00") for i, form in enumerate(forms)]


def test_strict_alignment_preserves_word_speaker_and_confidence(monkeypatch) -> None:
    _patch(monkeypatch, ["Hey Edgar."])
    words = [
        WordTimestamp("hey", 0.0, 0.5, speaker="SPEAKER_00", confidence=0.9),
        WordTimestamp("edgar", 1.0, 1.5, speaker="SPEAKER_01", confidence=0.8),
    ]
    result = _result(words)
    out = punctuation.restore_punctuation(result)
    assert [(w.text, w.speaker, w.confidence) for w in out.words] == [
        ("Hey", "SPEAKER_00", 0.9),
        ("Edgar.", "SPEAKER_01", 0.8),
    ]