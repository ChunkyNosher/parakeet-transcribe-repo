import pytest

from parakeet_transcribe.backend import (
    _extract_language,
    _words_from_timestamp_payload,
    is_triton_compiler_error,
    raise_if_triton_compiler_error,
)
from parakeet_transcribe.types import TranscriptionError


def test_tdt_token_spans_align_to_visible_words() -> None:
    payload = [
        {"token": "W", "start": 0.32, "end": 0.40},
        {"token": "ell", "start": 0.40, "end": 0.56},
        {"token": ",", "start": 0.56, "end": 0.56},
        {"token": "I", "start": 0.64, "end": 0.80},
        {"token": "don", "start": 0.80, "end": 0.96},
        {"token": "'t", "start": 0.96, "end": 1.04},
    ]
    words = _words_from_timestamp_payload(payload, "Well, I don't")
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Well,", 0.32, 0.56),
        ("I", 0.64, 0.80),
        ("don't", 0.80, 1.04),
    ]


def test_tdt_space_prefixed_tokens_align_to_visible_words() -> None:
    payload = [
        {"token": "W", "start": 0.96, "end": 1.12},
        {"token": "hat", "start": 1.12, "end": 1.28},
        {"token": " are", "start": 1.28, "end": 1.44},
        {"token": " you", "start": 1.44, "end": 1.52},
        {"token": "?", "start": 1.52, "end": 1.52},
    ]
    words = _words_from_timestamp_payload(payload, "What are you?")
    assert [(word.text, word.start, word.end) for word in words] == [
        ("What", 0.96, 1.28),
        ("are", 1.28, 1.44),
        ("you?", 1.44, 1.52),
    ]


def test_language_tag_is_removed_from_nemotron_transcript() -> None:
    assert _extract_language("Bonjour tout le monde. <fr-FR>") == ("Bonjour tout le monde.", "fr-FR")


def test_triton_compiler_error_is_detected() -> None:
    error = RuntimeError(
        "Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl."
    )
    assert is_triton_compiler_error(error)
    assert not is_triton_compiler_error(RuntimeError("CUDA out of memory"))


def test_triton_compiler_error_becomes_transcription_error() -> None:
    error = RuntimeError("Failed to find C compiler. Please specify via CC")
    with pytest.raises(TranscriptionError, match="build-essential"):
        raise_if_triton_compiler_error(error)
