from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .types import Segment, TranscriptionError, TranscriptResult, WordTimestamp

_PUNCTUATOR_MODEL_ID = "pcs_en"
_punctuator_model: Any | None = None

# Strip leading/trailing punctuation/whitespace for stem comparison.
_STEM_STRIP = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")


def _normalize_stem(token: str) -> str:
    return _STEM_STRIP.sub("", token).lower()


def unload_punctuation_model() -> None:
    """Release the ONNX punctuation model (CPU) and clear the singleton."""

    global _punctuator_model
    _punctuator_model = None


def _load_punctuator() -> Any:
    global _punctuator_model
    if _punctuator_model is not None:
        return _punctuator_model
    try:
        from punctuators.models import PunctCapSegModelONNX
    except ImportError as exc:  # pragma: no cover - container-only dependency
        raise TranscriptionError(
            "Punctuation restoration requires the `punctuators` and `onnxruntime` "
            "packages. Rebuild the Docker image (Linux GPU) where they are installed."
        ) from exc
    try:
        _punctuator_model = PunctCapSegModelONNX.from_pretrained(_PUNCTUATOR_MODEL_ID)
    except Exception as exc:  # pragma: no cover - network/model load
        _punctuator_model = None
        raise TranscriptionError(f"Failed to load the punctuation restoration model: {exc}") from exc
    return _punctuator_model


def _infer_sentences(text: str) -> list[str]:
    model = _load_punctuator()
    results = model.infer([text])
    if not results or not isinstance(results, (list, tuple)):
        return []
    sentences = results[0]
    if not isinstance(sentences, (list, tuple)):
        return []
    return [str(s).strip() for s in sentences if str(s).strip()]


def _restore_strict(
    sentences: list[str], words: list[WordTimestamp]
) -> tuple[list[str] | None, list[Segment], bool]:
    """Strict 1:1 token alignment: rewrite each word's surface (case + punctuation)."""

    restored_tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for sentence in sentences:
        tokens = sentence.split()
        spans.append((len(restored_tokens), len(restored_tokens) + len(tokens)))
        restored_tokens.extend(tokens)

    if not words or len(restored_tokens) != len(words):
        return None, [], False
    if not all(
        _normalize_stem(rt) == _normalize_stem(w.text) for rt, w in zip(restored_tokens, words, strict=True)
    ):
        return None, [], False

    segments: list[Segment] = []
    for start, end in spans:
        seg_words = words[start:end]
        segments.append(
            Segment(
                text=" ".join(restored_tokens[start:end]),
                start=seg_words[0].start,
                end=seg_words[-1].end,
            )
        )
    return restored_tokens, segments, True


def _restore_sentence_level(
    sentences: list[str], words: list[WordTimestamp]
) -> list[Segment]:
    """Greedy sentence-to-word alignment by stem match (fallback for merges/drops)."""

    segments: list[Segment] = []
    word_idx = 0
    total = len(words)
    for sentence in sentences:
        stems = [_normalize_stem(token) for token in sentence.split()]
        if not stems:
            continue
        # Advance to the first word whose stem matches the sentence's first token.
        while word_idx < total and _normalize_stem(words[word_idx].text) != stems[0]:
            word_idx += 1
        start_idx = word_idx
        ti = 0
        while word_idx < total and ti < len(stems) and _normalize_stem(words[word_idx].text) == stems[ti]:
            word_idx += 1
            ti += 1
        if word_idx > start_idx:
            segments.append(Segment(sentence, words[start_idx].start, words[word_idx - 1].end))
    return segments


def restore_punctuation(result: TranscriptResult) -> TranscriptResult:
    """Restore punctuation, capitalization, and sentence cues for a lowercase-vocab model.

    The 1-800-BAD-CODE ONNX model takes lowercased unpunctuated English and emits
    punctuated, true-cased, sentence-segmented text in one pass. We map it back onto
    the ASR word timestamps so SRT/VTT cues become sentence-based.
    """

    words = list(result.words)
    source_text = " ".join(word.text.strip() for word in words) if words else result.text
    if not source_text.strip():
        return replace(result, runtime={**result.runtime, "punctuation_restored": False})

    sentences = _infer_sentences(source_text)
    if not sentences:
        return replace(
            result,
            warnings=[*result.warnings, "Punctuation restoration model returned no output."],
            runtime={**result.runtime, "punctuation_restored": False},
        )

    restored_full = " ".join(sentences)
    surfaces, strict_segments, aligned = _restore_strict(sentences, words)
    warnings: list[str] = []
    if aligned and surfaces is not None:
        new_words = [
            WordTimestamp(
                surface or word.text,
                word.start,
                word.end,
                word.speaker,
                word.confidence,
            )
            for word, surface in zip(words, surfaces, strict=True)
        ]
        segments = strict_segments
    else:
        new_words = words
        segments = _restore_sentence_level(sentences, words)
        if not segments:
            # Sentence alignment failed entirely; keep original timing, restore text only.
            segments = list(result.segments)
            warnings.append(
                "Punctuation restored in full text only; word/segment timing unchanged."
            )
        else:
            warnings.append(
                "Punctuation restored at sentence level; some word surfaces kept lowercase "
                "where the restoration model merged tokens."
            )

    return replace(
        result,
        text=restored_full,
        words=new_words,
        segments=segments,
        warnings=[*result.warnings, *warnings],
        schema_version="1.1",
        runtime={**result.runtime, "punctuation_restored": True},
    )