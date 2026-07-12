from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from .types import AudioChunk, Segment, WordTimestamp


def _rms_windows(samples: np.ndarray, sample_rate: int, window_seconds: float = 0.1) -> np.ndarray:
    window = max(1, int(sample_rate * window_seconds))
    usable = len(samples) - (len(samples) % window)
    if usable <= 0:
        return np.asarray([], dtype=np.float32)
    values = samples[:usable].reshape(-1, window)
    return np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=1))


def _nearest_silence_boundary(
    samples: np.ndarray,
    sample_rate: int,
    target: int,
    search_seconds: float,
) -> int:
    radius = int(search_seconds * sample_rate)
    lower = max(1, target - radius)
    upper = min(len(samples) - 1, target + radius)
    if lower >= upper:
        return target

    window = max(1, int(sample_rate * 0.1))
    rms = _rms_windows(samples[lower:upper], sample_rate)
    if not len(rms):
        return target
    return min(len(samples) - 1, lower + int(np.argmin(rms)) * window)


def split_audio(
    samples: np.ndarray,
    sample_rate: int,
    source_name: str,
    *,
    chunk_seconds: int = 120,
    overlap_seconds: float = 2.0,
    silence_search_seconds: float = 5.0,
) -> list[AudioChunk]:
    """Create bounded, overlap-aware chunks with boundaries shifted toward quiet audio."""

    if samples.ndim != 1:
        raise ValueError("Audio must be mono before chunking")
    if len(samples) == 0:
        return []

    maximum = max(sample_rate, int(chunk_seconds * sample_rate))
    overlap = max(0, int(overlap_seconds * sample_rate))
    chunks: list[AudioChunk] = []
    content_start = 0

    while content_start < len(samples):
        target_end = min(len(samples), content_start + maximum)
        content_end = (
            target_end
            if target_end == len(samples)
            else _nearest_silence_boundary(samples, sample_rate, target_end, silence_search_seconds)
        )
        content_end = max(content_start + sample_rate, content_end)
        actual_start = max(0, content_start - overlap)
        actual_end = min(len(samples), content_end + overlap)
        chunks.append(
            AudioChunk(
                samples=samples[actual_start:actual_end],
                start=actual_start / sample_rate,
                end=actual_end / sample_rate,
                content_start=content_start / sample_rate,
                source_name=source_name,
            )
        )
        content_start = content_end
    return chunks


def _normalize_token(value: str) -> str:
    return re.sub(r"[^\\w']+", "", value.lower())


def merge_text(existing: str, incoming: str, maximum_overlap_words: int = 24) -> str:
    """Join chunk text while dropping an exact normalized overlap at the seam."""

    if not existing:
        return incoming.strip()
    if not incoming:
        return existing.strip()
    left = existing.split()
    right = incoming.split()
    max_size = min(maximum_overlap_words, len(left), len(right))
    overlap = 0
    for size in range(max_size, 0, -1):
        if [_normalize_token(word) for word in left[-size:]] == [
            _normalize_token(word) for word in right[:size]
        ]:
            overlap = size
            break
    return " ".join(left + right[overlap:]).strip()


def merge_words(chunks: Iterable[tuple[AudioChunk, list[WordTimestamp]]]) -> list[WordTimestamp]:
    """Offset chunk-relative timestamps and remove only duplicated overlap words."""

    merged: list[WordTimestamp] = []
    for chunk, words in chunks:
        for word in words:
            adjusted = WordTimestamp(
                word.text, max(0.0, word.start + chunk.start), max(0.0, word.end + chunk.start)
            )
            if adjusted.end <= chunk.content_start + 0.01 and merged:
                continue
            if (
                merged
                and adjusted.start < merged[-1].end
                and _normalize_token(adjusted.text) == _normalize_token(merged[-1].text)
            ):
                continue
            merged.append(adjusted)
    return merged


def segments_from_words(
    words: list[WordTimestamp], max_words: int = 12, max_duration: float = 8.0
) -> list[Segment]:
    if not words:
        return []
    segments: list[Segment] = []
    current: list[WordTimestamp] = []
    for word in words:
        current.append(word)
        duration = current[-1].end - current[0].start
        punctuated = word.text.rstrip().endswith((".", "?", "!"))
        if len(current) >= max_words or duration >= max_duration or punctuated:
            segments.append(
                Segment(" ".join(item.text for item in current), current[0].start, current[-1].end)
            )
            current = []
    if current:
        segments.append(Segment(" ".join(item.text for item in current), current[0].start, current[-1].end))
    return segments
