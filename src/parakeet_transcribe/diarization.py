from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from .types import Segment, TranscriptResult, WordTimestamp

SAMPLE_RATE = 16000


def _frame_features(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_times_sec, feature_matrix) for voiced frames using MFCC + energy."""

    import librosa

    if samples.size == 0:
        return np.asarray([], dtype=np.float64), np.zeros((0, 1), dtype=np.float64)

    hop = int(0.02 * sample_rate)
    frame = int(0.04 * sample_rate)
    mfcc = librosa.feature.mfcc(y=samples.astype(np.float32), sr=sample_rate, n_mfcc=13, hop_length=hop, n_fft=frame)
    rms = librosa.feature.rms(y=samples.astype(np.float32), frame_length=frame, hop_length=hop)[0]
    features = np.vstack([mfcc, rms]).T
    times = librosa.frames_to_time(np.arange(len(features)), sr=sample_rate, hop_length=hop)
    if not len(features):
        return times, features
    threshold = max(float(np.percentile(rms, 30)), 1e-6)
    voiced = rms >= threshold
    if not np.any(voiced):
        voiced = np.ones_like(rms, dtype=bool)
    return times[voiced], features[voiced]


def _kmeans(features: np.ndarray, k: int, *, seed: int = 0, iters: int = 25) -> np.ndarray:
    if len(features) == 0:
        return np.asarray([], dtype=np.int64)
    k = max(1, min(k, len(features)))
    rng = np.random.default_rng(seed)
    centers = features[rng.choice(len(features), size=k, replace=False)].astype(np.float64)
    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(iters):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for cluster in range(k):
            members = features[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def _choose_speaker_count(features: np.ndarray, maximum: int = 4) -> int:
    if len(features) < 8:
        return 1
    # Prefer 2 speakers when enough speech; cap exploration for long files.
    return 2 if len(features) >= 40 else 1


def _labels_for_times(
    frame_times: np.ndarray, frame_labels: np.ndarray, starts: list[float], ends: list[float]
) -> list[str | None]:
    if not len(frame_times):
        return [None] * len(starts)
    assigned: list[str | None] = []
    for start, end in zip(starts, ends, strict=True):
        mask = (frame_times >= start) & (frame_times <= max(end, start + 0.01))
        if not np.any(mask):
            # Nearest frame fallback.
            index = int(np.argmin(np.abs(frame_times - (start + end) / 2)))
            label = int(frame_labels[index])
        else:
            values, counts = np.unique(frame_labels[mask], return_counts=True)
            label = int(values[int(np.argmax(counts))])
        assigned.append(f"SPEAKER_{label:02d}")
    return assigned


def _speaker_for_segment(segment: Segment, words: list[WordTimestamp]) -> str | None:
    """Majority speaker among words overlapping the native NeMo segment time range."""

    overlapping = [
        word.speaker
        for word in words
        if word.speaker and word.end > segment.start and word.start < segment.end
    ]
    if not overlapping:
        return segment.speaker
    return Counter(overlapping).most_common(1)[0][0]


def diarize_transcript(
    result: TranscriptResult,
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    num_speakers: int | None = None,
) -> TranscriptResult:
    """Attach local speaker labels to words/segments via MFCC clustering (CPU-only)."""

    if not result.words:
        warning = "Diarization skipped: transcript has no word timestamps to align speakers."
        return replace(
            result,
            warnings=[*result.warnings, warning],
            schema_version="1.1",
        )

    times, features = _frame_features(samples, sample_rate)
    if not len(features):
        warning = "Diarization skipped: no voiced frames detected."
        return replace(result, warnings=[*result.warnings, warning], schema_version="1.1")

    speaker_count = num_speakers or _choose_speaker_count(features)
    frame_labels = _kmeans(features, speaker_count)
    speakers = _labels_for_times(
        times, frame_labels, [word.start for word in result.words], [word.end for word in result.words]
    )
    words = [
        WordTimestamp(word.text, word.start, word.end, speaker=speaker)
        for word, speaker in zip(result.words, speakers, strict=True)
    ]
    # Keep NeMo native cue boundaries; only attach speaker labels.
    segments = [
        Segment(segment.text, segment.start, segment.end, _speaker_for_segment(segment, words))
        for segment in result.segments
    ]
    return replace(
        result,
        words=words,
        segments=segments,
        schema_version="1.1",
        warnings=[*result.warnings, f"Local diarization labeled {speaker_count} speaker cluster(s)."],
        runtime={**result.runtime, "diarization": {"speakers": speaker_count, "method": "mfcc-kmeans"}},
    )
