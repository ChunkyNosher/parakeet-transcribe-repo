from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .types import Segment, TranscriptResult, WordTimestamp

SAMPLE_RATE = 16000
SORTFORMER_MODEL_ID = "nvidia/diar_sortformer_4spk-v1"

_sortformer_model: Any | None = None


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


def _parse_sortformer_segment(raw: Any) -> tuple[float, float, int] | None:
    """Normalize Sortformer outputs to (begin_s, end_s, speaker_index)."""

    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        begin, end, speaker = raw[0], raw[1], raw[2]
        if isinstance(begin, (int, float)) and isinstance(end, (int, float)):
            return float(begin), float(end), int(speaker)
    if isinstance(raw, str):
        parts = raw.replace(",", " ").split()
        if len(parts) >= 3:
            try:
                return float(parts[0]), float(parts[1]), int(float(parts[2]))
            except ValueError:
                return None
    begin = getattr(raw, "start", getattr(raw, "begin", None))
    end = getattr(raw, "end", None)
    speaker = getattr(raw, "speaker", getattr(raw, "speaker_index", None))
    if (
        isinstance(begin, (int, float))
        and isinstance(end, (int, float))
        and isinstance(speaker, (int, float, str))
    ):
        return float(begin), float(end), int(speaker)
    return None


def _speakers_from_rttm(
    rttm: Sequence[tuple[float, float, int]], starts: list[float], ends: list[float]
) -> list[str | None]:
    """Assign SPEAKER_XX by maximum overlap with Sortformer segments."""

    assigned: list[str | None] = []
    for start, end in zip(starts, ends, strict=True):
        best_speaker: int | None = None
        best_overlap = 0.0
        midpoint = (start + end) / 2.0
        nearest_speaker: int | None = None
        nearest_distance = float("inf")
        for begin, finish, speaker in rttm:
            overlap = max(0.0, min(end, finish) - max(start, begin))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
            center = (begin + finish) / 2.0
            distance = abs(midpoint - center)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_speaker = speaker
        label = best_speaker if best_overlap > 0 else nearest_speaker
        assigned.append(None if label is None else f"SPEAKER_{int(label):02d}")
    return assigned


def unload_sortformer() -> None:
    """Release Sortformer weights and free CUDA cache when possible."""

    global _sortformer_model
    if _sortformer_model is None:
        return
    try:
        import torch

        try:
            _sortformer_model.to("cpu")
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        _sortformer_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        _sortformer_model = None


def _load_sortformer() -> Any:
    global _sortformer_model
    if _sortformer_model is not None:
        return _sortformer_model
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for Sortformer diarization.")
    model = _load_sortformer_model()
    model.eval()
    model.to(torch.device("cuda"))
    _sortformer_model = model
    return model


def _load_sortformer_model() -> Any:
    """Instantiate Sortformer, preferring a persistent extracted checkpoint.

    Mirrors the ASR backend's speedups: restore from the unpacked directory so
    NeMo skips tar decompression on every diarize job (Sortformer is loaded
    fresh per diarization run), using the same fast restore path (overlapped
    GPU weight I/O, FP16 safetensors once converted) with a NeMo fallback.
    """
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

    from .models import get_model
    from .modelstore import ensure_extracted, extract_after_load, restore_extracted_model

    sortformer_spec = get_model("sortformer")
    extracted = ensure_extracted(sortformer_spec)
    if extracted is not None:
        try:
            return restore_extracted_model(sortformer_spec, SortformerEncLabelModel, extracted)
        except Exception as exc:
            print(
                f"Fast checkpoint restore failed for {sortformer_spec.model_id} ({exc}); "
                "falling back to NeMo restore_from.",
                flush=True,
            )
        connector = SaveRestoreConnector()
        connector.model_extracted_dir = str(extracted)
        return SortformerEncLabelModel.restore_from(
            str(extracted),
            save_restore_connector=connector,
        )
    model = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL_ID)
    extract_after_load(sortformer_spec)
    return model


def _sortformer_rttm(audio_path: str | Path) -> list[tuple[float, float, int]]:
    model = _load_sortformer()
    import torch

    with torch.inference_mode():
        raw = model.diarize(audio=[str(audio_path)], batch_size=1)
    if isinstance(raw, tuple):
        raw = raw[0]
    if not raw:
        return []
    first = raw[0]
    if _parse_sortformer_segment(first) is not None:
        file_segments = raw
    elif isinstance(first, (list, tuple)):
        file_segments = first
    else:
        file_segments = raw
    parsed: list[tuple[float, float, int]] = []
    for item in file_segments or []:
        segment = _parse_sortformer_segment(item)
        if segment is not None and segment[1] >= segment[0]:
            parsed.append(segment)
    return parsed


def _apply_speaker_labels(
    result: TranscriptResult,
    speakers: Sequence[str | None],
    *,
    method: str,
    speaker_count: int,
    warning: str,
) -> TranscriptResult:
    words = [
        WordTimestamp(word.text, word.start, word.end, speaker=speaker, confidence=word.confidence)
        for word, speaker in zip(result.words, speakers, strict=True)
    ]
    segments = [
        Segment(segment.text, segment.start, segment.end, _speaker_for_segment(segment, words))
        for segment in result.segments
    ]
    return replace(
        result,
        words=words,
        segments=segments,
        schema_version="1.1",
        warnings=[*result.warnings, warning],
        runtime={
            **result.runtime,
            "diarization": {"speakers": speaker_count, "method": method},
        },
    )


def _diarize_mfcc(
    result: TranscriptResult,
    samples: np.ndarray,
    sample_rate: int,
    *,
    num_speakers: int | None,
    fallback_note: str | None = None,
) -> TranscriptResult:
    times, features = _frame_features(samples, sample_rate)
    if not len(features):
        warning = "Diarization skipped: no voiced frames detected."
        if fallback_note:
            warning = f"{fallback_note} {warning}"
        return replace(result, warnings=[*result.warnings, warning], schema_version="1.1")

    speaker_count = num_speakers or _choose_speaker_count(features)
    frame_labels = _kmeans(features, speaker_count)
    speakers = _labels_for_times(
        times, frame_labels, [word.start for word in result.words], [word.end for word in result.words]
    )
    warning = f"Local MFCC diarization labeled {speaker_count} speaker cluster(s)."
    if fallback_note:
        warning = f"{fallback_note} {warning}"
    return _apply_speaker_labels(
        result,
        speakers,
        method="mfcc-kmeans",
        speaker_count=speaker_count,
        warning=warning,
    )


def diarize_transcript(
    result: TranscriptResult,
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    num_speakers: int | None = None,
    audio_path: str | Path | None = None,
    prefer_sortformer: bool = True,
    release_vram: Callable[[], None] | None = None,
) -> TranscriptResult:
    """Attach speaker labels to words/segments.

    Prefers NeMo Sortformer on CUDA (commercial-parity GPU diarization). Callers should
    pass ``release_vram`` to unload the ASR model first so Sortformer fits in VRAM.
    Falls back to CPU MFCC + k-means when Sortformer is unavailable or fails.
    """

    if not result.words:
        warning = "Diarization skipped: transcript has no word timestamps to align speakers."
        return replace(
            result,
            warnings=[*result.warnings, warning],
            schema_version="1.1",
        )

    if prefer_sortformer and audio_path is not None:
        try:
            if release_vram is not None:
                release_vram()
            rttm = _sortformer_rttm(audio_path)
            unload_sortformer()
            if rttm:
                speakers = _speakers_from_rttm(
                    rttm,
                    [word.start for word in result.words],
                    [word.end for word in result.words],
                )
                speaker_ids = {speaker for speaker in speakers if speaker}
                return _apply_speaker_labels(
                    result,
                    speakers,
                    method="sortformer",
                    speaker_count=len(speaker_ids) or len({spk for _, _, spk in rttm}),
                    warning=(
                        f"Sortformer GPU diarization labeled {len(speaker_ids) or len({spk for _, _, spk in rttm})} "
                        "speaker(s)."
                    ),
                )
            fallback_note = "Sortformer returned no speaker segments;"
        except Exception as exc:
            unload_sortformer()
            fallback_note = f"Sortformer unavailable ({exc});"
        return _diarize_mfcc(
            result,
            samples,
            sample_rate,
            num_speakers=num_speakers,
            fallback_note=fallback_note,
        )

    return _diarize_mfcc(result, samples, sample_rate, num_speakers=num_speakers)
