from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelCapabilities:
    timestamps: bool
    automatic_language_detection: bool
    supported_languages: int


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_id: str
    model_class: str
    capabilities: ModelCapabilities
    default_batch_size: int = 1


@dataclass(frozen=True)
class WordTimestamp:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass
class TranscriptResult:
    schema_version: str
    source_name: str
    duration_seconds: float
    model_id: str
    text: str
    detected_language: str | None = None
    words: list[WordTimestamp] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    chapters: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_timestamps(self) -> bool:
        return bool(self.words or self.segments)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioChunk:
    samples: Any
    start: float
    end: float
    content_start: float
    source_name: str


@dataclass(frozen=True)
class ChunkResult:
    text: str
    words: list[WordTimestamp]
    detected_language: str | None = None
    segments: list[Segment] = field(default_factory=list)


@dataclass(frozen=True)
class PreparedAudio:
    source_path: Path
    canonical_path: Path
    samples: Any
    sample_rate: int
    duration_seconds: float


class TranscriptionError(RuntimeError):
    """An expected user-facing transcription failure."""


class CancelledError(TranscriptionError):
    """Raised when a queued task is cancelled before publication."""
