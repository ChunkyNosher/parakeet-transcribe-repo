from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


TextNormalizer = Callable[[str], str]


@dataclass
class TranscriptionStats:
    """Statistics about a transcription run."""

    model_choice: str
    gpu_name: str
    total_duration: float
    total_time: float
    inference_time: float
    load_time: float
    chunk_size: int
    rtfx: float
    vram_used: float
    apply_itn: bool


@dataclass
class TranscriptionContext:
    """Context for a transcription operation."""

    transcription: str
    timestamps: List[Dict[str, Any]]
    timestamp_level: str
    file_info: List[Dict[str, Any]]


@dataclass
class BatchTranscriptionContext:
    """Context for batch transcription operations."""

    all_transcriptions: List[str]
    all_timestamps: List[Tuple[List[Dict[str, Any]], str]]
    file_info: List[Dict[str, Any]]
    per_file_stats: List[str]
    per_file_errors: List[str]


@dataclass
class OutputFilesConfig:
    """Configuration for output file generation."""

    file_list: List[str]
    file_info: List[Dict[str, Any]]
    is_batch: bool
    include_timestamps: bool
    model_choice: str
    total_duration: float
    total_time: float
    apply_itn: bool
    transcription: Optional[str] = None
    timestamps: Optional[List[Dict[str, Any]]] = None
    timestamp_level: str = "none"
    all_transcriptions: Optional[List[str]] = None
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]] = None
    itn_available: bool = False


@dataclass
class ResultProcessingContext:
    """Context for processing transcription results."""

    stats: TranscriptionStats
    file_list: List[str]
    file_info: List[Dict[str, Any]]
    include_timestamps: bool
    video_status: str = ""
    load_time: float = 0.0
    apply_itn_final: bool = False
    had_itn_per_chunk: bool = False
    all_transcriptions: Optional[List[str]] = None
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]] = None
    text_normalizer: Optional[TextNormalizer] = None
    itn_available: bool = False


@dataclass
class SimpleHypothesis:
    """Minimal normalized transcription result shared across backends."""

    text: str
    timestamp: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    chunk_timestamps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LoadedModelHandle:
    """Loaded backend runtime plus metadata needed for inference dispatch."""

    model_key: str
    backend: str
    runtime: Any
    processor: Optional[Any] = None
    source: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    supports_timestamps: bool = False
    supports_chunking: bool = False
    default_language: Optional[str] = None
    warning: Optional[str] = None


__all__ = [
    "BatchTranscriptionContext",
    "LoadedModelHandle",
    "OutputFilesConfig",
    "ResultProcessingContext",
    "SimpleHypothesis",
    "TextNormalizer",
    "TranscriptionContext",
    "TranscriptionStats",
]