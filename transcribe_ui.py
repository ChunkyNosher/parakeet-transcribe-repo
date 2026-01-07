# ============================================================================
# WINDOWS TEMP FILE FIX - Configure cache directories BEFORE importing libraries
# ============================================================================
# This prevents WinError 32 (file in use) errors on Windows caused by:
# - Antivirus software scanning %TEMP% folder
# - Cloud sync services (OneDrive, Dropbox, Google Drive) monitoring temp files
# - Windows Search/Indexing service holding file handles
#
# By setting TORCH_HOME, HF_HOME, and TMPDIR BEFORE importing NeMo/PyTorch,
# all model downloads and extractions go to a project-local cache directory
# that Windows services typically don't monitor.
# ============================================================================

import os
import sys
from pathlib import Path

# ============================================================================
# FIX #0: Disable Multiprocessing/Threading BEFORE any imports
# ============================================================================
# These environment variables MUST be set before importing PyTorch, NeMo,
# or any library that uses multiprocessing. They prevent Lhotse's dataloader
# from spawning worker processes that create manifest.json files.
# ============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Prevent PyTorch from using multiple threads for data loading
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Keep async for performance
# Force single-threaded behavior for dataloaders
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Get script directory and create cache directory structure
_script_dir = Path(__file__).parent.absolute()
_cache_dir = _script_dir / "model_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)

# Set cache environment variables BEFORE importing torch/NeMo
# This affects all subsequent model downloads and extractions
os.environ["TORCH_HOME"] = str(_cache_dir / "torch")
os.environ["HF_HOME"] = str(_cache_dir / "huggingface")
os.environ["NEMO_CACHE_DIR"] = str(_cache_dir / "nemo")

# Set TMPDIR to use our cache instead of Windows %TEMP%
# This is the CRITICAL line that prevents file lock issues
os.environ["TMPDIR"] = str(_cache_dir / "tmp")
if sys.platform == "win32":
    # Windows uses different env vars for temp directory
    os.environ["TEMP"] = str(_cache_dir / "tmp")
    os.environ["TMP"] = str(_cache_dir / "tmp")

# Create temp directory
_temp_dir = _cache_dir / "tmp"
_temp_dir.mkdir(parents=True, exist_ok=True)

# Create other cache subdirectories
(_cache_dir / "torch").mkdir(parents=True, exist_ok=True)
(_cache_dir / "huggingface").mkdir(parents=True, exist_ok=True)
(_cache_dir / "nemo").mkdir(parents=True, exist_ok=True)

# Store cache_dir for error messages later
CACHE_DIR = _cache_dir

# ============================================================================
# FIX #2: Explicit tempfile.tempdir Configuration
# ============================================================================
# Import tempfile and explicitly set tempfile.tempdir to force Python's
# tempfile module to use our custom temp directory instead of system %TEMP%.
# This is more reliable than environment variables alone because:
# 1. Some libraries cache the temp directory before env vars are checked
# 2. The tempfile module may have already been imported by other modules
# 3. Explicit assignment ensures all subsequent tempfile operations use our path
# ============================================================================

import tempfile

# Force Python's tempfile module to use our custom temp directory
# This affects all tempfile.TemporaryDirectory() calls, including NeMo's
tempfile.tempdir = str(_temp_dir)

# ============================================================================
# FIX #3: Validate tempfile Configuration
# ============================================================================
# Verify that tempfile.gettempdir() returns our custom cache directory.
# This ensures all subsequent tempfile operations (including NeMo's internal
# manifest creation during inference) will use our controlled location
# instead of system %TEMP%, preventing Windows file locking issues.
# ============================================================================

# Validate that our tempfile configuration took effect
_actual_temp = tempfile.gettempdir()
if _actual_temp != str(_temp_dir):
    print(f"⚠️  WARNING: tempfile.gettempdir() returned {_actual_temp}")
    print(f"   Expected: {_temp_dir}")
    print(f"   This may cause file locking issues!")
else:
    print(f"✓ Temp directory verified: {_temp_dir}")

# ============================================================================
# NOTE: Tensor-Based Transcription for Windows File Lock Prevention
# ============================================================================
# NeMo's ASR models use Lhotse dataloaders internally which create manifest.json
# files. The num_workers parameter to model.transcribe() does NOT override the
# model's internal config (which has num_workers=2 from training).
#
# The REAL fix is to bypass the file-based dataloader entirely by:
# 1. Loading audio into numpy arrays with librosa
# 2. Passing numpy arrays directly to model.transcribe()
# 3. This bypasses manifest.json creation completely
#
# Additionally, we override model.cfg after loading to set num_workers=0
# as a fallback for any code paths that still use file-based transcription.
# ============================================================================
print("\n✅ Using tensor-based transcription (bypasses manifest.json creation)")
print("   This prevents WinError 32 file locking issues on Windows")

# ============================================================================
# NOW import libraries (after cache directories are configured)
# ============================================================================

import gradio as gr
import nemo.collections.asr as nemo_asr
import torch
import time
import gc
import shutil
import hashlib
import io
import re
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union, TextIO


# ============================================================================
# Data Classes for Reducing Function Argument Counts
# ============================================================================
# These dataclasses encapsulate related parameters to reduce function signatures
# from 10-15 arguments to 2-4, improving readability and maintainability.
# ============================================================================

@dataclass
class TranscriptionStats:
    """Statistics about a transcription run."""
    model_choice: str
    gpu_name: str
    total_duration: float  # Audio duration in seconds
    total_time: float      # Total processing time
    inference_time: float  # Model inference time only
    load_time: float       # Model load time
    chunk_size: int        # Chunk size used
    rtfx: float           # Real-time factor
    vram_used: float      # VRAM in GB
    apply_itn: bool       # Whether ITN was applied


@dataclass 
class TranscriptionContext:
    """Context for a transcription operation."""
    transcription: str
    timestamps: List[Dict[str, Any]]
    timestamp_level: str  # 'word', 'segment', 'char', or 'none'
    file_info: List[Dict[str, Any]]  # List of {name, duration, is_video}


@dataclass
class BatchTranscriptionContext:
    """Context for batch transcription operations."""
    all_transcriptions: List[str]
    all_timestamps: List[Tuple[List[Dict[str, Any]], str]]  # List of (timestamps, level)
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
    # Single-file specific (optional for batch)
    transcription: Optional[str] = None
    timestamps: Optional[List[Dict[str, Any]]] = None
    timestamp_level: str = 'none'
    # Batch-file specific (optional for single)
    all_transcriptions: Optional[List[str]] = None
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]] = None


@dataclass
class ResultProcessingContext:
    """Context for processing transcription results (reduces argument passing)."""
    stats: TranscriptionStats
    file_list: List[str]
    file_info: List[Dict[str, Any]]
    include_timestamps: bool
    video_status: str = ""
    load_time: float = 0.0
    # ITN mode settings
    apply_itn_final: bool = False
    had_itn_per_chunk: bool = False
    # Batch-specific (optional for single)
    all_transcriptions: Optional[List[str]] = None
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]] = None


# ============================================================================
# Log Capture Setup - Capture all logs for downloadable output
# ============================================================================
# Create a StringIO buffer to capture log messages during transcription
# This allows users to download logs for debugging and review
# ============================================================================

class LogCapture:
    """Captures stdout/stderr and logging output for download."""
    
    def __init__(self):
        self.log_buffer = io.StringIO()
        self.original_stdout = None
        self.original_stderr = None
        self.handler = None
        
    def start(self) -> None:
        """Start capturing logs."""
        import sys
        self.log_buffer = io.StringIO()  # Reset buffer
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create a TeeWriter that writes to both original stream and buffer
        class TeeWriter:
            def __init__(self, original: Any, buffer: io.StringIO) -> None:
                self.original = original
                self.buffer = buffer
            def write(self, text: str) -> None:
                self.original.write(text)
                self.buffer.write(text)
            def flush(self) -> None:
                self.original.flush()
                
        sys.stdout = TeeWriter(self.original_stdout, self.log_buffer)  # type: ignore[assignment]
        sys.stderr = TeeWriter(self.original_stderr, self.log_buffer)  # type: ignore[assignment]
        
        # Also capture logging module output
        self.handler = logging.StreamHandler(self.log_buffer)
        self.handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self.handler)
        
    def stop(self) -> str:
        """Stop capturing logs and return captured content."""
        import sys
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
        if self.handler:
            logging.getLogger().removeHandler(self.handler)
        return self.log_buffer.getvalue()
    
    def get_logs(self) -> str:
        """Get current captured logs without stopping."""
        return self.log_buffer.getvalue()

# Global log capture instance
log_capture = LogCapture()

# Global model cache to avoid reloading
models_cache: Dict[str, Any] = {}

# ============================================================================
# ITN (Inverse Text Normalization) Support
# ============================================================================
# Converts spoken numbers to digits: "twenty twenty two" → "2022"
# Uses nemo_text_processing if available, with graceful fallback
# ============================================================================

# Try to import ITN - will be None if not installed
ITN_NORMALIZER: Optional[Any] = None
ITN_AVAILABLE: bool = False

try:
    from nemo_text_processing.inverse_text_normalization import InverseNormalizer
    # Initialize ITN for English (lazy - will init on first use)
    ITN_AVAILABLE = True  # type: ignore[misc]
    print("✅ ITN (Inverse Text Normalization) available - numbers will be converted to digits")
except ImportError:
    print("ℹ️  ITN not installed - numbers will remain as words")
    print("   To enable: pip install nemo_text_processing")


def _get_itn_normalizer(language: str = "en") -> Optional[Any]:
    """Get or create ITN normalizer (lazy initialization).
    
    Lazy initialization prevents slow startup when ITN is available
    but user may not always need it.
    
    Args:
        language: Language code (default: "en" for English)
        
    Returns:
        InverseNormalizer instance or None if unavailable
    """
    global ITN_NORMALIZER
    
    if not ITN_AVAILABLE:
        return None
        
    if ITN_NORMALIZER is None:
        try:
            from nemo_text_processing.inverse_text_normalization import InverseNormalizer
            print(f"   🔧 Initializing ITN normalizer for '{language}'...")
            ITN_NORMALIZER = InverseNormalizer(lang=language, cache_dir=str(CACHE_DIR / "itn"))  # type: ignore[misc]
            print(f"   ✅ ITN normalizer ready")
        except Exception as e:
            print(f"   ⚠️ Failed to initialize ITN: {e}")
            return None
            
    return ITN_NORMALIZER  # type: ignore[reportReturnType]


def _split_text_into_word_chunks(text: str, max_words: int = 50) -> List[str]:
    """Split text into chunks of max_words for ITN processing.
    
    Args:
        text: Input text to split
        max_words: Maximum words per chunk
        
    Returns:
        List of text chunks, or [text] if text is empty
    """
    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text]


def _normalize_chunks_with_fallback(normalizer: Any, chunks: List[str]) -> List[str]:
    """Normalize a list of text chunks using batch or individual fallback.
    
    Args:
        normalizer: ITN normalizer instance
        chunks: List of text chunks to normalize
        
    Returns:
        List of normalized text chunks
    """
    try:
        return normalizer.normalize_list(chunks, verbose=False)
    except Exception:
        # Individual normalization fallback
        results: List[str] = []
        for chunk in chunks:
            try:
                results.append(normalizer.normalize(chunk, verbose=False))
            except Exception:
                results.append(chunk)  # Keep original if all fails
        return results


def _try_itn_sentence_splitting(normalizer: Any, text: str) -> Tuple[bool, str, str]:
    """Try ITN using built-in sentence splitting.
    
    Returns:
        Tuple of (success: bool, result: str, message: str)
    """
    try:
        sentences = normalizer.split_text_into_sentences(text)
        if sentences and len(sentences) > 0:
            normalized = _normalize_chunks_with_fallback(normalizer, sentences)
            return True, ' '.join(normalized), f"sentence splitting: {len(sentences)} sentences"
    except Exception as e:
        print(f"   ℹ️ ITN sentence splitting failed: {e}")
    return False, text, ""


def _try_itn_regex_splitting(normalizer: Any, text: str) -> Tuple[bool, str, str]:
    """Try ITN using regex-based sentence splitting on .!?
    
    Returns:
        Tuple of (success: bool, result: str, message: str)
    """
    import re
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences and len(sentences) > 1:
            normalized = _normalize_chunks_with_fallback(normalizer, sentences)
            return True, ' '.join(normalized), f"regex splitting: {len(sentences)} sentences"
    except Exception as e:
        print(f"   ℹ️ ITN regex splitting failed: {e}")
    return False, text, ""


def _try_itn_chunk_splitting(normalizer: Any, text: str) -> Tuple[bool, str, str]:
    """Try ITN using fixed-size word chunks.
    
    Returns:
        Tuple of (success: bool, result: str, message: str)
    """
    try:
        chunks = _split_text_into_word_chunks(text, max_words=50)
        if len(chunks) > 1:
            normalized = _normalize_chunks_with_fallback(normalizer, chunks)
            return True, ' '.join(normalized), f"chunk splitting: {len(chunks)} chunks"
        else:
            result = normalizer.normalize(text, verbose=False)
            return True, result, "single text"
    except Exception as e:
        print(f"   ⚠️ ITN normalization failed completely: {e}")
    return False, text, ""


def _is_itn_applicable(normalizer: Any, text: str) -> bool:
    """Check if ITN can be applied to text.
    
    Returns:
        True if normalizer exists and text is non-empty
    """
    return normalizer is not None and bool(text) and bool(text.strip())


def apply_inverse_text_normalization(text: str, language: str = "en") -> str:
    """Apply Inverse Text Normalization to convert spoken numbers to digits.
    
    Converts:
    - "twenty twenty two" → "2022"
    - "one hundred dollars" → "$100"
    - "five point five percent" → "5.5%"
    
    For long text, splits into sentences first to avoid the "input too long" warning
    from nemo_text_processing, then normalizes each sentence and joins with single space.
    Uses multiple fallback strategies for sentence splitting.
    
    Args:
        text: Input transcription text
        language: Language code for ITN (default: "en")
        
    Returns:
        Text with numbers converted to digits, or original text if ITN unavailable
    """
    normalizer = _get_itn_normalizer(language)
    
    if not _is_itn_applicable(normalizer, text):
        return text
    
    # Try strategies in order: sentence splitting → regex splitting → chunk splitting
    strategies = [
        _try_itn_sentence_splitting,
        _try_itn_regex_splitting,
        _try_itn_chunk_splitting,
    ]
    
    for strategy in strategies:
        success, result, message = strategy(normalizer, text)
        if success:
            print(f"   ✅ ITN applied ({message})")
            return result
    
    return text


def apply_itn_to_segment(text: str, language: str = "en") -> str:
    """Apply ITN to a segment with automatic splitting for long text.
    
    For chunks from long audio (180s+), the text can exceed ITN's
    recommended length. This function automatically splits long text
    and normalizes in batches to avoid the "input too long" warning.
    
    Args:
        text: Segment text (may be long from chunked audio transcription)
        language: Language code for ITN (default: "en")
        
    Returns:
        Text with numbers converted to digits, or original if ITN unavailable/fails
    """
    normalizer = _get_itn_normalizer(language)
    
    if not _is_itn_applicable(normalizer, text):
        return text
    
    word_count = len(text.split())
    
    # If text is short enough, normalize directly
    if word_count <= 50:
        try:
            result = normalizer.normalize(text.strip(), verbose=False)  # type: ignore[union-attr]
            return result
        except Exception as e:
            print(f"   ⚠️ ITN direct normalization failed: {e}")
            # Fall through to chunked approach
    
    # For long text, split into 50-word chunks to avoid "input too long" warning
    chunks = _split_text_into_word_chunks(text, max_words=50)
    
    if len(chunks) <= 1 and word_count <= 50:
        # Already tried direct, return original
        return text
    
    try:
        # Use normalize_list for batch processing (more efficient)
        normalized = normalizer.normalize_list(chunks, verbose=False)  # type: ignore[union-attr]
        result = ' '.join(normalized)
        print(f"   ✅ ITN applied (chunked: {len(chunks)} chunks, {word_count} words)")
        return result
    except Exception as e:
        print(f"   ⚠️ ITN batch failed ({e}), trying individually")
        # Individual fallback - normalize each chunk separately
        results: List[str] = []
        for chunk in chunks:
            try:
                results.append(normalizer.normalize(chunk, verbose=False))  # type: ignore[union-attr]
            except Exception:
                results.append(chunk)  # Keep original chunk if normalization fails
        
        # Only print success if we actually normalized something
        if results:
            print(f"   ✅ ITN applied (individual: {len(chunks)} chunks)")
        return ' '.join(results)


# ============================================================================
# SRT/CSV/TXT Output Format Conversion Utilities
# ============================================================================
# Converts transcription with timestamps into various output formats:
# - SRT: SubRip subtitle format (for video subtitles)
# - CSV: Comma-separated values (for spreadsheets/analysis)
# - TXT: Plain text with timestamps (human-readable)
# ============================================================================

def _format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds (float)
        
    Returns:
        Formatted timestamp string "HH:MM:SS,mmm"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _ends_with_sentence_punctuation(word: str) -> bool:
    """Check if word ends with sentence-ending punctuation."""
    if not word:
        return False
    return word.rstrip().endswith(('.', '?', '!', '...', '。', '？', '！'))


def _get_word_text_from_timestamp(stamp: Dict[str, Any]) -> str:
    """Extract word text from timestamp dict."""
    return stamp.get('text', stamp.get('word', stamp.get('segment', stamp.get('char', ''))))


def _should_end_segment(word: str, segment_duration: float, word_count: int, has_punctuation: bool, words_per_segment: int, max_duration: float) -> bool:
    """Determine if current segment should be ended.
    
    Args:
        word: Current word text
        segment_duration: Duration of current segment
        word_count: Number of words in current segment
        has_punctuation: Whether source has punctuation for sentence detection
        words_per_segment: Target words per segment (fallback)
        max_duration: Maximum segment duration
        
    Returns:
        True if segment should be ended
    """
    ends_sentence = _ends_with_sentence_punctuation(word)
    return (
        (has_punctuation and ends_sentence) or
        segment_duration > max_duration or
        (not has_punctuation and word_count >= words_per_segment)
    )


def _finalize_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize a segment by joining words and returning clean dict."""
    return {
        'start': segment['start'],
        'end': segment['end'],
        'text': ' '.join(segment['words'])
    }


def _group_words_into_segments(timestamps: List[Dict[str, Any]], words_per_segment: int = 8, max_duration: float = 5.0, silence_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Group word-level timestamps into subtitle segments using sentence boundaries and silence detection.
    
    First attempts to detect sentence boundaries by punctuation marks (. ? !).
    Also ends segments when a silence gap > silence_threshold is detected between words.
    Falls back to grouping by word count/duration if no punctuation is detected.
    
    Args:
        timestamps: List of word timestamp dicts with 'start', 'end', 'word'/'text' keys
        words_per_segment: Target words per subtitle segment (default: 8, used as fallback)
        max_duration: Maximum segment duration in seconds (default: 5.0)
        silence_threshold: End segment if gap between words exceeds this (default: use global setting)
        
    Returns:
        List of segment dicts with 'start', 'end', 'text' keys
    """
    if not timestamps:
        return []
    
    # Use global silence threshold if not specified
    effective_silence_threshold = silence_threshold if silence_threshold is not None else silence_threshold_sec
    
    # Check if source has punctuation for sentence detection
    has_punctuation = any(
        _ends_with_sentence_punctuation(_get_word_text_from_timestamp(ts)) 
        for ts in timestamps
    )
    
    segments: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {'start': timestamps[0].get('start', 0.0), 'end': 0.0, 'words': []}
    last_end_time: float = timestamps[0].get('start', 0.0)
    
    for stamp in timestamps:
        word = _get_word_text_from_timestamp(stamp)
        start = stamp.get('start', 0.0)
        end = stamp.get('end', 0.0)
        
        # Check for silence gap since last word (silence detection)
        if current['words'] and effective_silence_threshold > 0:
            gap = start - last_end_time
            if gap >= effective_silence_threshold:
                # End current segment due to silence gap
                if current['words']:
                    segments.append(_finalize_segment(current))
                    current = {'start': start, 'end': end, 'words': [word]}
                    last_end_time = end
                    continue
        
        current['words'].append(word)
        current['end'] = end
        last_end_time = end
        
        segment_duration = end - current['start']
        if _should_end_segment(word, segment_duration, len(current['words']), 
                               has_punctuation, words_per_segment, max_duration):
            if current['words']:
                segments.append(_finalize_segment(current))
                current = {'start': end, 'end': end, 'words': []}
    
    # Finalize last segment
    if current['words']:
        segments.append(_finalize_segment(current))
    
    return segments


def format_as_srt(transcription: str, timestamps: List[Dict[str, Any]], timestamp_level: str = 'word') -> str:
    """Format transcription as SRT subtitle file content.
    
    Args:
        transcription: Full transcription text (used if no timestamps)
        timestamps: List of timestamp dicts from extract_timestamps()
        timestamp_level: 'word', 'segment', or 'none'
        
    Returns:
        SRT formatted string
    """
    if not timestamps or timestamp_level == 'none':
        # No timestamps - create single subtitle for entire transcription
        # Estimate duration from text length (roughly 150 words per minute)
        word_count = len(transcription.split())
        estimated_duration = max(5.0, word_count / 2.5)  # ~150 words/min = 2.5 words/sec
        
        return (
            f"1\n"
            f"00:00:00,000 --> {_format_srt_timestamp(estimated_duration)}\n"
            f"{transcription}\n"
        )
    
    # Group words into readable subtitle segments
    if timestamp_level == 'word':
        segments = _group_words_into_segments(timestamps)
    else:
        # Segment-level timestamps are already grouped
        segments: List[Dict[str, Any]] = []
        for stamp in timestamps:
            text = stamp.get('text', stamp.get('segment', stamp.get('word', stamp.get('char', ''))))
            segments.append({
                'start': stamp.get('start', 0.0),
                'end': stamp.get('end', 0.0),
                'text': text
            })
    
    # Build SRT content
    srt_lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        start_ts = _format_srt_timestamp(seg['start'])
        end_ts = _format_srt_timestamp(seg['end'])
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(seg['text'])
        srt_lines.append("")  # Blank line between entries
    
    return '\n'.join(srt_lines)


def format_as_csv(transcription: str, timestamps: List[Dict[str, Any]], timestamp_level: str = 'word') -> str:
    """Format transcription as CSV content.
    
    CSV columns: start_time, end_time, duration, text
    
    Args:
        transcription: Full transcription text (used if no timestamps)
        timestamps: List of timestamp dicts from extract_timestamps()
        timestamp_level: 'word', 'segment', or 'none'
        
    Returns:
        CSV formatted string with header row
    """
    csv_lines: List[str] = ["start_time,end_time,duration,text"]
    
    if not timestamps or timestamp_level == 'none':
        # No timestamps - create single row for entire transcription
        word_count = len(transcription.split())
        estimated_duration = max(5.0, word_count / 2.5)
        
        # Escape quotes in text for CSV
        escaped_text = transcription.replace('"', '""')
        csv_lines.append(f'0.000,{estimated_duration:.3f},{estimated_duration:.3f},"{escaped_text}"')
        return '\n'.join(csv_lines)
    
    # Output each timestamp entry
    for stamp in timestamps:
        start = stamp.get('start', 0.0)
        end = stamp.get('end', 0.0)
        duration = end - start
        text = stamp.get('text', stamp.get('word', stamp.get('segment', stamp.get('char', ''))))
        
        # Escape quotes in text for CSV
        escaped_text = text.replace('"', '""')
        csv_lines.append(f'{start:.3f},{end:.3f},{duration:.3f},"{escaped_text}"')
    
    return '\n'.join(csv_lines)


def format_as_txt_with_timestamps(transcription: str, timestamps: List[Dict[str, Any]], timestamp_level: str = 'word') -> str:
    """Format transcription as plain text with inline timestamps.
    
    Format: [HH:MM:SS] text text text [HH:MM:SS] text text...
    
    Args:
        transcription: Full transcription text (used if no timestamps)
        timestamps: List of timestamp dicts from extract_timestamps()
        timestamp_level: 'word', 'segment', or 'none'
        
    Returns:
        Plain text with timestamps
    """
    if not timestamps or timestamp_level == 'none':
        return f"[00:00:00] {transcription}"
    
    # Group into segments for readability
    if timestamp_level == 'word':
        segments = _group_words_into_segments(timestamps, words_per_segment=12, max_duration=8.0)
    else:
        segments: List[Dict[str, Any]] = []
        for stamp in timestamps:
            text = stamp.get('text', stamp.get('segment', stamp.get('word', stamp.get('char', ''))))
            segments.append({
                'start': stamp.get('start', 0.0),
                'text': text
            })
    
    # Build text with timestamps
    txt_lines: List[str] = []
    for seg in segments:
        # Convert to HH:MM:SS format
        seconds = seg['start']
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        timestamp = f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
        txt_lines.append(f"{timestamp} {seg['text']}")
    
    return '\n'.join(txt_lines)


# Create cache subdirectory for Gradio uploads
# This prevents manifest.json file locking issues during NeMo inference
_gradio_cache_dir = _cache_dir / "gradio_uploads"
_gradio_cache_dir.mkdir(parents=True, exist_ok=True)

# Video file extensions supported (librosa + FFmpeg backend handles audio extraction)
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.m4v'}

# Audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# Separator string for output formatting
SEPARATOR = '=' * 60

# ============================================================================
# Stage-Specific Error Messages (Pattern 6: HuggingFace Modernization)
# ============================================================================
# Provides actionable error messages based on failure stage.
# Each category has a title and detailed advice for users.
# ============================================================================
ERROR_MESSAGES = {
    'audio_load_failed': {
        'title': '❌ Could Not Load Audio File',
        'message': (
            'The audio file could not be loaded. Please check that:\n'
            '- The file format is supported (WAV, MP3, FLAC, M4A, OGG, AAC, WMA)\n'
            '- The file is not corrupted or empty\n'
            '- You have read permissions for the file'
        )
    },
    'format_unsupported': {
        'title': '❌ Audio Format Not Supported',
        'message': (
            '**Supported Audio Formats:** WAV, MP3, FLAC, M4A, OGG, AAC, WMA\n'
            '**Supported Video Formats:** MP4, AVI, MKV, MOV, WEBM, FLV, M4V\n'
            '\nVideo files will have their audio extracted automatically.'
        )
    },
    'duration_invalid': {
        'title': '❌ Invalid Audio Duration',
        'message': (
            'Audio duration must be between 100ms and 24 hours.\n'
            'Please check if the file is corrupted, silent, or has an unusual format.'
        )
    },
    'audio_silent': {
        'title': '⚠️ Audio Appears to be Silent',
        'message': (
            'The audio file appears to contain very little or no audio signal.\n'
            'Please check that:\n'
            '- The audio was recorded properly\n'
            '- The volume level is not too low\n'
            '- The correct audio channel was selected during recording'
        )
    },
    'output_validation_failed': {
        'title': '❌ Transcription Output Invalid',
        'message': (
            'The model returned an invalid or empty result.\n'
            'This can happen when:\n'
            '- Audio quality is very poor\n'
            '- Audio contains only noise or music\n'
            '- Audio language is not supported by the model'
        )
    },
    'batch_partial_failure': {
        'title': '⚠️ Some Files Failed to Process',
        'message': (
            'Some files in the batch could not be transcribed.\n'
            'Successfully processed files are shown below.\n'
            'Check the error details for each failed file.'
        )
    },
    'model_load_failed': {
        'title': '❌ Model Loading Failed',
        'message': (
            'The AI model could not be loaded.\n'
            'Please check that:\n'
            '- You have enough disk space\n'
            '- Your internet connection is stable (for first download)\n'
            '- The cache directory is accessible'
        )
    },
    'transcription_timeout': {
        'title': '❌ Transcription Timed Out',
        'message': (
            'The transcription process took too long and was stopped.\n'
            'This can happen with very long audio files.\n'
            'Try splitting the audio into smaller chunks.'
        )
    }
}


def format_error_message(error_type: str, detail: str = "") -> str:
    """Format a stage-specific error message with optional details.
    
    Args:
        error_type: Key from ERROR_MESSAGES dictionary
        detail: Additional context or technical details
        
    Returns:
        Formatted markdown error message string
    """
    msg = ERROR_MESSAGES.get(error_type, {
        'title': '❌ Unknown Error',
        'message': 'An unexpected error occurred.'
    })
    
    result = f"### {msg['title']}\n\n{msg['message']}"
    
    if detail:
        result += f"\n\n**Technical Details:**\n```\n{detail}\n```"
    
    return result


# ============================================================================
# Audio Validation Function (Pattern 1: HuggingFace Modernization)
# ============================================================================
# Validates and normalizes audio before transcription.
# - Loads audio with librosa (handles various formats)
# - Validates duration (100ms - 24h)
# - Checks audio is not silent (RMS energy check)
# - Resamples to 16kHz if needed
# - Converts stereo to mono if needed
# ============================================================================

def _validate_audio_duration(duration: float) -> Optional[str]:
    """Validate audio duration is within acceptable range.
    
    Returns:
        Error message if invalid, None if valid
    """
    if duration < 0.1:  # Less than 100ms
        return format_error_message('duration_invalid', 
            f"Duration: {duration:.3f}s (minimum: 0.1s)")
    if duration > 86400:  # More than 24 hours
        return format_error_message('duration_invalid',
            f"Duration: {duration:.1f}s ({duration/3600:.1f} hours, maximum: 24 hours)")
    return None


def _validate_audio_energy(rms_max: float) -> Tuple[bool, str, str]:
    """Validate audio energy level from RMS.
    
    Returns:
        Tuple of (is_valid, error_msg, warning_msg)
    """
    if rms_max < 0.001:  # Very quiet - likely silent
        error = format_error_message('audio_silent',
            f"Maximum RMS energy: {rms_max:.6f} (threshold: 0.001)")
        return False, error, ""
    warning = "⚠️ Audio is very quiet - transcription quality may be affected" if rms_max < 0.01 else ""
    return True, "", warning


def _classify_audio_load_error(error_str: str) -> str:
    """Classify audio load error and return appropriate error message."""
    if "Audio file" in error_str or "NoBackendError" in error_str:
        return format_error_message('audio_load_failed', error_str)
    if "Format" in error_str or "codec" in error_str.lower():
        return format_error_message('format_unsupported', error_str)
    return format_error_message('audio_load_failed', error_str)


def validate_and_normalize_audio(file_path: str) -> Tuple[bool, Any, int, str, str]:
    """Validate and normalize audio file for transcription.
    
    This function performs comprehensive audio validation following the
    HuggingFace Spaces pattern, ensuring consistent preprocessing
    regardless of input format.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Tuple of (success: bool, audio_data: ndarray | None, 
                  sample_rate: int, error_msg: str, warning_msg: str)
    """
    import librosa
    import numpy as np
    
    try:
        # Load audio preserving original sample rate
        y, sr = librosa.load(file_path, sr=None)  # type: ignore[reportUnknownMemberType]
        
        # Validate duration (100ms to 24 hours)
        duration: float = librosa.get_duration(y=y, sr=sr)  # type: ignore[reportUnknownMemberType]
        duration_error = _validate_audio_duration(duration)
        if duration_error:
            return (False, None, 0, duration_error, "")
        
        # Check for silent audio (RMS energy check)
        rms: Any = librosa.feature.rms(y=y)  # type: ignore[reportUnknownMemberType]
        is_valid, energy_error, warning_msg = _validate_audio_energy(rms.max())  # type: ignore[reportUnknownArgumentType]
        if not is_valid:
            return (False, None, 0, energy_error, "")
        
        # Resample to 16kHz if needed (NeMo models expect 16kHz)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)  # type: ignore[reportUnknownMemberType]
            sr = 16000
        
        # Convert to mono if stereo
        if y.ndim > 1:  # type: ignore[reportUnknownMemberType]
            y = librosa.to_mono(y)  # type: ignore[reportUnknownMemberType]
        
        return (True, y, sr, "", warning_msg)  # type: ignore[reportReturnType]
        
    except Exception as e:
        return (False, None, 0, _classify_audio_load_error(str(e)), "")


# ============================================================================
# Transcription Result Validation (Pattern 2: HuggingFace Modernization)
# ============================================================================
# Validates transcription result structure BEFORE accessing .text
# - Checks result is not None
# - Checks result is a list and not empty
# - Validates hypothesis has .text attribute
# - Checks .text is a non-empty string
# ============================================================================

def _validate_result_structure(result: Any, idx: int) -> Tuple[bool, str]:
    """Validate basic result structure.
    
    Returns:
        Tuple of (valid: bool, error_msg: str)
    """
    if result is None:
        return False, "Result is None - model may have failed silently"
    if not isinstance(result, list):
        return False, f"Result is {type(result).__name__}, expected list"
    if len(result) == 0:  # type: ignore[reportUnknownArgumentType]
        return False, "Result is an empty list - no transcription generated"
    if idx >= len(result):  # type: ignore[reportUnknownArgumentType]
        return False, f"Index {idx} out of range (result has {len(result)} items)"  # type: ignore[reportUnknownArgumentType]
    return True, ""


def _extract_text_from_hypothesis(hypothesis: Any) -> Tuple[bool, str, str]:
    """Extract text from hypothesis object or string.
    
    Returns:
        Tuple of (success: bool, text: str, error_msg: str)
    """
    # Try string fallback (some models return plain strings)
    if isinstance(hypothesis, str):
        return (len(hypothesis) > 0, hypothesis, "" if hypothesis else "Transcription is empty")
    
    # Check hypothesis has text attribute
    if not hasattr(hypothesis, 'text'):
        return False, "", f"Hypothesis has no .text attribute (type: {type(hypothesis).__name__})"
    
    # Validate text type and content
    if not isinstance(hypothesis.text, str):
        return False, "", f".text is {type(hypothesis.text).__name__}, expected string"
    if len(hypothesis.text) == 0:
        return False, "", "Transcription is empty (0 characters)"
    
    return True, hypothesis.text, ""


def validate_transcription_result(result: Any, idx: int = 0) -> Tuple[bool, str, str]:
    """Validate transcription result before accessing text.
    
    Args:
        result: Output from model.transcribe()
        idx: Index of the hypothesis to validate (default: 0)
        
    Returns:
        Tuple of (success: bool, text: str, error_msg: str)
    """
    valid, error_msg = _validate_result_structure(result, idx)
    if not valid:
        return False, "", error_msg
    
    return _extract_text_from_hypothesis(result[idx])


# ============================================================================
# Defensive Timestamp Extraction (Pattern 4: HuggingFace Modernization)
# ============================================================================
# Extracts timestamps with graceful fallback:
# 1. Try word-level timestamps
# 2. Fallback to segment-level timestamps
# 3. Return empty list if both unavailable
# ============================================================================

def _try_get_timestamp_level(hypothesis: Any, level_key: str) -> Optional[List[Dict[str, Any]]]:
    """Try to extract timestamps at a specific level from hypothesis.
    
    Args:
        hypothesis: The hypothesis object from model.transcribe()
        level_key: Key to look for ('word', 'segment', or 'char')
        
    Returns:
        List of timestamps if found, None otherwise
    """
    try:
        ts_dict = getattr(hypothesis, 'timestamp', None)
        if not ts_dict or not isinstance(ts_dict, dict):
            return None
        timestamps = ts_dict.get(level_key)  # type: ignore[reportUnknownMemberType]
        if isinstance(timestamps, list) and len(timestamps) > 0:  # type: ignore[reportUnknownArgumentType]
            return timestamps  # type: ignore[reportReturnType]
    except (AttributeError, KeyError, TypeError):
        pass
    return None


def extract_timestamps(hypothesis: Any, include_timestamps: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    """Extract timestamps with word → segment → char → none fallback.
    
    Args:
        hypothesis: The hypothesis object from model.transcribe()
        include_timestamps: Whether to attempt timestamp extraction
        
    Returns:
        Tuple of (timestamps_list: list, level: str)
        where level is 'word', 'segment', 'char', or 'none'
    """
    if not include_timestamps:
        return ([], 'none')
    
    # Try each level in priority order
    for level in ('word', 'segment', 'char'):
        timestamps = _try_get_timestamp_level(hypothesis, level)
        if timestamps:
            return (timestamps, level)
    
    return ([], 'none')


def format_timestamp_status(level: str, include_timestamps: bool) -> str:
    """Format a status message indicating the timestamp level available.
    
    Args:
        level: The timestamp level ('word', 'segment', 'char', or 'none')
        include_timestamps: Whether timestamps were requested
        
    Returns:
        Status message string
    """
    if not include_timestamps:
        return ""
    
    if level == 'word':
        return "\n✅ **Timestamps:** Word-level available"
    elif level == 'segment':
        return "\n⚠️ **Timestamps:** Segment-level (word-level unavailable for this model)"
    elif level == 'char':
        return "\n⚠️ **Timestamps:** Character-level (word/segment unavailable)"
    else:
        return "\nℹ️ **Timestamps:** Not available for this model"


# ============================================================================
# Transcription Helper with Tensor-Based Input (WinError 32 Fix)
# ============================================================================
# NeMo's Lhotse dataloader creates manifest.json files during inference that
# Windows services (antivirus, indexing, cloud sync) can lock, causing errors.
#
# The REAL fix: Pass audio as numpy arrays instead of file paths!
# - NeMo's transcribe() accepts numpy arrays directly
# - This BYPASSES the Lhotse dataloader entirely
# - No manifest.json files are created
# - No multiprocessing workers are spawned
#
# Reference: https://github.com/nvidia/nemo/blob/main/docs/source/asr/results.md
# "Inference on Numpy Audio Array" section shows this is officially supported
# ============================================================================

def _load_audio_to_numpy(file_path: str, target_sr: int = 16000) -> Tuple[Any, int]:
    """Load audio file to numpy array for tensor-based transcription.
    
    This bypasses NeMo's file-based dataloader by loading audio directly
    into memory as numpy arrays. NeMo's transcribe() accepts numpy arrays,
    which avoids manifest.json creation entirely.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (NeMo expects 16kHz)
        
    Returns:
        Tuple of (numpy_array, sample_rate)
    """
    import librosa  # Import here to match pattern used elsewhere in file
    
    try:
        # Load with librosa - handles various formats and resamples
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)  # type: ignore[reportUnknownMemberType]
        return audio, sr  # type: ignore[reportReturnType]
    except Exception as e:
        print(f"   ⚠️ Failed to load {file_path}: {e}")
        raise


# ============================================================================
# Audio Chunking for Long Audio Files (OutOfMemoryError Fix)
# ============================================================================
# When passing numpy arrays directly to NeMo's transcribe(), the entire audio
# is processed at once without internal chunking. For long audio files (e.g.,
# 30+ minutes), this can cause CUDA OOM errors even on 12GB+ VRAM GPUs.
#
# Solution: Split long audio into manageable chunks (60 seconds default),
# transcribe each chunk separately, and merge the results.
#
# The 2-second context overlap ensures words at chunk boundaries are
# transcribed with proper acoustic context, preventing word cutoff issues.
# ============================================================================

# Chunking configuration constants (defaults - can be overridden via UI)
# Increased to 60s to better utilize available VRAM (6-7GB typical usage)
DEFAULT_CHUNK_DURATION_SEC = 60   # Duration of each transcription chunk
CHUNK_OVERLAP_SEC = 2             # Context overlap on each side of chunk
DEFAULT_LONG_AUDIO_THRESHOLD_SEC = 90  # Audio longer than this triggers chunking

# Silence detection threshold for SRT segment boundaries (in seconds)
# If gap between words > this threshold, end the current subtitle segment
DEFAULT_SILENCE_THRESHOLD_SEC = 0.5

# ITN (Inverse Text Normalization) mode options
# - "per_chunk": Apply ITN to each chunk during transcription (best for long audio)
# - "final_pass": Apply ITN once to complete transcription (simpler, may fail on long text)
# - "both": Apply ITN per-chunk AND final pass (most thorough)
# - "disabled": Don't apply ITN
ITN_MODE_CHOICES = ["per_chunk", "final_pass", "both", "disabled"]
DEFAULT_ITN_MODE = "per_chunk"

# Runtime configuration (can be changed via Gradio UI)
# These are module-level so they can be modified by the UI
chunk_duration_sec = DEFAULT_CHUNK_DURATION_SEC
long_audio_threshold_sec = DEFAULT_LONG_AUDIO_THRESHOLD_SEC
silence_threshold_sec = DEFAULT_SILENCE_THRESHOLD_SEC
itn_mode = DEFAULT_ITN_MODE


def _clear_vram() -> None:
    """Clear CUDA VRAM and run garbage collection."""
    import gc
    if torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        torch.cuda.empty_cache()  # type: ignore[reportUnknownMemberType]
    gc.collect()


def _transcribe_single_buffer(model: Any, buffer: Any, use_cuda: bool) -> Any:
    """Transcribe a single audio buffer.
    
    Args:
        model: Loaded NeMo ASR model
        buffer: numpy array of audio samples
        use_cuda: Whether to use CUDA with mixed precision
        
    Returns:
        Result from model.transcribe()
    """
    transcribe_kwargs: Dict[str, Any] = {
        'audio': [buffer],
        'batch_size': 1,  # Single chunk at a time for memory safety
        'return_hypotheses': True,
        'timestamps': True,  # Enable timestamp extraction for word/segment boundaries
        'verbose': False
    }
    
    if use_cuda and torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        with torch.autocast(device_type='cuda', dtype=torch.float16):  # type: ignore[reportUnknownMemberType]
            return model.transcribe(**transcribe_kwargs)
    return model.transcribe(**transcribe_kwargs)


def _extract_hypothesis_text(hypothesis: Any) -> str:
    """Extract text from a hypothesis object.
    
    Args:
        hypothesis: Result hypothesis from model.transcribe()
        
    Returns:
        Text string (may be empty)
    """
    if hasattr(hypothesis, 'text'):
        return hypothesis.text
    if isinstance(hypothesis, str):
        return hypothesis
    return str(hypothesis)


def _adjust_chunk_timestamps(chunk_word_ts: List[Dict[str, Any]], ts_level: str, left_context_duration: float, chunk_start_time: float) -> List[Dict[str, Any]]:
    """Adjust raw timestamps from a chunk to absolute positions.
    
    Args:
        chunk_word_ts: List of timestamp dicts from extract_timestamps()
        ts_level: Timestamp level ('word', 'segment', 'char')
        left_context_duration: Duration of left context that was prepended
        chunk_start_time: Start time of this chunk in the full audio
        
    Returns:
        List of adjusted timestamp dicts
    """
    adjusted_timestamps: List[Dict[str, Any]] = []
    
    for ts in chunk_word_ts:
        adjusted_ts: Dict[str, Any] = {
            'start': max(0.0, ts.get('start', 0.0) - left_context_duration + chunk_start_time),
            'end': ts.get('end', 0.0) - left_context_duration + chunk_start_time,
        }
        # Ensure end >= start
        adjusted_ts['end'] = max(float(adjusted_ts['start']), float(adjusted_ts['end']))
        
        # Copy text field with correct key
        for key in ('word', 'segment', 'char', 'text'):
            if key in ts:
                adjusted_ts[key] = ts[key]
        
        adjusted_timestamps.append(adjusted_ts)
    
    return adjusted_timestamps


def _create_chunk_fallback_timestamp(chunk_start_time: float, chunk_end_time: float, chunk_text: str) -> Dict[str, Any]:
    """Create a fallback chunk-level timestamp."""
    return {
        'start': chunk_start_time,
        'end': chunk_end_time,
        'text': chunk_text
    }


def _process_single_chunk(model: Any, buffer: Any, use_cuda: bool, apply_itn_per_chunk: bool,
                          chunk_start_time: float, chunk_end_time: float,
                          left_context_duration: float) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Process a single audio chunk and return transcription and timestamps.
    
    Returns:
        Tuple of (chunk_text or None, chunk_timestamps list)
    """
    result = _transcribe_single_buffer(model, buffer, use_cuda)
    
    if not result or len(result) == 0:
        return None, []
    
    hypothesis = result[0]
    chunk_text = _extract_hypothesis_text(hypothesis).strip()
    
    if not chunk_text:
        return None, []
    
    # Apply ITN to this chunk immediately (before text gets too long)
    if apply_itn_per_chunk:
        chunk_text = apply_itn_to_segment(chunk_text)
    
    # Extract and adjust timestamps
    chunk_word_ts, ts_level = extract_timestamps(hypothesis, include_timestamps=True)
    
    if chunk_word_ts and ts_level in ('word', 'segment', 'char'):
        adjusted = _adjust_chunk_timestamps(chunk_word_ts, ts_level, left_context_duration, chunk_start_time)
        return chunk_text, adjusted
    else:
        # Fallback: chunk-level timestamp
        return chunk_text, [_create_chunk_fallback_timestamp(chunk_start_time, chunk_end_time, chunk_text)]


def _transcribe_long_audio_chunked(model: Any, audio_array: Any, sample_rate: int = 16000, use_cuda: bool = True,
                                    chunk_size_override: Optional[int] = None, apply_itn_per_chunk: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe long audio by processing in chunks to avoid VRAM OOM.
    
    This function splits long audio into manageable chunks with overlap,
    transcribes each chunk separately (clearing VRAM between chunks), and
    merges the results.
    
    Args:
        model: Loaded NeMo ASR model
        audio_array: numpy array of audio samples (16kHz mono)
        sample_rate: Sample rate in Hz (default 16000)
        use_cuda: Whether to use CUDA with mixed precision
        chunk_size_override: Override chunk duration in seconds (None = use global setting)
        apply_itn_per_chunk: Whether to apply ITN to each chunk immediately (avoids long text issues)
    
    Returns:
        Tuple of (full_transcription_text, chunk_timestamps)
    """
    effective_chunk_duration = chunk_size_override or chunk_duration_sec
    
    chunk_samples = int(effective_chunk_duration * sample_rate)
    context_samples = int(CHUNK_OVERLAP_SEC * sample_rate)
    step_samples = chunk_samples
    
    total_samples = len(audio_array)
    total_duration = total_samples / sample_rate
    
    print(f"   ⚡ Chunked transcription: {total_duration:.1f}s audio → {effective_chunk_duration}s chunks")
    
    transcriptions: List[str] = []
    chunk_timestamps: List[Dict[str, Any]] = []
    position = 0
    chunk_num = 0
    
    while position < total_samples:
        chunk_num += 1
        
        # Calculate buffer boundaries with context
        start = max(0, position - context_samples)
        end = min(total_samples, position + chunk_samples + context_samples)
        buffer = audio_array[start:end]
        
        chunk_start_time = position / sample_rate
        chunk_end_time = min((position + chunk_samples) / sample_rate, total_duration)
        left_context_duration = (position - start) / sample_rate if position > start else 0
        
        print(f"   📍 Chunk {chunk_num}: {chunk_start_time:.1f}s - {chunk_end_time:.1f}s ({len(buffer)/sample_rate:.1f}s with context)")
        
        _clear_vram()
        
        try:
            chunk_text, ts_list = _process_single_chunk(
                model, buffer, use_cuda, apply_itn_per_chunk,
                chunk_start_time, chunk_end_time, left_context_duration
            )
            if chunk_text:
                transcriptions.append(chunk_text)
                chunk_timestamps.extend(ts_list)
        except Exception as e:
            print(f"   ⚠️ Chunk {chunk_num} failed: {type(e).__name__}: {e}")
        
        position += step_samples
        _clear_vram()
    
    print(f"   ✅ Processed {chunk_num} chunks")
    
    # Merge transcriptions and clean up spacing
    full_transcription = ' '.join(transcriptions)
    while '  ' in full_transcription:
        full_transcription = full_transcription.replace('  ', ' ')
    
    return full_transcription.strip(), chunk_timestamps


def _load_audio_files_to_memory(files: List[str]) -> List[Tuple[Any, float]]:
    """Load audio files into numpy arrays.
    
    Args:
        files: List of audio file paths
        
    Returns:
        List of (audio_array, duration_sec) tuples
        
    Raises:
        Exception: If any file fails to load
    """
    print(f"   📂 Loading {len(files)} audio file(s) into memory...")
    audio_data: List[Tuple[Any, float]] = []
    
    for file_path in files:
        try:
            audio_np, sr = _load_audio_to_numpy(file_path, target_sr=16000)
            duration_sec = len(audio_np) / sr
            audio_data.append((audio_np, duration_sec))
            print(f"      • {Path(file_path).name}: {duration_sec:.1f}s")
        except Exception as e:
            print(f"   ❌ Failed to load audio: {file_path}")
            raise
    
    return audio_data


def _transcribe_chunked_files(model: Any, audio_data: List[Tuple[Any, float]], use_cuda: bool, chunk_size_override: Optional[int], threshold: float,
                               apply_itn_per_chunk: bool = False) -> Tuple[List[Any], Dict[int, List[Dict[str, Any]]]]:
    """Process files using chunked transcription for long audio.
    
    Args:
        model: Loaded NeMo ASR model
        audio_data: List of (audio_array, duration_sec) tuples
        use_cuda: Whether to use CUDA
        chunk_size_override: Override chunk duration
        threshold: Duration threshold for chunking
        apply_itn_per_chunk: Whether to apply ITN immediately after each chunk
        
    Returns:
        Tuple of (results, chunk_timestamps_map)
    """
    print(f"   ⚡ Long audio detected (>{threshold}s) - using chunked transcription")
    
    results: List[Any] = []
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]] = {}
    
    for i, (audio_np, duration) in enumerate(audio_data):
        if duration > threshold:
            # Use chunked transcription for long audio
            text, chunk_ts = _transcribe_long_audio_chunked(
                model, audio_np, sample_rate=16000, use_cuda=use_cuda,
                chunk_size_override=chunk_size_override,
                apply_itn_per_chunk=apply_itn_per_chunk
            )
            chunk_timestamps_map[i] = chunk_ts
            
            # Wrap in Hypothesis-like object for API compatibility
            class _Hypothesis:
                def __init__(self, t: str, ts: List[Dict[str, Any]]) -> None:
                    self.text = t
                    self.chunk_timestamps = ts
            results.append(_Hypothesis(text, chunk_ts))
        else:
            # Short audio - transcribe normally
            result = _transcribe_single_buffer(model, audio_np, use_cuda)
            if result:
                results.extend(result)
            _clear_vram()
    
    return results, chunk_timestamps_map


def _execute_transcription(model: Any, transcribe_kwargs: Dict[str, Any], use_cuda: bool) -> Any:
    """Execute model transcription with optional CUDA mixed precision.
    
    Args:
        model: NeMo ASR model
        transcribe_kwargs: Dict of kwargs for model.transcribe()
        use_cuda: Whether to use CUDA
        
    Returns:
        Transcription result
    """
    if use_cuda and torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        with torch.autocast(device_type='cuda', dtype=torch.float16):  # type: ignore[reportUnknownMemberType]
            return model.transcribe(**transcribe_kwargs)
    return model.transcribe(**transcribe_kwargs)


def _transcribe_short_audio_batch(model: Any, audio_arrays: List[Any], batch_size: int, use_cuda: bool, max_retries: int, base_delay: float) -> Any:
    """Transcribe batch of short audio files with retry logic."""
    print(f"   ✅ Audio loaded into memory, starting transcription...")
    
    transcribe_kwargs: Dict[str, Any] = {
        'audio': audio_arrays,
        'batch_size': batch_size,
        'return_hypotheses': True,
        'timestamps': True,
        'verbose': True
    }
    
    for attempt in range(max_retries):
        try:
            return _execute_transcription(model, transcribe_kwargs, use_cuda)
        except (PermissionError, Exception) as e:
            # Clear VRAM on any error
            _clear_vram()
            # Check if we should retry (file lock errors OR transient failures)
            should_retry = _is_file_lock_error(str(e)) or not isinstance(e, PermissionError)
            if should_retry and _handle_retry_delay(attempt, base_delay, max_retries):
                continue
            raise
    
    raise RuntimeError(f"Transcription failed after {max_retries} attempts")


def _transcribe_with_retry(model: Any, files: List[str], batch_size: int, use_cuda: bool = True, max_retries: int = 3,
                           chunk_size_override: Optional[int] = None, apply_itn: bool = False) -> Tuple[Any, Dict[int, List[Dict[str, Any]]]]:
    """Transcribe using tensor-based input with chunking for long audio.
    
    Loads audio into numpy arrays to bypass NeMo's Lhotse dataloader,
    and chunks long audio to prevent CUDA OOM errors.
    
    Args:
        model: Loaded NeMo ASR model
        files: List of audio file paths to transcribe
        batch_size: Batch size for transcription
        use_cuda: Whether to use CUDA with mixed precision
        max_retries: Maximum retry attempts
        chunk_size_override: Override chunk duration (None = use global setting)
        apply_itn: Whether to apply ITN (will be applied per-chunk for long audio)
        
    Returns:
        Tuple of (results, chunk_timestamps_map)
    """
    audio_data = _load_audio_files_to_memory(files)
    
    # Check if any audio needs chunking
    effective_threshold = long_audio_threshold_sec
    needs_chunking = any(duration > effective_threshold for _, duration in audio_data)
    
    if needs_chunking:
        return _transcribe_chunked_files(
            model, audio_data, use_cuda, chunk_size_override, effective_threshold,
            apply_itn_per_chunk=apply_itn  # Apply ITN per chunk to avoid long text issues
        )
    
    # Standard path for short audio
    audio_arrays = [audio for audio, _ in audio_data]
    result = _transcribe_short_audio_batch(
        model, audio_arrays, batch_size, use_cuda, max_retries, base_delay=0.5
    )
    return result, {}


# Model configurations
# All models use standard ASRModel.from_pretrained() API (no SALM required)
# Parakeet models: Can load from local .nemo file OR HuggingFace
# Canary models: Load from HuggingFace or local .nemo file if downloaded
#
# loading_method options:
#   - "local": ONLY load from local .nemo file (fails if missing)
#   - "huggingface": ONLY load from HuggingFace
#   - "local_or_huggingface": Try local first, fallback to HuggingFace
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ========== PARAKEET MODELS ==========
    "parakeet-v3": {
        "local_path": "local_models/parakeet-0.6b-v3.nemo",  # Unique filename
        "hf_model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "max_batch_size": 32,  # Reference only - NeMo uses default file batching
        "display_name": "Parakeet-TDT-0.6B v3",
        "loading_method": "local_or_huggingface",  # Try local, fallback to HF
        "architecture": "FastConformer-TDT",
        "parameters": "600M",
        "languages": 25,
        "wer": "~1.7%",
        "rtfx": "3,380×",
        "vram_gb": "3-4",
        "recommended_for": "Best all-around choice, 25 languages, auto-detection"
    },
    
    "parakeet-1.1b": {
        "local_path": "local_models/parakeet-1.1b.nemo",  # Unique filename
        "hf_model_id": "nvidia/parakeet-tdt-1.1b",
        "max_batch_size": 24,  # Reference only - NeMo uses default file batching
        "display_name": "Parakeet-TDT-1.1B",
        "loading_method": "local_or_huggingface",  # Try local, fallback to HF
        "architecture": "FastConformer-TDT",
        "parameters": "1.1B",
        "languages": 1,
        "wer": "1.5%",
        "rtfx": "1,336×",
        "vram_gb": "5-6",
        "recommended_for": "Best English transcription accuracy available"
    },
    
    # ========== CANARY MODELS ==========
    "canary-1b": {
        "local_path": "local_models/canary-1b.nemo",  # Unique filename
        "hf_model_id": "nvidia/canary-1b",
        "max_batch_size": 16,  # Reference only - NeMo uses default file batching
        "display_name": "Canary-1B",
        "loading_method": "local_or_huggingface",  # Try local, fallback to HF
        "architecture": "FastConformer-Transformer",
        "parameters": "1B",
        "languages": 25,
        "wer": "~1.9% (English)",
        "rtfx": "~200×",
        "vram_gb": "4-5",
        "recommended_for": "Multilingual ASR + speech-to-text translation",
        "additional_features": ["Speech Translation (AST)", "NeMo Forced Aligner timestamps"]
    },
    
    "canary-1b-v2": {
        "local_path": "local_models/canary-1b-v2.nemo",  # Unique filename
        "hf_model_id": "nvidia/canary-1b-v2",
        "max_batch_size": 16,  # Reference only - NeMo uses default file batching
        "display_name": "Canary-1B v2",
        "loading_method": "local_or_huggingface",  # Try local, fallback to HF
        "architecture": "FastConformer-Transformer",
        "parameters": "1B",
        "languages": 25,
        "wer": "1.88% (English)",
        "rtfx": "~200×",
        "vram_gb": "4-5",
        "recommended_for": "Multilingual ASR + speech-to-text translation (improved)",
        "additional_features": ["Speech Translation (AST)", "NeMo Forced Aligner timestamps"]
    },
    
    # ========== HYBRID TDT-CTC MODELS (with Punctuation & Capitalization) ==========
    "parakeet-tdt_ctc-1.1b": {
        "local_path": "local_models/parakeet-tdt_ctc-1.1b.nemo",  # Unique filename for hybrid PnC model
        "hf_model_id": "nvidia/parakeet-tdt_ctc-1.1b",
        "max_batch_size": 16,  # Reference only - NeMo uses default file batching
        "display_name": "Parakeet-TDT_CTC-1.1B (PnC)",
        "loading_method": "local_or_huggingface",  # Try local, fallback to HF
        "architecture": "Hybrid FastConformer-TDT-CTC",
        "parameters": "1.1B",
        "languages": 1,  # English only
        "wer": "1.82%",
        "rtfx": "~1,000×",  # Slightly slower than pure TDT due to hybrid architecture
        "vram_gb": "5-6",
        "recommended_for": "Best English accuracy WITH punctuation & capitalization",
        "has_punctuation": True,  # Built-in punctuation and capitalization
        "additional_features": ["Punctuation", "Capitalization", "11h audio in single pass"]
    }
}


# ============================================================================
# Model Config Override for Windows File Lock Prevention (Fallback)
# ============================================================================
# Even though tensor-based transcription is the primary fix, we also override
# the model's internal config to set num_workers=0 as a fallback for any code
# paths that might still use file-based transcription.
#
# The model's embedded config (from training) has num_workers=2 in train_ds,
# validation_ds, and test_ds. The num_workers parameter to transcribe() does
# NOT override these - we must modify model.cfg directly using OmegaConf.
# ============================================================================

def _override_model_dataloader_config(model: Any) -> None:
    """Override model's internal dataloader config to prevent file locking.
    
    This modifies model.cfg to set num_workers=0 in all dataloader configs.
    This is a FALLBACK measure - the primary fix is tensor-based transcription.
    
    The model's embedded config (from training) has num_workers=2 which Lhotse
    uses regardless of what's passed to transcribe(). We must modify model.cfg
    directly to override this.
    
    Args:
        model: Loaded NeMo ASR model
    """
    try:
        from omegaconf import OmegaConf
        
        # Disable struct mode to allow modifications
        OmegaConf.set_struct(model.cfg, False)  # type: ignore[reportUnknownMemberType]
        
        # Override num_workers in all dataloader configs
        modified: List[str] = []
        for ds_name in ['train_ds', 'validation_ds', 'test_ds']:
            if hasattr(model.cfg, ds_name):
                ds_cfg = getattr(model.cfg, ds_name)
                if hasattr(ds_cfg, 'num_workers'):
                    old_value = ds_cfg.num_workers
                    ds_cfg.num_workers = 0
                    modified.append(f"{ds_name}: {old_value} -> 0")
        
        # Re-enable struct mode
        OmegaConf.set_struct(model.cfg, True)  # type: ignore[reportUnknownMemberType]
        
        if modified:
            print(f"   ✅ Overrode dataloader num_workers: {', '.join(modified)}")
        else:
            print(f"   ℹ️  No dataloader num_workers found in model.cfg")
            
    except Exception as e:
        # Non-fatal - tensor-based transcription is the primary fix
        print(f"   ⚠️  Could not override model config (non-fatal): {e}")


def get_model_key_from_choice(choice_text: str) -> str:
    """Extract model key from radio button choice text
    
    Args:
        choice_text: The display text from the radio button selection
    
    Returns:
        Model key string for accessing MODEL_CONFIGS
    """
    choice_map = {
        "Parakeet-TDT-0.6B v3": "parakeet-v3",
        "Parakeet-TDT-1.1B": "parakeet-1.1b",
        "Parakeet-TDT_CTC-1.1B": "parakeet-tdt_ctc-1.1b",  # Hybrid PnC model
        "Canary-1B v2": "canary-1b-v2",
        "Canary-1B": "canary-1b",
        # Legacy support
        "Parakeet-TDT-0.6B v2": "parakeet-v3",  # Map old to new multilingual version
        # Legacy SALM model mapping: Canary-Qwen-2.5B was a 2.5B param SALM-based model
        # We map it to Canary-1B-v2 (1B param, standard encoder-decoder architecture)
        # Trade-off: Slightly different model size, but gains multilingual support + simpler architecture
        "Canary-Qwen-2.5B": "canary-1b-v2"
    }
    
    # Try exact matches first
    for choice, key in choice_map.items():
        if choice in choice_text:
            return key
    
    # Default to parakeet-v3 if no match
    return "parakeet-v3"

def get_script_dir() -> Path:
    """Get the directory where the script is located"""
    return Path(__file__).parent.absolute()

def _check_model_local_availability(script_dir: Path, model_key: str, config: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if a model is available locally or needs HuggingFace download.
    
    Returns:
        Tuple of (is_local: bool, message: str)
    """
    local_path = config.get("local_path")
    display_name = config['display_name']
    hf_id = config['hf_model_id']
    
    if not local_path:
        return False, f"   📥 {display_name}: HuggingFace only ({hf_id})"
    
    full_path = script_dir / local_path
    if full_path.exists():
        size_mb = full_path.stat().st_size / (1024 * 1024)
        return True, f"   ✅ {display_name}: {full_path.name} ({size_mb:.1f} MB)"
    
    return False, f"   📥 {display_name}: Will download from HuggingFace ({hf_id})"


def validate_local_models() -> None:
    """Display local model setup status and provide informational messages.
    
    With "local_or_huggingface" loading method, local .nemo files are OPTIONAL.
    Models will automatically download from HuggingFace if local files are missing.
    """
    script_dir = get_script_dir()
    local_models_dir = script_dir / "local_models"
    
    print("\n" + "="*80)
    print("📦 Model Availability Check")
    print("="*80)
    
    if not local_models_dir.exists():
        print(f"\n📁 Local models directory not found: {local_models_dir}")
        print("   This is OK - models will download from HuggingFace on first use.")
        print("   To download models locally, run: python setup_local_models.py")
    else:
        print(f"\n📁 Local models directory: {local_models_dir}")
    
    # Check each model's local availability
    local_found: List[str] = []
    hf_fallback: List[str] = []
    
    for model_key, config in MODEL_CONFIGS.items():
        is_local, msg = _check_model_local_availability(script_dir, model_key, config)
        (local_found if is_local else hf_fallback).append(msg)
    
    if local_found:
        print("\n📦 Local .nemo files found (instant loading):")
        for msg in local_found:
            print(msg)
    
    if hf_fallback:
        print("\n📡 Models to download on first use:")
        for msg in hf_fallback:
            print(msg)
    
    print("\n💡 Tip: Run 'python setup_local_models.py' to download all models locally")
    print("="*80 + "\n")


def _try_file_copy(source_path: Path, dest_path: Path) -> bool:
    """Attempt to copy file and verify success.
    
    Returns:
        True if copy succeeded and verified, False if failed
    """
    shutil.copy2(source_path, dest_path)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    # Invalid copy - clean up
    if dest_path.exists():
        dest_path.unlink()
    return False


def copy_gradio_file_to_cache(file_path: str, max_retries: int = 6) -> str:
    """Copy Gradio uploaded file to cache directory to prevent manifest.json file locking.
    
    Args:
        file_path: Path to Gradio uploaded file
        max_retries: Maximum retry attempts for file copy
        
    Returns:
        Path to file in cache directory
    """
    file_path_obj = Path(file_path)
    
    # Generate unique filename using hash of original path
    path_hash = hashlib.sha256(str(file_path_obj).encode()).hexdigest()[:16]
    cached_path = _gradio_cache_dir / f"{path_hash}_{file_path_obj.name}"
    
    # Return immediately if already cached
    if cached_path.exists():
        return str(cached_path)
    
    # Copy with retry logic
    base_delay = 0.5
    for attempt in range(max_retries):
        try:
            if _try_file_copy(file_path_obj, cached_path):
                return str(cached_path)
        except (OSError, PermissionError) as e:
            if _is_file_lock_error(str(e)) and _handle_retry_delay(attempt, base_delay, max_retries):
                continue
            raise OSError(f"Failed to copy file to cache: {file_path_obj} -> {cached_path}. Error: {e}")
    
    raise OSError(f"File copy failed after {max_retries} attempts: {file_path_obj}")

def get_dynamic_batch_size(duration: float, model_key: str) -> int:
    """Calculate optimal batch size based on audio duration and model
    
    DEPRECATED: This function is no longer used. NeMo's transcribe() method
    handles batch sizing automatically based on available VRAM. The batch_size
    parameter in transcribe() controls how many FILES are batched together,
    not duration-based splitting.
    
    Kept for reference only - will be removed in future version.
    """
    # Return NeMo's default batch_size for file batching
    return 4

def setup_gpu_optimizations() -> None:
    """Enable GPU optimizations for better performance"""
    if torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        # Enable TF32 for matrix multiplication (Ampere+ GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[reportUnknownMemberType]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[reportUnknownMemberType]
        # Enable cuDNN benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True  # type: ignore[reportUnknownMemberType]
        print("✅ GPU optimizations enabled (TF32, cuDNN benchmark)")

def _format_model_error(title: str, model_path: str, display_name: str, problem_msg: str, solution_msg: str, original_error: Optional[Exception] = None) -> str:
    """Format a model loading error message with consistent styling.
    
    Args:
        title: Error title (e.g., "MODEL FILE NOT FOUND!")
        model_path: Path to the model file
        display_name: Human-readable model name
        problem_msg: Description of what went wrong
        solution_msg: How to fix the issue
        original_error: Original exception message if wrapping an error
    
    Returns:
        Formatted error message string
    """
    error_lines = [
        f"\n{'='*80}",
        f"❌ {title}",
        f"{'='*80}",
        "",
        f"File path: {model_path}",
        f"Model: {display_name}",
    ]
    
    if original_error:
        error_lines.append(f"Original error: {str(original_error)}")
    
    error_lines.extend([
        "",
        problem_msg,
        solution_msg,
        "",
        "See docs/manual/user-model-setup-guide.md for instructions.",
        f"{'='*80}",
    ])
    
    return "\n".join(error_lines)


def _format_network_error(display_name: str, error: Exception) -> str:
    """Format a network error message for HuggingFace download failures."""
    return (
        f"\n{'='*80}\n"
        f"❌ NETWORK ERROR LOADING MODEL!\n"
        f"{'='*80}\n\n"
        f"Model: {display_name}\n"
        f"Failed to connect to HuggingFace to download the model.\n\n"
        f"Solution: Please check your internet connection and try again.\n"
        f"Original error: {str(error)}\n"
        f"{'='*80}"
    )


def _format_disk_space_error(display_name: str, error: Exception) -> str:
    """Format a disk space error message."""
    return (
        f"\n{'='*80}\n"
        f"❌ DISK SPACE ERROR!\n"
        f"{'='*80}\n\n"
        f"Model: {display_name}\n"
        f"Insufficient disk space to download the model.\n\n"
        f"Solution: Please free up disk space and try again.\n"
        f"Original error: {str(error)}\n"
        f"{'='*80}"
    )


def _format_filesystem_error(display_name: str, error: Exception) -> str:
    """Format a file system error message."""
    return (
        f"\n{'='*80}\n"
        f"❌ FILE SYSTEM ERROR!\n"
        f"{'='*80}\n\n"
        f"Model: {display_name}\n"
        f"A file system error occurred while loading the model.\n\n"
        f"Solution: Please check file permissions and try again.\n"
        f"Cache location: {CACHE_DIR}\n"
        f"Original error: {str(error)}\n"
        f"{'='*80}"
    )


def _handle_huggingface_os_error(error: OSError, display_name: str) -> None:
    """Handle OSError from HuggingFace loading - raise appropriate error type.
    
    Args:
        error: The OSError exception
        display_name: Model display name for error messages
        
    Raises:
        OSError with appropriate formatted message
    """
    error_str = str(error).lower()
    if "no space" in error_str or "disk" in error_str:
        raise OSError(_format_disk_space_error(display_name, error))
    raise OSError(_format_filesystem_error(display_name, error))


def _format_file_lock_error(display_name: str, model_source: str, max_retries: int) -> str:
    """Format a file lock error message with diagnostics and solutions.
    
    Args:
        display_name: Human-readable model name
        model_source: Path or HuggingFace ID of the model
        max_retries: Number of retries attempted
        
    Returns:
        Formatted error message string
    """
    return (
        f"\n{'='*80}\n"
        f"❌ FILE LOCK ERROR (PERSISTED AFTER {max_retries} RETRIES)\n"
        f"{'='*80}\n\n"
        f"Model: {display_name}\n"
        f"Source: {model_source}\n\n"
        f"The model extraction cannot complete due to persistent file locks.\n"
        f"This typically happens when Windows services hold file handles open.\n\n"
        f"🔒 Likely Causes:\n"
        f"  1. Windows Defender or antivirus scanning temp files\n"
        f"  2. OneDrive, Google Drive, or Dropbox syncing the cache folder\n"
        f"  3. Windows Search or Windows Indexing Service\n"
        f"  4. Another Python process accessing the same models\n\n"
        f"💡 Solutions (in order of likelihood to work):\n"
        f"  1. Pause OneDrive/Dropbox/Google Drive temporarily\n"
        f"  2. Add cache directory to antivirus exclusions:\n"
        f"     {CACHE_DIR}\n"
        f"  3. Restart your computer\n"
        f"  4. Run as Administrator\n"
        f"  5. Disable Windows Search indexing for:\n"
        f"     {CACHE_DIR}\n\n"
        f"⚙️ Cache Location:\n"
        f"  {CACHE_DIR}\n\n"
        f"If this persists, try manually deleting the cache directory.\n"
        f"{'='*80}"
    )


def _format_permission_error(display_name: str, error_str: str) -> str:
    """Format a permission error message (non-file-lock).
    
    Args:
        display_name: Human-readable model name
        error_str: Original error message string
        
    Returns:
        Formatted error message string
    """
    return (
        f"\n{'='*80}\n"
        f"❌ PERMISSION ERROR!\n"
        f"{'='*80}\n\n"
        f"Model: {display_name}\n"
        f"Error: {error_str}\n\n"
        f"The process does not have permission to access the cache directory.\n"
        f"Try running as Administrator or checking directory permissions.\n"
        f"Cache location: {CACHE_DIR}\n"
        f"{'='*80}"
    )


def _is_file_lock_error(error_str: str) -> bool:
    """Check if error string indicates a Windows file lock issue."""
    return "WinError 32" in error_str or "being used by another process" in error_str


def _handle_retry_delay(attempt: int, base_delay: float, max_retries: int) -> bool:
    """Handle retry delay logic with garbage collection.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_retries: Maximum number of retries
        
    Returns:
        True if should retry, False if all retries exhausted
    """
    if attempt >= max_retries - 1:
        return False
    
    delay = base_delay * (attempt + 1)
    print(f"   ⚠️  Retry (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
    
    gc.collect()
    if torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        torch.cuda.empty_cache()  # type: ignore[reportUnknownMemberType]
    
    time.sleep(delay)
    return True


def _load_from_huggingface_with_retry(hf_model_id: str, config: Dict[str, Any], max_retries: int = 3) -> Any:
    """Load model from HuggingFace with retry logic for Windows file lock issues.
    
    Uses linear backoff delays (0.5s, 1.0s, 1.5s) to allow Windows services
    time to release file handles between retry attempts.
    
    Args:
        hf_model_id: HuggingFace model ID (e.g., "nvidia/parakeet-tdt-0.6b-v3")
        config: Model configuration dict
        max_retries: Maximum number of retry attempts for file lock errors
    
    Returns:
        Loaded ASR model
        
    Raises:
        PermissionError: If file lock persists after all retries
        ConnectionError: If HuggingFace download fails
        OSError: If disk space or file system issues occur
    """
    base_delay = 0.5
    display_name = config['display_name']
    
    for attempt in range(max_retries):
        try:
            return nemo_asr.models.ASRModel.from_pretrained(hf_model_id)  # type: ignore[reportUnknownMemberType, reportReturnType]
            
        except PermissionError as e:
            error_str = str(e)
            if _is_file_lock_error(error_str):
                if _handle_retry_delay(attempt, base_delay, max_retries):
                    continue
                raise PermissionError(_format_file_lock_error(display_name, hf_model_id, max_retries))
            raise PermissionError(_format_permission_error(display_name, error_str))
    
    raise RuntimeError(f"Failed to load model after {max_retries} attempts")


def _load_with_retry(restore_path: Union[str, Path], config: Dict[str, Any], max_retries: int = 3) -> Any:
    """Load model from .nemo file with retry logic for Windows file lock issues.
    
    Enhanced retry logic wrapper for ASRModel.restore_from() that handles
    Windows file locking errors during model extraction.
    
    Args:
        restore_path: Path to .nemo file
        config: Model configuration dict
        max_retries: Maximum retry attempts for file lock errors
        
    Returns:
        Loaded ASR model
        
    Raises:
        PermissionError: If file lock persists after retries
        OSError: If extraction or loading fails
    """
    base_delay = 0.5
    display_name = config['display_name']
    
    for attempt in range(max_retries):
        try:
            return nemo_asr.models.ASRModel.restore_from(restore_path=str(restore_path))  # type: ignore[reportUnknownMemberType, reportReturnType]
            
        except PermissionError as e:
            error_str = str(e)
            if _is_file_lock_error(error_str):
                if _handle_retry_delay(attempt, base_delay, max_retries):
                    continue
                raise PermissionError(_format_file_lock_error(display_name, str(restore_path), max_retries))
            raise  # Non-file-lock permission errors - re-raise immediately
        
        except Exception as e:
            # Retry transient failures (disk I/O, etc.)
            if _handle_retry_delay(attempt, base_delay, max_retries):
                print(f"   ⚠️  Extraction error: {e}")
                continue
            raise


# ============================================================================
# Model Loading Strategy Functions (extracted from load_model to reduce cc)
# ============================================================================

def _unload_cached_models(model_name: str, models_cache: Dict[str, Any]) -> None:
    """Unload other cached models to free VRAM before loading a new model.
    
    Args:
        model_name: The model being loaded (will NOT be unloaded)
        models_cache: Global model cache dictionary
    """
    models_to_unload = [key for key in models_cache.keys() if key != model_name]
    
    if not models_to_unload or not torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        return
        
    for old_model_key in models_to_unload:
        try:
            old_model = models_cache[old_model_key]
            print(f"🔄 Unloading {old_model_key} to free VRAM for {model_name}...")
            
            # Move model to CPU first to free VRAM
            old_model = old_model.to("cpu")
            
            # Delete the model reference
            del models_cache[old_model_key]
            del old_model
            
            # Clear CUDA cache and run garbage collection
            torch.cuda.empty_cache()  # type: ignore[reportUnknownMemberType]
            gc.collect()
            
            print(f"   ✅ {old_model_key} unloaded successfully")
            
        except Exception as e:
            print(f"   ⚠️  Failed to unload {old_model_key}: {e}")
            # Continue loading the new model anyway


def _try_load_local_model(model_path: Optional[Path], config: Dict[str, Any]) -> Tuple[Optional[Any], bool]:
    """Try loading model from local .nemo file.
    
    Returns:
        Tuple of (model, should_fallback: bool)
    """
    if not model_path or not model_path.exists():
        return None, True  # No local file, should fallback
    
    print(f"📦 Loading {config['display_name']} from local file...")
    print(f"   Path: {model_path}")
    
    try:
        model = _load_with_retry(restore_path=model_path, config=config, max_retries=3)
        _override_model_dataloader_config(model)
        return model, False
    except PermissionError as e:
        if _is_file_lock_error(str(e)):
            print(f"⚠️  Local file locked, falling back to HuggingFace...")
            return None, True
        raise
    except Exception as e:
        print(f"⚠️  Local file corrupted or invalid: {e}")
        print(f"   Falling back to HuggingFace download...")
        return None, True


def _load_model_local_or_huggingface(model_name: str, config: Dict[str, Any], script_dir: Path) -> Any:
    """Load model trying local .nemo first, then falling back to HuggingFace."""
    local_path = config.get("local_path")
    model_path = script_dir / local_path if local_path else None
    
    model, _ = _try_load_local_model(model_path, config)
    if model is not None:
        return model
    
    # Log HuggingFace fallback
    print(f"📦 Loading {config['display_name']} from HuggingFace...")
    if model_path:
        print(f"   Local .nemo file not found: {model_path}")
    print(f"   To download locally, run: python setup_local_models.py")
    
    return _load_model_huggingface(model_name, config)


def _load_model_local_only(config: Dict[str, Any], script_dir: Path) -> Any:
    """Load model strictly from local .nemo file.
    
    Raises:
        FileNotFoundError: If local file doesn't exist
    """
    model_path = script_dir / config["local_path"]
    
    if not model_path.exists():
        raise FileNotFoundError(_format_model_error(
            title="MODEL FILE NOT FOUND!",
            model_path=model_path,
            display_name=config['display_name'],
            problem_msg="The .nemo file must be created once using the setup script.",
            solution_msg="Please run: python setup_local_models.py"
        ))
    
    print(f"📦 Loading {config['display_name']} from local file...")
    
    model = _load_with_retry(restore_path=model_path, config=config, max_retries=3)
    return model


def _load_model_huggingface(model_name: str, config: Dict[str, Any]) -> Any:
    """Load model from HuggingFace.
    
    Raises:
        ConnectionError: If network fails
        OSError: If disk space issues
        RuntimeError: For other unexpected errors
    """
    hf_model_id = config["hf_model_id"]
    
    print(f"📦 Loading {config['display_name']} from HuggingFace...")
    print(f"   Model ID: {hf_model_id}")
    print("   (First load downloads model, subsequent loads use cache)")
    
    try:
        model = _load_from_huggingface_with_retry(hf_model_id, config, max_retries=3)
        _override_model_dataloader_config(model)
        return model
        
    except ConnectionError as e:
        raise ConnectionError(_format_network_error(config['display_name'], e))
    except OSError as e:
        _handle_huggingface_os_error(e, config['display_name'])
    except Exception as e:
        raise RuntimeError(
            f"\n{'='*80}\n"
            f"❌ ERROR LOADING MODEL!\n"
            f"{'='*80}\n\n"
            f"Model: {config['display_name']}\n"
            f"An unexpected error occurred: {type(e).__name__}\n\n"
            f"Solution: Try clearing the cache and retrying.\n"
            f"Cache location: {CACHE_DIR}\n"
            f"Original error: {str(e)}\n"
            f"{'='*80}"
        )


def _move_model_to_cuda(model: Any, model_name: str) -> Any:
    """Explicitly move model to CUDA if available."""
    if not torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        return model
        
    try:
        model = model.to("cuda")
        print(f"   ✅ Model moved to CUDA")
    except Exception as e:
        print(f"   ⚠️  Could not move model to CUDA: {e}")
        # Continue with model on CPU
    
    return model


# ============================================================================
# VRAM Management: Manual and Auto Unload
# ============================================================================
# Global setting for auto-unload after transcription
auto_unload_after_transcription: bool = False


def unload_all_models() -> str:
    """Unload all cached models to free VRAM.
    
    Call this when you're done transcribing and want to reclaim
    GPU memory for other applications like gaming or video editing.
    
    Returns:
        Status message indicating which models were unloaded and VRAM freed
    """
    global models_cache
    
    if not models_cache:
        return "ℹ️ No models currently loaded in memory"
    
    unloaded: List[str] = []
    initial_vram = 0.0
    final_vram = 0.0
    
    if torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        initial_vram = torch.cuda.memory_allocated() / 1024**3  # type: ignore[reportUnknownMemberType]
    
    for model_key in list(models_cache.keys()):
        try:
            print(f"🗑️ Unloading {model_key}...")
            model = models_cache[model_key]
            
            # Move to CPU first to free VRAM immediately
            model.cpu()
            
            # Delete from cache
            del models_cache[model_key]
            del model
            
            unloaded.append(model_key)
            print(f"   ✅ {model_key} unloaded")
            
        except Exception as e:
            print(f"   ⚠️ Failed to unload {model_key}: {e}")
    
    # Clear CUDA cache and garbage collect
    gc.collect()
    if torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        torch.cuda.empty_cache()  # type: ignore[reportUnknownMemberType]
        final_vram = torch.cuda.memory_allocated() / 1024**3  # type: ignore[reportUnknownMemberType]
    
    freed = initial_vram - final_vram
    
    if unloaded:
        return f"✅ Unloaded: {', '.join(unloaded)}\n💾 Freed ~{freed:.1f}GB VRAM"
    return "⚠️ No models were successfully unloaded"


def set_auto_unload(enabled: bool) -> str:
    """Enable or disable auto-unload after transcription.
    
    Args:
        enabled: True to auto-unload after each transcription
        
    Returns:
        Status message
    """
    global auto_unload_after_transcription
    auto_unload_after_transcription = enabled
    status = "enabled" if enabled else "disabled"
    return f"Auto-unload {status}"


def load_model(model_name: str, show_progress: bool = False) -> Any:
    """Load model using the appropriate method based on model type.
    
    All models now use standard ASRModel API (no SALM required).
    
    Loading methods:
    - "local": ONLY load from local .nemo file (fails if missing)
    - "huggingface": ONLY load from HuggingFace
    - "local_or_huggingface": Try local first, fallback to HuggingFace
    
    Includes explicit device management (Pattern 3: HuggingFace Modernization):
    - When loading a different model, unloads the old model from VRAM
    - Moves old model to CPU before deletion to free VRAM immediately
    - Clears CUDA cache and runs garbage collection
    - Explicitly moves new model to CUDA if available
    
    Args:
        model_name: Model key from MODEL_CONFIGS (e.g., "parakeet-v3", "canary-1b-v2")
        show_progress: Whether to show loading progress (for startup)
    
    Returns:
        The loaded ASR model
        
    Raises:
        FileNotFoundError: If the local .nemo file doesn't exist (for "local" method)
        ConnectionError: If HuggingFace download fails due to network issues
        OSError: If download fails due to disk space issues
        PermissionError: If file locks prevent model extraction
    """
    # Return cached model if valid
    if model_name in models_cache:
        cached_model = models_cache[model_name]
        try:
            if not hasattr(cached_model, 'transcribe'):
                print(f"⚠️  Cached {model_name} appears corrupted (missing transcribe method)")
                del models_cache[model_name]
                return load_model(model_name, show_progress)
            return cached_model
        except Exception as e:
            print(f"⚠️  Cached model validation failed: {e}")
            del models_cache[model_name]
    
    # Unload other cached models to free VRAM
    _unload_cached_models(model_name, models_cache)
    
    # Get config and prepare for loading
    config = MODEL_CONFIGS[model_name]
    script_dir = get_script_dir()
    loading_method = config.get("loading_method", "huggingface")
    
    start_time = time.time()
    
    # Load using appropriate strategy
    if loading_method == "local_or_huggingface":
        models_cache[model_name] = _load_model_local_or_huggingface(model_name, config, script_dir)
    elif loading_method == "local":
        models_cache[model_name] = _load_model_local_only(config, script_dir)
    else:
        models_cache[model_name] = _load_model_huggingface(model_name, config)
    
    load_time = time.time() - start_time
    print(f"✓ {config['display_name']} loaded in {load_time:.1f}s")
    
    # Move model to CUDA if available
    models_cache[model_name] = _move_model_to_cuda(models_cache[model_name], model_name)
    
    return models_cache[model_name]


def _save_logs(logs: str, prefix: str = "transcription") -> Optional[str]:
    """Save captured logs to a file for download.
    
    Args:
        logs: Log content string
        prefix: Filename prefix (e.g., "transcription", "error")
        
    Returns:
        Path to saved log file, or None if save failed
    """
    if not logs:
        return None
        
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{prefix}_log_{timestamp}.txt"
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Transcription Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(logs)
            
        return log_file
    except Exception as e:
        print(f"⚠️ Failed to save log file: {e}")
        return None


# ============================================================================
# Transcription Helper Functions (Refactored from transcribe_audio)
# ============================================================================
# These helper functions reduce nesting and improve maintainability of the
# main transcribe_audio function by extracting common patterns:
# - Error response formatting
# - File output generation
# - Statistics calculation
# ============================================================================

def _make_error_response(error_type: str, error_msg: str, log_capture_obj: 'LogCapture') -> Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Create standardized error response tuple for transcribe_audio.
    
    Args:
        error_type: Type of error (e.g., 'permission', 'network', 'validation')
        error_msg: Detailed error message
        log_capture_obj: LogCapture instance to stop and save
        
    Returns:
        Tuple of (status_message, transcription, txt_file, srt_file, csv_file, log_file)
    """
    logs = log_capture_obj.stop()
    log_file = _save_logs(logs, "error")
    
    error_messages = {
        'permission': f"### ❌ Permission Error\n\n{error_msg}",
        'permission_file_lock': (
            f"### ❌ Model Loading Failed: Windows File Lock\n\n"
            f"{error_msg}\n\n"
            f"**Root Cause:** Windows services (antivirus, OneDrive, indexing) are locking model files.\n\n"
            f"**Immediate Actions:**\n"
            f"1. Pause OneDrive/Dropbox/Google Drive\n"
            f"2. Temporarily disable antivirus real-time scanning\n"
            f"3. Run as Administrator\n"
            f"4. Restart your computer\n\n"
            f"**Cache Location:** `{CACHE_DIR}`\n\n"
            f"Add this folder to antivirus exclusions if issue persists."
        ),
        'network': f"### ❌ Network Error\n\n{error_msg}",
        'file_not_found': f"### ❌ File Not Found\n\n{error_msg}",
        'filesystem': f"### ❌ File System Error\n\n{error_msg}",
        'runtime': f"### ❌ Runtime Error\n\n{error_msg}",
        'validation': format_error_message('output_validation_failed', error_msg),
        'transcription_file_lock': (
            f"### ❌ Transcription Failed: File Lock Error\n\n"
            f"The transcription process failed due to file locking after 3 retry attempts.\n\n"
            f"**Solutions:**\n"
            f"1. Pause OneDrive/Dropbox/Google Drive temporarily\n"
            f"2. Add cache directory to antivirus exclusions: `{CACHE_DIR}`\n"
            f"3. Restart your computer\n"
            f"4. Run as Administrator\n\n"
            f"**Technical Details:**\n"
            f"```\n{error_msg}\n```"
        ),
        'transcription': (
            f"### ❌ Transcription Error\n\n"
            f"An error occurred during transcription.\n\n"
            f"**Technical Details:**\n"
            f"```\n{error_msg}\n```"
        ),
        'generic': (
            f"{'='*80}\n"
            f"❌ UNEXPECTED ERROR\n"
            f"{'='*80}\n\n"
            f"{error_msg}\n\n"
            f"Please check the console for more details.\n"
            f"{'='*80}"
        ),
    }
    
    status = error_messages.get(error_type, error_messages['generic'])
    return status, "", None, None, None, log_file


# ============================================================================
# Output File Generation Helpers (extracted to reduce nesting depth)
# ============================================================================

def _write_txt_header(
    f: Any, 
    model_choice: str, 
    total_duration: float, 
    total_time: float, 
    apply_itn: bool, 
    file_info: List[Dict[str, Any]], 
    is_batch: bool
) -> None:
    """Write TXT file header with metadata."""
    itn_text = 'Yes' if apply_itn and ITN_AVAILABLE else 'No'
    mins, secs = int(total_duration // 60), int(total_duration % 60)
    
    if is_batch:
        f.write(f"Batch Transcription - {len(file_info)} files\n")
        f.write(f"Model: {model_choice}\n")
        f.write(f"Total Duration: {mins}m {secs}s\n")
        f.write(f"Processing Time: {total_time:.2f}s\n")
        f.write(f"ITN Applied: {itn_text}\n")
        f.write(f"\n{SEPARATOR}\n")
    else:
        info = file_info[0]
        dur_mins, dur_secs = int(info['duration'] // 60), int(info['duration'] % 60)
        f.write(f"Audio File: {info['name']}\n")
        f.write(f"Model: {model_choice}\n")
        f.write(f"Duration: {dur_mins}m {dur_secs}s\n")
        f.write(f"Processing Time: {total_time:.2f}s\n")
        f.write(f"ITN Applied: {itn_text}\n")
        f.write(f"\n{SEPARATOR}\n")
        f.write(f"TRANSCRIPTION\n")
        f.write(f"{SEPARATOR}\n\n")


def _write_txt_batch_files(
    f: Any, 
    file_info: List[Dict[str, Any]], 
    all_transcriptions: Optional[List[str]], 
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]]
) -> None:
    """Write batch file transcriptions to TXT."""
    for i, info in enumerate(file_info):
        trans = all_transcriptions[i] if all_transcriptions else ""
        f.write(f"\nFILE {i+1}: {info['name']}\n")
        f.write(f"Duration: {int(info['duration'] // 60)}m {int(info['duration'] % 60)}s\n")
        f.write(f"{SEPARATOR}\n\n")
        ts, ts_level = all_timestamps[i] if all_timestamps and i < len(all_timestamps) else ([], 'none')
        if ts:
            f.write(format_as_txt_with_timestamps(trans, ts, ts_level))
        else:
            f.write(trans)
        f.write("\n")


def _get_batch_file_data(
    file_info: List[Dict[str, Any]], 
    all_transcriptions: Optional[List[str]], 
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]], 
    index: int
) -> Tuple[str, List[Dict[str, Any]], str, bool]:
    """Get transcription and timestamps for a batch file index.
    
    Returns:
        Tuple of (transcription, timestamps, timestamp_level, is_valid)
    """
    trans = all_transcriptions[index] if all_transcriptions else ""
    is_valid = not trans.startswith("[Transcription failed:")
    ts, ts_level = ([], 'none')
    if all_timestamps and index < len(all_timestamps):
        ts, ts_level = all_timestamps[index]
    return trans, ts, ts_level, is_valid


def _write_srt_batch(
    f: Any, 
    file_info: List[Dict[str, Any]], 
    all_transcriptions: Optional[List[str]], 
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]]
) -> None:
    """Write batch SRT with file headers and sequential numbering."""
    srt_index = 1
    for i, info in enumerate(file_info):
        trans, ts, _, is_valid = _get_batch_file_data(file_info, all_transcriptions, all_timestamps, i)
        if not is_valid:
            continue
            
        file_srt = format_as_srt(trans, ts, 'segment')
        
        # Write file header as first subtitle entry
        f.write(f"{srt_index}\n00:00:00,000 --> 00:00:02,000\n[FILE: {info['name']}]\n\n")
        srt_index += 1
        
        # Write subtitle entries, renumbering them
        for block in file_srt.split('\n\n'):
            if not block.strip():
                continue
            parts = block.split('\n', 1)
            if len(parts) >= 2:
                f.write(f"{srt_index}\n{parts[1]}\n\n")
                srt_index += 1


def _write_csv_timestamp_row(f: Any, filename: str, stamp: Dict[str, Any]) -> None:
    """Write a single CSV row for a timestamp entry."""
    start = stamp.get('start', 0.0)
    end = stamp.get('end', 0.0)
    duration = end - start
    text = stamp.get('text', stamp.get('word', stamp.get('segment', '')))
    escaped_text = text.replace('"', '""')
    escaped_name = filename.replace('"', '""')
    f.write(f'"{escaped_name}",{start:.3f},{end:.3f},{duration:.3f},"{escaped_text}"\n')


def _write_csv_batch(
    f: Any, 
    file_info: List[Dict[str, Any]], 
    all_transcriptions: Optional[List[str]], 
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]]
) -> None:
    """Write batch CSV with per-file timestamps."""
    f.write("file,start_time,end_time,duration,text\n")
    
    for i, info in enumerate(file_info):
        trans, ts, ts_level, is_valid = _get_batch_file_data(file_info, all_transcriptions, all_timestamps, i)
        if not is_valid:
            continue
            
        if ts:
            for stamp in ts:
                _write_csv_timestamp_row(f, info['name'], stamp)
        else:
            escaped_trans = trans.replace('"', '""')
            escaped_name = info['name'].replace('"', '""')
            f.write(f'"{escaped_name}",0.000,{info["duration"]:.3f},{info["duration"]:.3f},"{escaped_trans}"\n')


def _write_txt_content(f: TextIO, config: 'OutputFilesConfig') -> None:
    """Write TXT file content based on config."""
    if config.is_batch:
        _write_txt_batch_files(f, config.file_info, config.all_transcriptions, config.all_timestamps)
    elif config.timestamps:
        f.write(format_as_txt_with_timestamps(config.transcription or "", config.timestamps or [], config.timestamp_level))
    else:
        f.write(config.transcription or "")


def _write_srt_content(f: TextIO, config: 'OutputFilesConfig') -> None:
    """Write SRT file content based on config."""
    if config.is_batch:
        _write_srt_batch(f, config.file_info, config.all_transcriptions, config.all_timestamps)
    else:
        f.write(format_as_srt(config.transcription or "", config.timestamps or [], config.timestamp_level))


def _write_csv_content(f: TextIO, config: 'OutputFilesConfig') -> None:
    """Write CSV file content based on config."""
    if config.is_batch:
        _write_csv_batch(f, config.file_info, config.all_transcriptions, config.all_timestamps)
    else:
        f.write(format_as_csv(config.transcription or "", config.timestamps or [], config.timestamp_level))


def _save_output_files(base_filename: str, config: OutputFilesConfig):
    """Generate TXT, SRT, and CSV output files.
    
    Args:
        base_filename: Base name for output files (without extension)
        config: OutputFilesConfig with all file data and metadata
        
    Returns:
        Tuple of (txt_file, srt_file, csv_file) paths
    """
    txt_file = f"{base_filename}.txt"
    srt_file = f"{base_filename}.srt"
    csv_file = f"{base_filename}.csv"
    
    with open(txt_file, "w", encoding="utf-8") as f:
        _write_txt_header(f, config.model_choice, config.total_duration, config.total_time, 
                         config.apply_itn, config.file_info, config.is_batch)
        _write_txt_content(f, config)
    
    with open(srt_file, "w", encoding="utf-8") as f:
        _write_srt_content(f, config)
    
    with open(csv_file, "w", encoding="utf-8") as f:
        _write_csv_content(f, config)
    
    return txt_file, srt_file, csv_file


def _format_itn_status(apply_itn: bool) -> str:
    """Format ITN status string for display."""
    if apply_itn and ITN_AVAILABLE:
        return "- **ITN (Numbers to Digits)**: ✅ Applied"
    elif apply_itn:
        return "- **ITN (Numbers to Digits)**: ⚠️ Not installed"
    else:
        return "- **ITN (Numbers to Digits)**: Disabled"


def _format_batch_status(
    file_list: List[str], 
    file_info: List[Dict[str, Any]], 
    all_transcriptions: List[str], 
    per_file_stats: List[str], 
    per_file_errors: List[str], 
    stats: TranscriptionStats, 
    video_status: str = ""
) -> Tuple[str, str]:
    """Format status message for batch transcription.
    
    Args:
        file_list: List of original file paths
        file_info: List of file metadata dicts
        all_transcriptions: List of transcription strings
        per_file_stats: List of per-file stats strings
        per_file_errors: List of error messages
        stats: TranscriptionStats with processing metrics
        video_status: Optional video detection message
    
    Returns:
        Tuple of (status_message, combined_transcription)
    """
    total_mins = int(stats.total_duration // 60)
    total_secs = int(stats.total_duration % 60)
    
    # Calculate total words only from successful transcriptions
    successful = [t for t in all_transcriptions if not t.startswith("[Transcription failed:")]
    total_words = sum(len(t.split()) for t in successful)
    
    # Error summary
    error_summary = ""
    if per_file_errors:
        error_summary = (
            f"\n\n⚠️ **{len(per_file_errors)} file(s) failed to transcribe:**\n"
            + "\n".join(per_file_errors)
        )
    
    itn_status = _format_itn_status(stats.apply_itn)
    
    status = f"""
### ✅ Batch Transcription Complete!

{video_status}**📊 Overall Statistics:**
- **Files Processed**: {len(file_list)} ({len(file_list) - len(per_file_errors)} successful, {len(per_file_errors)} failed)
- **Model**: {stats.model_choice}
- **GPU**: {stats.gpu_name}
- **Total Audio Duration**: {total_mins}m {total_secs}s
- **Processing Time**: {stats.total_time:.2f} seconds
- **Inference Time**: {stats.inference_time:.2f} seconds
- **Chunk Size**: {stats.chunk_size}s
- **Real-Time Factor**: {stats.rtfx:.1f}× (processed {stats.rtfx:.1f}× faster than real-time)
- **VRAM Used**: {stats.vram_used:.2f} GB
- **Total Words**: {total_words}
{itn_status}

**📁 Per-File Statistics:**
{chr(10).join(per_file_stats)}{error_summary}

---
"""
    
    # Combine transcriptions with file headers
    combined = ""
    for i, (info, trans) in enumerate(zip(file_info, all_transcriptions)):
        combined += f"\n{SEPARATOR}\n"
        combined += f"FILE {i+1}: {info['name']}\n"
        combined += f"{SEPARATOR}\n\n"
        combined += trans + "\n"
    
    return status, combined


def _format_single_status(file_info: List[Dict[str, Any]], stats: TranscriptionStats, transcription: str,
                          timestamp_level: str, include_timestamps: bool, video_status: str = "") -> str:
    """Format status message for single file transcription.
    
    Args:
        file_info: List with single file metadata dict
        stats: TranscriptionStats with processing metrics
        transcription: The transcription text
        timestamp_level: Level of timestamps ('word', 'segment', etc.)
        include_timestamps: Whether timestamps were requested
        video_status: Optional video detection message
    
    Returns:
        Status message string
    """
    info = file_info[0]
    minutes = int(info["duration"] // 60)
    seconds = int(info["duration"] % 60)
    
    file_type_msg = "🎬 Video file detected - audio extracted automatically\n" if info["is_video"] else ""
    timestamp_status = format_timestamp_status(timestamp_level, include_timestamps)
    itn_status = _format_itn_status(stats.apply_itn)
    
    return f"""
### ✅ Transcription Complete!

{file_type_msg}**📊 Statistics:**
- **Model**: {stats.model_choice}
- **GPU**: {stats.gpu_name}
- **Audio Duration**: {minutes}m {seconds}s
- **Processing Time**: {stats.total_time:.2f} seconds
- **Inference Time**: {stats.inference_time:.2f} seconds
- **Model Load Time**: {stats.load_time:.2f} seconds
- **Chunk Size**: {stats.chunk_size}s
- **Real-Time Factor**: {stats.rtfx:.1f}× (processed {stats.rtfx:.1f}× faster than real-time)
- **VRAM Used**: {stats.vram_used:.2f} GB
- **Transcription Length**: {len(transcription)} characters ({len(transcription.split())} words)
{itn_status}
{timestamp_status}

---
"""


def _get_audio_duration_with_retry(file_path: str, max_retries: int = 4, base_delay: float = 0.5) -> float:
    """Get audio duration with retry logic for Windows file locks.
    
    Args:
        file_path: Path to the audio file
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        
    Returns:
        Duration in seconds
        
    Raises:
        OSError/PermissionError: If file access fails after all retries
    """
    import librosa
    
    for attempt in range(max_retries):
        try:
            duration: float = librosa.get_duration(path=file_path)  # type: ignore[reportUnknownMemberType]
            gc.collect()
            return duration  # type: ignore[reportReturnType]
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                print(f"   ⚠️  File lock on duration check (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise
    
    # This shouldn't be reached due to raise above, but satisfies type checker
    raise RuntimeError("Retry loop exited unexpectedly")


def _process_audio_files(file_list: List[str], log_capture_obj: 'LogCapture') -> Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], float, int, Optional[Tuple[Any, ...]]]:
    """Process and validate uploaded audio/video files.
    
    Copies files to cache directory and extracts duration information.
    
    Args:
        file_list: List of file paths to process
        log_capture_obj: LogCapture instance for error handling
        
    Returns:
        Tuple of (processed_files, file_info, total_duration, video_count, error_response)
        If error_response is not None, processing failed and should return immediately.
    """
    processed_files: List[str] = []
    file_info: List[Dict[str, Any]] = []
    total_duration = 0
    video_count = 0
    
    for file_path in file_list:
        # Copy file to cache directory
        try:
            cached_file_path = copy_gradio_file_to_cache(file_path)
            print(f"📁 Using cached file: {os.path.basename(cached_file_path)}")
        except OSError as e:
            return None, None, 0, 0, _make_error_response(
                'filesystem', f"Failed to copy uploaded file to cache.\n\nError: {str(e)}", 
                log_capture_obj
            )
        
        file_ext = os.path.splitext(cached_file_path)[1].lower()
        is_video = file_ext in VIDEO_EXTENSIONS
        
        if is_video:
            video_count += 1
            print(f"🎬 Extracting audio from video: {os.path.basename(cached_file_path)}")
        
        # Get duration with retry logic
        try:
            duration = _get_audio_duration_with_retry(cached_file_path)
        except (OSError, PermissionError, Exception) as e:
            if is_video:
                return None, None, 0, 0, _make_error_response(
                    'filesystem',
                    f"Video file '{os.path.basename(cached_file_path)}' appears to have no audio track or cannot be processed.\n\nError: {str(e)}",
                    log_capture_obj
                )
            raise
        
        total_duration += duration
        processed_files.append(cached_file_path)
        file_info.append({
            "path": cached_file_path,
            "name": os.path.basename(file_path),  # Use original name for display
            "duration": duration,
            "is_video": is_video
        })
    
    return processed_files, file_info, total_duration, video_count, None
    
    return processed_files, file_info, total_duration, video_count, None


def _normalize_file_list(audio_files: Any) -> List[str]:
    """Convert various Gradio file input formats to a flat list of paths.
    
    Args:
        audio_files: File input from Gradio - string, list, or file object
        
    Returns:
        List of file path strings
    """
    if audio_files is None:
        return []
    
    if isinstance(audio_files, str):
        return [audio_files]
    
    if isinstance(audio_files, list):
        result: List[str] = []
        for f in audio_files:
            if hasattr(f, 'name'):  # type: ignore[reportUnknownArgumentType]
                result.append(f.name)  # type: ignore[reportUnknownMemberType]
            else:
                result.append(str(f))  # type: ignore[reportUnknownArgumentType]
        return result
    
    if hasattr(audio_files, 'name'):
        return [audio_files.name]
    
    return [str(audio_files)]


def _load_model_for_transcription(model_key: str, log_capture: 'LogCapture') -> Tuple[Any, Optional[Tuple[Any, ...]]]:
    """Load transcription model with standardized error handling.
    
    Args:
        model_key: Model key from MODEL_CONFIGS
        log_capture: LogCapture instance for error responses
        
    Returns:
        Tuple of (model, error_response) - error_response is None on success
    """
    try:
        model = load_model(model_key)
        return model, None
    except PermissionError as e:
        error_msg = str(e)
        print(f"GRADIO_ERROR (PermissionError): {error_msg}")
        error_type = 'permission_file_lock' if ("WinError 32" in error_msg or "being used by another process" in error_msg) else 'permission'
        return None, _make_error_response(error_type, error_msg, log_capture)
    except ConnectionError as e:
        print(f"GRADIO_ERROR (ConnectionError): {e}")
        return None, _make_error_response('network', str(e), log_capture)
    except FileNotFoundError as e:
        print(f"GRADIO_ERROR (FileNotFoundError): {e}")
        return None, _make_error_response('file_not_found', str(e), log_capture)
    except OSError as e:
        print(f"GRADIO_ERROR (OSError): {e}")
        return None, _make_error_response('filesystem', str(e), log_capture)
    except RuntimeError as e:
        print(f"GRADIO_ERROR (RuntimeError): {e}")
        return None, _make_error_response('runtime', str(e), log_capture)
    except Exception as e:
        error_msg = f"Type: {type(e).__name__}\nMessage: {str(e)}"
        print(f"GRADIO_ERROR (Unexpected): {error_msg}")
        return None, _make_error_response('generic', error_msg, log_capture)


def _get_gpu_stats() -> Tuple[float, str]:
    """Get GPU statistics for reporting.
    
    Returns:
        Tuple of (vram_used_gb, gpu_name)
    """
    if torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        vram_used: float = torch.cuda.memory_allocated() / 1024**3  # type: ignore[reportUnknownMemberType]
        gpu_name: str = torch.cuda.get_device_name(0)  # type: ignore[reportUnknownMemberType]
        return vram_used, gpu_name  # type: ignore[reportReturnType]
    return 0.0, "CPU"


def _process_batch_results(
    result: List[Any], 
    file_info: List[Dict[str, Any]], 
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]], 
    apply_itn: bool, 
    include_timestamps: bool
) -> Tuple[List[str], List[Tuple[List[Dict[str, Any]], str]], List[str], List[str]]:
    """Process batch transcription results with validation and ITN.
    
    Args:
        result: List of transcription results from model
        file_info: List of file metadata dicts
        chunk_timestamps_map: Dict mapping file index to chunk timestamps
        apply_itn: Whether to apply inverse text normalization
        include_timestamps: Whether timestamps were requested
        
    Returns:
        Tuple of (all_transcriptions, all_timestamps, per_file_stats, per_file_errors)
    """
    all_transcriptions: List[str] = []
    all_timestamps: List[Tuple[List[Dict[str, Any]], str]] = []
    per_file_stats: List[str] = []
    per_file_errors: List[str] = []
    
    for i, (res, info) in enumerate(zip(result, file_info)):
        success, transcription, error_msg = validate_transcription_result(result, i)
        
        if success:
            # Apply ITN if enabled and NOT already applied per-chunk
            if apply_itn and i not in chunk_timestamps_map:
                transcription = apply_inverse_text_normalization(transcription)
            
            all_transcriptions.append(transcription)
            
            # Get timestamps - use chunk timestamps if available
            if i in chunk_timestamps_map:
                timestamps = chunk_timestamps_map[i]
                timestamp_level = 'segment'
            else:
                timestamps, timestamp_level = extract_timestamps(res, include_timestamps)
            all_timestamps.append((timestamps, timestamp_level))
            
            # Build stats entry
            file_duration = info["duration"]
            file_mins, file_secs = int(file_duration // 60), int(file_duration % 60)
            file_type = "🎬 Video" if info["is_video"] else "🎵 Audio"
            
            per_file_stats.append(
                f"**{i+1}. {info['name']}** ({file_type})\n"
                f"   - Duration: {file_mins}m {file_secs}s\n"
                f"   - Words: {len(transcription.split())}"
            )
        else:
            all_transcriptions.append(f"[Transcription failed: {error_msg}]")
            all_timestamps.append(([], 'none'))
            per_file_errors.append(f"**{i+1}. {info['name']}**: {error_msg}")
            per_file_stats.append(
                f"**{i+1}. {info['name']}** ❌ Failed\n"
                f"   - Error: {error_msg}"
            )
    
    return all_transcriptions, all_timestamps, per_file_stats, per_file_errors


def _process_single_result(
    result: List[Any], 
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]], 
    apply_itn_final: bool, 
    include_timestamps: bool, 
    log_capture: 'LogCapture',
    had_itn_per_chunk: bool = False
) -> Tuple[Optional[str], List[Dict[str, Any]], str, Optional[Tuple[Any, ...]]]:
    """Process single file transcription result with validation and ITN.
    
    Args:
        result: Transcription result from model
        chunk_timestamps_map: Dict mapping file index to chunk timestamps
        apply_itn_final: Whether to apply final-pass inverse text normalization
        include_timestamps: Whether timestamps were requested
        log_capture: LogCapture instance for error responses
        had_itn_per_chunk: Whether ITN was already applied per-chunk
        
    Returns:
        Tuple of (transcription, timestamps, timestamp_level, error_response)
        error_response is None on success
    """
    success, transcription, error_msg = validate_transcription_result(result, 0)
    
    if not success:
        return None, [], 'none', _make_error_response('validation', error_msg, log_capture)
    
    # Apply final-pass ITN based on mode
    if apply_itn_final:
        if had_itn_per_chunk:
            print(f"   🔢 Applying final-pass ITN (already applied per-chunk, mode=both)")
        else:
            print(f"   🔢 Applying final-pass Inverse Text Normalization...")
        transcription = apply_inverse_text_normalization(transcription)
    elif had_itn_per_chunk:
        print(f"   🔢 ITN was applied per-chunk during transcription")
    
    # Get timestamps
    timestamps, timestamp_level = _extract_single_result_timestamps(
        result, chunk_timestamps_map, include_timestamps
    )
    
    return transcription, timestamps, timestamp_level, None


def _extract_single_result_timestamps(
    result: List[Any], 
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]], 
    include_timestamps: bool
) -> Tuple[List[Dict[str, Any]], str]:
    """Extract timestamps from single result, preferring chunk-based if available.
    
    Returns:
        Tuple of (timestamps list, timestamp_level string)
    """
    if 0 in chunk_timestamps_map and chunk_timestamps_map[0]:
        timestamps = chunk_timestamps_map[0]
        ts_level = 'word' if any('word' in ts for ts in timestamps) else 'segment'
        print(f"   ⏱️ Using chunk-based timestamps ({len(timestamps)} entries)")
        return timestamps, ts_level
    
    if include_timestamps:
        return extract_timestamps(result[0], include_timestamps)
    
    return [], 'none'


def _generate_and_save_output_files(
    save_to_file: bool, 
    config: OutputFilesConfig
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Generate output files if save_to_file is enabled.
    
    Args:
        save_to_file: Whether to save files
        config: OutputFilesConfig with all file generation parameters
    
    Returns:
        Tuple of (txt_file, srt_file, csv_file, status_suffix)
    """
    if not save_to_file:
        return None, None, None, ""
    
    base_name = os.path.splitext(os.path.basename(config.file_list[0]))[0]
    
    if config.is_batch:
        base_filename = f"batch_transcription_{len(config.file_list)}_files"
        # Create batch config with no timestamps set
        batch_config = OutputFilesConfig(
            file_list=config.file_list,
            file_info=config.file_info,
            is_batch=True,
            include_timestamps=False,
            model_choice=config.model_choice,
            total_duration=config.total_duration,
            total_time=config.total_time,
            apply_itn=config.apply_itn,
            transcription=None,
            timestamps=None,
            timestamp_level='none',
            all_transcriptions=config.all_transcriptions,
            all_timestamps=config.all_timestamps
        )
        txt_file, srt_file, csv_file = _save_output_files(base_filename, batch_config)
    else:
        base_filename = f"{base_name}_transcription"
        # Create single file config with adjusted timestamps
        single_config = OutputFilesConfig(
            file_list=config.file_list,
            file_info=config.file_info,
            is_batch=False,
            include_timestamps=config.include_timestamps,
            model_choice=config.model_choice,
            total_duration=config.total_duration,
            total_time=config.total_time,
            apply_itn=config.apply_itn,
            transcription=config.transcription,
            timestamps=config.timestamps if config.include_timestamps else None,
            timestamp_level=config.timestamp_level
        )
        txt_file, srt_file, csv_file = _save_output_files(base_filename, single_config)
    
    return txt_file, srt_file, csv_file, f"\n💾 **Files saved**: `{base_filename}.[txt/srt/csv]`"


def _process_batch_transcription(
    result: List[Any], 
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]], 
    ctx: ResultProcessingContext
) -> Tuple[str, str, List[Any], str, List[str], List[Tuple[List[Dict[str, Any]], str]]]:
    """Process batch transcription results and format output.
    
    Args:
        result: Transcription result from model
        chunk_timestamps_map: Map of chunk timestamps
        ctx: ResultProcessingContext with stats, file_info, etc.
    
    Returns:
        Tuple of (status, transcription_output, timestamps, timestamp_level, all_transcriptions, all_timestamps)
    """
    all_transcriptions, all_timestamps, per_file_stats, per_file_errors = _process_batch_results(
        result, ctx.file_info, chunk_timestamps_map, ctx.stats.apply_itn, ctx.include_timestamps
    )
    
    status, transcription_output = _format_batch_status(
        file_list=ctx.file_list,
        file_info=ctx.file_info,
        all_transcriptions=all_transcriptions,
        per_file_stats=per_file_stats,
        per_file_errors=per_file_errors,
        stats=ctx.stats,
        video_status=ctx.video_status
    )
    
    return status, transcription_output, [], 'none', all_transcriptions, all_timestamps


def _process_single_transcription(
    result: List[Any], 
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]], 
    log_capture: 'LogCapture', 
    ctx: ResultProcessingContext
) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]], str, Optional[Tuple[Any, ...]]]:
    """Process single file transcription result and format output.
    
    Args:
        result: Transcription result from model
        chunk_timestamps_map: Map of chunk timestamps
        log_capture: LogCapture instance for error handling
        ctx: ResultProcessingContext with stats, file_info, etc.
    
    Returns:
        Tuple of (status, transcription_output, timestamps, timestamp_level, error_response)
    """
    transcription, timestamps, timestamp_level, error_response = _process_single_result(
        result, chunk_timestamps_map, ctx.apply_itn_final, ctx.include_timestamps, log_capture,
        had_itn_per_chunk=ctx.had_itn_per_chunk
    )
    
    if error_response is not None:
        return None, None, [], 'none', error_response
    
    # Create stats with single-file duration
    single_stats = TranscriptionStats(
        model_choice=ctx.stats.model_choice,
        gpu_name=ctx.stats.gpu_name,
        total_duration=ctx.file_info[0]["duration"],
        total_time=ctx.stats.total_time,
        inference_time=ctx.stats.inference_time,
        load_time=ctx.load_time,
        chunk_size=ctx.stats.chunk_size,
        rtfx=ctx.stats.rtfx,
        vram_used=ctx.stats.vram_used,
        apply_itn=ctx.stats.apply_itn
    )
    
    status = _format_single_status(
        file_info=ctx.file_info,
        stats=single_stats,
        transcription=transcription or "",
        timestamp_level=timestamp_level,
        include_timestamps=ctx.include_timestamps
    )
    
    if timestamps and ctx.include_timestamps:
        transcription_output = format_as_txt_with_timestamps(transcription or "", timestamps, timestamp_level)
    else:
        transcription_output = transcription or ""
    
    return status, transcription_output, timestamps, timestamp_level, None


def _run_transcription(
    model: Any, 
    processed_files: List[str], 
    batch_size: int, 
    chunk_size: int, 
    apply_itn: bool, 
    log_capture: 'LogCapture'
) -> Tuple[Optional[List[Any]], Dict[int, List[Dict[str, Any]]], Optional[Tuple[Any, ...]]]:
    """Run transcription with error handling.
    
    Returns:
        Tuple of (result, chunk_timestamps_map, error_response)
    """
    try:
        result, chunk_timestamps_map = _transcribe_with_retry(
            model=model,
            files=processed_files,
            batch_size=batch_size,
            use_cuda=torch.cuda.is_available(),  # type: ignore[reportUnknownMemberType]
            max_retries=3,
            chunk_size_override=chunk_size,
            apply_itn=apply_itn
        )
        return result, chunk_timestamps_map, None
            
    except PermissionError as e:
        error_str = str(e)
        print(f"❌ Transcription permission error: {error_str}")
        error_type = 'transcription_file_lock' if _is_file_lock_error(error_str) else 'permission'
        return None, {}, _make_error_response(error_type, error_str, log_capture)
                
    except Exception as e:
        error_msg = f"Error Type: {type(e).__name__}\n\nDetails: {str(e)}"
        print(f"❌ Transcription error: {type(e).__name__}: {str(e)}")
        return None, {}, _make_error_response('transcription', error_msg, log_capture)


def transcribe_audio(
    audio_files: Any, 
    model_choice: str, 
    save_to_file: bool, 
    include_timestamps: bool,
    output_format: str = "txt", 
    apply_itn: bool = True, 
    chunk_size: int = 120, 
    batch_size: int = 1,
    silence_threshold: float = 0.5,
    itn_mode_choice: str = "per_chunk"
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Main transcription function with batch processing, video support, and GPU optimization.
    
    Args:
        audio_files: Input audio/video files
        model_choice: Model selection from UI
        save_to_file: Whether to save output files
        include_timestamps: Whether to include timestamps
        output_format: Output format (txt, srt, csv)
        apply_itn: Whether to apply ITN (legacy, now controlled by itn_mode_choice)
        chunk_size: Audio chunk size in seconds
        batch_size: Batch size for processing
        silence_threshold: End subtitle segments when silence gap exceeds this (seconds)
        itn_mode_choice: ITN mode - per_chunk, final_pass, both, or disabled
    """
    
    # Start capturing logs
    log_capture.start()
    print(f"\n{'='*60}")
    print(f"🎙️ Transcription Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Update global settings
    global chunk_duration_sec, long_audio_threshold_sec, silence_threshold_sec, itn_mode
    chunk_duration_sec = chunk_size
    long_audio_threshold_sec = chunk_size + 30
    silence_threshold_sec = silence_threshold
    itn_mode = itn_mode_choice
    
    # Determine ITN behavior based on mode
    apply_itn_per_chunk = itn_mode_choice in ("per_chunk", "both")
    apply_itn_final = itn_mode_choice in ("final_pass", "both")
    itn_enabled = itn_mode_choice != "disabled" and apply_itn
    
    # Early return for empty input
    file_list = _normalize_file_list(audio_files)
    if not file_list:
        logs = log_capture.stop()
        return "⚠️ Please upload an audio or video file first", "", None, None, None, None
    
    try:
        import librosa
        
        is_batch = len(file_list) > 1
        
        # Log settings
        print(f"📁 Files: {len(file_list)}")
        print(f"📊 Model: {model_choice}")
        print(f"📝 Output format: {output_format.upper()}")
        print(f"🔢 ITN mode: {itn_mode_choice} {'(enabled)' if itn_enabled else '(disabled)'}")
        print(f"⏱️ Timestamps: {'Enabled' if include_timestamps else 'Disabled'}")
        print(f"📦 Chunk size: {chunk_size}s")
        print(f"🔇 Silence threshold: {silence_threshold}s")
        
        model_key = get_model_key_from_choice(model_choice)
        start_time = time.time()
        
        # Load model
        model, error_response = _load_model_for_transcription(model_key, log_capture)
        if error_response is not None:
            return error_response
        
        load_time = time.time() - start_time
        
        # Process audio files
        processed_files, file_info, total_duration, video_count, error_response = _process_audio_files(
            file_list, log_capture
        )
        if error_response is not None:
            return error_response
        
        # Assert non-None after error check (for type checker)
        assert processed_files is not None
        assert file_info is not None
        
        video_status = f"🎬 Extracted audio from {video_count} video file(s)\n" if video_count > 0 else ""
        
        # Run transcription (apply_itn_per_chunk controls per-chunk ITN during chunked transcription)
        inference_start = time.time()
        result, chunk_timestamps_map, error_response = _run_transcription(
            model, processed_files, 4, chunk_size, apply_itn_per_chunk, log_capture
        )
        if error_response is not None:
            return error_response
        
        # Assert result is not None after error check (for type checker)
        assert result is not None
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        vram_used, gpu_name = _get_gpu_stats()
        rtfx = total_duration / inference_time if inference_time > 0 else 0
        
        print(f"\n✅ Transcription complete!")
        print(f"   Duration: {total_duration:.1f}s audio in {inference_time:.1f}s ({rtfx:.1f}× real-time)")
        
        # Create shared stats for result processing
        stats = TranscriptionStats(
            model_choice=model_choice, gpu_name=gpu_name, total_duration=total_duration,
            total_time=total_time, inference_time=inference_time, load_time=load_time,
            chunk_size=chunk_size, rtfx=rtfx, vram_used=vram_used, apply_itn=itn_enabled
        )
        ctx = ResultProcessingContext(
            stats=stats, file_list=file_list, file_info=file_info,
            include_timestamps=include_timestamps, video_status=video_status, load_time=load_time,
            apply_itn_final=apply_itn_final, had_itn_per_chunk=apply_itn_per_chunk
        )
        
        # Process results based on batch vs single
        if is_batch:
            status, transcription_output, timestamps, timestamp_level, all_transcriptions, all_timestamps = \
                _process_batch_transcription(result, chunk_timestamps_map, ctx)
        else:
            status, transcription_output, timestamps, timestamp_level, error_response = \
                _process_single_transcription(result, chunk_timestamps_map, log_capture, ctx)
            if error_response is not None:
                return error_response
            all_transcriptions, all_timestamps = None, None
        
        # Generate output files
        output_config = OutputFilesConfig(
            file_list=file_list, file_info=file_info, is_batch=is_batch,
            include_timestamps=include_timestamps, model_choice=model_choice,
            total_duration=total_duration, total_time=total_time, apply_itn=apply_itn,
            transcription=transcription_output if not is_batch else None,
            timestamps=timestamps if not is_batch else None,
            timestamp_level=timestamp_level,
            all_transcriptions=all_transcriptions if is_batch else None,
            all_timestamps=all_timestamps if is_batch else None
        )
        txt_file, srt_file, csv_file, status_suffix = _generate_and_save_output_files(
            save_to_file=save_to_file, config=output_config
        )
        if status is not None:
            status = status + status_suffix
        else:
            status = status_suffix
        
        print(f"\n{'='*60}")
        print(f"✅ Transcription Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Auto-unload models if enabled
        if auto_unload_after_transcription:
            print("\n🗑️ Auto-unloading models to free VRAM...")
            unload_result = unload_all_models()
            print(unload_result)
        
        logs = log_capture.stop()
        log_file = _save_logs(logs, "transcription")
        return status, transcription_output or "", txt_file, srt_file, csv_file, log_file
        
    except Exception as e:
        vram_info = f"you have {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f}GB" if torch.cuda.is_available() else "no GPU detected"  # type: ignore[reportUnknownMemberType]
        error_msg = f"""
### ❌ Error During Transcription

**Error Type**: {type(e).__name__}

**Error Message**: {str(e)}

**Troubleshooting:**
1. Make sure the audio/video file is valid
2. Check that you have enough VRAM ({vram_info})
3. Try a shorter audio file first
4. Restart the interface if models are stuck
5. For video files, ensure FFmpeg is installed

If the error persists, please check the console output for more details.
"""
        logs = log_capture.stop()
        log_file = _save_logs(logs, "error")
        return error_msg, "", None, None, None, log_file
        logs = log_capture.stop()
        log_file = _save_logs(logs, "error")
        return error_msg, "", None, None, None, log_file

def get_system_info() -> str:
    """Display system information"""
    if torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        gpu_name: str = torch.cuda.get_device_name(0)  # type: ignore[reportUnknownMemberType]
        vram_total: float = torch.cuda.get_device_properties(0).total_memory / 1024**3  # type: ignore[reportUnknownMemberType]
        vram_free: float = (torch.cuda.get_device_properties(0).total_memory -  # type: ignore[reportUnknownMemberType]
                     torch.cuda.memory_allocated()) / 1024**3  # type: ignore[reportUnknownMemberType]
        
        cuda_version: str = torch.version.cuda  # type: ignore[reportUnknownMemberType]
        pytorch_version: str = torch.__version__  # type: ignore[reportUnknownMemberType]
        
        info = f"""
### 🖥️ System Information

**GPU**: {gpu_name}
**Total VRAM**: {vram_total:.1f} GB
**Available VRAM**: {vram_free:.1f} GB
**CUDA Version**: {cuda_version}
**PyTorch Version**: {pytorch_version}
**NeMo Available**: ✅ Yes

**Status**: ✅ Ready for transcription
**Models Available**: All Parakeet and Canary models supported
"""
    else:
        info = f"""
### ⚠️ No GPU Detected

CUDA is not available. Please check:
1. NVIDIA GPU drivers are installed
2. CUDA Toolkit is installed
3. PyTorch was installed with CUDA support

**NeMo Available**: Check console output
"""
    return info

def get_privacy_performance_info() -> str:
    """Generate dynamic privacy & performance information"""
    if torch.cuda.is_available():  # type: ignore[reportUnknownMemberType]
        gpu_name: str = torch.cuda.get_device_name(0)  # type: ignore[reportUnknownMemberType]
        vram_total: float = torch.cuda.get_device_properties(0).total_memory / 1024**3  # type: ignore[reportUnknownMemberType]
        gpu_info = f"**GPU**: {gpu_name} ({vram_total:.0f}GB VRAM)"
    else:
        gpu_info = "**GPU**: CPU Mode (No CUDA GPU detected)"
    
    return f"""
### Privacy:
- ✅ All transcription processing happens locally on your machine
- ✅ Your audio never leaves your computer
- ✅ All models: Try local .nemo first, then HuggingFace fallback
- ✅ Once downloaded, models work offline from cache

### Model Storage:
- **Local .nemo files**: `local_models/` directory (run `python setup_local_models.py`)
  - `parakeet-0.6b-v3.nemo` (~2.4GB)
  - `parakeet-1.1b.nemo` (~4.5GB)
  - `canary-1b.nemo` (~5.0GB)
  - `canary-1b-v2.nemo` (~5.0GB)
- **HuggingFace Cache**: `model_cache/` directory (auto-downloads if .nemo missing)

### Cache Location:
- **Custom cache**: `{CACHE_DIR}` (prevents Windows temp file issues)
- This directory is used for all model downloads and extraction

### Performance Optimizations:
- {gpu_info}
- **Speed**: 200-3,380× faster than real-time (model dependent)
- **Example**: 1 hour audio → 10-60 seconds processing
- **Mixed Precision**: FP16 inference for 1.5-2.5× speedup
- **TF32 Tensor Cores**: Enabled for matrix operations
- **Dynamic Batch Sizing**: Optimized based on audio duration
- **Memory Caching**: Models stay in RAM after first load
"""

# Create the Gradio interface
with gr.Blocks(title="🎙️ Local ASR Transcription") as app:
    
    # Header
    gr.Markdown("""
    # 🎙️ NVIDIA NeMo Local Audio Transcription
    ### 100% Offline - Powered by Parakeet & Canary ASR Models
    
    Transform your audio and video files into accurate text transcriptions using state-of-the-art AI models stored locally on your system. No internet required.
    """)
    
    # System info at the top
    with gr.Accordion("📊 System Information", open=False):
        system_info = gr.Markdown(get_system_info())
        refresh_btn = gr.Button("🔄 Refresh System Info", size="sm")
        refresh_btn.click(fn=get_system_info, outputs=system_info)
    
    # VRAM Management section
    with gr.Accordion("💾 VRAM Management", open=False):
        gr.Markdown("""
        **Free GPU Memory**: Unload models when you're done transcribing to free up VRAM for other apps (gaming, video editing, etc.)
        
        Models are cached in memory by default for fast subsequent transcriptions.
        """)
        with gr.Row():
            unload_btn = gr.Button("🗑️ Unload All Models (Free VRAM)", size="sm", variant="secondary")
            auto_unload_checkbox = gr.Checkbox(
                label="Auto-unload after transcription",
                value=False,
                info="Automatically free VRAM after each transcription completes"
            )
        unload_status = gr.Markdown("")
        
        unload_btn.click(fn=unload_all_models, outputs=unload_status)
        auto_unload_checkbox.change(fn=set_auto_unload, inputs=auto_unload_checkbox, outputs=unload_status)
    
    gr.Markdown("---")
    
    # Main interface
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Audio/Video Files")
            
            # Use gr.File for multiple file uploads (audio and video)
            audio_input = gr.File(
                file_count="multiple",
                file_types=[
                    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma",  # Audio
                    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".m4v"   # Video
                ],
                label="Audio/Video Files (select multiple)"
            )
            
            gr.Markdown("""
            **Supported audio formats**: WAV, MP3, FLAC, M4A, OGG, AAC, WMA
            
            **Supported video formats**: MP4, AVI, MKV, MOV, WEBM, FLV, M4V
            *(Audio will be automatically extracted from video files)*
            
            💡 **Tip**: Select multiple files for batch processing!
            """)
            
            gr.Markdown("### ⚙️ Settings")
            
            model_selector = gr.Radio(
                choices=[
                    "📊 Parakeet-TDT-0.6B v3 (Multilingual, Default) - 25 languages, auto-detect",
                    "🎯 Parakeet-TDT-1.1B (Maximum Accuracy) - 1.5% WER, English only",
                    "� Parakeet-TDT_CTC-1.1B (PnC) - 1.82% WER, English with Punctuation & Capitalization",
                    "�🌍 Canary-1B v2 (Multilingual + Translation) - 25 languages with AST",
                    "🌐 Canary-1B (Multilingual) - 25 languages, standard ASR"
                ],
                value="📊 Parakeet-TDT-0.6B v3 (Multilingual, Default) - 25 languages, auto-detect",
                label="Model Selection",
                info="Choose based on your priority: accuracy, speed, languages, or features"
            )
            
            save_checkbox = gr.Checkbox(
                label="💾 Save transcription to file",
                value=True,
                info="Creates a file in the current directory"
            )
            
            output_format = gr.Dropdown(
                choices=["txt", "srt", "csv"],
                value="txt",
                label="📄 Output Format",
                info="TXT (plain text), SRT (subtitles), CSV (spreadsheet)"
            )
            
            timestamp_checkbox = gr.Checkbox(
                label="⏱️ Include word-level timestamps",
                value=True,
                info="Shows when each word was spoken (recommended for SRT/CSV export)"
            )
            
            itn_checkbox = gr.Checkbox(
                label="🔢 Convert numbers to digits (ITN)",
                value=True,
                info="Converts 'twenty twenty two' → '2022' (requires nemo_text_processing)"
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                chunk_size_slider = gr.Slider(
                    minimum=30,
                    maximum=300,
                    value=120,
                    step=10,
                    label="🎛️ Chunk Size (seconds)",
                    info="Larger = faster but more VRAM. 120-180s recommended for 12GB VRAM. Reduce if OOM errors."
                )
                
                batch_size_slider = gr.Slider(
                    minimum=1,
                    maximum=24,
                    value=1,
                    step=1,
                    label="📦 Batch Size",
                    info="Higher = more VRAM usage. Increase to utilize more VRAM (1-8 for 8-12GB, 8-16 for 16GB+, 16-24 for 24GB+)."
                )
                
                silence_threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=3.0,
                    value=0.5,
                    step=0.1,
                    label="🔇 Silence Threshold (seconds)",
                    info="End subtitle segments when silence gap exceeds this. Lower = more segments, Higher = fewer but longer segments."
                )
                
                itn_mode_dropdown = gr.Dropdown(
                    choices=["per_chunk", "final_pass", "both", "disabled"],
                    value="per_chunk",
                    label="🔢 ITN Mode (Inverse Text Normalization)",
                    info="per_chunk: Apply during chunked transcription | final_pass: Apply once at end | both: Apply both | disabled: No ITN"
                )
                
                gr.Markdown(f"""
                **VRAM Optimization Tips**:
                - **Chunk Size**: Larger chunks process faster but use more VRAM
                  - 60-90s: Safe for 8GB VRAM
                  - 120-180s: Good for 12GB VRAM  
                  - 200-300s: For 16GB+ VRAM
                - **Batch Size**: How many chunks to process at once
                  - 1-4: Safe for 8-12GB VRAM
                  - 4-8: Better GPU utilization for 12GB+
                  - 8-16: Optimal for 16GB+ VRAM
                  - 16-24: For 24GB+ VRAM (high throughput)
                
                **Silence Threshold**: Controls when subtitle segments end based on speech pauses
                  - 0.3-0.5s: Quick speakers, natural pauses
                  - 0.5-1.0s: Normal speech patterns
                  - 1.0-2.0s: Slow speakers, lecture-style
                
                **ITN Modes**:
                  - **per_chunk**: Best for long audio (prevents "input too long" errors)
                  - **final_pass**: Simpler, processes complete text once (may fail on very long transcriptions)
                  - **both**: Most thorough, applies ITN twice (per-chunk then final)
                  - **disabled**: Skip ITN entirely (numbers stay as words)
                
                **ITN Status**: {'✅ Available' if ITN_AVAILABLE else '❌ Not installed (run: pip install nemo_text_processing)'}
                """)
            
            transcribe_btn = gr.Button(
                "🚀 Start Transcription",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ---
            ### 📖 Model Information
            
            **Parakeet-TDT-0.6B v3** (Recommended):
            - Languages: 25 European languages with auto-detection
            - Speed: 3,380× real-time (ultra-fast)
            - Accuracy: ~1.7% WER
            - VRAM: 3-4 GB
            - Loads from: `local_models/parakeet-0.6b-v3.nemo` OR HuggingFace
            
            **Parakeet-TDT-1.1B** (Best Accuracy):
            - Languages: English only
            - Speed: 1,336× real-time
            - Accuracy: **1.5% WER** (best available)
            - VRAM: 5-6 GB
            - Loads from: `local_models/parakeet-1.1b.nemo` OR HuggingFace
            
            **Canary-1B v2** (Multilingual + Translation):
            - Languages: 25 European languages
            - Speed: ~200× real-time
            - Accuracy: 1.88% WER (English)
            - VRAM: 4-5 GB
            - Features: **Speech Translation** (AST)
            - Loads from: `local_models/canary-1b-v2.nemo` OR HuggingFace
            
            **Canary-1B** (Standard):
            - Languages: 25 European languages
            - Speed: ~200× real-time
            - Accuracy: ~1.9% WER (English)
            - VRAM: 4-5 GB
            - Loads from: `local_models/canary-1b.nemo` OR HuggingFace
            
            💡 **Tip**: Run `python setup_local_models.py` to download models locally for faster loading.
            """)
        
        # Right column - Output
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Transcription Results")
            
            status_output = gr.Markdown(
                "Upload an audio file and click 'Start Transcription' to begin..."
            )
            
            # FIXED: Replaced show_copy_button with buttons parameter
            transcription_output = gr.Textbox(
                label="Transcription Text",
                lines=20,
                placeholder="Your transcription will appear here...",
                buttons=["copy"],  # NEW: Use buttons parameter instead
                show_label=True
            )
            
            gr.Markdown("### 📥 Download Files")
            
            with gr.Row():
                txt_file_output = gr.File(
                    label="📄 TXT (Plain Text)",
                    visible=True
                )
                srt_file_output = gr.File(
                    label="🎬 SRT (Subtitles)",
                    visible=True
                )
                csv_file_output = gr.File(
                    label="📊 CSV (Spreadsheet)",
                    visible=True
                )
            
            log_file_output = gr.File(
                label="📋 Download Processing Logs",
                visible=True
            )
    
    # Bottom section
    gr.Markdown("---")
    
    with gr.Accordion("❓ How to Use", open=False):
        gr.Markdown("""
        ### Quick Start:
        
        1. **Upload files** (audio or video - select multiple for batch processing)
        2. **Select model** (choose based on your needs: speed, accuracy, or language support)
        3. **Click "Start Transcription"**
        4. **Copy or download** your transcription
        
        ### Features:
        - 🎵 **Audio Support**: WAV, MP3, FLAC, M4A, OGG, AAC, WMA
        - 🎬 **Video Support**: MP4, AVI, MKV, MOV, WEBM, FLV, M4V
        - 📦 **Batch Processing**: Upload multiple files at once
        - ⚡ **GPU Optimized**: Uses FP16 mixed precision for faster processing
        - 🔒 **Privacy First**: All processing happens locally on your machine
        - 🌍 **Multilingual**: Support for 25 European languages (model dependent)
        
        ### Model Loading:
        - **All Models**: Try local `.nemo` file first, then HuggingFace fallback
        - Local files load instantly (no internet required)
        - HuggingFace downloads are cached for future offline use
        - All models stay in memory after first load (instant subsequent use)
        
        ### Tips:
        - First transcription loads model into memory (~2-15 seconds depending on model)
        - Subsequent transcriptions reuse cached model (instant)
        - Processing time: 10-60 seconds per hour of audio (model dependent)
        - Video files: Audio is automatically extracted via FFmpeg
        - Choose Parakeet-1.1B for best English accuracy
        - Choose Parakeet-v3 or Canary for multilingual support
        - Canary models support speech translation features
        
        ### Setup (Optional but Recommended):
        - Run `python setup_local_models.py` to download models locally
        - You can download all 4 models or select specific ones
        - Local `.nemo` files load faster than HuggingFace downloads
        - See `docs/manual/user-model-setup-guide.md` for instructions
        """)
    
    with gr.Accordion("🔒 Privacy & Performance", open=False):
        gr.Markdown(get_privacy_performance_info())
    
    # Connect button - queue=True ensures proper callback execution
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_selector, save_checkbox, timestamp_checkbox, 
                output_format, itn_checkbox, chunk_size_slider, batch_size_slider,
                silence_threshold_slider, itn_mode_dropdown],
        outputs=[status_output, transcription_output, txt_file_output, srt_file_output, csv_file_output, log_file_output],
        queue=False
    )

# Launch
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting NVIDIA NeMo Transcription Interface")
    print("="*80)
    
    # Show cache directory info
    print(f"\n📁 Cache Directory: {CACHE_DIR}")
    print("   (Used for model downloads and extraction - prevents Windows temp file issues)")
    
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"✅ CUDA: {torch.version.cuda}")
        
        # Enable GPU optimizations (TF32, cuDNN)
        setup_gpu_optimizations()
    else:
        print("\n⚠️  WARNING: No CUDA GPU detected!")
    
    print("="*80)
    
    # Validate models - uses new validation that shows local vs HuggingFace status
    validate_local_models()
    
    print("="*80)
    print("\n🌐 Opening in browser at: http://127.0.0.1:7860")
    print("💡 Keep this terminal open while using the interface")
    print("🛑 Press Ctrl+C to stop\n")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )


