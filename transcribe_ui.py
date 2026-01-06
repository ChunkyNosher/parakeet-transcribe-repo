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
        
    def start(self):
        """Start capturing logs."""
        import sys
        self.log_buffer = io.StringIO()  # Reset buffer
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create a TeeWriter that writes to both original stream and buffer
        class TeeWriter:
            def __init__(self, original, buffer):
                self.original = original
                self.buffer = buffer
            def write(self, text):
                self.original.write(text)
                self.buffer.write(text)
            def flush(self):
                self.original.flush()
                
        sys.stdout = TeeWriter(self.original_stdout, self.log_buffer)
        sys.stderr = TeeWriter(self.original_stderr, self.log_buffer)
        
        # Also capture logging module output
        self.handler = logging.StreamHandler(self.log_buffer)
        self.handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self.handler)
        
    def stop(self):
        """Stop capturing logs and return captured content."""
        import sys
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
        if self.handler:
            logging.getLogger().removeHandler(self.handler)
        return self.log_buffer.getvalue()
    
    def get_logs(self):
        """Get current captured logs without stopping."""
        return self.log_buffer.getvalue()

# Global log capture instance
log_capture = LogCapture()

# Global model cache to avoid reloading
models_cache = {}

# ============================================================================
# ITN (Inverse Text Normalization) Support
# ============================================================================
# Converts spoken numbers to digits: "twenty twenty two" → "2022"
# Uses nemo_text_processing if available, with graceful fallback
# ============================================================================

# Try to import ITN - will be None if not installed
ITN_NORMALIZER = None
ITN_AVAILABLE = False

try:
    from nemo_text_processing.inverse_text_normalization import InverseNormalizer
    # Initialize ITN for English (lazy - will init on first use)
    ITN_AVAILABLE = True
    print("✅ ITN (Inverse Text Normalization) available - numbers will be converted to digits")
except ImportError:
    print("ℹ️  ITN not installed - numbers will remain as words")
    print("   To enable: pip install nemo_text_processing")


def _get_itn_normalizer(language="en"):
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
            ITN_NORMALIZER = InverseNormalizer(lang=language, cache_dir=str(CACHE_DIR / "itn"))
            print(f"   ✅ ITN normalizer ready")
        except Exception as e:
            print(f"   ⚠️ Failed to initialize ITN: {e}")
            return None
            
    return ITN_NORMALIZER


def apply_inverse_text_normalization(text, language="en"):
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
    import re
    
    normalizer = _get_itn_normalizer(language)
    
    if normalizer is None:
        return text
    
    if not text or not text.strip():
        return text
    
    def _split_into_chunks(text, max_words=50):
        """Fallback: Split text into chunks of max_words for ITN processing."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]
    
    def _normalize_chunks(chunks):
        """Normalize a list of text chunks, handling errors gracefully."""
        try:
            return normalizer.normalize_list(chunks, verbose=False)
        except Exception:
            # Individual normalization fallback
            results = []
            for chunk in chunks:
                try:
                    results.append(normalizer.normalize(chunk, verbose=False))
                except Exception:
                    results.append(chunk)  # Keep original if all fails
            return results
        
    try:
        # Strategy 1: Try built-in sentence splitting
        sentences = normalizer.split_text_into_sentences(text)
        
        if sentences and len(sentences) > 0:
            # Success - normalize each sentence
            normalized = _normalize_chunks(sentences)
            result = ' '.join(normalized)
            print(f"   ✅ ITN applied (sentence splitting: {len(sentences)} sentences)")
            return result
            
    except Exception as e:
        print(f"   ℹ️ ITN sentence splitting failed: {e}")
    
    try:
        # Strategy 2: Regex-based sentence splitting on .!?
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences and len(sentences) > 1:
            normalized = _normalize_chunks(sentences)
            result = ' '.join(normalized)
            print(f"   ✅ ITN applied (regex splitting: {len(sentences)} sentences)")
            return result
            
    except Exception as e:
        print(f"   ℹ️ ITN regex splitting failed: {e}")
    
    try:
        # Strategy 3: Split into fixed-size word chunks
        chunks = _split_into_chunks(text, max_words=50)
        
        if len(chunks) > 1:
            normalized = _normalize_chunks(chunks)
            result = ' '.join(normalized)
            print(f"   ✅ ITN applied (chunk splitting: {len(chunks)} chunks)")
            return result
        else:
            # Single chunk - normalize directly
            result = normalizer.normalize(text, verbose=False)
            print(f"   ✅ ITN applied (single text)")
            return result
            
    except Exception as e:
        print(f"   ⚠️ ITN normalization failed completely: {e}")
        return text


# ============================================================================
# SRT/CSV/TXT Output Format Conversion Utilities
# ============================================================================
# Converts transcription with timestamps into various output formats:
# - SRT: SubRip subtitle format (for video subtitles)
# - CSV: Comma-separated values (for spreadsheets/analysis)
# - TXT: Plain text with timestamps (human-readable)
# ============================================================================

def _format_srt_timestamp(seconds):
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


def _group_words_into_segments(timestamps, words_per_segment=8, max_duration=5.0):
    """Group word-level timestamps into subtitle segments.
    
    SRT subtitles work best with phrase-length segments rather than
    individual words. This groups words into readable chunks.
    
    Args:
        timestamps: List of word timestamp dicts with 'start', 'end', 'word' keys
        words_per_segment: Target words per subtitle segment (default: 8)
        max_duration: Maximum segment duration in seconds (default: 5.0)
        
    Returns:
        List of segment dicts with 'start', 'end', 'text' keys
    """
    if not timestamps:
        return []
        
    segments = []
    current_segment = {
        'start': timestamps[0].get('start', 0.0),
        'end': timestamps[0].get('end', 0.0),
        'words': []
    }
    
    for stamp in timestamps:
        word = stamp.get('text', stamp.get('word', stamp.get('segment', stamp.get('char', ''))))
        start = stamp.get('start', 0.0)
        end = stamp.get('end', 0.0)
        
        # Check if we should start a new segment
        segment_duration = end - current_segment['start']
        word_count = len(current_segment['words'])
        
        if word_count >= words_per_segment or segment_duration > max_duration:
            # Finalize current segment
            if current_segment['words']:
                current_segment['text'] = ' '.join(current_segment['words'])
                segments.append({
                    'start': current_segment['start'],
                    'end': current_segment['end'],
                    'text': current_segment['text']
                })
            # Start new segment
            current_segment = {
                'start': start,
                'end': end,
                'words': [word]
            }
        else:
            # Add word to current segment
            current_segment['words'].append(word)
            current_segment['end'] = end
    
    # Don't forget the last segment
    if current_segment['words']:
        current_segment['text'] = ' '.join(current_segment['words'])
        segments.append({
            'start': current_segment['start'],
            'end': current_segment['end'],
            'text': current_segment['text']
        })
    
    return segments


def format_as_srt(transcription, timestamps, timestamp_level='word'):
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
        segments = []
        for stamp in timestamps:
            text = stamp.get('text', stamp.get('segment', stamp.get('word', stamp.get('char', ''))))
            segments.append({
                'start': stamp.get('start', 0.0),
                'end': stamp.get('end', 0.0),
                'text': text
            })
    
    # Build SRT content
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start_ts = _format_srt_timestamp(seg['start'])
        end_ts = _format_srt_timestamp(seg['end'])
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(seg['text'])
        srt_lines.append("")  # Blank line between entries
    
    return '\n'.join(srt_lines)


def format_as_csv(transcription, timestamps, timestamp_level='word'):
    """Format transcription as CSV content.
    
    CSV columns: start_time, end_time, duration, text
    
    Args:
        transcription: Full transcription text (used if no timestamps)
        timestamps: List of timestamp dicts from extract_timestamps()
        timestamp_level: 'word', 'segment', or 'none'
        
    Returns:
        CSV formatted string with header row
    """
    csv_lines = ["start_time,end_time,duration,text"]
    
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


def format_as_txt_with_timestamps(transcription, timestamps, timestamp_level='word'):
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
        segments = []
        for stamp in timestamps:
            text = stamp.get('text', stamp.get('segment', stamp.get('word', stamp.get('char', ''))))
            segments.append({
                'start': stamp.get('start', 0.0),
                'text': text
            })
    
    # Build text with timestamps
    txt_lines = []
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


def format_error_message(error_type, detail=""):
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

def validate_and_normalize_audio(file_path):
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
    
    warning_msg = ""
    
    try:
        # Load audio preserving original sample rate
        y, sr = librosa.load(file_path, sr=None)
        
        # Validate duration (100ms to 24 hours)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration < 0.1:  # Less than 100ms
            return (False, None, 0, 
                    format_error_message('duration_invalid', 
                        f"Duration: {duration:.3f}s (minimum: 0.1s)"), "")
        
        if duration > 86400:  # More than 24 hours
            return (False, None, 0,
                    format_error_message('duration_invalid',
                        f"Duration: {duration:.1f}s ({duration/3600:.1f} hours, maximum: 24 hours)"), "")
        
        # Check for silent audio (RMS energy check)
        rms = librosa.feature.rms(y=y)
        if rms.max() < 0.001:  # Very quiet - likely silent
            return (False, None, 0,
                    format_error_message('audio_silent',
                        f"Maximum RMS energy: {rms.max():.6f} (threshold: 0.001)"), "")
        
        if rms.max() < 0.01:  # Quiet but not silent - add warning
            warning_msg = "⚠️ Audio is very quiet - transcription quality may be affected"
        
        # Resample to 16kHz if needed (NeMo models expect 16kHz)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Convert to mono if stereo
        if y.ndim > 1:
            y = librosa.to_mono(y)
        
        return (True, y, sr, "", warning_msg)
        
    except Exception as e:
        error_str = str(e)
        
        # Provide specific error messages based on error type
        if "Audio file" in error_str or "NoBackendError" in error_str:
            return (False, None, 0,
                    format_error_message('audio_load_failed', error_str), "")
        elif "Format" in error_str or "codec" in error_str.lower():
            return (False, None, 0,
                    format_error_message('format_unsupported', error_str), "")
        else:
            return (False, None, 0,
                    format_error_message('audio_load_failed', error_str), "")


# ============================================================================
# Transcription Result Validation (Pattern 2: HuggingFace Modernization)
# ============================================================================
# Validates transcription result structure BEFORE accessing .text
# - Checks result is not None
# - Checks result is a list and not empty
# - Validates hypothesis has .text attribute
# - Checks .text is a non-empty string
# ============================================================================

def validate_transcription_result(result, idx=0):
    """Validate transcription result before accessing text.
    
    Performs comprehensive validation of the model.transcribe() output
    to prevent AttributeError and other crashes from malformed results.
    
    Args:
        result: Output from model.transcribe()
        idx: Index of the hypothesis to validate (default: 0)
        
    Returns:
        Tuple of (success: bool, text: str, error_msg: str)
    """
    # Check result is not None
    if result is None:
        return (False, "", "Result is None - model may have failed silently")
    
    # Check result is a list
    if not isinstance(result, list):
        return (False, "", f"Result is {type(result).__name__}, expected list")
    
    # Check result is not empty
    if len(result) == 0:
        return (False, "", "Result is an empty list - no transcription generated")
    
    # Check index is valid
    if idx >= len(result):
        return (False, "", f"Index {idx} out of range (result has {len(result)} items)")
    
    hypothesis = result[idx]
    
    # Check hypothesis has text attribute
    if not hasattr(hypothesis, 'text'):
        # Try string fallback (some models return plain strings)
        if isinstance(hypothesis, str):
            if len(hypothesis) == 0:
                return (False, "", "Transcription is empty")
            return (True, hypothesis, "")
        return (False, "", f"Hypothesis has no .text attribute (type: {type(hypothesis).__name__})")
    
    # Check text is a string
    if not isinstance(hypothesis.text, str):
        return (False, "", f".text is {type(hypothesis.text).__name__}, expected string")
    
    # Check text is not empty
    if len(hypothesis.text) == 0:
        return (False, "", "Transcription is empty (0 characters)")
    
    return (True, hypothesis.text, "")


# ============================================================================
# Defensive Timestamp Extraction (Pattern 4: HuggingFace Modernization)
# ============================================================================
# Extracts timestamps with graceful fallback:
# 1. Try word-level timestamps
# 2. Fallback to segment-level timestamps
# 3. Return empty list if both unavailable
# ============================================================================

def _try_get_timestamp_level(hypothesis, level_key):
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
        timestamps = ts_dict.get(level_key)
        if isinstance(timestamps, list) and len(timestamps) > 0:
            return timestamps
    except (AttributeError, KeyError, TypeError):
        pass
    return None


def extract_timestamps(hypothesis, include_timestamps=False):
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


def format_timestamp_status(level, include_timestamps):
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

def _load_audio_to_numpy(file_path, target_sr=16000):
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
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
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

# Runtime configuration (can be changed via Gradio UI)
# These are module-level so they can be modified by the UI
chunk_duration_sec = DEFAULT_CHUNK_DURATION_SEC
long_audio_threshold_sec = DEFAULT_LONG_AUDIO_THRESHOLD_SEC


def _clear_vram():
    """Clear CUDA VRAM and run garbage collection."""
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _transcribe_single_buffer(model, buffer, use_cuda):
    """Transcribe a single audio buffer.
    
    Args:
        model: Loaded NeMo ASR model
        buffer: numpy array of audio samples
        use_cuda: Whether to use CUDA with mixed precision
        
    Returns:
        Result from model.transcribe()
    """
    transcribe_kwargs = {
        'audio': [buffer],
        'batch_size': 1,  # Single chunk at a time for memory safety
        'return_hypotheses': True,
        'verbose': False
    }
    
    if use_cuda and torch.cuda.is_available():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return model.transcribe(**transcribe_kwargs)
    return model.transcribe(**transcribe_kwargs)


def _extract_hypothesis_text(hypothesis):
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


def _adjust_chunk_timestamps(chunk_word_ts, ts_level, left_context_duration, chunk_start_time):
    """Adjust raw timestamps from a chunk to absolute positions.
    
    Args:
        chunk_word_ts: List of timestamp dicts from extract_timestamps()
        ts_level: Timestamp level ('word', 'segment', 'char')
        left_context_duration: Duration of left context that was prepended
        chunk_start_time: Start time of this chunk in the full audio
        
    Returns:
        List of adjusted timestamp dicts
    """
    adjusted_timestamps = []
    
    for ts in chunk_word_ts:
        adjusted_ts = {
            'start': max(0.0, ts.get('start', 0.0) - left_context_duration + chunk_start_time),
            'end': ts.get('end', 0.0) - left_context_duration + chunk_start_time,
        }
        # Ensure end >= start
        adjusted_ts['end'] = max(adjusted_ts['start'], adjusted_ts['end'])
        
        # Copy text field with correct key
        for key in ('word', 'segment', 'char', 'text'):
            if key in ts:
                adjusted_ts[key] = ts[key]
        
        adjusted_timestamps.append(adjusted_ts)
    
    return adjusted_timestamps


def _transcribe_long_audio_chunked(model, audio_array, sample_rate=16000, use_cuda=True,
                                    chunk_size_override=None):
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
    
    transcriptions = []
    chunk_timestamps = []
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
        
        print(f"   📍 Chunk {chunk_num}: {chunk_start_time:.1f}s - {chunk_end_time:.1f}s ({len(buffer)/sample_rate:.1f}s with context)")
        
        _clear_vram()
        
        try:
            result = _transcribe_single_buffer(model, buffer, use_cuda)
            
            if not result or len(result) == 0:
                position += step_samples
                continue
            
            hypothesis = result[0]
            chunk_text = _extract_hypothesis_text(hypothesis).strip()
            
            if not chunk_text:
                position += step_samples
                continue
            
            transcriptions.append(chunk_text)
            
            # Extract and adjust timestamps
            left_context_duration = (position - start) / sample_rate if position > start else 0
            chunk_word_ts, ts_level = extract_timestamps(hypothesis, include_timestamps=True)
            
            if chunk_word_ts and ts_level in ('word', 'segment', 'char'):
                adjusted = _adjust_chunk_timestamps(chunk_word_ts, ts_level, left_context_duration, chunk_start_time)
                chunk_timestamps.extend(adjusted)
            else:
                # Fallback: chunk-level timestamp
                chunk_timestamps.append({
                    'start': chunk_start_time,
                    'end': chunk_end_time,
                    'text': chunk_text
                })
                
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


def _load_audio_files_to_memory(files):
    """Load audio files into numpy arrays.
    
    Args:
        files: List of audio file paths
        
    Returns:
        List of (audio_array, duration_sec) tuples
        
    Raises:
        Exception: If any file fails to load
    """
    print(f"   📂 Loading {len(files)} audio file(s) into memory...")
    audio_data = []
    
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


def _transcribe_chunked_files(model, audio_data, use_cuda, chunk_size_override, threshold):
    """Process files using chunked transcription for long audio.
    
    Args:
        model: Loaded NeMo ASR model
        audio_data: List of (audio_array, duration_sec) tuples
        use_cuda: Whether to use CUDA
        chunk_size_override: Override chunk duration
        threshold: Duration threshold for chunking
        
    Returns:
        Tuple of (results, chunk_timestamps_map)
    """
    print(f"   ⚡ Long audio detected (>{threshold}s) - using chunked transcription")
    
    results = []
    chunk_timestamps_map = {}
    
    for i, (audio_np, duration) in enumerate(audio_data):
        if duration > threshold:
            # Use chunked transcription for long audio
            text, chunk_ts = _transcribe_long_audio_chunked(
                model, audio_np, sample_rate=16000, use_cuda=use_cuda,
                chunk_size_override=chunk_size_override
            )
            chunk_timestamps_map[i] = chunk_ts
            
            # Wrap in Hypothesis-like object for API compatibility
            class _Hypothesis:
                def __init__(self, t, ts):
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


def _transcribe_short_audio_batch(model, audio_arrays, batch_size, use_cuda, max_retries, base_delay):
    """Transcribe batch of short audio files with retry logic.
    
    Args:
        model: Loaded NeMo ASR model
        audio_arrays: List of numpy audio arrays
        batch_size: Batch size for transcription
        use_cuda: Whether to use CUDA
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        
    Returns:
        Transcription result from model.transcribe()
        
    Raises:
        Exception: If transcription fails after all retries
    """
    import gc
    
    print(f"   ✅ Audio loaded into memory, starting transcription...")
    last_error = None
    
    for attempt in range(max_retries):
        try:
            transcribe_kwargs = {
                'audio': audio_arrays,
                'batch_size': batch_size,
                'return_hypotheses': True,
                'verbose': True
            }
            
            if use_cuda and torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    return model.transcribe(**transcribe_kwargs)
            return model.transcribe(**transcribe_kwargs)
            
        except PermissionError as e:
            last_error = e
            error_str = str(e)
            is_file_lock = "WinError 32" in error_str or "being used by another process" in error_str
            
            if is_file_lock and attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                print(f"   ⚠️ File lock detected (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
                _clear_vram()
                time.sleep(delay)
                continue
            raise
                
        except Exception as e:
            if attempt < max_retries - 1:
                last_error = e
                delay = base_delay * (attempt + 1)
                print(f"   ⚠️ Transcription error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                _clear_vram()
                time.sleep(delay)
                continue
            raise
    
    if last_error:
        raise last_error
    raise RuntimeError(f"Transcription failed after {max_retries} attempts")


def _transcribe_with_retry(model, files, batch_size, use_cuda=True, max_retries=3,
                           chunk_size_override=None):
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
        
    Returns:
        Tuple of (results, chunk_timestamps_map)
    """
    audio_data = _load_audio_files_to_memory(files)
    
    # Check if any audio needs chunking
    effective_threshold = long_audio_threshold_sec
    needs_chunking = any(duration > effective_threshold for _, duration in audio_data)
    
    if needs_chunking:
        return _transcribe_chunked_files(
            model, audio_data, use_cuda, chunk_size_override, effective_threshold
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
MODEL_CONFIGS = {
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

def _override_model_dataloader_config(model):
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
        OmegaConf.set_struct(model.cfg, False)
        
        # Override num_workers in all dataloader configs
        modified = []
        for ds_name in ['train_ds', 'validation_ds', 'test_ds']:
            if hasattr(model.cfg, ds_name):
                ds_cfg = getattr(model.cfg, ds_name)
                if hasattr(ds_cfg, 'num_workers'):
                    old_value = ds_cfg.num_workers
                    ds_cfg.num_workers = 0
                    modified.append(f"{ds_name}: {old_value} -> 0")
        
        # Re-enable struct mode
        OmegaConf.set_struct(model.cfg, True)
        
        if modified:
            print(f"   ✅ Overrode dataloader num_workers: {', '.join(modified)}")
        else:
            print(f"   ℹ️  No dataloader num_workers found in model.cfg")
            
    except Exception as e:
        # Non-fatal - tensor-based transcription is the primary fix
        print(f"   ⚠️  Could not override model config (non-fatal): {e}")


def get_model_key_from_choice(choice_text):
    """Extract model key from radio button choice text
    
    Args:
        choice_text: The display text from the radio button selection
    
    Returns:
        Model key string for accessing MODEL_CONFIGS
    """
    choice_map = {
        "Parakeet-TDT-0.6B v3": "parakeet-v3",
        "Parakeet-TDT-1.1B": "parakeet-1.1b",
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

def get_script_dir():
    """Get the directory where the script is located"""
    return Path(__file__).parent.absolute()

def validate_local_models():
    """
    Display local model setup status and provide informational messages.
    
    With "local_or_huggingface" loading method, local .nemo files are OPTIONAL.
    Models will automatically download from HuggingFace if local files are missing.
    
    This function is informational only and does not block startup.
    """
    script_dir = get_script_dir()
    local_models_dir = script_dir / "local_models"
    
    print("\n" + "="*80)
    print("📦 Model Availability Check")
    print("="*80)
    
    # Check if local_models directory exists
    if not local_models_dir.exists():
        print(f"\n📁 Local models directory not found: {local_models_dir}")
        print("   This is OK - models will download from HuggingFace on first use.")
        print("   To download models locally, run: python setup_local_models.py")
    else:
        print(f"\n📁 Local models directory: {local_models_dir}")
    
    # Check each model's local availability
    local_found = []
    hf_fallback = []
    
    for model_key, config in MODEL_CONFIGS.items():
        local_path = config.get("local_path")
        if local_path:
            full_path = script_dir / local_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                local_found.append(f"   ✅ {config['display_name']}: {full_path.name} ({size_mb:.1f} MB)")
            else:
                hf_fallback.append(f"   📥 {config['display_name']}: Will download from HuggingFace ({config['hf_model_id']})")
        else:
            hf_fallback.append(f"   📥 {config['display_name']}: HuggingFace only ({config['hf_model_id']})")
    
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

def copy_gradio_file_to_cache(file_path, max_retries=6):
    """Copy Gradio uploaded file to cache directory to prevent manifest.json file locking.
    
    Gradio uploads files to its own temp directory. When NeMo reads these files,
    it may create manifest.json in that location or in system temp, which can
    cause WinError 32 (file in use) issues on Windows.
    
    This function copies uploaded files from Gradio's temp to our controlled
    cache directory BEFORE passing to NeMo, ensuring all NeMo operations
    (including internal manifest creation) happen in our cache location.
    
    Args:
        file_path: Path to Gradio uploaded file
        max_retries: Maximum retry attempts for file copy (default: 6 for Windows antivirus)
        
    Returns:
        Path to file in cache directory
        
    Raises:
        OSError: If file copy fails after all retries
    """
    file_path = Path(file_path)
    
    # Generate unique filename using hash of original path + filename
    # This prevents collisions from multiple uploads of same filename
    # SHA-256 is used for secure, collision-resistant filename generation
    path_hash = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
    cached_filename = f"{path_hash}_{file_path.name}"
    cached_path = _gradio_cache_dir / cached_filename
    
    # If file already cached, return immediately
    if cached_path.exists():
        return str(cached_path)
    
    # Copy with retry logic for Windows file locks
    # Use linear backoff to accommodate antivirus scanning (500ms-4000ms)
    base_delay = 0.5  # 500ms base delay
    
    for attempt in range(max_retries):
        try:
            shutil.copy2(file_path, cached_path)
            
            # Validate file was actually written and is readable
            if cached_path.exists() and cached_path.stat().st_size > 0:
                return str(cached_path)
            else:
                # File exists but is empty or invalid - treat as failure
                if cached_path.exists():
                    cached_path.unlink()  # Remove invalid file
                raise OSError(f"File copy succeeded but file is empty or invalid: {cached_path}")
            
        except (OSError, PermissionError) as e:
            error_str = str(e)
            is_file_lock = "WinError 32" in error_str or "being used by another process" in error_str
            
            if is_file_lock and attempt < max_retries - 1:
                # Linear backoff: 0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s = 10.5s total
                delay = base_delay * (attempt + 1)
                print(f"   ⚠️  File copy lock detected (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
                time.sleep(delay)
                continue
            
            # Final retry failed or non-file-lock error
            raise OSError(
                f"Failed to copy file to cache after {attempt + 1} attempts.\n"
                f"Source: {file_path}\n"
                f"Destination: {cached_path}\n"
                f"Error: {error_str}"
            )

def get_dynamic_batch_size(duration, model_key):
    """Calculate optimal batch size based on audio duration and model
    
    DEPRECATED: This function is no longer used. NeMo's transcribe() method
    handles batch sizing automatically based on available VRAM. The batch_size
    parameter in transcribe() controls how many FILES are batched together,
    not duration-based splitting.
    
    Kept for reference only - will be removed in future version.
    """
    # Return NeMo's default batch_size for file batching
    return 4

def setup_gpu_optimizations():
    """Enable GPU optimizations for better performance"""
    if torch.cuda.is_available():
        # Enable TF32 for matrix multiplication (Ampere+ GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cuDNN benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True
        print("✅ GPU optimizations enabled (TF32, cuDNN benchmark)")

def _format_model_error(title, model_path, display_name, problem_msg, solution_msg, original_error=None):
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


def _format_network_error(display_name, error):
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


def _format_disk_space_error(display_name, error):
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


def _format_filesystem_error(display_name, error):
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


def _handle_huggingface_os_error(error, display_name):
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


def _load_from_huggingface_with_retry(hf_model_id, config, max_retries=3):
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
    base_delay = 0.5  # 500ms base delay - Windows services need time to release handles
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return nemo_asr.models.ASRModel.from_pretrained(hf_model_id)
            
        except PermissionError as e:
            last_error = e
            error_str = str(e)
            is_file_lock = "WinError 32" in error_str or "being used by another process" in error_str
            
            if is_file_lock and attempt < max_retries - 1:
                # File lock detected - retry with linear backoff
                delay = base_delay * (attempt + 1)  # 0.5s, 1.0s, 1.5s
                print(f"⏳ File lock detected (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
                
                # Force garbage collection and cache cleanup before retry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                time.sleep(delay)
                continue
            
            elif is_file_lock:
                # Final retry failed - provide detailed diagnostics
                raise PermissionError(
                    f"\n{'='*80}\n"
                    f"❌ FILE LOCK ERROR (PERSISTED AFTER {max_retries} RETRIES)\n"
                    f"{'='*80}\n\n"
                    f"Model: {config['display_name']}\n"
                    f"HuggingFace ID: {hf_model_id}\n\n"
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
                    f"  5. Disable Windows Search indexing\n\n"
                    f"⚙️ Cache Location:\n"
                    f"  {CACHE_DIR}\n\n"
                    f"If this persists, try manually deleting the cache directory.\n"
                    f"{'='*80}"
                )
            else:
                # Different PermissionError (not file lock) - re-raise immediately
                raise PermissionError(
                    f"\n{'='*80}\n"
                    f"❌ PERMISSION ERROR!\n"
                    f"{'='*80}\n\n"
                    f"Model: {config['display_name']}\n"
                    f"Error: {error_str}\n\n"
                    f"The process does not have permission to access the cache directory.\n"
                    f"Try running as Administrator or checking directory permissions.\n"
                    f"Cache location: {CACHE_DIR}\n"
                    f"{'='*80}"
                )
    
    # This is reached if all attempts fail without raising an exception
    # (should not happen with current logic, but provides safety)
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to load model after {max_retries} attempts")


def _load_with_retry(restore_path, config, max_retries=3):
    """Load model from .nemo file with retry logic for Windows file lock issues.
    
    Enhanced retry logic wrapper for ASRModel.restore_from() that handles
    Windows file locking errors during model extraction.
    
    Since we've set tempfile.tempdir to use our custom cache directory (Fix #2),
    NeMo's internal extraction will use that location instead of system %TEMP%.
    This function adds retry logic to handle any remaining transient file locks.
    
    How it works:
    1. Attempts to load model using ASRModel.restore_from()
    2. If WinError 32 occurs, cleanup and retry with linear backoff
    3. Force garbage collection between retries to release file handles
    4. Provides detailed error messages after all retries exhausted
    
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
    base_delay = 0.5  # 500ms base delay - Windows services need time to release handles
    
    for attempt in range(max_retries):
        try:
            # Load model - NeMo will extract to tempfile.tempdir location
            model = nemo_asr.models.ASRModel.restore_from(
                restore_path=str(restore_path)
            )
            return model
            
        except PermissionError as e:
            error_str = str(e)
            is_file_lock = "WinError 32" in error_str or "being used by another process" in error_str
            
            if is_file_lock and attempt < max_retries - 1:
                # File lock detected - retry with linear backoff
                delay = base_delay * (attempt + 1)  # 0.5s, 1.0s, 1.5s
                print(f"   ⚠️  File lock detected (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
                
                # Force garbage collection and cache cleanup before retry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                time.sleep(delay)
                continue
            
            elif is_file_lock:
                # Final retry failed - provide detailed diagnostics
                raise PermissionError(
                    f"\n{'='*80}\n"
                    f"❌ FILE LOCK ERROR (PERSISTED AFTER {max_retries} RETRIES)\n"
                    f"{'='*80}\n\n"
                    f"Model: {config['display_name']}\n"
                    f"Source: {restore_path}\n\n"
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
                    f"     {CACHE_DIR}\n"
                    f"     (especially the tmp subdirectory)\n\n"
                    f"⚙️ Cache Location:\n"
                    f"  {CACHE_DIR}\n\n"
                    f"If this persists, try manually deleting the cache and temp directories.\n"
                    f"{'='*80}"
                )
            else:
                # Different PermissionError (not file lock) - re-raise immediately
                raise
        
        except Exception as e:
            # Other exceptions - retry for transient issues
            # Note: Broad exception handling is intentional here to handle various
            # transient failures (disk I/O, network, etc.) during model extraction
            if attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                print(f"   ⚠️  Extraction error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   Retrying in {delay:.1f}s...")
                
                # Force garbage collection before retry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                time.sleep(delay)
                continue
            else:
                # Final attempt failed - re-raise the exception
                raise


def load_model(model_name, show_progress=False):
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
    if model_name in models_cache:
        # Validate cached model before returning
        cached_model = models_cache[model_name]
        try:
            # Quick validation: model should have required methods
            if not hasattr(cached_model, 'transcribe'):
                print(f"⚠️  Cached {model_name} appears corrupted (missing transcribe method)")
                # Remove corrupted cache entry
                del models_cache[model_name]
                # Recursively reload
                return load_model(model_name, show_progress)
            # Cached model is valid, return it
            return cached_model
        except Exception as e:
            print(f"⚠️  Cached model validation failed: {e}")
            del models_cache[model_name]
            # Fall through to reload
    
    # ========================================================================
    # Explicit Device Management (Pattern 3: HuggingFace Modernization)
    # ========================================================================
    # Before loading a new model, unload any other models from VRAM to prevent
    # OOM errors. This is critical when switching between models (e.g., 
    # Parakeet → Canary) as multiple large models cannot fit in VRAM together.
    # ========================================================================
    
    # Check if other models are cached and unload them
    models_to_unload = [key for key in models_cache.keys() if key != model_name]
    
    if models_to_unload and torch.cuda.is_available():
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
                torch.cuda.empty_cache()
                gc.collect()
                
                print(f"   ✅ {old_model_key} unloaded successfully")
                
            except Exception as e:
                print(f"   ⚠️  Failed to unload {old_model_key}: {e}")
                # Continue loading the new model anyway
    
    # Model not in cache or cache was invalid, proceed to load
    config = MODEL_CONFIGS[model_name]
    script_dir = get_script_dir()
    loading_method = config.get("loading_method", "huggingface")
    
    start_time = time.time()
    
    # ============================================================
    # LOCAL_OR_HUGGINGFACE: Try local first, fallback to HuggingFace
    # ============================================================
    if loading_method == "local_or_huggingface":
        local_path = config.get("local_path")
        model_path = script_dir / local_path if local_path else None
        
        # Try loading from local .nemo file first
        if model_path and model_path.exists():
            print(f"📦 Loading {config['display_name']} from local file...")
            print(f"   Path: {model_path}")
            
            try:
                # Use retry logic for file lock handling
                models_cache[model_name] = _load_with_retry(
                    restore_path=model_path,
                    config=config,
                    max_retries=3
                )
                
                # Override model config to disable multiprocessing in dataloader
                # This is a FALLBACK - the primary fix is tensor-based transcription
                _override_model_dataloader_config(models_cache[model_name])
                
                load_time = time.time() - start_time
                print(f"✓ {config['display_name']} loaded from local file in {load_time:.1f}s")
                return models_cache[model_name]
                
            except PermissionError as e:
                error_str = str(e)
                if "WinError 32" in error_str or "being used by another process" in error_str:
                    print(f"⚠️  Local file locked, falling back to HuggingFace...")
                    # Fall through to HuggingFace download
                else:
                    raise
                    
            except Exception as e:
                # Local file exists but is corrupted or invalid
                print(f"⚠️  Local file corrupted or invalid: {e}")
                print(f"   Falling back to HuggingFace download...")
                # Fall through to HuggingFace download
        else:
            # Local file not found - inform user
            print(f"📦 Loading {config['display_name']} from HuggingFace...")
            if model_path:
                print(f"   Local .nemo file not found: {model_path}")
            print(f"   To download locally, run: python setup_local_models.py")
        
        # Load from HuggingFace (either no local file, or local file failed)
        hf_model_id = config["hf_model_id"]
        print(f"   Model ID: {hf_model_id}")
        print("   (First load downloads model, subsequent loads use cache)")
        
        try:
            models_cache[model_name] = _load_from_huggingface_with_retry(
                hf_model_id, config, max_retries=3
            )
            
            # Override model config to disable multiprocessing in dataloader
            # This is a FALLBACK - the primary fix is tensor-based transcription
            _override_model_dataloader_config(models_cache[model_name])
            
        except ConnectionError as e:
            raise ConnectionError(_format_network_error(config['display_name'], e))
        except OSError as e:
            _handle_huggingface_os_error(e, config['display_name'])
    
    # ============================================================
    # LOCAL: Strictly load from local .nemo file only
    # ============================================================
    elif loading_method == "local":
        model_path = script_dir / config["local_path"]
        
        # Check if .nemo file exists
        if not model_path.exists():
            raise FileNotFoundError(_format_model_error(
                title="MODEL FILE NOT FOUND!",
                model_path=model_path,
                display_name=config['display_name'],
                problem_msg="The .nemo file must be created once using the setup script.",
                solution_msg="Please run: python setup_local_models.py"
            ))
        
        print(f"📦 Loading {config['display_name']} from local file...")
        
        # Use retry logic for file lock handling
        # _load_with_retry provides comprehensive error messages for file locks
        models_cache[model_name] = _load_with_retry(
            restore_path=model_path,
            config=config,
            max_retries=3
        )
    
    # ============================================================
    # HUGGINGFACE: Load from HuggingFace only
    # ============================================================
    else:
        hf_model_id = config["hf_model_id"]
        
        print(f"📦 Loading {config['display_name']} from HuggingFace...")
        print(f"   Model ID: {hf_model_id}")
        print("   (First load downloads model, subsequent loads use cache)")
        
        try:
            models_cache[model_name] = _load_from_huggingface_with_retry(
                hf_model_id, config, max_retries=3
            )
            
            # Override model config to disable multiprocessing in dataloader
            # This is a FALLBACK - the primary fix is tensor-based transcription
            _override_model_dataloader_config(models_cache[model_name])
            
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
    
    load_time = time.time() - start_time
    print(f"✓ {config['display_name']} loaded in {load_time:.1f}s")
    
    # ====================================================================
    # Explicit CUDA Placement (Pattern 3: HuggingFace Modernization)
    # ====================================================================
    # Ensure model is explicitly moved to CUDA after loading.
    # While NeMo usually handles this, explicit placement ensures
    # consistent behavior across different loading methods.
    # ====================================================================
    if torch.cuda.is_available():
        try:
            models_cache[model_name] = models_cache[model_name].to("cuda")
            print(f"   ✅ Model moved to CUDA")
        except Exception as e:
            print(f"   ⚠️  Could not move model to CUDA: {e}")
            # Continue with model on CPU
    
    return models_cache[model_name]


def _save_logs(logs, prefix="transcription"):
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

def _make_error_response(error_type, error_msg, log_capture_obj):
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


def _save_output_files(base_filename, file_info, transcription, timestamps, timestamp_level,
                       all_transcriptions=None, all_timestamps=None, is_batch=False,
                       model_choice="", total_duration=0, total_time=0, apply_itn=False):
    """Generate TXT, SRT, and CSV output files.
    
    Args:
        base_filename: Base name for output files (without extension)
        file_info: List of dicts with file metadata (name, duration, is_video)
        transcription: Transcription text (single file) or None (batch)
        timestamps: Timestamp list (single file) or None (batch)
        timestamp_level: 'word', 'segment', 'char', or 'none'
        all_transcriptions: List of transcriptions (batch mode)
        all_timestamps: List of (timestamps, level) tuples (batch mode)
        is_batch: Whether this is batch processing
        model_choice: Model name for metadata
        total_duration: Total audio duration in seconds
        total_time: Total processing time in seconds
        apply_itn: Whether ITN was applied
        
    Returns:
        Tuple of (txt_file, srt_file, csv_file) paths
    """
    txt_file = f"{base_filename}.txt"
    srt_file = f"{base_filename}.srt"
    csv_file = f"{base_filename}.csv"
    
    # Generate TXT file
    with open(txt_file, "w", encoding="utf-8") as f:
        if is_batch:
            f.write(f"Batch Transcription - {len(file_info)} files\n")
            f.write(f"Model: {model_choice}\n")
            f.write(f"Total Duration: {int(total_duration // 60)}m {int(total_duration % 60)}s\n")
            f.write(f"Processing Time: {total_time:.2f}s\n")
            f.write(f"ITN Applied: {'Yes' if apply_itn and ITN_AVAILABLE else 'No'}\n")
            f.write(f"\n{SEPARATOR}\n")
            
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
        else:
            info = file_info[0]
            f.write(f"Audio File: {info['name']}\n")
            f.write(f"Model: {model_choice}\n")
            f.write(f"Duration: {int(info['duration'] // 60)}m {int(info['duration'] % 60)}s\n")
            f.write(f"Processing Time: {total_time:.2f}s\n")
            f.write(f"ITN Applied: {'Yes' if apply_itn and ITN_AVAILABLE else 'No'}\n")
            f.write(f"\n{SEPARATOR}\n")
            f.write(f"TRANSCRIPTION\n")
            f.write(f"{SEPARATOR}\n\n")
            
            if timestamps:
                f.write(format_as_txt_with_timestamps(transcription, timestamps, timestamp_level))
            else:
                f.write(transcription)
    
    # Generate SRT file
    with open(srt_file, "w", encoding="utf-8") as f:
        if is_batch:
            srt_index = 1
            for i, info in enumerate(file_info):
                trans = all_transcriptions[i] if all_transcriptions else ""
                if not trans.startswith("[Transcription failed:"):
                    ts, ts_level = all_timestamps[i] if all_timestamps and i < len(all_timestamps) else ([], 'none')
                    file_srt = format_as_srt(trans, ts, ts_level)
                    f.write(f"{srt_index}\n")
                    f.write(f"00:00:00,000 --> 00:00:02,000\n")
                    f.write(f"[FILE: {info['name']}]\n\n")
                    srt_index += 1
                    for line in file_srt.split('\n\n'):
                        if line.strip():
                            parts = line.split('\n', 1)
                            if len(parts) >= 2:
                                f.write(f"{srt_index}\n{parts[1]}\n\n")
                                srt_index += 1
        else:
            f.write(format_as_srt(transcription, timestamps, timestamp_level))
    
    # Generate CSV file
    with open(csv_file, "w", encoding="utf-8") as f:
        if is_batch:
            f.write("file,start_time,end_time,duration,text\n")
            for i, info in enumerate(file_info):
                trans = all_transcriptions[i] if all_transcriptions else ""
                if not trans.startswith("[Transcription failed:"):
                    ts, ts_level = all_timestamps[i] if all_timestamps and i < len(all_timestamps) else ([], 'none')
                    if ts:
                        for stamp in ts:
                            start = stamp.get('start', 0.0)
                            end = stamp.get('end', 0.0)
                            duration = end - start
                            text = stamp.get('text', stamp.get('word', stamp.get('segment', '')))
                            escaped = text.replace('"', '""')
                            escaped_name = info['name'].replace('"', '""')
                            f.write(f'"{escaped_name}",{start:.3f},{end:.3f},{duration:.3f},"{escaped}"\n')
                    else:
                        escaped = trans.replace('"', '""')
                        escaped_name = info['name'].replace('"', '""')
                        f.write(f'"{escaped_name}",0.000,{info["duration"]:.3f},{info["duration"]:.3f},"{escaped}"\n')
        else:
            f.write(format_as_csv(transcription, timestamps, timestamp_level))
    
    return txt_file, srt_file, csv_file


def _format_batch_status(file_list, file_info, all_transcriptions, per_file_stats, 
                         per_file_errors, model_choice, gpu_name, total_duration,
                         total_time, inference_time, chunk_size, rtfx, vram_used,
                         apply_itn, video_status=""):
    """Format status message for batch transcription.
    
    Returns:
        Tuple of (status_message, combined_transcription)
    """
    total_mins = int(total_duration // 60)
    total_secs = int(total_duration % 60)
    
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
    
    # ITN status
    itn_status = "- **ITN (Numbers to Digits)**: " + (
        "✅ Applied" if apply_itn and ITN_AVAILABLE else 
        ("⚠️ Not installed" if apply_itn else "Disabled")
    )
    
    status = f"""
### ✅ Batch Transcription Complete!

{video_status}**📊 Overall Statistics:**
- **Files Processed**: {len(file_list)} ({len(file_list) - len(per_file_errors)} successful, {len(per_file_errors)} failed)
- **Model**: {model_choice}
- **GPU**: {gpu_name}
- **Total Audio Duration**: {total_mins}m {total_secs}s
- **Processing Time**: {total_time:.2f} seconds
- **Inference Time**: {inference_time:.2f} seconds
- **Chunk Size**: {chunk_size}s
- **Real-Time Factor**: {rtfx:.1f}× (processed {rtfx:.1f}× faster than real-time)
- **VRAM Used**: {vram_used:.2f} GB
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


def _format_single_status(file_info, model_choice, gpu_name, total_time, inference_time,
                          load_time, chunk_size, rtfx, vram_used, transcription,
                          timestamp_level, include_timestamps, apply_itn, video_status=""):
    """Format status message for single file transcription.
    
    Returns:
        Status message string
    """
    info = file_info[0]
    minutes = int(info["duration"] // 60)
    seconds = int(info["duration"] % 60)
    
    file_type_msg = "🎬 Video file detected - audio extracted automatically\n" if info["is_video"] else ""
    timestamp_status = format_timestamp_status(timestamp_level, include_timestamps)
    
    itn_status = "- **ITN (Numbers to Digits)**: " + (
        "✅ Applied" if apply_itn and ITN_AVAILABLE else 
        ("⚠️ Not installed" if apply_itn else "Disabled")
    )
    
    return f"""
### ✅ Transcription Complete!

{file_type_msg}**📊 Statistics:**
- **Model**: {model_choice}
- **GPU**: {gpu_name}
- **Audio Duration**: {minutes}m {seconds}s
- **Processing Time**: {total_time:.2f} seconds
- **Inference Time**: {inference_time:.2f} seconds
- **Model Load Time**: {load_time:.2f} seconds
- **Chunk Size**: {chunk_size}s
- **Real-Time Factor**: {rtfx:.1f}× (processed {rtfx:.1f}× faster than real-time)
- **VRAM Used**: {vram_used:.2f} GB
- **Transcription Length**: {len(transcription)} characters ({len(transcription.split())} words)
{itn_status}
{timestamp_status}

---
"""


def _get_audio_duration_with_retry(file_path, max_retries=4, base_delay=0.5):
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
            duration = librosa.get_duration(path=file_path)
            gc.collect()
            return duration
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                print(f"   ⚠️  File lock on duration check (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise


def _process_audio_files(file_list, log_capture_obj):
    """Process and validate uploaded audio/video files.
    
    Copies files to cache directory and extracts duration information.
    
    Args:
        file_list: List of file paths to process
        log_capture_obj: LogCapture instance for error handling
        
    Returns:
        Tuple of (processed_files, file_info, total_duration, video_count, error_response)
        If error_response is not None, processing failed and should return immediately.
    """
    processed_files = []
    file_info = []
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


def transcribe_audio(audio_files, model_choice, save_to_file, include_timestamps,
                     output_format="txt", apply_itn=True, chunk_size=120, batch_size=1):
    """
    Main transcription function with batch processing, video support, and GPU optimization
    
    Args:
        audio_files: Path to uploaded audio/video file OR list of paths for batch processing
        model_choice: Model selection from radio button
        save_to_file: Boolean - whether to save output file
        include_timestamps: Boolean - whether to generate timestamps
        output_format: Output format - "txt", "srt", or "csv" (primary format for UI display)
        apply_itn: Boolean - whether to apply Inverse Text Normalization (numbers as digits)
        chunk_size: Chunk size in seconds for long audio (30-300)
        batch_size: Batch size for processing (1-8, higher uses more VRAM)
    
    Returns:
        Tuple of (status_message, transcription_text, txt_file, srt_file, csv_file, log_file)
    """
    
    # Start capturing logs
    log_capture.start()
    print(f"\n{'='*60}")
    print(f"🎙️ Transcription Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Update global chunk settings
    global chunk_duration_sec, long_audio_threshold_sec
    chunk_duration_sec = chunk_size
    long_audio_threshold_sec = chunk_size + 30  # Threshold slightly above chunk size
    
    if audio_files is None:
        logs = log_capture.stop()
        return "⚠️ Please upload an audio or video file first", "", None, None, None, None
    
    try:
        import librosa
        
        # Handle both single file and multiple files
        # Gradio gr.File with file_count="multiple" returns list of file objects
        # Each file object has a .name attribute with the path
        if audio_files is None or (isinstance(audio_files, list) and len(audio_files) == 0):
            logs = log_capture.stop()
            return "⚠️ Please upload an audio or video file first", "", None, None, None, None
        
        # Convert file objects to paths
        if isinstance(audio_files, str):
            file_list = [audio_files]
        elif isinstance(audio_files, list):
            # Handle list of file objects or strings
            file_list = []
            for f in audio_files:
                if hasattr(f, 'name'):
                    file_list.append(f.name)
                else:
                    file_list.append(str(f))
        elif hasattr(audio_files, 'name'):
            file_list = [audio_files.name]
        else:
            file_list = [str(audio_files)]
        
        is_batch = len(file_list) > 1
        
        # Log settings
        print(f"📁 Files: {len(file_list)}")
        print(f"📊 Model: {model_choice}")
        print(f"📝 Output format: {output_format.upper()}")
        print(f"🔢 ITN (numbers to digits): {'Enabled' if apply_itn else 'Disabled'}")
        print(f"⏱️ Timestamps: {'Enabled' if include_timestamps else 'Disabled'}")
        print(f"📦 Chunk size: {chunk_size}s")
        
        # Determine which model to use - extract key from choice text
        model_key = get_model_key_from_choice(model_choice)
        
        # Start timing
        start_time = time.time()
        
        # Load model with explicit error handling for Gradio display
        # Uses helper function to standardize error responses
        try:
            model = load_model(model_key)
        except PermissionError as e:
            error_msg = str(e)
            print(f"GRADIO_ERROR (PermissionError): {error_msg}")
            error_type = 'permission_file_lock' if ("WinError 32" in error_msg or "being used by another process" in error_msg) else 'permission'
            return _make_error_response(error_type, error_msg, log_capture)
        except ConnectionError as e:
            print(f"GRADIO_ERROR (ConnectionError): {e}")
            return _make_error_response('network', str(e), log_capture)
        except FileNotFoundError as e:
            print(f"GRADIO_ERROR (FileNotFoundError): {e}")
            return _make_error_response('file_not_found', str(e), log_capture)
        except OSError as e:
            print(f"GRADIO_ERROR (OSError): {e}")
            return _make_error_response('filesystem', str(e), log_capture)
        except RuntimeError as e:
            print(f"GRADIO_ERROR (RuntimeError): {e}")
            return _make_error_response('runtime', str(e), log_capture)
        except Exception as e:
            error_msg = f"Type: {type(e).__name__}\nMessage: {str(e)}"
            print(f"GRADIO_ERROR (Unexpected): {error_msg}")
            return _make_error_response('generic', error_msg, log_capture)
        
        load_time = time.time() - start_time
        
        # ============================================================================
        # FIX #4: Copy Gradio Uploads to Cache Directory
        # ============================================================================
        # Process files using helper function that handles copying to cache,
        # duration extraction, and video detection with proper error handling.
        # ============================================================================
        
        processed_files, file_info, total_duration, video_count, error_response = _process_audio_files(
            file_list, log_capture
        )
        
        if error_response is not None:
            return error_response
        
        # Use NeMo's default batch_size for file batching
        # NeMo handles optimal batching automatically based on VRAM
        batch_size = 4  # NeMo default for file-count batching
        
        # Prepare status update for video processing
        video_status = ""
        if video_count > 0:
            video_status = f"🎬 Extracted audio from {video_count} video file(s)\n"
        
        # ============================================================================
        # Transcription with Retry Logic (WinError 32 Fix)
        # ============================================================================
        # Uses _transcribe_with_retry() which:
        # 1. Uses num_workers=0 to disable multiprocessing (prevents manifest.json locking)
        # 2. Uses NeMo's official transcribe() API with direct parameters
        # 3. Has retry logic with linear backoff for transient file locks
        # 4. Runs garbage collection between retries to release handles
        #
        # This fixes the WinError 32 "file being used by another process" error
        # that occurs when Windows services lock manifest.json files.
        # ============================================================================
        
        # Transcribe with retry wrapper (handles file locks gracefully)
        inference_start = time.time()
        
        try:
            result, chunk_timestamps_map = _transcribe_with_retry(
                model=model,
                files=processed_files,
                batch_size=batch_size,
                use_cuda=torch.cuda.is_available(),
                max_retries=3,
                chunk_size_override=chunk_size
            )
                
        except PermissionError as e:
            error_str = str(e)
            print(f"❌ Transcription permission error: {error_str}")
            error_type = 'transcription_file_lock' if ("WinError 32" in error_str or "being used by another process" in error_str) else 'permission'
            return _make_error_response(error_type, error_str, log_capture)
                    
        except Exception as e:
            error_msg = f"Error Type: {type(e).__name__}\n\nDetails: {str(e)}"
            print(f"❌ Transcription error: {type(e).__name__}: {str(e)}")
            return _make_error_response('transcription', error_msg, log_capture)
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        # Get GPU stats
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
        else:
            vram_used = 0
            gpu_name = "CPU"
        
        # Calculate real-time factor
        rtfx = total_duration / inference_time if inference_time > 0 else 0
        
        print(f"\n✅ Transcription complete!")
        print(f"   Duration: {total_duration:.1f}s audio in {inference_time:.1f}s ({rtfx:.1f}× real-time)")
        
        # ========================================================================
        # Output Validation (Pattern 2: HuggingFace Modernization)
        # ========================================================================
        # Validate transcription result before accessing .text to prevent
        # crashes from malformed results.
        # ========================================================================
        
        # Build output based on single vs batch processing
        if is_batch:
            # ================================================================
            # Batch processing with per-file validation (Pattern 5)
            # ================================================================
            all_transcriptions = []
            all_timestamps = []  # Store timestamps for each file
            per_file_stats = []
            per_file_errors = []
            
            for i, (res, info) in enumerate(zip(result, file_info)):
                # Validate each result individually
                success, transcription, error_msg = validate_transcription_result(result, i)
                
                if success:
                    # Apply ITN if enabled
                    if apply_itn:
                        transcription = apply_inverse_text_normalization(transcription)
                    
                    all_transcriptions.append(transcription)
                    
                    # Get timestamps - use chunk timestamps if available, else model timestamps
                    if i in chunk_timestamps_map:
                        timestamps = chunk_timestamps_map[i]
                        timestamp_level = 'segment'  # Chunk timestamps are segment-level
                    else:
                        timestamps, timestamp_level = extract_timestamps(res, include_timestamps)
                    all_timestamps.append((timestamps, timestamp_level))
                    
                    file_duration = info["duration"]
                    file_mins = int(file_duration // 60)
                    file_secs = int(file_duration % 60)
                    file_type = "🎬 Video" if info["is_video"] else "🎵 Audio"
                    
                    per_file_stats.append(
                        f"**{i+1}. {info['name']}** ({file_type})\n"
                        f"   - Duration: {file_mins}m {file_secs}s\n"
                        f"   - Words: {len(transcription.split())}"
                    )
                else:
                    # Record error for this file
                    all_transcriptions.append(f"[Transcription failed: {error_msg}]")
                    all_timestamps.append(([], 'none'))
                    per_file_errors.append(f"**{i+1}. {info['name']}**: {error_msg}")
                    per_file_stats.append(
                        f"**{i+1}. {info['name']}** ❌ Failed\n"
                        f"   - Error: {error_msg}"
                    )
            
            # Use helper function to format batch status
            status, transcription_output = _format_batch_status(
                file_list=file_list,
                file_info=file_info,
                all_transcriptions=all_transcriptions,
                per_file_stats=per_file_stats,
                per_file_errors=per_file_errors,
                model_choice=model_choice,
                gpu_name=gpu_name,
                total_duration=total_duration,
                total_time=total_time,
                inference_time=inference_time,
                chunk_size=chunk_size,
                rtfx=rtfx,
                vram_used=vram_used,
                apply_itn=apply_itn,
                video_status=video_status
            )
            
            timestamps = []  # For single-file compatibility
            timestamp_level = 'none'
            
        else:
            # ================================================================
            # Single file processing with validation (Pattern 2)
            # ================================================================
            
            # Validate result before accessing
            success, transcription, error_msg = validate_transcription_result(result, 0)
            
            if not success:
                return _make_error_response('validation', error_msg, log_capture)
            
            # Apply ITN if enabled
            if apply_itn:
                print(f"   🔢 Applying Inverse Text Normalization...")
                transcription = apply_inverse_text_normalization(transcription)
            
            # ================================================================
            # Timestamp Extraction - Use chunk timestamps if available
            # ================================================================
            timestamp_level = 'none'
            timestamps = []
            
            # Check if we have chunk timestamps (from long audio chunking)
            if 0 in chunk_timestamps_map and chunk_timestamps_map[0]:
                timestamps = chunk_timestamps_map[0]
                timestamp_level = 'word' if any('word' in ts for ts in timestamps) else 'segment'
                print(f"   ⏱️ Using chunk-based timestamps ({len(timestamps)} entries)")
            elif include_timestamps:
                timestamps, timestamp_level = extract_timestamps(result[0], include_timestamps)
            
            # Use helper function to format single file status
            status = _format_single_status(
                file_info=file_info,
                model_choice=model_choice,
                gpu_name=gpu_name,
                total_time=total_time,
                inference_time=inference_time,
                load_time=load_time,
                chunk_size=chunk_size,
                rtfx=rtfx,
                vram_used=vram_used,
                transcription=transcription,
                timestamp_level=timestamp_level,
                include_timestamps=include_timestamps,
                apply_itn=apply_itn
            )
            # Format transcription output for UI display with timestamps
            if timestamps and include_timestamps:
                transcription_output = format_as_txt_with_timestamps(transcription, timestamps, timestamp_level)
            else:
                transcription_output = transcription
        
        # Generate all 3 file formats for download using helper function
        txt_file = None
        srt_file = None  
        csv_file = None
        
        if save_to_file:
            base_name = os.path.splitext(os.path.basename(file_list[0]))[0]
            
            if is_batch:
                base_filename = f"batch_transcription_{len(file_list)}_files"
                txt_file, srt_file, csv_file = _save_output_files(
                    base_filename=base_filename,
                    file_info=file_info,
                    transcription=None,
                    timestamps=None,
                    timestamp_level='none',
                    all_transcriptions=all_transcriptions,
                    all_timestamps=all_timestamps,
                    is_batch=True,
                    model_choice=model_choice,
                    total_duration=total_duration,
                    total_time=total_time,
                    apply_itn=apply_itn
                )
            else:
                base_filename = f"{base_name}_transcription"
                txt_file, srt_file, csv_file = _save_output_files(
                    base_filename=base_filename,
                    file_info=file_info,
                    transcription=transcription,
                    timestamps=timestamps if include_timestamps else None,
                    timestamp_level=timestamp_level,
                    is_batch=False,
                    model_choice=model_choice,
                    total_duration=total_duration,
                    total_time=total_time,
                    apply_itn=apply_itn
                )
            
            status += f"\n💾 **Files saved**: `{base_filename}.[txt/srt/csv]`"
        
        # Save logs and return
        print(f"\n{'='*60}")
        print(f"✅ Transcription Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        logs = log_capture.stop()
        log_file = _save_logs(logs, "transcription")
        
        return status, transcription_output, txt_file, srt_file, csv_file, log_file
        
    except Exception as e:
        # Get actual VRAM info for error message
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_info = f"you have {vram_total:.0f}GB"
        else:
            vram_info = "no GPU detected"
        
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

def get_system_info():
    """Display system information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_free = (torch.cuda.get_device_properties(0).total_memory - 
                     torch.cuda.memory_allocated()) / 1024**3
        
        info = f"""
### 🖥️ System Information

**GPU**: {gpu_name}
**Total VRAM**: {vram_total:.1f} GB
**Available VRAM**: {vram_free:.1f} GB
**CUDA Version**: {torch.version.cuda}
**PyTorch Version**: {torch.__version__}
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

def get_privacy_performance_info():
    """Generate dynamic privacy & performance information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
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
                    "🌍 Canary-1B v2 (Multilingual + Translation) - 25 languages with AST",
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
                    maximum=8,
                    value=1,
                    step=1,
                    label="📦 Batch Size",
                    info="Higher = more VRAM usage. Increase to utilize more VRAM (try 2-4 for 12GB)."
                )
                
                gr.Markdown(f"""
                **VRAM Optimization Tips**:
                - **Chunk Size**: Larger chunks process faster but use more VRAM
                  - 60-90s: Safe for 8GB VRAM
                  - 120-180s: Good for 12GB VRAM  
                  - 200-300s: For 16GB+ VRAM
                - **Batch Size**: How many chunks to process at once
                  - 1: Safest, lowest VRAM
                  - 2-4: Better GPU utilization for 12GB+
                
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
                output_format, itn_checkbox, chunk_size_slider, batch_size_slider],
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


