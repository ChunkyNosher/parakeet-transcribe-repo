import gc
import hashlib
import importlib
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .env_bootstrap import GRADIO_UPLOADS_DIR
from .output_formats import format_error_message


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}


def _is_file_lock_error(error_str: str) -> bool:
    return "WinError 32" in error_str or "being used by another process" in error_str


def _clear_cuda_cache_if_available() -> None:
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _handle_retry_delay(attempt: int, base_delay: float, max_retries: int) -> bool:
    if attempt >= max_retries - 1:
        return False

    delay = base_delay * (attempt + 1)
    print(f"   ⚠️  Retry (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
    gc.collect()
    _clear_cuda_cache_if_available()
    time.sleep(delay)
    return True


def _try_file_copy(source_path: Path, dest_path: Path) -> bool:
    shutil.copy2(source_path, dest_path)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    if dest_path.exists():
        dest_path.unlink()
    return False


def copy_gradio_file_to_cache(file_path: str, max_retries: int = 6) -> str:
    file_path_obj = Path(file_path)
    path_hash = hashlib.sha256(str(file_path_obj).encode()).hexdigest()[:16]
    cached_path = GRADIO_UPLOADS_DIR / f"{path_hash}_{file_path_obj.name}"

    if cached_path.exists():
        return str(cached_path)

    base_delay = 0.5
    for attempt in range(max_retries):
        try:
            if _try_file_copy(file_path_obj, cached_path):
                return str(cached_path)
        except (OSError, PermissionError) as exc:
            if _is_file_lock_error(str(exc)) and _handle_retry_delay(attempt, base_delay, max_retries):
                continue
            raise OSError(f"Failed to copy file to cache: {file_path_obj} -> {cached_path}. Error: {exc}") from exc

    raise OSError(f"File copy failed after {max_retries} attempts: {file_path_obj}")


def _validate_audio_duration(duration: float) -> Optional[str]:
    if duration < 0.1:
        return format_error_message("duration_invalid", f"Duration: {duration:.3f}s (minimum: 0.1s)")
    if duration > 86400:
        return format_error_message(
            "duration_invalid",
            f"Duration: {duration:.1f}s ({duration / 3600:.1f} hours, maximum: 24 hours)",
        )
    return None


def _validate_audio_energy(rms_max: float) -> Tuple[bool, str, str]:
    if rms_max < 0.001:
        error = format_error_message("audio_silent", f"Maximum RMS energy: {rms_max:.6f} (threshold: 0.001)")
        return False, error, ""

    warning = "⚠️ Audio is very quiet - transcription quality may be affected" if rms_max < 0.01 else ""
    return True, "", warning


def _classify_audio_load_error(error_str: str) -> str:
    if "Audio file" in error_str or "NoBackendError" in error_str:
        return format_error_message("audio_load_failed", error_str)
    if "Format" in error_str or "codec" in error_str.lower():
        return format_error_message("format_unsupported", error_str)
    return format_error_message("audio_load_failed", error_str)


def validate_and_normalize_audio(file_path: str) -> Tuple[bool, Any, int, str, str]:
    import librosa

    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        duration = float(librosa.get_duration(y=audio_data, sr=sample_rate))
        duration_error = _validate_audio_duration(duration)
        if duration_error:
            return False, None, 0, duration_error, ""

        rms = librosa.feature.rms(y=audio_data)
        is_valid, energy_error, warning_msg = _validate_audio_energy(float(rms.max()))
        if not is_valid:
            return False, None, 0, energy_error, ""

        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        if audio_data.ndim > 1:
            audio_data = librosa.to_mono(audio_data)

        return True, audio_data, sample_rate, "", warning_msg
    except Exception as exc:
        return False, None, 0, _classify_audio_load_error(str(exc)), ""


def load_audio_to_numpy(file_path: str, target_sr: int = 16000) -> Tuple[Any, int]:
    import librosa

    try:
        audio, sample_rate = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sample_rate
    except Exception as exc:
        print(f"   ⚠️ Failed to load {file_path}: {exc}")
        raise


_load_audio_to_numpy = load_audio_to_numpy


def get_audio_duration_with_retry(file_path: str, max_retries: int = 4, base_delay: float = 0.5) -> float:
    import librosa

    for attempt in range(max_retries):
        try:
            duration = float(librosa.get_duration(path=file_path))
            gc.collect()
            return duration
        except (OSError, PermissionError):
            if attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                print(
                    f"   ⚠️  File lock on duration check (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s..."
                )
                time.sleep(delay)
                continue
            raise

    raise RuntimeError("Retry loop exited unexpectedly")


_get_audio_duration_with_retry = get_audio_duration_with_retry


def normalize_file_list(audio_files: Any) -> List[str]:
    if audio_files is None:
        return []
    if isinstance(audio_files, str):
        return [audio_files]
    if isinstance(audio_files, list):
        result: List[str] = []
        for item in audio_files:
            if hasattr(item, "name"):
                result.append(item.name)
            else:
                result.append(str(item))
        return result
    if hasattr(audio_files, "name"):
        return [audio_files.name]
    return [str(audio_files)]


_normalize_file_list = normalize_file_list


def process_audio_files(file_list: List[str]) -> Tuple[List[str], List[Dict[str, Any]], float, int]:
    processed_files: List[str] = []
    file_info: List[Dict[str, Any]] = []
    total_duration = 0.0
    video_count = 0

    for file_path in file_list:
        cached_file_path = copy_gradio_file_to_cache(file_path)
        print(f"📁 Using cached file: {os.path.basename(cached_file_path)}")

        file_ext = os.path.splitext(cached_file_path)[1].lower()
        is_video = file_ext in VIDEO_EXTENSIONS
        if is_video:
            video_count += 1
            print(f"🎬 Extracting audio from video: {os.path.basename(cached_file_path)}")

        try:
            duration = get_audio_duration_with_retry(cached_file_path)
        except (OSError, PermissionError, Exception) as exc:
            if is_video:
                raise OSError(
                    f"Video file '{os.path.basename(cached_file_path)}' appears to have no audio track or cannot be processed.\n\nError: {exc}"
                ) from exc
            raise

        total_duration += duration
        processed_files.append(cached_file_path)
        file_info.append(
            {
                "path": cached_file_path,
                "name": os.path.basename(file_path),
                "duration": duration,
                "is_video": is_video,
            }
        )

    return processed_files, file_info, total_duration, video_count


def _process_audio_files(
    file_list: List[str],
    log_capture_obj: Optional[Any] = None,
    error_factory: Optional[Callable[[str, str, Any], Tuple[Any, ...]]] = None,
) -> Tuple[Optional[List[str]], Optional[List[Dict[str, Any]]], float, int, Optional[Tuple[Any, ...]]]:
    try:
        processed_files, file_info, total_duration, video_count = process_audio_files(file_list)
        return processed_files, file_info, total_duration, video_count, None
    except OSError as exc:
        if error_factory is not None and log_capture_obj is not None:
            return None, None, 0, 0, error_factory("filesystem", str(exc), log_capture_obj)
        raise


__all__ = [
    "AUDIO_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "_get_audio_duration_with_retry",
    "_load_audio_to_numpy",
    "_normalize_file_list",
    "_process_audio_files",
    "copy_gradio_file_to_cache",
    "get_audio_duration_with_retry",
    "load_audio_to_numpy",
    "normalize_file_list",
    "process_audio_files",
    "validate_and_normalize_audio",
]