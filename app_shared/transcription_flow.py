import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .env_bootstrap import CACHE_DIR
from .logging_utils import LogCapture, save_logs
from .output_formats import SEPARATOR, _save_output_files, format_as_txt_with_timestamps, format_error_message
from .result_types import OutputFilesConfig, ResultProcessingContext, TranscriptionStats


def _validate_result_structure(result: Any, idx: int) -> Tuple[bool, str]:
    if result is None:
        return False, "Result is None - model may have failed silently"
    if not isinstance(result, list):
        return False, f"Result is {type(result).__name__}, expected list"
    if len(result) == 0:
        return False, "Result is an empty list - no transcription generated"
    if idx >= len(result):
        return False, f"Index {idx} out of range (result has {len(result)} items)"
    return True, ""


def _extract_text_from_hypothesis(hypothesis: Any) -> Tuple[bool, str, str]:
    if isinstance(hypothesis, str):
        return len(hypothesis) > 0, hypothesis, "" if hypothesis else "Transcription is empty"
    if not hasattr(hypothesis, "text"):
        return False, "", f"Hypothesis has no .text attribute (type: {type(hypothesis).__name__})"
    if not isinstance(hypothesis.text, str):
        return False, "", f".text is {type(hypothesis.text).__name__}, expected string"
    if len(hypothesis.text) == 0:
        return False, "", "Transcription is empty (0 characters)"
    return True, hypothesis.text, ""


def validate_transcription_result(result: Any, idx: int = 0) -> Tuple[bool, str, str]:
    valid, error_msg = _validate_result_structure(result, idx)
    if not valid:
        return False, "", error_msg
    return _extract_text_from_hypothesis(result[idx])


def _try_get_timestamp_level(hypothesis: Any, level_key: str) -> Optional[List[Dict[str, Any]]]:
    try:
        timestamp_dict = getattr(hypothesis, "timestamp", None)
        if not timestamp_dict or not isinstance(timestamp_dict, dict):
            return None
        timestamps = timestamp_dict.get(level_key)
        if isinstance(timestamps, list) and len(timestamps) > 0:
            return timestamps
    except (AttributeError, KeyError, TypeError):
        pass
    return None


def extract_timestamps(hypothesis: Any, include_timestamps: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    if not include_timestamps:
        return [], "none"

    for level in ("word", "segment", "char"):
        timestamps = _try_get_timestamp_level(hypothesis, level)
        if timestamps:
            return timestamps, level

    return [], "none"


def format_timestamp_status(level: str, include_timestamps: bool) -> str:
    if not include_timestamps:
        return ""
    if level == "word":
        return "\n✅ **Timestamps:** Word-level available"
    if level == "segment":
        return "\n⚠️ **Timestamps:** Segment-level (word-level unavailable for this model)"
    if level == "char":
        return "\n⚠️ **Timestamps:** Character-level (word/segment unavailable)"
    return "\nℹ️ **Timestamps:** Not available for this model"


def _format_itn_status(apply_itn: bool, itn_available: bool) -> str:
    if apply_itn and itn_available:
        return "- **ITN (Numbers to Digits)**: ✅ Applied"
    if apply_itn:
        return "- **ITN (Numbers to Digits)**: ⚠️ Not installed"
    return "- **ITN (Numbers to Digits)**: Disabled"


def _format_batch_status(
    file_list: List[str],
    file_info: List[Dict[str, Any]],
    all_transcriptions: List[str],
    per_file_stats: List[str],
    per_file_errors: List[str],
    stats: TranscriptionStats,
    itn_available: bool = False,
    video_status: str = "",
) -> Tuple[str, str]:
    total_mins = int(stats.total_duration // 60)
    total_secs = int(stats.total_duration % 60)
    successful = [item for item in all_transcriptions if not item.startswith("[Transcription failed:")]
    total_words = sum(len(item.split()) for item in successful)

    error_summary = ""
    if per_file_errors:
        error_summary = (
            f"\n\n⚠️ **{len(per_file_errors)} file(s) failed to transcribe:**\n"
            + "\n".join(per_file_errors)
        )

    itn_status = _format_itn_status(stats.apply_itn, itn_available)
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

    combined = ""
    for index, (info, transcription) in enumerate(zip(file_info, all_transcriptions)):
        combined += f"\n{SEPARATOR}\n"
        combined += f"FILE {index + 1}: {info['name']}\n"
        combined += f"{SEPARATOR}\n\n"
        combined += transcription + "\n"

    return status, combined


def _format_single_status(
    file_info: List[Dict[str, Any]],
    stats: TranscriptionStats,
    transcription: str,
    timestamp_level: str,
    include_timestamps: bool,
    itn_available: bool = False,
    video_status: str = "",
) -> str:
    info = file_info[0]
    minutes = int(info["duration"] // 60)
    seconds = int(info["duration"] % 60)

    file_type_msg = "🎬 Video file detected - audio extracted automatically\n" if info["is_video"] else ""
    timestamp_status = format_timestamp_status(timestamp_level, include_timestamps)
    itn_status = _format_itn_status(stats.apply_itn, itn_available)

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


def _make_error_response(
    error_type: str,
    error_msg: str,
    log_capture_obj: LogCapture,
    cache_dir: Optional[Path] = None,
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    logs = log_capture_obj.stop()
    log_file = save_logs(logs, "error")
    effective_cache_dir = cache_dir or CACHE_DIR

    error_messages = {
        "permission": f"### ❌ Permission Error\n\n{error_msg}",
        "permission_file_lock": (
            f"### ❌ Model Loading Failed: Windows File Lock\n\n"
            f"{error_msg}\n\n"
            f"**Root Cause:** Windows services (antivirus, OneDrive, indexing) are locking model files.\n\n"
            f"**Immediate Actions:**\n"
            f"1. Pause OneDrive/Dropbox/Google Drive\n"
            f"2. Temporarily disable antivirus real-time scanning\n"
            f"3. Run as Administrator\n"
            f"4. Restart your computer\n\n"
            f"**Cache Location:** `{effective_cache_dir}`\n\n"
            f"Add this folder to antivirus exclusions if issue persists."
        ),
        "network": f"### ❌ Network Error\n\n{error_msg}",
        "file_not_found": f"### ❌ File Not Found\n\n{error_msg}",
        "filesystem": f"### ❌ File System Error\n\n{error_msg}",
        "runtime": f"### ❌ Runtime Error\n\n{error_msg}",
        "validation": format_error_message("output_validation_failed", error_msg),
        "transcription_file_lock": (
            f"### ❌ Transcription Failed: File Lock Error\n\n"
            f"The transcription process failed due to file locking after 3 retry attempts.\n\n"
            f"**Solutions:**\n"
            f"1. Pause OneDrive/Dropbox/Google Drive temporarily\n"
            f"2. Add cache directory to antivirus exclusions: `{effective_cache_dir}`\n"
            f"3. Restart your computer\n"
            f"4. Run as Administrator\n\n"
            f"**Technical Details:**\n"
            f"```\n{error_msg}\n```"
        ),
        "transcription": (
            f"### ❌ Transcription Error\n\n"
            f"An error occurred during transcription.\n\n"
            f"**Technical Details:**\n"
            f"```\n{error_msg}\n```"
        ),
        "generic": (
            f"{'=' * 80}\n"
            f"❌ UNEXPECTED ERROR\n"
            f"{'=' * 80}\n\n"
            f"{error_msg}\n\n"
            f"Please check the console for more details.\n"
            f"{'=' * 80}"
        ),
    }

    status = error_messages.get(error_type, error_messages["generic"])
    return status, "", None, None, None, log_file


def _process_batch_results(
    result: List[Any],
    file_info: List[Dict[str, Any]],
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]],
    apply_itn: bool,
    include_timestamps: bool,
    text_normalizer: Optional[Any] = None,
) -> Tuple[List[str], List[Tuple[List[Dict[str, Any]], str]], List[str], List[str]]:
    all_transcriptions: List[str] = []
    all_timestamps: List[Tuple[List[Dict[str, Any]], str]] = []
    per_file_stats: List[str] = []
    per_file_errors: List[str] = []

    for index, (res, info) in enumerate(zip(result, file_info)):
        success, transcription, error_msg = validate_transcription_result(result, index)

        if success:
            if apply_itn and index not in chunk_timestamps_map and text_normalizer is not None:
                transcription = text_normalizer(transcription)

            all_transcriptions.append(transcription)

            if index in chunk_timestamps_map:
                timestamps = chunk_timestamps_map[index]
                timestamp_level = "word" if any("word" in stamp for stamp in timestamps) else "segment"
            else:
                timestamps, timestamp_level = extract_timestamps(res, include_timestamps)
            all_timestamps.append((timestamps, timestamp_level))

            file_duration = info["duration"]
            file_mins = int(file_duration // 60)
            file_secs = int(file_duration % 60)
            file_type = "🎬 Video" if info["is_video"] else "🎵 Audio"

            per_file_stats.append(
                f"**{index + 1}. {info['name']}** ({file_type})\n"
                f"   - Duration: {file_mins}m {file_secs}s\n"
                f"   - Words: {len(transcription.split())}"
            )
        else:
            all_transcriptions.append(f"[Transcription failed: {error_msg}]")
            all_timestamps.append(([], "none"))
            per_file_errors.append(f"**{index + 1}. {info['name']}**: {error_msg}")
            per_file_stats.append(
                f"**{index + 1}. {info['name']}** ❌ Failed\n"
                f"   - Error: {error_msg}"
            )

    return all_transcriptions, all_timestamps, per_file_stats, per_file_errors


def _extract_single_result_timestamps(
    result: List[Any],
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]],
    include_timestamps: bool,
) -> Tuple[List[Dict[str, Any]], str]:
    if 0 in chunk_timestamps_map and chunk_timestamps_map[0]:
        timestamps = chunk_timestamps_map[0]
        timestamp_level = "word" if any("word" in stamp for stamp in timestamps) else "segment"
        print(f"   ⏱️ Using chunk-based timestamps ({len(timestamps)} entries)")
        return timestamps, timestamp_level

    if include_timestamps:
        return extract_timestamps(result[0], include_timestamps)

    return [], "none"


def _process_single_result(
    result: List[Any],
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]],
    apply_itn_final: bool,
    include_timestamps: bool,
    log_capture: LogCapture,
    text_normalizer: Optional[Any] = None,
    had_itn_per_chunk: bool = False,
) -> Tuple[Optional[str], List[Dict[str, Any]], str, Optional[Tuple[Any, ...]]]:
    success, transcription, error_msg = validate_transcription_result(result, 0)
    if not success:
        return None, [], "none", _make_error_response("validation", error_msg, log_capture)

    if apply_itn_final and text_normalizer is not None:
        if had_itn_per_chunk:
            print("   🔢 Applying final-pass ITN (already applied per-chunk, mode=both)")
        else:
            print("   🔢 Applying final-pass Inverse Text Normalization...")
        transcription = text_normalizer(transcription)
    elif had_itn_per_chunk:
        print("   🔢 ITN was applied per-chunk during transcription")

    timestamps, timestamp_level = _extract_single_result_timestamps(result, chunk_timestamps_map, include_timestamps)
    return transcription, timestamps, timestamp_level, None


def _generate_and_save_output_files(
    save_to_file: bool,
    config: OutputFilesConfig,
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    if not save_to_file:
        return None, None, None, ""

    base_name = os.path.splitext(os.path.basename(config.file_list[0]))[0]
    if config.is_batch:
        base_filename = f"batch_transcription_{len(config.file_list)}_files"
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
            timestamp_level="none",
            all_transcriptions=config.all_transcriptions,
            all_timestamps=config.all_timestamps,
            itn_available=config.itn_available,
        )
        txt_file, srt_file, csv_file = _save_output_files(base_filename, batch_config)
    else:
        base_filename = f"{base_name}_transcription"
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
            timestamp_level=config.timestamp_level,
            itn_available=config.itn_available,
        )
        txt_file, srt_file, csv_file = _save_output_files(base_filename, single_config)

    return txt_file, srt_file, csv_file, f"\n💾 **Files saved**: `{base_filename}.[txt/srt/csv]`"


def _process_batch_transcription(
    result: List[Any],
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]],
    ctx: ResultProcessingContext,
) -> Tuple[str, str, List[Any], str, List[str], List[Tuple[List[Dict[str, Any]], str]]]:
    all_transcriptions, all_timestamps, per_file_stats, per_file_errors = _process_batch_results(
        result,
        ctx.file_info,
        chunk_timestamps_map,
        ctx.stats.apply_itn,
        ctx.include_timestamps,
        ctx.text_normalizer,
    )

    status, transcription_output = _format_batch_status(
        file_list=ctx.file_list,
        file_info=ctx.file_info,
        all_transcriptions=all_transcriptions,
        per_file_stats=per_file_stats,
        per_file_errors=per_file_errors,
        stats=ctx.stats,
        itn_available=ctx.itn_available,
        video_status=ctx.video_status,
    )

    return status, transcription_output, [], "none", all_transcriptions, all_timestamps


def _process_single_transcription(
    result: List[Any],
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]],
    log_capture: LogCapture,
    ctx: ResultProcessingContext,
) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]], str, Optional[Tuple[Any, ...]]]:
    transcription, timestamps, timestamp_level, error_response = _process_single_result(
        result,
        chunk_timestamps_map,
        ctx.apply_itn_final,
        ctx.include_timestamps,
        log_capture,
        text_normalizer=ctx.text_normalizer,
        had_itn_per_chunk=ctx.had_itn_per_chunk,
    )

    if error_response is not None:
        return None, None, [], "none", error_response

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
        apply_itn=ctx.stats.apply_itn,
    )

    status = _format_single_status(
        file_info=ctx.file_info,
        stats=single_stats,
        transcription=transcription or "",
        timestamp_level=timestamp_level,
        include_timestamps=ctx.include_timestamps,
        itn_available=ctx.itn_available,
        video_status=ctx.video_status,
    )

    if timestamps and ctx.include_timestamps:
        transcription_output = format_as_txt_with_timestamps(transcription or "", timestamps, timestamp_level)
    else:
        transcription_output = transcription or ""

    return status, transcription_output, timestamps, timestamp_level, None


__all__ = [
    "_extract_text_from_hypothesis",
    "_extract_single_result_timestamps",
    "_format_batch_status",
    "_format_itn_status",
    "_format_single_status",
    "_generate_and_save_output_files",
    "_make_error_response",
    "_process_batch_results",
    "_process_batch_transcription",
    "_process_single_result",
    "_process_single_transcription",
    "_try_get_timestamp_level",
    "_validate_result_structure",
    "extract_timestamps",
    "format_timestamp_status",
    "validate_transcription_result",
]