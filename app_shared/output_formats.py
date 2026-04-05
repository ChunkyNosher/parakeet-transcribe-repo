from typing import Any, Dict, List, Optional, TextIO, Tuple

from .result_types import OutputFilesConfig


DEFAULT_SILENCE_THRESHOLD_SEC = 0.5
DEFAULT_MAX_WORD_DURATION_SEC = 2.0

silence_threshold_sec = DEFAULT_SILENCE_THRESHOLD_SEC
max_word_duration_sec = DEFAULT_MAX_WORD_DURATION_SEC

SEPARATOR = "=" * 60

ERROR_MESSAGES = {
    "audio_load_failed": {
        "title": "❌ Could Not Load Audio File",
        "message": (
            "The audio file could not be loaded. Please check that:\n"
            "- The file format is supported (WAV, MP3, FLAC, M4A, OGG, AAC, WMA)\n"
            "- The file is not corrupted or empty\n"
            "- You have read permissions for the file"
        ),
    },
    "format_unsupported": {
        "title": "❌ Audio Format Not Supported",
        "message": (
            "**Supported Audio Formats:** WAV, MP3, FLAC, M4A, OGG, AAC, WMA\n"
            "**Supported Video Formats:** MP4, AVI, MKV, MOV, WEBM, FLV, M4V\n"
            "\nVideo files will have their audio extracted automatically."
        ),
    },
    "duration_invalid": {
        "title": "❌ Invalid Audio Duration",
        "message": (
            "Audio duration must be between 100ms and 24 hours.\n"
            "Please check if the file is corrupted, silent, or has an unusual format."
        ),
    },
    "audio_silent": {
        "title": "⚠️ Audio Appears to be Silent",
        "message": (
            "The audio file appears to contain very little or no audio signal.\n"
            "Please check that:\n"
            "- The audio was recorded properly\n"
            "- The volume level is not too low\n"
            "- The correct audio channel was selected during recording"
        ),
    },
    "output_validation_failed": {
        "title": "❌ Transcription Output Invalid",
        "message": (
            "The model returned an invalid or empty result.\n"
            "This can happen when:\n"
            "- Audio quality is very poor\n"
            "- Audio contains only noise or music\n"
            "- Audio language is not supported by the model"
        ),
    },
    "batch_partial_failure": {
        "title": "⚠️ Some Files Failed to Process",
        "message": (
            "Some files in the batch could not be transcribed.\n"
            "Successfully processed files are shown below.\n"
            "Check the error details for each failed file."
        ),
    },
    "model_load_failed": {
        "title": "❌ Model Loading Failed",
        "message": (
            "The AI model could not be loaded.\n"
            "Please check that:\n"
            "- You have enough disk space\n"
            "- Your internet connection is stable (for first download)\n"
            "- The cache directory is accessible"
        ),
    },
    "transcription_timeout": {
        "title": "❌ Transcription Timed Out",
        "message": (
            "The transcription process took too long and was stopped.\n"
            "This can happen with very long audio files.\n"
            "Try splitting the audio into smaller chunks."
        ),
    },
}


def configure_output_timing(
    silence_threshold: Optional[float] = None,
    max_word_duration: Optional[float] = None,
) -> None:
    """Update default segmentation timing used by TXT/SRT formatting."""

    global silence_threshold_sec, max_word_duration_sec

    if silence_threshold is not None:
        silence_threshold_sec = silence_threshold
    if max_word_duration is not None:
        max_word_duration_sec = max_word_duration


def format_error_message(error_type: str, detail: str = "") -> str:
    """Format a stage-specific error message with optional details."""

    msg = ERROR_MESSAGES.get(
        error_type,
        {
            "title": "❌ Unknown Error",
            "message": "An unexpected error occurred.",
        },
    )

    result = f"### {msg['title']}\n\n{msg['message']}"
    if detail:
        result += f"\n\n**Technical Details:**\n```\n{detail}\n```"
    return result


def _format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _ends_with_sentence_punctuation(word: str) -> bool:
    if not word:
        return False
    return word.rstrip().endswith((".", "?", "!", "...", "。", "？", "！"))


def _get_word_text_from_timestamp(stamp: Dict[str, Any]) -> str:
    return stamp.get("text", stamp.get("word", stamp.get("segment", stamp.get("char", ""))))


def _should_end_segment(
    word: str,
    segment_duration: float,
    word_count: int,
    has_punctuation: bool,
    words_per_segment: int,
    max_duration: float,
) -> bool:
    ends_sentence = _ends_with_sentence_punctuation(word)
    return (
        (has_punctuation and ends_sentence)
        or segment_duration > max_duration
        or (not has_punctuation and word_count >= words_per_segment)
    )


def _normalize_word_timing(start: float, end: float, max_word_duration: float) -> Tuple[float, float]:
    safe_start = float(start)
    safe_end = float(end)

    if safe_end < safe_start:
        safe_end = safe_start

    if max_word_duration > 0 and (safe_end - safe_start) > max_word_duration:
        safe_start = max(safe_start, safe_end - max_word_duration)
        safe_end = min(safe_end, safe_start + max_word_duration)

    if safe_end < safe_start:
        safe_end = safe_start

    return safe_start, safe_end


def _merge_orphan_sentence_segments(
    segments: List[Dict[str, Any]],
    silence_threshold: float,
) -> List[Dict[str, Any]]:
    if len(segments) < 2:
        return segments

    merged: List[Dict[str, Any]] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        prev_text = str(prev.get("text", "")).strip()
        seg_text = str(seg.get("text", "")).strip()

        seg_word_count = len(seg_text.split())
        gap = float(seg.get("start", 0.0)) - float(prev.get("end", 0.0))
        should_merge = (
            seg_word_count <= 2
            and gap <= max(0.0, silence_threshold)
            and not _ends_with_sentence_punctuation(prev_text)
        )

        if should_merge:
            prev["text"] = f"{prev_text} {seg_text}".strip()
            prev["end"] = max(float(prev.get("end", 0.0)), float(seg.get("end", 0.0)))
            continue

        merged.append(seg)

    return merged


def _enforce_segment_boundaries(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segments:
        return []

    cleaned: List[Dict[str, Any]] = []
    for index, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()

        if not text:
            continue

        if end < start:
            end = start

        if index + 1 < len(segments):
            next_start = float(segments[index + 1].get("start", end))
            end = min(end, next_start)
            if end < start:
                end = start

        cleaned.append({"start": start, "end": end, "text": text})

    return cleaned


def _finalize_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start": segment["start"],
        "end": segment["end"],
        "text": " ".join(segment["words"]),
    }


def _group_words_into_segments(
    timestamps: List[Dict[str, Any]],
    words_per_segment: int = 8,
    max_duration: float = 5.0,
    silence_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if not timestamps:
        return []

    effective_silence_threshold = silence_threshold if silence_threshold is not None else silence_threshold_sec
    has_punctuation = any(_ends_with_sentence_punctuation(_get_word_text_from_timestamp(ts)) for ts in timestamps)

    segments: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"start": timestamps[0].get("start", 0.0), "end": 0.0, "words": []}
    last_end_time = float(timestamps[0].get("start", 0.0))

    for stamp in timestamps:
        word = _get_word_text_from_timestamp(stamp)
        raw_start = float(stamp.get("start", 0.0))
        raw_end = float(stamp.get("end", 0.0))
        start, end = _normalize_word_timing(raw_start, raw_end, max_word_duration_sec)

        if current["words"] and effective_silence_threshold > 0:
            gap = start - last_end_time
            if gap >= effective_silence_threshold:
                if current["words"]:
                    segments.append(_finalize_segment(current))
                    current = {"start": start, "end": end, "words": [word]}
                    last_end_time = end
                    continue

        current["words"].append(word)
        current["end"] = end
        last_end_time = end

        segment_duration = end - current["start"]
        if _should_end_segment(
            word,
            segment_duration,
            len(current["words"]),
            has_punctuation,
            words_per_segment,
            max_duration,
        ):
            if current["words"]:
                segments.append(_finalize_segment(current))
                current = {"start": end, "end": end, "words": []}

    if current["words"]:
        segments.append(_finalize_segment(current))

    segments = _merge_orphan_sentence_segments(segments, effective_silence_threshold)
    return _enforce_segment_boundaries(segments)


def format_as_srt(
    transcription: str,
    timestamps: List[Dict[str, Any]],
    timestamp_level: str = "word",
) -> str:
    if not timestamps or timestamp_level == "none":
        word_count = len(transcription.split())
        estimated_duration = max(5.0, word_count / 2.5)
        return (
            "1\n"
            f"00:00:00,000 --> {_format_srt_timestamp(estimated_duration)}\n"
            f"{transcription}\n"
        )

    if timestamp_level == "word":
        segments = _group_words_into_segments(timestamps)
    else:
        segments: List[Dict[str, Any]] = []
        for stamp in timestamps:
            text = stamp.get("text", stamp.get("segment", stamp.get("word", stamp.get("char", ""))))
            segments.append(
                {
                    "start": stamp.get("start", 0.0),
                    "end": stamp.get("end", 0.0),
                    "text": text,
                }
            )

    srt_lines: List[str] = []
    for index, seg in enumerate(segments, 1):
        start_ts = _format_srt_timestamp(seg["start"])
        end_ts = _format_srt_timestamp(seg["end"])
        srt_lines.append(f"{index}")
        srt_lines.append(f"{start_ts} --> {end_ts}")
        srt_lines.append(seg["text"])
        srt_lines.append("")

    return "\n".join(srt_lines)


def format_as_csv(
    transcription: str,
    timestamps: List[Dict[str, Any]],
    timestamp_level: str = "word",
) -> str:
    csv_lines: List[str] = ["start_time,end_time,duration,text"]

    if not timestamps or timestamp_level == "none":
        word_count = len(transcription.split())
        estimated_duration = max(5.0, word_count / 2.5)
        escaped_text = transcription.replace('"', '""')
        csv_lines.append(f'0.000,{estimated_duration:.3f},{estimated_duration:.3f},"{escaped_text}"')
        return "\n".join(csv_lines)

    for stamp in timestamps:
        start = stamp.get("start", 0.0)
        end = stamp.get("end", 0.0)
        duration = end - start
        text = stamp.get("text", stamp.get("word", stamp.get("segment", stamp.get("char", ""))))
        escaped_text = text.replace('"', '""')
        csv_lines.append(f'{start:.3f},{end:.3f},{duration:.3f},"{escaped_text}"')

    return "\n".join(csv_lines)


def format_as_txt_with_timestamps(
    transcription: str,
    timestamps: List[Dict[str, Any]],
    timestamp_level: str = "word",
) -> str:
    if not timestamps or timestamp_level == "none":
        return f"[00:00:00] {transcription}"

    if timestamp_level == "word":
        segments = _group_words_into_segments(timestamps, words_per_segment=12, max_duration=8.0)
    else:
        segments: List[Dict[str, Any]] = []
        for stamp in timestamps:
            text = stamp.get("text", stamp.get("segment", stamp.get("word", stamp.get("char", ""))))
            segments.append({"start": stamp.get("start", 0.0), "text": text})

    txt_lines: List[str] = []
    for seg in segments:
        seconds = seg["start"]
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        timestamp = f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
        txt_lines.append(f"{timestamp} {seg['text']}")

    return "\n".join(txt_lines)


def _write_txt_header(handle: TextIO, config: OutputFilesConfig) -> None:
    itn_text = "Yes" if config.apply_itn and config.itn_available else "No"
    mins, secs = int(config.total_duration // 60), int(config.total_duration % 60)

    if config.is_batch:
        handle.write(f"Batch Transcription - {len(config.file_info)} files\n")
        handle.write(f"Model: {config.model_choice}\n")
        handle.write(f"Total Duration: {mins}m {secs}s\n")
        handle.write(f"Processing Time: {config.total_time:.2f}s\n")
        handle.write(f"ITN Applied: {itn_text}\n")
        handle.write(f"\n{SEPARATOR}\n")
        return

    info = config.file_info[0]
    dur_mins = int(info["duration"] // 60)
    dur_secs = int(info["duration"] % 60)
    handle.write(f"Audio File: {info['name']}\n")
    handle.write(f"Model: {config.model_choice}\n")
    handle.write(f"Duration: {dur_mins}m {dur_secs}s\n")
    handle.write(f"Processing Time: {config.total_time:.2f}s\n")
    handle.write(f"ITN Applied: {itn_text}\n")
    handle.write(f"\n{SEPARATOR}\n")
    handle.write("TRANSCRIPTION\n")
    handle.write(f"{SEPARATOR}\n\n")


def _write_txt_batch_files(
    handle: TextIO,
    file_info: List[Dict[str, Any]],
    all_transcriptions: Optional[List[str]],
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]],
) -> None:
    for index, info in enumerate(file_info):
        transcription = all_transcriptions[index] if all_transcriptions else ""
        handle.write(f"\nFILE {index + 1}: {info['name']}\n")
        handle.write(f"Duration: {int(info['duration'] // 60)}m {int(info['duration'] % 60)}s\n")
        handle.write(f"{SEPARATOR}\n\n")
        timestamps, timestamp_level = all_timestamps[index] if all_timestamps and index < len(all_timestamps) else ([], "none")
        if timestamps:
            handle.write(format_as_txt_with_timestamps(transcription, timestamps, timestamp_level))
        else:
            handle.write(transcription)
        handle.write("\n")


def _get_batch_file_data(
    file_info: List[Dict[str, Any]],
    all_transcriptions: Optional[List[str]],
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]],
    index: int,
) -> Tuple[str, List[Dict[str, Any]], str, bool]:
    transcription = all_transcriptions[index] if all_transcriptions else ""
    is_valid = not transcription.startswith("[Transcription failed:")
    timestamps: List[Dict[str, Any]] = []
    timestamp_level = "none"
    if all_timestamps and index < len(all_timestamps):
        timestamps, timestamp_level = all_timestamps[index]
    return transcription, timestamps, timestamp_level, is_valid


def _write_srt_batch(
    handle: TextIO,
    file_info: List[Dict[str, Any]],
    all_transcriptions: Optional[List[str]],
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]],
) -> None:
    srt_index = 1
    for index, info in enumerate(file_info):
        transcription, timestamps, _timestamp_level, is_valid = _get_batch_file_data(
            file_info,
            all_transcriptions,
            all_timestamps,
            index,
        )
        if not is_valid:
            continue

        file_srt = format_as_srt(transcription, timestamps, "segment")
        handle.write(f"{srt_index}\n00:00:00,000 --> 00:00:02,000\n[FILE: {info['name']}]\n\n")
        srt_index += 1

        for block in file_srt.split("\n\n"):
            if not block.strip():
                continue
            parts = block.split("\n", 1)
            if len(parts) >= 2:
                handle.write(f"{srt_index}\n{parts[1]}\n\n")
                srt_index += 1


def _write_csv_timestamp_row(handle: TextIO, filename: str, stamp: Dict[str, Any]) -> None:
    start = stamp.get("start", 0.0)
    end = stamp.get("end", 0.0)
    duration = end - start
    text = stamp.get("text", stamp.get("word", stamp.get("segment", "")))
    escaped_text = text.replace('"', '""')
    escaped_name = filename.replace('"', '""')
    handle.write(f'"{escaped_name}",{start:.3f},{end:.3f},{duration:.3f},"{escaped_text}"\n')


def _write_csv_batch(
    handle: TextIO,
    file_info: List[Dict[str, Any]],
    all_transcriptions: Optional[List[str]],
    all_timestamps: Optional[List[Tuple[List[Dict[str, Any]], str]]],
) -> None:
    handle.write("file,start_time,end_time,duration,text\n")

    for index, info in enumerate(file_info):
        transcription, timestamps, _timestamp_level, is_valid = _get_batch_file_data(
            file_info,
            all_transcriptions,
            all_timestamps,
            index,
        )
        if not is_valid:
            continue

        if timestamps:
            for stamp in timestamps:
                _write_csv_timestamp_row(handle, info["name"], stamp)
            continue

        escaped_transcription = transcription.replace('"', '""')
        escaped_name = info["name"].replace('"', '""')
        handle.write(
            f'"{escaped_name}",0.000,{info["duration"]:.3f},{info["duration"]:.3f},"{escaped_transcription}"\n'
        )


def _write_txt_content(handle: TextIO, config: OutputFilesConfig) -> None:
    if config.is_batch:
        _write_txt_batch_files(handle, config.file_info, config.all_transcriptions, config.all_timestamps)
        return

    if config.timestamps:
        handle.write(format_as_txt_with_timestamps(config.transcription or "", config.timestamps or [], config.timestamp_level))
        return

    handle.write(config.transcription or "")


def _write_srt_content(handle: TextIO, config: OutputFilesConfig) -> None:
    if config.is_batch:
        _write_srt_batch(handle, config.file_info, config.all_transcriptions, config.all_timestamps)
        return

    handle.write(format_as_srt(config.transcription or "", config.timestamps or [], config.timestamp_level))


def _write_csv_content(handle: TextIO, config: OutputFilesConfig) -> None:
    if config.is_batch:
        _write_csv_batch(handle, config.file_info, config.all_transcriptions, config.all_timestamps)
        return

    handle.write(format_as_csv(config.transcription or "", config.timestamps or [], config.timestamp_level))


def save_output_files(base_filename: str, config: OutputFilesConfig) -> Tuple[str, str, str]:
    txt_file = f"{base_filename}.txt"
    srt_file = f"{base_filename}.srt"
    csv_file = f"{base_filename}.csv"

    with open(txt_file, "w", encoding="utf-8") as handle:
        _write_txt_header(handle, config)
        _write_txt_content(handle, config)

    with open(srt_file, "w", encoding="utf-8") as handle:
        _write_srt_content(handle, config)

    with open(csv_file, "w", encoding="utf-8") as handle:
        _write_csv_content(handle, config)

    return txt_file, srt_file, csv_file


_save_output_files = save_output_files


__all__ = [
    "DEFAULT_MAX_WORD_DURATION_SEC",
    "DEFAULT_SILENCE_THRESHOLD_SEC",
    "ERROR_MESSAGES",
    "SEPARATOR",
    "_save_output_files",
    "configure_output_timing",
    "format_as_csv",
    "format_as_srt",
    "format_as_txt_with_timestamps",
    "format_error_message",
    "save_output_files",
]