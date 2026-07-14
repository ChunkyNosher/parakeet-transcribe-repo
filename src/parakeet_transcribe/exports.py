from __future__ import annotations

import csv
import json
import os
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .types import TranscriptResult

SCRATCH_DIR_NAMES = frozenset({".work", ".youtube"})


def resolved_output_dir() -> Path:
    return Path(os.environ.get("PARAKEET_OUTPUT_DIR", "outputs")).expanduser().resolve()


def _timestamp(seconds: float, *, vtt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_part, milliseconds = divmod(milliseconds, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}{separator}{milliseconds:03d}"


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(value).stem).strip(".-")
    return cleaned or "transcript"


def _segment_line(segment) -> str:
    if segment.speaker:
        return f"[{segment.speaker}] {segment.text}"
    return segment.text


def _subtitle(result: TranscriptResult, *, vtt: bool) -> str:
    if not result.segments:
        return ""
    lines = ["WEBVTT", ""] if vtt else []
    for index, segment in enumerate(result.segments, start=1):
        if not vtt:
            lines.append(str(index))
        lines.extend(
            [
                f"{_timestamp(segment.start, vtt=vtt)} --> {_timestamp(segment.end, vtt=vtt)}",
                _segment_line(segment),
                "",
            ]
        )
    return "\n".join(lines)


def create_run_directory(base_dir: Path | None = None) -> Path:
    base_dir = (base_dir or resolved_output_dir()).resolve()
    name = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    path = base_dir / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_result(result: TranscriptResult, run_dir: Path) -> dict[str, Path]:
    stem = _safe_stem(result.source_name)
    files: dict[str, Path] = {}
    text_path = run_dir / f"{stem}.txt"
    if any(segment.speaker for segment in result.segments):
        text_body = "\n".join(_segment_line(segment) for segment in result.segments)
    else:
        text_body = result.text
    text_path.write_text(text_body + "\n", encoding="utf-8")
    files["txt"] = text_path

    json_path = run_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files["json"] = json_path

    csv_path = run_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_time", "end_time", "duration", "text", "speaker", "confidence"])
        for word in result.words:
            writer.writerow(
                [
                    f"{word.start:.3f}",
                    f"{word.end:.3f}",
                    f"{word.end - word.start:.3f}",
                    word.text,
                    word.speaker or "",
                    "" if word.confidence is None else f"{word.confidence:.6f}",
                ]
            )
    files["csv"] = csv_path

    if result.segments:
        srt_path = run_dir / f"{stem}.srt"
        srt_path.write_text(_subtitle(result, vtt=False), encoding="utf-8")
        files["srt"] = srt_path
        vtt_path = run_dir / f"{stem}.vtt"
        vtt_path.write_text(_subtitle(result, vtt=True), encoding="utf-8")
        files["vtt"] = vtt_path

    if result.summary:
        summary_path = run_dir / f"{stem}.summary.txt"
        summary_path.write_text(result.summary + "\n", encoding="utf-8")
        files["summary"] = summary_path
    if result.chapters:
        chapters_path = run_dir / f"{stem}.chapters.txt"
        chapter_lines = [
            f"{_timestamp(chapter['start'])} {_timestamp(chapter['end'])} {chapter['title']}"
            for chapter in result.chapters
        ]
        chapters_path.write_text("\n".join(chapter_lines) + "\n", encoding="utf-8")
        files["chapters"] = chapters_path
    return files


def write_bundle(results: list[TranscriptResult], run_dir: Path) -> Path:
    """Zip transcript artifacts only — never scratch dirs like `.work` or `.youtube`."""

    manifest = run_dir / "manifest.json"
    manifest.write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    archive_path = Path(f"{run_dir}.zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(run_dir)
            if any(part in SCRATCH_DIR_NAMES for part in relative.parts):
                continue
            archive.write(path, arcname=relative.as_posix())
    return archive_path


def readable_summary(result: TranscriptResult) -> str:
    timing = "word and segment timestamps" if result.has_timestamps else "no timestamps"
    language = result.detected_language or "automatic detection unavailable"
    return f"{result.source_name}: {result.duration_seconds:.1f}s, {language}, {timing}"
