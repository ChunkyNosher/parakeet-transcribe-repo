from __future__ import annotations

import csv
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .types import TranscriptResult


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
                segment.text,
                "",
            ]
        )
    return "\n".join(lines)


def create_run_directory(base_dir: Path | None = None) -> Path:
    base_dir = base_dir or Path(os.environ.get("PARAKEET_OUTPUT_DIR", "outputs"))
    name = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    path = base_dir / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_result(result: TranscriptResult, run_dir: Path) -> dict[str, Path]:
    stem = _safe_stem(result.source_name)
    files: dict[str, Path] = {}
    text_path = run_dir / f"{stem}.txt"
    text_path.write_text(result.text + "\n", encoding="utf-8")
    files["txt"] = text_path

    json_path = run_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files["json"] = json_path

    csv_path = run_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_time", "end_time", "duration", "text"])
        for word in result.words:
            writer.writerow(
                [f"{word.start:.3f}", f"{word.end:.3f}", f"{word.end - word.start:.3f}", word.text]
            )
    files["csv"] = csv_path

    if result.segments:
        srt_path = run_dir / f"{stem}.srt"
        srt_path.write_text(_subtitle(result, vtt=False), encoding="utf-8")
        files["srt"] = srt_path
        vtt_path = run_dir / f"{stem}.vtt"
        vtt_path.write_text(_subtitle(result, vtt=True), encoding="utf-8")
        files["vtt"] = vtt_path
    return files


def write_bundle(results: list[TranscriptResult], run_dir: Path) -> Path:
    manifest = run_dir / "manifest.json"
    manifest.write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    return Path(archive)


def readable_summary(result: TranscriptResult) -> str:
    timing = "word and segment timestamps" if result.has_timestamps else "no timestamps"
    language = result.detected_language or "automatic detection unavailable"
    return f"{result.source_name}: {result.duration_seconds:.1f}s, {language}, {timing}"
