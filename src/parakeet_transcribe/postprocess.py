from __future__ import annotations

import re
from dataclasses import replace

from .types import Segment, TranscriptResult, WordTimestamp

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def redact_pii_text(text: str) -> str:
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _SSN.sub("[REDACTED_SSN]", text)
    text = _CARD.sub("[REDACTED_CARD]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    return text


def clean_format_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)

    def _capitalize(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2).upper()

    if text:
        text = text[0].upper() + text[1:]
    text = re.sub(r"([.!?]\s+)([a-z])", _capitalize, text)
    return text


def build_chapters(segments: list[Segment], *, gap_seconds: float = 8.0) -> list[dict[str, object]]:
    if not segments:
        return []
    chapters: list[dict[str, object]] = []
    current = [segments[0]]
    for segment in segments[1:]:
        gap = segment.start - current[-1].end
        if gap >= gap_seconds:
            chapters.append(_chapter_from_segments(current))
            current = [segment]
        else:
            current.append(segment)
    if current:
        chapters.append(_chapter_from_segments(current))
    return chapters


def _chapter_from_segments(segments: list[Segment]) -> dict[str, object]:
    title = segments[0].text.strip()
    if len(title) > 80:
        title = title[:77].rstrip() + "..."
    return {
        "start": segments[0].start,
        "end": segments[-1].end,
        "title": title or "Chapter",
    }


def build_summary(result: TranscriptResult, *, max_sentences: int = 5) -> str:
    if result.chapters:
        lines = [f"- {chapter['title']}" for chapter in result.chapters[:max_sentences]]
        return "Extractive summary from chapter openings:\n" + "\n".join(lines)
    sentences = re.split(r"(?<=[.!?])\s+", result.text.strip())
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return ""
    selected = sentences[:max_sentences]
    return "Extractive summary:\n" + "\n".join(f"- {sentence}" for sentence in selected)


def apply_postprocess(
    result: TranscriptResult,
    *,
    summarize: bool = False,
    redact_pii: bool = False,
    clean_format: bool = False,
) -> TranscriptResult:
    text = result.text
    words = list(result.words)
    segments = list(result.segments)
    warnings = list(result.warnings)
    chapters = list(result.chapters)
    summary = result.summary
    schema = result.schema_version

    if clean_format:
        text = clean_format_text(text)
        segments = [
            Segment(clean_format_text(segment.text), segment.start, segment.end, segment.speaker)
            for segment in segments
        ]
        warnings.append("Applied light clean/smart formatting to transcript text.")
        schema = "1.1"

    if redact_pii:
        text = redact_pii_text(text)
        words = [
            WordTimestamp(
                redact_pii_text(word.text),
                word.start,
                word.end,
                word.speaker,
                word.confidence,
            )
            for word in words
        ]
        segments = [
            Segment(redact_pii_text(segment.text), segment.start, segment.end, segment.speaker)
            for segment in segments
        ]
        warnings.append("Applied local regex PII redaction to transcript text.")
        schema = "1.1"

    if summarize:
        chapters = build_chapters(segments)
        summary = build_summary(replace(result, text=text, segments=segments, chapters=chapters))
        warnings.append("Added extractive local summary and chapters (no cloud LLM).")
        schema = "1.1"

    return replace(
        result,
        text=text,
        words=words,
        segments=segments,
        warnings=warnings,
        summary=summary,
        chapters=chapters,
        schema_version=schema,
        runtime={
            **result.runtime,
            "postprocess": {
                "summarize": summarize,
                "redact_pii": redact_pii,
                "clean_format": clean_format,
            },
        },
    )
