from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .backend import TransformersASRBackend, is_cuda_oom
from .chunking import merge_text, merge_words, segments_from_words, split_audio
from .media import prepare_audio
from .models import get_model
from .types import CancelledError, ChunkResult, TranscriptionError, TranscriptResult
from .youtube import download_youtube_audio

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


def _noop_progress(_: float, __: str) -> None:
    return None


def _not_cancelled() -> bool:
    return False


class TranscriptionService:
    def __init__(self, cache_dir: Path = Path("model_cache/huggingface")) -> None:
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.cache_dir))
        self._backend: TransformersASRBackend | None = None
        self._model_key: str | None = None

    def unload(self) -> str:
        if self._backend is None:
            return "No model is loaded."
        self._backend.unload()
        self._backend = None
        self._model_key = None
        return "Model unloaded and CUDA cache released."

    def _get_backend(self, model_key: str) -> TransformersASRBackend:
        if self._backend is not None and self._model_key == model_key:
            return self._backend
        self.unload()
        self._backend = TransformersASRBackend(get_model(model_key))
        self._model_key = model_key
        return self._backend

    def _transcribe_chunk_groups(
        self,
        backend: TransformersASRBackend,
        chunks: Sequence[Any],
        batch_size: int,
        language: str,
        progress: ProgressCallback,
        cancel: CancelCheck,
        progress_base: float,
        progress_span: float,
    ) -> list[ChunkResult]:
        results: list[ChunkResult] = []
        total = len(chunks)
        for index in range(0, total, batch_size):
            if cancel():
                raise CancelledError("Transcription cancelled before publishing outputs.")
            group = chunks[index : index + batch_size]
            progress(
                progress_base + progress_span * index / max(total, 1),
                f"Transcribing chunk {index + 1}/{total}",
            )
            try:
                results.extend(backend.transcribe([chunk.samples for chunk in group], language=language))
            except Exception as exc:
                if is_cuda_oom(exc):
                    raise TranscriptionError("CUDA out of memory") from exc
                raise
        return results

    def _transcribe_prepared(
        self,
        backend: TransformersASRBackend,
        prepared: Any,
        *,
        batch_size: int,
        language: str,
        progress: ProgressCallback,
        cancel: CancelCheck,
        progress_base: float,
        progress_span: float,
    ) -> TranscriptResult:
        warnings: list[str] = []
        attempts = ((120, batch_size), (120, 1), (60, 1))
        error: Exception | None = None
        for attempt_index, (chunk_seconds, effective_batch) in enumerate(attempts):
            chunks = split_audio(
                prepared.samples, prepared.sample_rate, prepared.source_path.name, chunk_seconds=chunk_seconds
            )
            try:
                chunk_results = self._transcribe_chunk_groups(
                    backend,
                    chunks,
                    effective_batch,
                    language,
                    progress,
                    cancel,
                    progress_base,
                    progress_span,
                )
                if attempt_index:
                    warnings.append(
                        f"Recovered from CUDA memory pressure using {chunk_seconds}s chunks and batch size {effective_batch}."
                    )
                break
            except TranscriptionError as exc:
                error = exc
                if str(exc) != "CUDA out of memory" or attempt_index == len(attempts) - 1:
                    raise
        else:  # pragma: no cover - defensive
            raise error or TranscriptionError("Transcription did not return a result.")

        text = ""
        language_result: str | None = None
        for item in chunk_results:
            text = merge_text(text, item.text)
            language_result = language_result or item.detected_language
        words = merge_words((chunk, result.words) for chunk, result in zip(chunks, chunk_results, strict=True))
        if not text:
            raise TranscriptionError(f"{prepared.source_path.name} produced an empty transcript.")
        return TranscriptResult(
            schema_version="1.0",
            source_name=prepared.source_path.name,
            duration_seconds=prepared.duration_seconds,
            model_id=backend.spec.model_id,
            text=text,
            detected_language=language_result,
            words=words,
            segments=segments_from_words(words),
            warnings=warnings,
        )

    def transcribe_files(
        self,
        paths: Sequence[str],
        *,
        model_key: str,
        language: str = "auto",
        batch_size: int = 1,
        work_dir: Path,
        progress: ProgressCallback | None = None,
        cancel: CancelCheck | None = None,
    ) -> list[TranscriptResult]:
        if not paths:
            raise TranscriptionError("Upload at least one audio or video file.")
        if batch_size < 1 or batch_size > 4:
            raise TranscriptionError("Batch size must be between 1 and 4.")
        spec = get_model(model_key)
        if not spec.capabilities.timestamps and language == "":
            language = "auto"
        progress = progress or _noop_progress
        cancel = cancel or _not_cancelled
        backend = self._get_backend(model_key)
        started = time.perf_counter()
        results: list[TranscriptResult] = []
        media_dir = work_dir / ".work"
        for index, path in enumerate(paths):
            if cancel():
                raise CancelledError("Transcription cancelled before publishing outputs.")
            base = index / len(paths)
            span = 1 / len(paths)
            progress(base, f"Normalizing {Path(path).name}")
            prepared = prepare_audio(path, media_dir / str(index))
            result = self._transcribe_prepared(
                backend,
                prepared,
                batch_size=batch_size,
                language=language or "auto",
                progress=progress,
                cancel=cancel,
                progress_base=base,
                progress_span=span,
            )
            result.runtime = {
                "model_key": model_key,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "requested_batch_size": batch_size,
            }
            results.append(result)
        progress(1.0, "Finalizing downloads")
        return results

    def transcribe_youtube(
        self,
        url: str,
        *,
        model_key: str,
        language: str = "auto",
        batch_size: int = 1,
        work_dir: Path,
        progress: ProgressCallback | None = None,
        cancel: CancelCheck | None = None,
    ) -> list[TranscriptResult]:
        progress = progress or _noop_progress
        cancel = cancel or _not_cancelled
        if cancel():
            raise CancelledError("Transcription cancelled before downloading YouTube audio.")
        progress(0.0, "Downloading YouTube audio")
        download = download_youtube_audio(url, work_dir / ".youtube")
        if cancel():
            raise CancelledError("Transcription cancelled after downloading YouTube audio.")
        results = self.transcribe_files(
            [str(download.path)],
            model_key=model_key,
            language=language,
            batch_size=batch_size,
            work_dir=work_dir,
            progress=progress,
            cancel=cancel,
        )
        result = results[0]
        result.source_name = download.source_name
        result.runtime.update({"source_type": "youtube", "source_url": download.webpage_url})
        return results
