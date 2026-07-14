from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .backend import TransformersASRBackend, is_cuda_oom, max_new_tokens_for_audio
from .chunking import merge_text, merge_words, segments_from_words, split_audio
from .diarization import diarize_transcript
from .media import prepare_audio
from .models import get_model
from .postprocess import apply_postprocess
from .types import CancelledError, ChunkResult, TranscriptionError, TranscriptResult
from .youtube import download_youtube_audio

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]

MAX_BATCH_SIZE = 16


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
        groups = [chunks[index : index + batch_size] for index in range(0, total, batch_size)]
        if not groups:
            return results

        def prepare(group: Sequence[Any]) -> tuple[Any, int]:
            audio = [chunk.samples for chunk in group]
            return backend.prepare_inputs(audio, language=language), max_new_tokens_for_audio(audio)

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending: Future[tuple[Any, int]] | None = pool.submit(prepare, groups[0])
            chunk_offset = 0
            for group_index, group in enumerate(groups):
                if cancel():
                    raise CancelledError("Transcription cancelled before publishing outputs.")
                progress(
                    progress_base + progress_span * chunk_offset / max(total, 1),
                    f"Transcribing chunk {chunk_offset + 1}/{total}",
                )
                assert pending is not None
                inputs, token_budget = pending.result()
                if group_index + 1 < len(groups):
                    pending = pool.submit(prepare, groups[group_index + 1])
                else:
                    pending = None
                try:
                    results.extend(backend.generate_from_inputs(inputs, max_new_tokens=token_budget))
                except Exception as exc:
                    if is_cuda_oom(exc):
                        raise TranscriptionError("CUDA out of memory") from exc
                    raise
                chunk_offset += len(group)
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
        diarize: bool = False,
        summarize: bool = False,
        redact_pii: bool = False,
        clean_format: bool = False,
    ) -> TranscriptResult:
        warnings: list[str] = []
        # Prefer shorter chunks with the requested batch for higher GPU occupancy; fall back under OOM.
        attempts = ((60, batch_size), (120, max(1, batch_size // 2)), (60, 1), (30, 1))
        error: Exception | None = None
        chunk_results: list[ChunkResult] = []
        chunks: list[Any] = []
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
                    progress_span * 0.9,
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
        result = TranscriptResult(
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
        if diarize:
            progress(progress_base + progress_span * 0.92, "Running local speaker diarization")
            result = diarize_transcript(result, prepared.samples, prepared.sample_rate)
        if summarize or redact_pii or clean_format:
            progress(progress_base + progress_span * 0.96, "Applying transcript post-processing")
            result = apply_postprocess(
                result,
                summarize=summarize,
                redact_pii=redact_pii,
                clean_format=clean_format,
            )
        return result

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
        diarize: bool = False,
        summarize: bool = False,
        redact_pii: bool = False,
        clean_format: bool = False,
    ) -> list[TranscriptResult]:
        if not paths:
            raise TranscriptionError("Upload at least one audio or video file.")
        if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
            raise TranscriptionError(f"Batch size must be between 1 and {MAX_BATCH_SIZE}.")
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
                diarize=diarize,
                summarize=summarize,
                redact_pii=redact_pii,
                clean_format=clean_format,
            )
            result.runtime = {
                **result.runtime,
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
        diarize: bool = False,
        summarize: bool = False,
        redact_pii: bool = False,
        clean_format: bool = False,
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
            diarize=diarize,
            summarize=summarize,
            redact_pii=redact_pii,
            clean_format=clean_format,
        )
        result = results[0]
        result.source_name = download.source_name
        result.runtime.update({"source_type": "youtube", "source_url": download.webpage_url})
        return results
