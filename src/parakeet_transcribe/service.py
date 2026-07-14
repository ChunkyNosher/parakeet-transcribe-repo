from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .backend import NeMoASRBackend, is_cuda_oom, parse_key_phrases
from .chunking import merge_segments, merge_text, merge_words, split_audio
from .diarization import diarize_transcript
from .media import prepare_audio
from .models import DEFAULT_MODEL_KEY, get_model
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


def _write_chunk_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="FLOAT")
    return path


class TranscriptionService:
    def __init__(self, cache_dir: Path = Path("model_cache/huggingface")) -> None:
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.cache_dir))
        self._backend: NeMoASRBackend | None = None
        self._model_key: str | None = None

    def unload(self) -> str:
        if self._backend is None:
            return "No model is loaded."
        self._backend.unload()
        self._backend = None
        self._model_key = None
        return "Model unloaded and CUDA cache released."

    def warm_default_model(self, model_key: str = DEFAULT_MODEL_KEY) -> str:
        """Load the default (or given) model into VRAM so the first job skips cold start."""

        spec = get_model(model_key)
        print(f"Warming up {spec.model_id}...", flush=True)
        try:
            backend = self._get_backend(model_key)
            backend.load()
        except Exception as exc:
            message = f"Model warm-up failed for {spec.model_id}: {exc}"
            print(message, flush=True)
            return message
        message = f"Warmed up {spec.model_id}."
        print(message, flush=True)
        return message

    def _get_backend(self, model_key: str) -> NeMoASRBackend:
        if self._backend is not None and self._model_key == model_key:
            return self._backend
        self.unload()
        self._backend = NeMoASRBackend(get_model(model_key))
        self._model_key = model_key
        return self._backend

    def _transcribe_chunk_groups(
        self,
        backend: NeMoASRBackend,
        chunks: Sequence[Any],
        batch_size: int,
        *,
        timestamps: bool,
        sample_rate: int,
        chunk_dir: Path,
        progress: ProgressCallback,
        cancel: CancelCheck,
        progress_base: float,
        progress_span: float,
    ) -> list[ChunkResult]:
        results: list[ChunkResult] = []
        total = len(chunks)
        groups = [chunks[index : index + batch_size] for index in range(0, total, batch_size)]
        chunk_offset = 0
        for group in groups:
            if cancel():
                raise CancelledError("Transcription cancelled before publishing outputs.")
            progress(
                progress_base + progress_span * chunk_offset / max(total, 1),
                f"Transcribing chunk {chunk_offset + 1}/{total}",
            )
            paths: list[Path] = []
            for local_index, chunk in enumerate(group):
                path = chunk_dir / f"chunk-{chunk_offset + local_index:05d}.wav"
                _write_chunk_wav(path, chunk.samples, sample_rate)
                paths.append(path)
            try:
                results.extend(
                    backend.transcribe_paths(
                        paths,
                        timestamps=timestamps,
                        batch_size=len(paths),
                    )
                )
            except Exception as exc:
                if is_cuda_oom(exc):
                    raise TranscriptionError("CUDA out of memory") from exc
                raise
            chunk_offset += len(group)
        return results

    def _transcribe_prepared(
        self,
        backend: NeMoASRBackend,
        prepared: Any,
        *,
        batch_size: int,
        language: str,
        key_phrases: Sequence[str],
        boost_alpha: float,
        progress: ProgressCallback,
        cancel: CancelCheck,
        progress_base: float,
        progress_span: float,
        work_dir: Path,
        diarize: bool = False,
        summarize: bool = False,
        redact_pii: bool = False,
        clean_format: bool = False,
    ) -> TranscriptResult:
        del language  # NeMo Parakeet auto-detects; Nemotron language prompting is model-specific.
        warnings: list[str] = []
        timestamps = backend.spec.capabilities.timestamps
        backend.configure_decoding(key_phrases, boost_alpha)

        chunk_results: list[ChunkResult] = []
        chunks: list[Any] = []
        used_chunking = False

        if cancel():
            raise CancelledError("Transcription cancelled before publishing outputs.")
        progress(progress_base, f"Transcribing {prepared.source_path.name} (NeMo long-form)")
        try:
            chunk_results = backend.transcribe_paths(
                [prepared.canonical_path],
                timestamps=timestamps,
                batch_size=1,
            )
        except TranscriptionError as exc:
            if str(exc) != "CUDA out of memory":
                raise
            used_chunking = True
            warnings.append(
                "Full-file NeMo long-form hit CUDA memory pressure; falling back to chunked transcription."
            )
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            used_chunking = True
            warnings.append(
                "Full-file NeMo long-form hit CUDA memory pressure; falling back to chunked transcription."
            )
        else:
            if not chunk_results:
                raise TranscriptionError(f"{prepared.source_path.name} produced an empty transcript.")

        if used_chunking:
            attempts = ((60, batch_size), (120, max(1, batch_size // 2)), (60, 1), (30, 1))
            error: Exception | None = None
            chunk_dir = work_dir / "chunks"
            for attempt_index, (chunk_seconds, effective_batch) in enumerate(attempts):
                chunks = split_audio(
                    prepared.samples,
                    prepared.sample_rate,
                    prepared.source_path.name,
                    chunk_seconds=chunk_seconds,
                )
                try:
                    chunk_results = self._transcribe_chunk_groups(
                        backend,
                        chunks,
                        effective_batch,
                        timestamps=timestamps,
                        sample_rate=prepared.sample_rate,
                        chunk_dir=chunk_dir,
                        progress=progress,
                        cancel=cancel,
                        progress_base=progress_base,
                        progress_span=progress_span * 0.9,
                    )
                    if attempt_index or used_chunking:
                        warnings.append(
                            f"Recovered using {chunk_seconds}s chunks and batch size {effective_batch}."
                        )
                    break
                except TranscriptionError as exc:
                    error = exc
                    if str(exc) != "CUDA out of memory" or attempt_index == len(attempts) - 1:
                        raise
            else:  # pragma: no cover - defensive
                raise error or TranscriptionError("Transcription did not return a result.")

        if used_chunking and chunks:
            text = ""
            language_result: str | None = None
            for item in chunk_results:
                text = merge_text(text, item.text)
                language_result = language_result or item.detected_language
            words = merge_words(
                (chunk, result.words) for chunk, result in zip(chunks, chunk_results, strict=True)
            )
            segments = merge_segments(
                (chunk, result.segments) for chunk, result in zip(chunks, chunk_results, strict=True)
            )
        else:
            first = chunk_results[0]
            text = first.text
            language_result = first.detected_language
            words = list(first.words)
            segments = list(first.segments)

        if not text:
            raise TranscriptionError(f"{prepared.source_path.name} produced an empty transcript.")
        warnings_out = list(warnings)
        if words and not segments:
            warnings_out.append(
                "NeMo returned word timestamps but no native segment cues; SRT/VTT preview will be empty."
            )
        result = TranscriptResult(
            schema_version="1.0",
            source_name=prepared.source_path.name,
            duration_seconds=prepared.duration_seconds,
            model_id=backend.spec.model_id,
            text=text,
            detected_language=language_result,
            words=words,
            segments=segments,
            warnings=warnings_out,
            runtime={
                "backend": "nemo",
                "longform_attention": not used_chunking,
                "key_phrase_count": len(list(key_phrases)),
                "boost_alpha": float(boost_alpha),
                "segment_source": "nemo_native" if segments else "none",
                "preview_audio_path": str(prepared.canonical_path),
            },
        )
        if diarize:
            progress(progress_base + progress_span * 0.92, "Running speaker diarization")
            result = diarize_transcript(
                result,
                prepared.samples,
                prepared.sample_rate,
                audio_path=prepared.canonical_path,
                release_vram=backend.unload,
            )
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
        key_phrases: Sequence[str] | str | None = None,
        boost_alpha: float = 1.0,
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
        phrases = (
            parse_key_phrases(key_phrases)
            if isinstance(key_phrases, str) or key_phrases is None
            else list(key_phrases)
        )
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
                key_phrases=phrases,
                boost_alpha=float(boost_alpha),
                progress=progress,
                cancel=cancel,
                progress_base=base,
                progress_span=span,
                work_dir=media_dir / str(index),
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
        key_phrases: Sequence[str] | str | None = None,
        boost_alpha: float = 1.0,
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
            key_phrases=key_phrases,
            boost_alpha=boost_alpha,
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
