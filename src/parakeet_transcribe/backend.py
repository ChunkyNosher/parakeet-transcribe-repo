from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .types import CancelledError, ChunkResult, ModelSpec, Segment, TranscriptionError, WordTimestamp

SAMPLE_RATE = 16000
WORD_CONFIDENCE_AGGREGATION_ERROR = "Something went wrong with word-level confidence aggregation."
NEMO_BUFFERED_LEFT_CONTEXT_SECONDS = 10.0
NEMO_BUFFERED_CHUNK_SECONDS = 10.0
NEMO_BUFFERED_RIGHT_CONTEXT_SECONDS = 5.0


def _torch_runtime() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - installation error
        raise TranscriptionError(
            "PyTorch is not installed. Run inference inside the Docker Compose Linux GPU container."
        ) from exc


def _extract_language(text: str) -> tuple[str, str | None]:
    match = re.search(r"\s*<([a-z]{2,3}(?:-[A-Z]{2})?)>\s*$", text)
    if not match:
        return text.strip(), None
    return text[: match.start()].strip(), match.group(1)


def capitalize_key_phrases(phrases: Sequence[str], *, casing: str = "title") -> list[str]:
    """Normalize key phrases for NeMo GPU-PB boosting.

    ``casing="title"`` (default) title-cases phrases and keeps short all-caps
    abbreviations, which Parakeet capitalization models expect. ``casing="lower"``
    lowercases phrases for lowercase-vocab models (e.g. Parakeet 1.1B) whose
    SentencePiece tokenizer has no uppercase tokens.
    """

    normalized: list[str] = []
    for raw in phrases:
        phrase = " ".join(str(raw).split())
        if not phrase:
            continue
        if casing == "lower":
            normalized.append(phrase.lower())
        elif phrase.isupper() and len(phrase) <= 6:
            normalized.append(phrase)
        else:
            normalized.append(phrase.title())
    return normalized


def parse_key_phrases(raw: str | None, *, casing: str = "title") -> list[str]:
    if not raw or not str(raw).strip():
        return []
    parts = re.split(r"[\n,;]+", str(raw))
    return capitalize_key_phrases([part.strip() for part in parts if part.strip()], casing=casing)


def _words_from_nemo_word_timestamps(
    payload: Any,
    confidences: Sequence[float] | None = None,
) -> list[WordTimestamp]:
    """Map NeMo ``timestamps=True`` word entries (start/end in seconds) into WordTimestamp."""

    if not payload:
        return []
    raw_items = payload.get("word", payload.get("words", payload)) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return []
    words: list[WordTimestamp] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        text = str(item.get("word", item.get("text", item.get("char", "")))).strip()
        start, end = item.get("start"), item.get("end")
        if text and isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            confidence: float | None = None
            raw_conf = item.get("confidence", item.get("score"))
            if isinstance(raw_conf, (int, float)):
                confidence = float(raw_conf)
            elif confidences is not None and index < len(confidences):
                value = confidences[index]
                if isinstance(value, (int, float)):
                    confidence = float(value)
            words.append(WordTimestamp(text, float(start), float(end), confidence=confidence))
    return words


def _segments_from_nemo_segment_timestamps(payload: Any) -> list[Segment]:
    """Map NeMo ``timestamps=True`` segment entries (start/end in seconds) into Segment cues."""

    if not payload or not isinstance(payload, dict):
        return []
    raw_items = payload.get("segment", payload.get("segments"))
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return []
    segments: list[Segment] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("segment", item.get("text", item.get("word", "")))).strip()
        start, end = item.get("start"), item.get("end")
        if text and isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            segments.append(Segment(text, float(start), float(end)))
    return segments


def _hypothesis_word_confidences(hypothesis: Any) -> list[float] | None:
    raw = getattr(hypothesis, "word_confidence", None)
    if raw is None:
        return None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [float(value) for value in raw if isinstance(value, (int, float))]
        return values or None
    return None


def chunk_result_from_hypothesis(hypothesis: Any, *, expect_timestamps: bool) -> ChunkResult:
    text = str(getattr(hypothesis, "text", hypothesis) or "").strip()
    text, detected_language = _extract_language(text)
    words: list[WordTimestamp] = []
    segments: list[Segment] = []
    if expect_timestamps:
        timestamp = getattr(hypothesis, "timestamp", None)
        words = _words_from_nemo_word_timestamps(timestamp, _hypothesis_word_confidences(hypothesis))
        if text and not words:
            raise TranscriptionError(
                "NeMo returned no usable word timestamps. The app will not fabricate subtitle timing."
            )
        segments = _segments_from_nemo_segment_timestamps(timestamp)
    return ChunkResult(text=text, words=words, detected_language=detected_language, segments=segments)


class NeMoASRBackend:
    """In-process NeMo ASR backend for NVIDIA Parakeet TDT and Nemotron RNNT checkpoints."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.model: Any | None = None
        self._key_phrases: list[str] = []
        self._boost_alpha: float = 1.0
        self._word_confidence_enabled = True
        self._word_confidence_fallback_used = False
        self._decoding_fingerprint: tuple[tuple[str, ...], float, bool, bool, bool] | None = None

    @property
    def word_confidence_fallback_used(self) -> bool:
        return self._word_confidence_fallback_used

    def load(self) -> None:
        if self.model is not None:
            return
        torch = _torch_runtime()
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:  # pragma: no cover - installation / platform error
            raise TranscriptionError(
                "NeMo ASR is required. Use Docker Compose on a Linux GPU host "
                "(`docker compose up --build`); native Windows inference is not supported."
            ) from exc

        if not torch.cuda.is_available():
            raise TranscriptionError(
                "CUDA is unavailable. Run inside the Docker Compose Linux GPU container."
            )
        try:
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.spec.model_id)
            self.model.eval()
            self.model.to(torch.device("cuda"))
            # Attention/decoding rebuild modules in FP32; cast precision after those calls.
            self._configure_longform_attention()
            self.configure_decoding(self._key_phrases, self._boost_alpha)
            self._apply_inference_dtype()
        except Exception as exc:
            self.model = None
            self._decoding_fingerprint = None
            raise_if_triton_compiler_error(exc)
            raise

    def _apply_inference_dtype(self) -> None:
        """Cast weights to FP16 after any NeMo reconfiguration that may recreate FP32 modules."""

        if self.model is None:
            return
        torch = _torch_runtime()
        self.model.to(dtype=torch.float16)

    def _configure_longform_attention(self) -> None:
        assert self.model is not None
        try:
            self.model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[256, 256],
            )
            change_chunking = getattr(self.model, "change_subsampling_conv_chunking_factor", None)
            if callable(change_chunking):
                change_chunking(1)
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to enable NeMo local-attention long-form mode: {exc}"
            ) from exc

    def unload(self) -> None:
        if self.model is None:
            return
        torch = _torch_runtime()
        try:
            self.model.to("cpu")
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        self.model = None
        self._decoding_fingerprint = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def enable_longform_attention(self) -> None:
        self.load()
        self._configure_longform_attention()
        self._apply_inference_dtype()

    def configure_decoding(
        self,
        key_phrases: Sequence[str],
        boost_alpha: float = 1.0,
        *,
        streaming: bool = False,
        timestamps: bool = True,
    ) -> None:
        phrases = capitalize_key_phrases(
            key_phrases, casing="lower" if self.spec.capabilities.lowercase_vocab else "title"
        )
        alpha = float(boost_alpha)
        fingerprint = (
            tuple(phrases),
            alpha,
            self._word_confidence_enabled,
            bool(streaming),
            bool(timestamps),
        )
        self._key_phrases = phrases
        self._boost_alpha = alpha
        if self.model is None:
            return
        if self._decoding_fingerprint == fingerprint:
            return
        self._apply_decoding_configuration(
            phrases,
            alpha,
            streaming=streaming,
            timestamps=timestamps,
        )
        self._decoding_fingerprint = fingerprint

    def _apply_decoding_configuration(
        self,
        phrases: Sequence[str],
        alpha: float,
        *,
        streaming: bool = False,
        timestamps: bool = True,
    ) -> None:
        assert self.model is not None
        try:
            from omegaconf import OmegaConf, open_dict
        except ImportError as exc:  # pragma: no cover
            raise TranscriptionError("omegaconf is required for NeMo decoding configuration.") from exc

        decoding_cfg = OmegaConf.create(
            OmegaConf.to_container(self.model.cfg.decoding, resolve=True)
        )
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy_batch"
            # Offline NeMo timestamps are reconstructed from preserved alignments. NeMo's
            # stateful RNNT/TDT streaming path instead carries token durations between
            # bounded encoder windows, so retaining frame alignments would defeat the
            # bounded-memory design.
            decoding_cfg.preserve_alignments = not streaming
            decoding_cfg.compute_timestamps = bool(timestamps and not streaming)
            decoding_cfg.tdt_include_token_duration = bool(timestamps and streaming)
            greedy = decoding_cfg.setdefault("greedy", {})
            greedy["use_cuda_graph_decoder"] = True
            greedy["loop_labels"] = True
            greedy["preserve_alignments"] = not streaming
            if phrases:
                greedy["boosting_tree"] = {
                    "key_phrases_list": list(phrases),
                    "context_score": 1.0,
                    "depth_scaling": 2.0,
                }
                greedy["boosting_tree_alpha"] = alpha
            else:
                greedy.pop("boosting_tree", None)
                greedy.pop("boosting_tree_alpha", None)
            preserve_confidence = self._word_confidence_enabled and not streaming
            decoding_cfg["confidence_cfg"] = {
                "preserve_frame_confidence": preserve_confidence,
                "preserve_token_confidence": preserve_confidence,
                "preserve_word_confidence": preserve_confidence,
                "aggregation": "min",
                "exclude_blank": True,
                "tdt_include_duration": False,
                "method_cfg": {
                    "name": "entropy",
                    "entropy_type": "tsallis",
                    "alpha": 0.33,
                    "entropy_norm": "exp",
                },
            }
        try:
            self.model.change_decoding_strategy(decoding_cfg)
        except Exception as exc:
            raise_if_triton_compiler_error(exc)
            raise TranscriptionError(f"Failed to configure NeMo decoding strategy: {exc}") from exc
        self._apply_inference_dtype()

    def _disable_word_confidence_after_aggregation_error(self) -> None:
        if not self._word_confidence_enabled:
            return
        self._word_confidence_enabled = False
        self._word_confidence_fallback_used = True
        self._decoding_fingerprint = None
        print(
            "NeMo word-confidence aggregation was inconsistent; retrying without confidence scores.",
            flush=True,
        )
        self._apply_decoding_configuration(self._key_phrases, self._boost_alpha)
        self._decoding_fingerprint = (
            tuple(self._key_phrases),
            self._boost_alpha,
            self._word_confidence_enabled,
            False,
            True,
        )

    def transcribe_streaming_audio(
        self,
        samples: Any,
        sample_rate: int,
        *,
        timestamps: bool,
        progress: Callable[[int, int], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        left_context_seconds: float = NEMO_BUFFERED_LEFT_CONTEXT_SECONDS,
        chunk_seconds: float = NEMO_BUFFERED_CHUNK_SECONDS,
        right_context_seconds: float = NEMO_BUFFERED_RIGHT_CONTEXT_SECONDS,
    ) -> ChunkResult:
        """Run NeMo's stateful RNNT/TDT buffered inference over bounded encoder windows."""

        self.load()
        assert self.model is not None
        self.configure_decoding(
            self._key_phrases,
            self._boost_alpha,
            streaming=True,
            timestamps=timestamps,
        )

        try:
            from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
            from nemo.collections.asr.parts.utils.streaming_utils import (
                ContextSize,
                StreamingBatchedAudioBuffer,
            )
            from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs
        except ImportError as exc:  # pragma: no cover - tied to the supported NeMo image
            raise TranscriptionError(
                "The installed NeMo build does not provide RNNT/TDT buffered inference utilities."
            ) from exc

        torch = _torch_runtime()
        audio = torch.as_tensor(samples, dtype=torch.float32, device="cpu")
        if audio.ndim != 1:
            raise TranscriptionError("NeMo buffered inference requires mono audio.")
        total_samples = int(audio.shape[0])
        if total_samples == 0:
            return ChunkResult(text="", words=[], detected_language=None, segments=[])
        if int(sample_rate) != int(self.model.cfg.preprocessor.sample_rate):
            raise TranscriptionError(
                f"NeMo buffered inference requires {self.model.cfg.preprocessor.sample_rate} Hz audio."
            )

        device = next(self.model.parameters()).device
        subsampling_factor = int(self.model.encoder.subsampling_factor)
        feature_stride_seconds = float(self.model.cfg.preprocessor.window_stride)
        feature_frame_samples = int(sample_rate * feature_stride_seconds)
        feature_frame_samples = max(
            subsampling_factor,
            (feature_frame_samples // subsampling_factor) * subsampling_factor,
        )
        features_per_second = 1.0 / feature_stride_seconds
        encoder_context = ContextSize(
            left=int(left_context_seconds * features_per_second / subsampling_factor),
            chunk=int(chunk_seconds * features_per_second / subsampling_factor),
            right=int(right_context_seconds * features_per_second / subsampling_factor),
        )
        encoder_frame_samples = feature_frame_samples * subsampling_factor
        sample_context = ContextSize(
            left=encoder_context.left * encoder_frame_samples,
            chunk=encoder_context.chunk * encoder_frame_samples,
            right=encoder_context.right * encoder_frame_samples,
        )
        if sample_context.chunk <= 0:
            raise TranscriptionError("NeMo buffered inference chunk size must be positive.")

        featurizer = getattr(getattr(self.model, "preprocessor", None), "featurizer", None)
        if featurizer is not None:
            featurizer.dither = 0.0
            featurizer.pad_to = 0

        try:
            decoding_computer = self.model.decoding.decoding.decoding_computer
        except AttributeError as exc:
            raise TranscriptionError(
                "The selected NeMo model does not expose stateful RNNT/TDT greedy decoding."
            ) from exc

        buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=sample_context,
            dtype=torch.float32,
            device=device,
        )
        remaining = torch.tensor([total_samples], dtype=torch.long, device=device)
        current_hypotheses: Any | None = None
        decoder_state: Any | None = None
        left_sample = 0
        right_sample = min(sample_context.chunk + sample_context.right, total_samples)
        total_chunks = max(
            1,
            (max(total_samples - sample_context.right, 0) + sample_context.chunk - 1)
            // sample_context.chunk,
        )
        completed_chunks = 0

        with torch.no_grad(), torch.inference_mode():
            while left_sample < total_samples:
                if cancel is not None and cancel():
                    raise CancelledError("Transcription cancelled before publishing outputs.")
                chunk_length = right_sample - left_sample
                is_last_for_stream = chunk_length >= int(remaining.item())
                is_last_chunk = right_sample >= total_samples
                chunk_lengths = torch.tensor(
                    [min(chunk_length, int(remaining.item()))],
                    dtype=torch.long,
                    device=device,
                )
                last_chunk_batch = torch.tensor(
                    [is_last_for_stream],
                    dtype=torch.bool,
                    device=device,
                )
                audio_chunk = audio[left_sample:right_sample].unsqueeze(0).to(device=device)
                buffer.add_audio_batch_(
                    audio_chunk,
                    audio_lengths=chunk_lengths,
                    is_last_chunk=is_last_chunk,
                    is_last_chunk_batch=last_chunk_batch,
                )

                encoder_output, encoder_output_length = self.model(
                    input_signal=buffer.samples,
                    input_signal_length=buffer.context_size_batch.total(),
                )
                encoder_output = encoder_output.transpose(1, 2)
                encoder_buffer_context = buffer.context_size.subsample(encoder_frame_samples)
                encoder_batch_context = buffer.context_size_batch.subsample(encoder_frame_samples)
                encoder_output = encoder_output[:, encoder_buffer_context.left :]
                decode_lengths = torch.where(
                    last_chunk_batch,
                    encoder_output_length - encoder_batch_context.left,
                    encoder_batch_context.chunk,
                )
                decode_result = decoding_computer(
                    x=encoder_output,
                    out_len=decode_lengths,
                    prev_batched_state=decoder_state,
                )
                if completed_chunks == 0:
                    graph_mode = getattr(decoding_computer, "cuda_graphs_mode", None)
                    print(
                        f"NeMo buffered decoder CUDA graphs mode: {graph_mode or 'disabled'}",
                        flush=True,
                    )
                # NeMo releases differ on whether disabled alignments remain as a
                # placeholder in the returned tuple.
                if len(decode_result) == 3:
                    chunk_hypotheses, _, decoder_state = decode_result
                else:
                    chunk_hypotheses, decoder_state = decode_result
                if current_hypotheses is None:
                    current_hypotheses = chunk_hypotheses
                else:
                    current_hypotheses.merge_(chunk_hypotheses)

                remaining -= chunk_lengths
                left_sample = right_sample
                right_sample = min(right_sample + sample_context.chunk, total_samples)
                completed_chunks += 1
                if progress is not None:
                    progress(completed_chunks, total_chunks)

        if current_hypotheses is None:  # pragma: no cover - non-empty audio always decodes
            return ChunkResult(text="", words=[], detected_language=None, segments=[])
        hypotheses = batched_hyps_to_hypotheses(current_hypotheses, batch_size=1)
        hypothesis = hypotheses[0]
        if timestamps:
            # NeMo 2.5 temporarily stores (decoded ids, alignments, token
            # repetitions) in ``text``; timestamp conversion must consume that
            # tuple before ``text`` is replaced by the final string.
            hypothesis = self.model.decoding.compute_rnnt_timestamps(hypothesis)
            processed_hypotheses = process_timestamp_outputs(
                [hypothesis],
                subsampling_factor=subsampling_factor,
                window_stride=feature_stride_seconds,
            )
            hypothesis = processed_hypotheses[0]
        token_ids = (
            hypothesis.y_sequence.tolist()
            if hasattr(hypothesis.y_sequence, "tolist")
            else list(hypothesis.y_sequence)
        )
        hypothesis.text = self.model.tokenizer.ids_to_text(token_ids)
        return chunk_result_from_hypothesis(
            hypothesis,
            expect_timestamps=timestamps and self.spec.capabilities.timestamps,
        )

    def transcribe_paths(
        self,
        paths: Sequence[str | Path],
        *,
        timestamps: bool,
        batch_size: int = 1,
    ) -> list[ChunkResult]:
        self.load()
        assert self.model is not None
        path_list = [str(Path(path)) for path in paths]
        if not path_list:
            return []
        def transcribe() -> Any:
            return self.model.transcribe(
                path_list,
                timestamps=bool(timestamps),
                batch_size=max(1, int(batch_size)),
                return_hypotheses=True,
            )

        try:
            hypotheses = transcribe()
        except Exception as exc:
            if self._word_confidence_enabled and WORD_CONFIDENCE_AGGREGATION_ERROR in str(exc):
                self._disable_word_confidence_after_aggregation_error()
                try:
                    hypotheses = transcribe()
                except Exception as retry_exc:
                    raise_if_triton_compiler_error(retry_exc)
                    raise
            else:
                raise_if_triton_compiler_error(exc)
                raise

        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]
        results: list[ChunkResult] = []
        for hypothesis in hypotheses:
            results.append(
                chunk_result_from_hypothesis(
                    hypothesis,
                    expect_timestamps=timestamps and self.spec.capabilities.timestamps,
                )
            )
        return results


def is_cuda_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() or "cuda error: out of memory" in str(error).lower()


def is_triton_compiler_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "c compiler" in message or "triton.knobs" in message


def raise_if_triton_compiler_error(error: BaseException) -> None:
    if not is_triton_compiler_error(error):
        return
    raise TranscriptionError(
        "Linux PyTorch needs a C compiler for Triton's CUDA helpers. "
        "Rebuild the Docker image (it installs build-essential) or install gcc and ensure CC points to it."
    ) from error
