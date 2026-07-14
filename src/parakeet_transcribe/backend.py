from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .types import ChunkResult, ModelSpec, Segment, TranscriptionError, WordTimestamp

SAMPLE_RATE = 16000


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


def capitalize_key_phrases(phrases: Sequence[str]) -> list[str]:
    """Parakeet capitalization models expect capitalized key phrases (and full caps for abbreviations)."""

    normalized: list[str] = []
    for raw in phrases:
        phrase = " ".join(str(raw).split())
        if not phrase:
            continue
        if phrase.isupper() and len(phrase) <= 6:
            normalized.append(phrase)
        else:
            normalized.append(phrase.title())
    return normalized


def parse_key_phrases(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    parts = re.split(r"[\n,;]+", str(raw))
    return capitalize_key_phrases([part.strip() for part in parts if part.strip()])


def _words_from_nemo_word_timestamps(payload: Any) -> list[WordTimestamp]:
    """Map NeMo ``timestamps=True`` word entries (start/end in seconds) into WordTimestamp."""

    if not payload:
        return []
    raw_items = payload.get("word", payload.get("words", payload)) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return []
    words: list[WordTimestamp] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("word", item.get("text", item.get("char", "")))).strip()
        start, end = item.get("start"), item.get("end")
        if text and isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            words.append(WordTimestamp(text, float(start), float(end)))
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


def chunk_result_from_hypothesis(hypothesis: Any, *, expect_timestamps: bool) -> ChunkResult:
    text = str(getattr(hypothesis, "text", hypothesis) or "").strip()
    text, detected_language = _extract_language(text)
    words: list[WordTimestamp] = []
    segments: list[Segment] = []
    if expect_timestamps:
        timestamp = getattr(hypothesis, "timestamp", None)
        words = _words_from_nemo_word_timestamps(timestamp)
        if not words:
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
        self._decoding_fingerprint: tuple[tuple[str, ...], float] | None = None

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

    def configure_decoding(self, key_phrases: Sequence[str], boost_alpha: float = 1.0) -> None:
        phrases = capitalize_key_phrases(key_phrases)
        alpha = float(boost_alpha)
        fingerprint = (tuple(phrases), alpha)
        self._key_phrases = phrases
        self._boost_alpha = alpha
        if self.model is None:
            return
        if self._decoding_fingerprint == fingerprint:
            return
        try:
            from omegaconf import OmegaConf, open_dict
        except ImportError as exc:  # pragma: no cover
            raise TranscriptionError("omegaconf is required for NeMo decoding configuration.") from exc

        decoding_cfg = OmegaConf.create(
            OmegaConf.to_container(self.model.cfg.decoding, resolve=True)
        )
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy_batch"
            greedy = decoding_cfg.setdefault("greedy", {})
            greedy["use_cuda_graph_decoder"] = True
            greedy["loop_labels"] = True
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
        try:
            self.model.change_decoding_strategy(decoding_cfg)
        except Exception as exc:
            raise_if_triton_compiler_error(exc)
            raise TranscriptionError(f"Failed to configure NeMo decoding strategy: {exc}") from exc
        self._decoding_fingerprint = fingerprint
        self._apply_inference_dtype()

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
        try:
            hypotheses = self.model.transcribe(
                path_list,
                timestamps=bool(timestamps),
                batch_size=max(1, int(batch_size)),
            )
        except Exception as exc:
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
