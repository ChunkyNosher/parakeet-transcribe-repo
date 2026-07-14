from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from .types import ChunkResult, ModelSpec, TranscriptionError, WordTimestamp

SAMPLE_RATE = 16000


def _token_piece(text: str) -> str:
    """Normalize tokenizer fragments for alignment (drop SentencePiece markers and spaces)."""

    return text.replace("▁", "").replace("Ġ", "").replace(" ", "")


def _torch_runtime() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - installation error
        raise TranscriptionError("PyTorch is not installed. Run `uv sync` before launching the app.") from exc


def _words_from_timestamp_payload(payload: Any, transcript: str) -> list[WordTimestamp]:
    """Join documented TDT token fragments into timestamped words."""

    if not payload:
        return []
    raw_items = payload.get("word", payload.get("words", payload)) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return []
    tokens: list[WordTimestamp] = []
    for item in raw_items:
        if isinstance(item, dict):
            text = str(item.get("word", item.get("text", item.get("token", "")))).strip()
            start, end = item.get("start"), item.get("end")
        elif isinstance(item, (tuple, list)) and len(item) >= 3:
            text, start, end = str(item[0]).strip(), item[1], item[2]
        else:
            continue
        if text and isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            tokens.append(WordTimestamp(text, float(start), float(end)))

    visible_words = list(re.finditer(r"\S+", transcript))
    if not tokens or not visible_words:
        return []
    compact_tokens = "".join(_token_piece(token.text) for token in tokens)
    compact_transcript = "".join(match.group(0) for match in visible_words)
    if compact_tokens != compact_transcript:
        return []

    words: list[WordTimestamp] = []
    token_index = 0
    token_offset = 0
    for match in visible_words:
        target_length = len(match.group(0))
        consumed = 0
        start = tokens[token_index].start
        end = start
        while consumed < target_length and token_index < len(tokens):
            token_length = len(_token_piece(tokens[token_index].text))
            if token_length == 0:
                token_index += 1
                token_offset = 0
                continue
            available = token_length - token_offset
            take = min(available, target_length - consumed)
            consumed += take
            end = tokens[token_index].end
            token_offset += take
            if token_offset >= token_length:
                token_index += 1
                token_offset = 0
        if consumed != target_length:
            return []
        words.append(WordTimestamp(match.group(0), start, end))
    return words


def _extract_language(text: str) -> tuple[str, str | None]:
    match = re.search(r"\s*<([a-z]{2,3}(?:-[A-Z]{2})?)>\s*$", text)
    if not match:
        return text.strip(), None
    return text[: match.start()].strip(), match.group(1)


def max_new_tokens_for_audio(audio: Sequence[np.ndarray], sample_rate: int = SAMPLE_RATE) -> int:
    """Bound decoder steps from the longest clip in the batch (seconds × generous ASR rate)."""

    max_seconds = max((len(samples) / sample_rate for samples in audio), default=1.0)
    return min(4096, max(64, int(max_seconds * 25) + 32))


def _compile_enabled() -> bool:
    return os.environ.get("PARAKEET_TORCH_COMPILE", "").strip().lower() in {"1", "true", "yes", "on"}


class TransformersASRBackend:
    """One in-process Transformers backend for NVIDIA TDT and RNNT checkpoints."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.processor: Any | None = None
        self.model: Any | None = None
        self.device: str | None = None
        self._compiled = False

    def load(self) -> None:
        if self.model is not None:
            return
        torch = _torch_runtime()
        try:
            from transformers import AutoModelForRNNT, AutoModelForTDT, AutoProcessor
        except ImportError as exc:  # pragma: no cover - installation error
            raise TranscriptionError(
                "Transformers 5.13.1 is required for the selected NVIDIA model."
            ) from exc

        if not torch.cuda.is_available():
            raise TranscriptionError(
                "CUDA is unavailable. This app requires the CUDA PyTorch wheel; run `uv run parakeet-transcribe doctor`."
            )
        self.device = "cuda"
        dtype = torch.float16
        try:
            self.processor = AutoProcessor.from_pretrained(self.spec.model_id)
            model_class = AutoModelForTDT if self.spec.model_class == "tdt" else AutoModelForRNNT
            self.model = model_class.from_pretrained(self.spec.model_id, dtype=dtype).eval().to(self.device)
            if _compile_enabled() and not self._compiled:
                self.model.generate = torch.compile(
                    self.model.generate, mode="reduce-overhead", fullgraph=False
                )
                self._compiled = True
        except Exception as exc:
            raise_if_triton_compiler_error(exc)
            raise

    def unload(self) -> None:
        if self.model is None:
            return
        torch = _torch_runtime()
        self.model.to("cpu")
        self.model = None
        self.processor = None
        self.device = None
        self._compiled = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_inputs(self, audio: Sequence[np.ndarray], *, language: str = "auto") -> Any:
        """Run the feature processor on CPU so the next batch can prefetch during GPU generate."""

        self.load()
        assert self.processor is not None
        processor_kwargs: dict[str, Any] = {
            "sampling_rate": SAMPLE_RATE,
            "return_tensors": "pt",
            "padding": True,
            "return_attention_mask": True,
        }
        if self.spec.model_class == "rnnt":
            processor_kwargs["language"] = language
        return self.processor(list(audio), **processor_kwargs)

    def generate_from_inputs(self, inputs: Any, *, max_new_tokens: int) -> list[ChunkResult]:
        self.load()
        assert self.model is not None and self.processor is not None and self.device is not None
        torch = _torch_runtime()
        inputs = inputs.to(self.device, dtype=torch.float16)
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    max_new_tokens=max_new_tokens,
                )
        except Exception as exc:
            raise_if_triton_compiler_error(exc)
            raise

        sequences = output.sequences
        durations = getattr(output, "durations", None)
        if getattr(sequences, "ndim", 0) == 1:
            sequences = sequences.unsqueeze(0)
            durations = durations.unsqueeze(0) if durations is not None else None
        if self.spec.model_class == "tdt":
            decoded_batch, timestamp_batches = self.processor.decode(
                sequences,
                durations=durations,
                skip_special_tokens=True,
            )
            if isinstance(decoded_batch, str):
                decoded_batch = [decoded_batch]
        else:
            decoded_batch = self.processor.batch_decode(sequences, skip_special_tokens=False)
            timestamp_batches = [None] * len(decoded_batch)
        results: list[ChunkResult] = []
        for text, timestamp_payload in zip(decoded_batch, timestamp_batches, strict=True):
            text = str(text)
            text, detected_language = _extract_language(text)
            words = (
                _words_from_timestamp_payload(timestamp_payload, text)
                if self.spec.capabilities.timestamps
                else []
            )
            if self.spec.capabilities.timestamps and not words:
                raise TranscriptionError(
                    "Parakeet returned no usable timestamp payload. The app will not fabricate subtitle timing."
                )
            results.append(ChunkResult(text=text, words=words, detected_language=detected_language))
        return results

    def transcribe(self, audio: Sequence[np.ndarray], *, language: str = "auto") -> list[ChunkResult]:
        inputs = self.prepare_inputs(audio, language=language)
        return self.generate_from_inputs(inputs, max_new_tokens=max_new_tokens_for_audio(audio))


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
