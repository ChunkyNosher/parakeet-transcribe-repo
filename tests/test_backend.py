from types import SimpleNamespace
from unittest.mock import patch

import pytest

from parakeet_transcribe.backend import (
    _extract_language,
    _segments_from_nemo_segment_timestamps,
    _words_from_nemo_word_timestamps,
    capitalize_key_phrases,
    chunk_result_from_hypothesis,
    is_triton_compiler_error,
    parse_key_phrases,
    raise_if_triton_compiler_error,
)
from parakeet_transcribe.types import TranscriptionError


def test_nemo_word_timestamps_map_directly() -> None:
    payload = [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "world.", "start": 0.45, "end": 0.9},
    ]
    words = _words_from_nemo_word_timestamps(payload)
    assert [(word.text, word.start, word.end) for word in words] == [
        ("Hello", 0.0, 0.4),
        ("world.", 0.45, 0.9),
    ]


def test_nemo_timestamp_dict_word_key() -> None:
    payload = {"word": [{"word": "Hi", "start": 1.0, "end": 1.2}]}
    words = _words_from_nemo_word_timestamps(payload)
    assert words[0].text == "Hi"


def test_nemo_native_segments_map_from_timestamp_dict() -> None:
    payload = {
        "word": [{"word": "Hello", "start": 0.0, "end": 0.4}],
        "segment": [
            {"segment": "Hello there.", "start": 0.0, "end": 1.2},
            {"segment": "Next line.", "start": 1.5, "end": 2.0},
        ],
    }
    segments = _segments_from_nemo_segment_timestamps(payload)
    assert [(item.text, item.start, item.end) for item in segments] == [
        ("Hello there.", 0.0, 1.2),
        ("Next line.", 1.5, 2.0),
    ]


def test_chunk_result_includes_native_segments() -> None:
    hyp = SimpleNamespace(
        text="Hello there. Next line.",
        timestamp={
            "word": [
                {"word": "Hello", "start": 0.0, "end": 0.4},
                {"word": "there.", "start": 0.45, "end": 0.9},
            ],
            "segment": [{"segment": "Hello there.", "start": 0.0, "end": 0.9}],
        },
    )
    result = chunk_result_from_hypothesis(hyp, expect_timestamps=True)
    assert len(result.words) == 2
    assert len(result.segments) == 1
    assert result.segments[0].text == "Hello there."


def test_chunk_result_allows_empty_native_segments() -> None:
    hyp = SimpleNamespace(
        text="hello",
        timestamp={"word": [{"word": "hello", "start": 0.0, "end": 0.3}]},
    )
    result = chunk_result_from_hypothesis(hyp, expect_timestamps=True)
    assert result.words
    assert result.segments == []


def test_chunk_result_requires_timestamps_when_expected() -> None:
    hyp = SimpleNamespace(text="hello", timestamp={"word": []})
    with pytest.raises(TranscriptionError, match="no usable word timestamps"):
        chunk_result_from_hypothesis(hyp, expect_timestamps=True)


def test_chunk_result_untimed_ok() -> None:
    hyp = SimpleNamespace(text="hello <en-US>", timestamp=None)
    result = chunk_result_from_hypothesis(hyp, expect_timestamps=False)
    assert result.text == "hello"
    assert result.detected_language == "en-US"
    assert result.words == []
    assert result.segments == []


def test_language_tag_is_removed_from_nemotron_transcript() -> None:
    assert _extract_language("Bonjour tout le monde. <fr-FR>") == ("Bonjour tout le monde.", "fr-FR")


def test_parse_key_phrases_title_cases() -> None:
    assert parse_key_phrases("acmeCorp, GPU\nnvidia") == ["Acmecorp", "GPU", "Nvidia"]
    assert capitalize_key_phrases(["already Title"]) == ["Already Title"]
    assert parse_key_phrases("  ") == []


def test_triton_compiler_error_is_detected() -> None:
    error = RuntimeError(
        "Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl."
    )
    assert is_triton_compiler_error(error)
    assert not is_triton_compiler_error(RuntimeError("CUDA out of memory"))


def test_triton_compiler_error_becomes_transcription_error() -> None:
    error = RuntimeError("Failed to find C compiler. Please specify via CC")
    with pytest.raises(TranscriptionError, match="build-essential"):
        raise_if_triton_compiler_error(error)


def test_chunk_result_maps_word_confidence_from_hypothesis() -> None:
    hyp = SimpleNamespace(
        text="Hello there.",
        timestamp={
            "word": [
                {"word": "Hello", "start": 0.0, "end": 0.4},
                {"word": "there.", "start": 0.45, "end": 0.9},
            ],
            "segment": [{"segment": "Hello there.", "start": 0.0, "end": 0.9}],
        },
        word_confidence=[0.91, 0.77],
    )
    result = chunk_result_from_hypothesis(hyp, expect_timestamps=True)
    assert [word.confidence for word in result.words] == [0.91, 0.77]


def test_configure_decoding_applies_gpu_pb_phrases() -> None:
    from omegaconf import OmegaConf

    from parakeet_transcribe.backend import NeMoASRBackend
    from parakeet_transcribe.models import PARAKEET_V3

    backend = NeMoASRBackend(PARAKEET_V3)
    captured: dict = {}
    fake_torch = SimpleNamespace(float16="float16")

    class FakeModel:
        cfg = SimpleNamespace(decoding=OmegaConf.create({"strategy": "greedy", "greedy": {}}))

        def change_decoding_strategy(self, cfg) -> None:
            captured["cfg"] = cfg
            captured.setdefault("order", []).append("change_decoding_strategy")

        def to(self, *args, **kwargs) -> None:
            captured.setdefault("order", []).append(("to", kwargs.get("dtype", args[0] if args else None)))

    backend.model = FakeModel()
    with patch("parakeet_transcribe.backend._torch_runtime", return_value=fake_torch):
        backend.configure_decoding(["acme"], boost_alpha=1.5)
    cfg = captured["cfg"]
    assert cfg.strategy == "greedy_batch"
    assert cfg.greedy.use_cuda_graph_decoder is True
    assert cfg.greedy.boosting_tree_alpha == 1.5
    assert list(cfg.greedy.boosting_tree.key_phrases_list) == ["Acme"]
    assert cfg.confidence_cfg.preserve_word_confidence is True
    assert captured["order"][0] == "change_decoding_strategy"
    assert captured["order"][1] == ("to", "float16")


def test_longform_attention_casts_dtype_after_reconfigure() -> None:
    from parakeet_transcribe.backend import NeMoASRBackend
    from parakeet_transcribe.models import PARAKEET_V3

    backend = NeMoASRBackend(PARAKEET_V3)
    order: list = []
    fake_torch = SimpleNamespace(float16="float16")

    class FakeModel:
        def change_attention_model(self, **kwargs) -> None:
            order.append(("change_attention_model", kwargs))

        def change_subsampling_conv_chunking_factor(self, factor: int) -> None:
            order.append(("change_subsampling_conv_chunking_factor", factor))

        def to(self, *args, **kwargs) -> None:
            order.append(("to", kwargs.get("dtype", args[0] if args else None)))

    backend.model = FakeModel()
    backend._configure_longform_attention()
    with patch("parakeet_transcribe.backend._torch_runtime", return_value=fake_torch):
        backend._apply_inference_dtype()

    assert order[0][0] == "change_attention_model"
    assert order[0][1]["self_attention_model"] == "rel_pos_local_attn"
    assert order[0][1]["att_context_size"] == [256, 256]
    assert order[1] == ("change_subsampling_conv_chunking_factor", 1)
    assert order[2] == ("to", "float16")


def test_configure_decoding_recasts_dtype_after_strategy_change() -> None:
    from omegaconf import OmegaConf

    from parakeet_transcribe.backend import NeMoASRBackend
    from parakeet_transcribe.models import PARAKEET_V3

    backend = NeMoASRBackend(PARAKEET_V3)
    order: list = []
    fake_torch = SimpleNamespace(float16="float16")

    class FakeModel:
        cfg = SimpleNamespace(decoding=OmegaConf.create({"strategy": "greedy", "greedy": {}}))

        def change_decoding_strategy(self, cfg) -> None:
            order.append("change_decoding_strategy")

        def to(self, *args, **kwargs) -> None:
            order.append("to")

    backend.model = FakeModel()
    with patch("parakeet_transcribe.backend._torch_runtime", return_value=fake_torch):
        backend.configure_decoding([])
        # Fingerprint hit must not re-cast.
        backend.configure_decoding([])
    assert order == ["change_decoding_strategy", "to"]
