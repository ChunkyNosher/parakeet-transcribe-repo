from types import SimpleNamespace

import pytest

from parakeet_transcribe.backend import (
    _extract_language,
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


def test_configure_decoding_applies_gpu_pb_phrases() -> None:
    from omegaconf import OmegaConf

    from parakeet_transcribe.backend import NeMoASRBackend
    from parakeet_transcribe.models import PARAKEET_V3

    backend = NeMoASRBackend(PARAKEET_V3)
    captured: dict = {}

    class FakeModel:
        cfg = SimpleNamespace(decoding=OmegaConf.create({"strategy": "greedy", "greedy": {}}))

        def change_decoding_strategy(self, cfg) -> None:
            captured["cfg"] = cfg

    backend.model = FakeModel()
    backend.configure_decoding(["acme"], boost_alpha=1.5)
    cfg = captured["cfg"]
    assert cfg.strategy == "greedy_batch"
    assert cfg.greedy.use_cuda_graph_decoder is True
    assert cfg.greedy.boosting_tree_alpha == 1.5
    assert list(cfg.greedy.boosting_tree.key_phrases_list) == ["Acme"]
