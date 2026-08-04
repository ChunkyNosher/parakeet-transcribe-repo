from types import SimpleNamespace
from unittest.mock import patch

import pytest

from parakeet_transcribe.backend import (
    _extract_language,
    _load_nemo_model,
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


def test_chunk_result_allows_empty_silent_timed_hypothesis() -> None:
    hyp = SimpleNamespace(text="", timestamp=None)
    result = chunk_result_from_hypothesis(hyp, expect_timestamps=True)
    assert result.text == ""
    assert result.words == []
    assert result.segments == []


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


def test_key_phrases_lower_cased_for_lowercase_vocab() -> None:
    assert capitalize_key_phrases(["Chunky", "NVIDIA", "multi word"], casing="lower") == [
        "chunky",
        "nvidia",
        "multi word",
    ]
    assert parse_key_phrases("Chunky, NVIDIA\nmulti word", casing="lower") == [
        "chunky",
        "nvidia",
        "multi word",
    ]
    # Default casing remains title-case for capitalization-vocab models.
    assert capitalize_key_phrases(["chunky"]) == ["Chunky"]


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
    assert cfg.preserve_alignments is True
    assert cfg.greedy.use_cuda_graph_decoder is True
    assert cfg.greedy.boosting_tree_alpha == 1.5
    assert list(cfg.greedy.boosting_tree.key_phrases_list) == ["Acme"]
    assert cfg.confidence_cfg.preserve_frame_confidence is True
    assert cfg.confidence_cfg.preserve_word_confidence is True
    assert captured["order"][0] == "change_decoding_strategy"
    assert captured["order"][1] == ("to", "float16")


def test_configure_decoding_lowercases_phrases_for_lowercase_vocab() -> None:
    from omegaconf import OmegaConf

    from parakeet_transcribe.backend import NeMoASRBackend
    from parakeet_transcribe.models import PARAKEET_11B

    backend = NeMoASRBackend(PARAKEET_11B)
    captured: dict = {}
    fake_torch = SimpleNamespace(float16="float16")

    class FakeModel:
        cfg = SimpleNamespace(decoding=OmegaConf.create({"strategy": "greedy", "greedy": {}}))

        def change_decoding_strategy(self, cfg) -> None:
            captured["cfg"] = cfg

        def to(self, *args, **kwargs) -> None:
            return None

    backend.model = FakeModel()
    with patch("parakeet_transcribe.backend._torch_runtime", return_value=fake_torch):
        backend.configure_decoding(["Chunky", "LIMC"], boost_alpha=1.0)
    assert list(captured["cfg"].greedy.boosting_tree.key_phrases_list) == ["chunky", "limc"]


def test_configure_decoding_stores_lowercased_phrases_without_model() -> None:
    from parakeet_transcribe.backend import NeMoASRBackend
    from parakeet_transcribe.models import PARAKEET_11B

    backend = NeMoASRBackend(PARAKEET_11B)
    assert backend.model is None
    backend.configure_decoding(["Chunky", "LIMC"], boost_alpha=1.0)
    assert backend._key_phrases == ["chunky", "limc"]


def test_streaming_decoding_uses_tdt_durations_without_frame_alignments() -> None:
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

        def to(self, *args, **kwargs) -> None:
            return None

    backend.model = FakeModel()
    with patch("parakeet_transcribe.backend._torch_runtime", return_value=fake_torch):
        backend.configure_decoding([], streaming=True, timestamps=True)

    cfg = captured["cfg"]
    assert cfg.preserve_alignments is False
    assert cfg.compute_timestamps is False
    assert cfg.tdt_include_token_duration is True
    assert cfg.greedy.preserve_alignments is False
    assert cfg.confidence_cfg.preserve_frame_confidence is False
    assert cfg.confidence_cfg.preserve_token_confidence is False
    assert cfg.confidence_cfg.preserve_word_confidence is False


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


def test_word_confidence_aggregation_mismatch_retries_without_confidence() -> None:
    from omegaconf import OmegaConf

    from parakeet_transcribe.backend import NeMoASRBackend
    from parakeet_transcribe.models import PARAKEET_V3

    backend = NeMoASRBackend(PARAKEET_V3)
    configs: list = []
    calls = 0
    fake_torch = SimpleNamespace(float16="float16")
    hypothesis = SimpleNamespace(
        text="Hello there.",
        timestamp={
            "word": [
                {"word": "Hello", "start": 0.0, "end": 0.4},
                {"word": "there.", "start": 0.45, "end": 0.9},
            ],
            "segment": [{"segment": "Hello there.", "start": 0.0, "end": 0.9}],
        },
        word_confidence=None,
    )

    class FakeModel:
        cfg = SimpleNamespace(decoding=OmegaConf.create({"strategy": "greedy", "greedy": {}}))

        def change_decoding_strategy(self, cfg) -> None:
            configs.append(cfg)

        def to(self, *args, **kwargs) -> None:
            return None

        def transcribe(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError(
                    "Something went wrong with word-level confidence aggregation. "
                    "len(words): 90, len(word_confidence): 91"
                )
            return [hypothesis]

    backend.model = FakeModel()
    with patch("parakeet_transcribe.backend._torch_runtime", return_value=fake_torch):
        results = backend.transcribe_paths(["sample.wav"], timestamps=True)

    assert calls == 2
    assert configs[-1].preserve_alignments is True
    assert configs[-1].confidence_cfg.preserve_frame_confidence is False
    assert configs[-1].confidence_cfg.preserve_token_confidence is False
    assert configs[-1].confidence_cfg.preserve_word_confidence is False
    assert backend.word_confidence_fallback_used is True
    assert results[0].text == "Hello there."
    assert [word.confidence for word in results[0].words] == [None, None]


def test_load_nemo_model_uses_extracted_dir_when_present() -> None:
    from pathlib import Path

    from parakeet_transcribe.models import PARAKEET_V3

    extracted = Path("/cache/extracted/parakeet-v3")
    captured: dict = {}

    class FakeConnector:
        def __init__(self) -> None:
            self.model_extracted_dir = None

    class FakeASRModel:
        @classmethod
        def restore_from(cls, restore_path: str, **kwargs) -> str:
            captured["path"] = restore_path
            captured["connector"] = kwargs["save_restore_connector"]
            return "restored-model"

        @classmethod
        def from_pretrained(cls, model_id: str) -> str:
            captured["pretrained"] = model_id
            return "pretrained-model"

    class FakeASRModels:
        models = SimpleNamespace(ASRModel=FakeASRModel)

    with (
        patch("parakeet_transcribe.backend.ensure_extracted", return_value=extracted),
        patch("parakeet_transcribe.backend._import_nemo_asr", return_value=FakeASRModels()),
        patch("parakeet_transcribe.backend._import_save_restore_connector", return_value=FakeConnector),
    ):
        model = _load_nemo_model(PARAKEET_V3)

    assert model == "restored-model"
    assert captured["path"] == str(extracted)
    assert captured["connector"].model_extracted_dir == str(extracted)
    assert "pretrained" not in captured


def test_load_nemo_model_falls_back_to_pretrained_without_extracted_dir() -> None:
    from parakeet_transcribe.models import PARAKEET_V3

    class FakeASRModel:
        @classmethod
        def restore_from(cls, restore_path: str, **kwargs) -> str:
            raise AssertionError("restore_from must not be called")

        @classmethod
        def from_pretrained(cls, model_id: str) -> str:
            return "pretrained-model"

    class FakeASRModels:
        models = SimpleNamespace(ASRModel=FakeASRModel)

    with (
        patch("parakeet_transcribe.backend.ensure_extracted", return_value=None),
        patch("parakeet_transcribe.backend.extract_after_load") as extract_after,
        patch("parakeet_transcribe.backend._import_nemo_asr", return_value=FakeASRModels()),
        patch("parakeet_transcribe.backend._import_save_restore_connector") as connector_import,
    ):
        model = _load_nemo_model(PARAKEET_V3)

    assert model == "pretrained-model"
    extract_after.assert_called_once_with(PARAKEET_V3)
    connector_import.assert_not_called()


def test_backend_load_uses_helper_and_keeps_existing_model() -> None:
    from omegaconf import OmegaConf

    from parakeet_transcribe.backend import NeMoASRBackend
    from parakeet_transcribe.models import PARAKEET_V3

    class FakeModel:
        cfg = SimpleNamespace(
            decoding=OmegaConf.create({"strategy": "greedy", "greedy": {}}),
        )

        def eval(self) -> None:
            return None

        def to(self, *args, **kwargs) -> None:
            return None

        def change_attention_model(self, **kwargs) -> None:
            return None

        def change_decoding_strategy(self, cfg) -> None:
            return None

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda *args, **kwargs: "cuda",
        float16="float16",
    )
    backend = NeMoASRBackend(PARAKEET_V3)
    with (
        patch("parakeet_transcribe.backend._import_nemo_asr", return_value=object()),
        patch("parakeet_transcribe.backend._load_nemo_model", return_value=FakeModel()) as loader,
        patch("parakeet_transcribe.backend._torch_runtime", return_value=fake_torch),
    ):
        backend.load()
        backend.load()

    assert loader.call_count == 1
    assert backend.model is not None
