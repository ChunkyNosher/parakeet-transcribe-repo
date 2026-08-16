"""Tests for persistent NeMo checkpoint pre-extraction (modelstore)."""

import os
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from parakeet_transcribe.models import PARAKEET_V3, SORTFORMER
from parakeet_transcribe.modelstore import (
    _ensure_fast_caches,
    _extract_members,
    _nemo_archive_path,
    _unwrap_state_dict,
    convert_ckpt_to_fp16_safetensors,
    ensure_extracted,
    extract_after_load,
    extract_nemo,
    extracted_dir_for,
    local_weights_dir,
    nemo_filename_for,
    prewarm_local_caches,
    ready_config_path_for,
    ready_snapshot_available,
    ready_weights_path_for,
    restore_extracted_model,
    restore_ready_model,
    safetensors_path_for,
    write_ready_snapshot,
)
from parakeet_transcribe.types import TranscriptionError


def _write_nemo_archive(path: Path) -> Path:
    """Create a minimal .nemo tar (uncompressed, like NeMo >= 1.7.0)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(path), "w") as tar:
        for name, content in (
            ("model_config.yaml", b"model:\n  encoder: ctc\n"),
            ("model_weights.ckpt", b"weights-payload"),
            ("README.md", b"readme"),
            ("tokenizer.model", b"vocab"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, __import__("io").BytesIO(content))
    return path


def test_nemo_filename_for_uses_repo_tail() -> None:
    assert nemo_filename_for(PARAKEET_V3) == "parakeet-tdt-0.6b-v3.nemo"


def test_extracted_dir_lives_under_cache_parent(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert extracted_dir_for(PARAKEET_V3) == tmp_path / "extracted" / "parakeet-v3"


def test_nemo_archive_path_resolves_via_hf_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    cached = tmp_path / "hub" / "models--nvidia--parakeet-tdt-0.6b-v3" / "snapshots" / "abc" / "parakeet-tdt-0.6b-v3.nemo"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"x")

    with patch("parakeet_transcribe.modelstore.try_to_load_from_cache", return_value=str(cached)):
        assert _nemo_archive_path(PARAKEET_V3) == cached


def test_extract_members_filters_safe_files(tmp_path) -> None:
    path = _write_nemo_archive(tmp_path / "model.nemo")
    with tarfile.open(str(path), "r:") as tar:
        names = [member.name for member in _extract_members(tar)]
    assert "model_config.yaml" in names
    assert "model_weights.ckpt" in names
    assert "README.md" not in names


def test_extract_nemo_unpacks_and_is_idempotent(tmp_path) -> None:
    archive = _write_nemo_archive(tmp_path / "model.nemo")
    dest = tmp_path / "extracted"
    assert extract_nemo(PARAKEET_V3, archive, dest) is True
    assert (dest / "model_config.yaml").is_file()
    assert (dest / "model_weights.ckpt").is_file()
    # Second call sees a complete destination and skips re-extraction.
    assert extract_nemo(PARAKEET_V3, archive, dest) is False


def test_extract_nemo_rejects_incomplete_archive(tmp_path) -> None:
    archive = tmp_path / "broken.nemo"
    with tarfile.open(str(archive), "w") as tar:
        info = tarfile.TarInfo("model_config.yaml")
        info.size = 3
        tar.addfile(info, __import__("io").BytesIO(b"cfg"))
    with pytest.raises(TranscriptionError, match="model_weights.ckpt"):
        extract_nemo(PARAKEET_V3, archive, tmp_path / "dest")


def test_ensure_extracted_returns_existing_dir(tmp_path) -> None:
    dest = tmp_path / "extracted" / PARAKEET_V3.key
    dest.mkdir(parents=True)
    (dest / "model_config.yaml").write_text("cfg")
    (dest / "model_weights.ckpt").write_bytes(b"weights")
    with patch("parakeet_transcribe.modelstore.extracted_dir_for", return_value=dest):
        assert ensure_extracted(PARAKEET_V3) == dest


def test_ensure_extracted_extracts_cached_archive(tmp_path) -> None:
    archive = _write_nemo_archive(tmp_path / "model.nemo")
    dest = tmp_path / "extracted" / PARAKEET_V3.key
    with (
        patch("parakeet_transcribe.modelstore.extracted_dir_for", return_value=dest),
        patch("parakeet_transcribe.modelstore._nemo_archive_path", return_value=archive),
    ):
        result = ensure_extracted(PARAKEET_V3)
    assert result == dest
    assert (dest / "model_weights.ckpt").is_file()


def test_ensure_extracted_returns_none_when_archive_missing(tmp_path) -> None:
    dest = tmp_path / "extracted" / PARAKEET_V3.key
    with (
        patch("parakeet_transcribe.modelstore.extracted_dir_for", return_value=dest),
        patch("parakeet_transcribe.modelstore._nemo_archive_path", return_value=None),
    ):
        assert ensure_extracted(PARAKEET_V3) is None


def test_extract_after_load_is_best_effort(tmp_path) -> None:
    dest = tmp_path / "extracted" / PARAKEET_V3.key
    with (
        patch("parakeet_transcribe.modelstore.extracted_dir_for", return_value=dest),
        patch("parakeet_transcribe.modelstore._nemo_archive_path", return_value=None),
        patch("parakeet_transcribe.modelstore.logger") as logger,
    ):
        extract_after_load(PARAKEET_V3)
    logger.warning.assert_not_called()
    assert not dest.exists()


def test_sortformer_spec_resolves_its_hf_id() -> None:
    assert nemo_filename_for(SORTFORMER) == "diar_sortformer_4spk-v1.nemo"


def test_extract_is_atomic_on_swap(tmp_path) -> None:
    archive = _write_nemo_archive(tmp_path / "model.nemo")
    dest = tmp_path / "extracted" / PARAKEET_V3.key
    dest.mkdir(parents=True)
    (dest / "stale.txt").write_text("old")
    with (
        patch("parakeet_transcribe.modelstore.extracted_dir_for", return_value=dest),
        patch("parakeet_transcribe.modelstore._nemo_archive_path", return_value=archive),
    ):
        result = ensure_extracted(PARAKEET_V3)
    assert result == dest
    assert (dest / "model_weights.ckpt").is_file()
    assert not (dest / "stale.txt").exists()


# ---------------------------------------------------------------------------
# Fast restore from an extracted directory
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, name: str, floating: bool = True) -> None:
        self.name = name
        self.floating = floating
        self.dtype = "fp32"

    def is_floating_point(self) -> bool:
        return self.floating

    def detach(self) -> "FakeTensor":
        return self

    def to(self, dtype: object) -> "FakeTensor":
        clone = FakeTensor(self.name, self.floating)
        clone.dtype = dtype
        return clone


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, init=lambda: None),
        device=lambda *args, **kwargs: f"device:{args[0]}" if args else "device",
        float16="fp16",
        float32="fp32",
    )


class FakeModelInstance:
    def __init__(self) -> None:
        self.state_calls: list[dict] = []
        self.to_calls: list[tuple] = []
        self.missing_keys: list = []
        self.unexpected_keys: list = []

    def load_state_dict(self, state: dict, *, strict: bool, assign: bool):
        self.state_calls.append(
            {"keys": sorted(state), "strict": strict, "assign": assign}
        )
        if not strict:
            # Mirror torch's IncompatibleKeys return value.
            return SimpleNamespace(
                missing_keys=list(self.missing_keys),
                unexpected_keys=list(self.unexpected_keys),
            )
        return None

    def to(self, *args) -> "FakeModelInstance":
        self.to_calls.append(args)
        return self


class FakeModelClass:
    restore_state_calls: list[tuple] = []
    construct_args: list[dict] = []
    instance = FakeModelInstance()

    @classmethod
    def _set_model_restore_state(cls, is_being_restored: bool, folder: str | None = None) -> None:
        cls.restore_state_calls.append((is_being_restored, folder))

    @classmethod
    def from_config_dict(cls, config: object, trainer: object = None) -> FakeModelInstance:
        cls.construct_args.append({"config": config, "trainer": trainer})
        return cls.instance


@pytest.fixture()
def fake_model_class() -> type:
    FakeModelClass.restore_state_calls = []
    FakeModelClass.construct_args = []
    FakeModelClass.instance = FakeModelInstance()
    return FakeModelClass


def _write_extracted_dir(tmp_path: Path, *, with_safetensors: bool = False) -> Path:
    extracted = tmp_path / "extracted" / PARAKEET_V3.key
    extracted.mkdir(parents=True)
    (extracted / "model_config.yaml").write_text("sample_rate: 16000\n")
    (extracted / "model_weights.ckpt").write_bytes(b"weights")
    if with_safetensors:
        safetensors_path_for(extracted).write_bytes(b"safetensors")
    return extracted


def test_restore_extracted_model_overlaps_io_and_construction(tmp_path, fake_model_class, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path)
    state = {"encoder.weight": FakeTensor("encoder.weight")}
    weight_calls: list[Path] = []

    def fake_load_weights(weights_path: Path, use_safetensors: bool) -> dict:
        weight_calls.append(weights_path)
        assert use_safetensors is False
        return state

    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._load_weights_on_cuda", side_effect=fake_load_weights),
        patch("parakeet_transcribe.modelstore._spawn_background_fast_cache") as spawn,
    ):
        model = restore_extracted_model(PARAKEET_V3, fake_model_class, extracted)

    assert model is fake_model_class.instance
    assert weight_calls == [extracted / "model_weights.ckpt"]
    # Construction ran with NeMo restore-state semantics for artifact resolution.
    assert fake_model_class.restore_state_calls == [(True, str(extracted)), (False, None)]
    assert fake_model_class.construct_args[0]["config"].sample_rate == 16000
    # Weights assigned directly (no CPU staging copy), strict key check.
    assert model.state_calls == [{"keys": ["encoder.weight"], "strict": True, "assign": True}]
    # Remaining buffers move to CUDA; fp32 checkpoint needs no precision cast.
    assert model.to_calls == [("device:cuda",)]
    spawn.assert_called_once_with(PARAKEET_V3, extracted)


def test_restore_extracted_model_prefers_safetensors_and_casts_fp32(tmp_path, fake_model_class, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path, with_safetensors=True)
    seen: dict = {}

    def fake_load_weights(weights_path: Path, use_safetensors: bool) -> dict:
        seen["path"] = weights_path
        seen["safetensors"] = use_safetensors
        return {"w": FakeTensor("w")}

    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._load_weights_on_cuda", side_effect=fake_load_weights),
        patch("parakeet_transcribe.modelstore._spawn_background_fast_cache") as spawn,
    ):
        model = restore_extracted_model(PARAKEET_V3, fake_model_class, extracted)

    assert seen == {"path": safetensors_path_for(extracted), "safetensors": True}
    # FP16 storage is cast back to FP32 so attention/decoding rebuilds match
    # the stock load sequence before its final FP16 cast.
    assert model.to_calls == [("device:cuda",), ("fp32",)]
    # The bind-mounted safetensors still triggers a local-mirror build.
    spawn.assert_called_once_with(PARAKEET_V3, extracted)


def test_restore_extracted_model_prefers_fresh_local_mirror(tmp_path, fake_model_class, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path, with_safetensors=True)
    local_dir = local_weights_dir(PARAKEET_V3)
    local_dir.mkdir(parents=True)
    local_safe = local_dir / "model_weights_fp16.safetensors"
    local_safe.write_bytes(b"local")
    # Keep the mirror newer than the bind-mounted sources.
    os.utime(local_safe, (time.time() + 60, time.time() + 60))
    seen: dict = {}

    def fake_load_weights(weights_path: Path, use_safetensors: bool) -> dict:
        seen["path"] = weights_path
        seen["safetensors"] = use_safetensors
        return {"w": FakeTensor("w")}

    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._load_weights_on_cuda", side_effect=fake_load_weights),
        patch("parakeet_transcribe.modelstore._spawn_background_fast_cache") as spawn,
    ):
        restore_extracted_model(PARAKEET_V3, fake_model_class, extracted)

    assert seen == {"path": local_safe, "safetensors": True}
    # Local mirror reads need no background cache work.
    spawn.assert_not_called()


def test_restore_extracted_model_skips_stale_local_mirror(tmp_path, fake_model_class, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path, with_safetensors=True)
    local_dir = local_weights_dir(PARAKEET_V3)
    local_dir.mkdir(parents=True)
    local_safe = local_dir / "model_weights_fp16.safetensors"
    local_safe.write_bytes(b"local")
    # Mirror is older than the checkpoint -> treated as stale.
    os.utime(local_safe, (1000.0, 1000.0))
    seen: dict = {}

    def fake_load_weights(weights_path: Path, use_safetensors: bool) -> dict:
        seen["path"] = weights_path
        return {"w": FakeTensor("w")}

    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._load_weights_on_cuda", side_effect=fake_load_weights),
        patch("parakeet_transcribe.modelstore._spawn_background_fast_cache"),
    ):
        restore_extracted_model(PARAKEET_V3, fake_model_class, extracted)

    assert seen["path"] == safetensors_path_for(extracted)


def test_restore_extracted_model_raises_when_weights_fail(tmp_path, fake_model_class) -> None:
    extracted = _write_extracted_dir(tmp_path)

    def fake_load_weights(weights_path: Path, use_safetensors: bool) -> dict:
        raise OSError("disk read failed")

    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._load_weights_on_cuda", side_effect=fake_load_weights),
    ):
        with pytest.raises(OSError, match="disk read failed"):
            restore_extracted_model(PARAKEET_V3, fake_model_class, extracted)
    # Restore state is reset even on failure.
    assert fake_model_class.restore_state_calls[-1] == (False, None)


def test_restore_extracted_model_requires_cuda(tmp_path, fake_model_class) -> None:
    extracted = _write_extracted_dir(tmp_path)
    torch_no_cuda = _fake_torch()
    torch_no_cuda.cuda.is_available = lambda: False
    with patch("parakeet_transcribe.modelstore._import_torch", return_value=torch_no_cuda):
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            restore_extracted_model(PARAKEET_V3, fake_model_class, extracted)


def test_unwrap_state_dict_handles_lightning_wrapper() -> None:
    raw = {"a": 1}
    assert _unwrap_state_dict(raw) == raw
    assert _unwrap_state_dict({"state_dict": raw, "epoch": 3}) == raw
    with pytest.raises(ValueError, match="state dict"):
        _unwrap_state_dict([1, 2, 3])


# ---------------------------------------------------------------------------
# FP16 safetensors conversion
# ---------------------------------------------------------------------------


def test_conversion_casts_floats_and_keeps_ints(tmp_path) -> None:
    extracted = _write_extracted_dir(tmp_path)
    state = {
        "encoder.weight": FakeTensor("encoder.weight", floating=True),
        "tokenizer.ids": FakeTensor("tokenizer.ids", floating=False),
    }
    saved: dict = {}

    def fake_save(tensors: dict, dest: Path) -> None:
        saved["tensors"] = tensors
        saved["dest"] = dest
        dest.write_bytes(b"converted")

    fake_torch = _fake_torch()
    fake_torch.load = lambda *args, **kwargs: state
    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=fake_torch),
        patch("parakeet_transcribe.modelstore._save_safetensors_atomic", side_effect=fake_save),
    ):
        assert convert_ckpt_to_fp16_safetensors(extracted) is True

    assert saved["dest"] == safetensors_path_for(extracted)
    assert saved["tensors"]["encoder.weight"].dtype == "fp16"
    assert saved["tensors"]["tokenizer.ids"].dtype == "fp32"


def test_conversion_skips_when_already_converted(tmp_path) -> None:
    extracted = _write_extracted_dir(tmp_path, with_safetensors=True)
    with patch("parakeet_transcribe.modelstore._import_torch") as torch_import:
        assert convert_ckpt_to_fp16_safetensors(extracted) is False
    torch_import.assert_not_called()


def test_conversion_skips_when_checkpoint_missing(tmp_path) -> None:
    extracted = tmp_path / "extracted" / PARAKEET_V3.key
    extracted.mkdir(parents=True)
    with patch("parakeet_transcribe.modelstore._import_torch") as torch_import:
        assert convert_ckpt_to_fp16_safetensors(extracted) is False
    torch_import.assert_not_called()


def test_conversion_unwraps_lightning_checkpoint(tmp_path) -> None:
    extracted = _write_extracted_dir(tmp_path)
    state = {"state_dict": {"w": FakeTensor("w")}, "epoch": 1}
    saved: dict = {}

    fake_torch = _fake_torch()
    fake_torch.load = lambda *args, **kwargs: state
    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=fake_torch),
        patch(
            "parakeet_transcribe.modelstore._save_safetensors_atomic",
            side_effect=lambda tensors, dest: saved.update(tensors=tensors),
        ),
    ):
        assert convert_ckpt_to_fp16_safetensors(extracted) is True
    assert saved["tensors"]["w"].dtype == "fp16"


# ---------------------------------------------------------------------------
# Container-local mirror / fast caches
# ---------------------------------------------------------------------------


def test_local_weights_dir_respects_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "fast"))
    assert local_weights_dir(PARAKEET_V3) == tmp_path / "fast" / "parakeet-v3"


def test_ensure_fast_caches_converts_and_mirrors(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path)
    state = {"w": FakeTensor("w")}
    saved: list[Path] = []

    fake_torch = _fake_torch()
    fake_torch.load = lambda *args, **kwargs: state
    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=fake_torch),
        patch(
            "parakeet_transcribe.modelstore._save_safetensors_atomic",
            side_effect=lambda tensors, dest: saved.append(dest),
        ),
    ):
        _ensure_fast_caches(PARAKEET_V3, extracted)

    # Local mirror is written first, then the persistent bind-mounted copy.
    assert saved == [
        local_weights_dir(PARAKEET_V3) / "model_weights_fp16.safetensors",
        safetensors_path_for(extracted),
    ]


def test_ensure_fast_caches_mirrors_existing_safetensors(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path, with_safetensors=True)
    copied: list[tuple[Path, Path]] = []

    with (
        patch(
            "parakeet_transcribe.modelstore._copy_file_atomic",
            side_effect=lambda src, dest: copied.append((src, dest)),
        ),
        patch("parakeet_transcribe.modelstore._import_torch") as torch_import,
    ):
        _ensure_fast_caches(PARAKEET_V3, extracted)

    assert copied == [(safetensors_path_for(extracted), local_weights_dir(PARAKEET_V3) / "model_weights_fp16.safetensors")]
    torch_import.assert_not_called()


def test_ensure_fast_caches_noop_when_mirror_fresh(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path, with_safetensors=True)
    local_dir = local_weights_dir(PARAKEET_V3)
    local_dir.mkdir(parents=True)
    local_safe = local_dir / "model_weights_fp16.safetensors"
    local_safe.write_bytes(b"local")
    os.utime(local_safe, (time.time() + 60, time.time() + 60))

    with (
        patch("parakeet_transcribe.modelstore._copy_file_atomic") as copy,
        patch("parakeet_transcribe.modelstore._save_safetensors_atomic") as save,
    ):
        _ensure_fast_caches(PARAKEET_V3, extracted)

    copy.assert_not_called()
    save.assert_not_called()


# ---------------------------------------------------------------------------
# Ready-state snapshot (config with local attention + default decoding baked in)
# ---------------------------------------------------------------------------


class FakeLiveModel:
    """Stand-in for a fully loaded, FP16 model at snapshot time."""

    def __init__(self, state: dict) -> None:
        self._state = state

    def state_dict(self) -> dict:
        return self._state


def _write_ready_config(extracted: Path) -> Path:
    ready_cfg = ready_config_path_for(extracted)
    ready_cfg.write_text(
        "encoder:\n"
        "  self_attention_model: rel_pos_local_attn\n"
        "  att_context_size:\n  - 256\n  - 256\n"
        "decoding:\n"
        "  strategy: greedy_batch\n"
        "  greedy:\n"
        "    use_cuda_graph_decoder: false\n"
    )
    return ready_cfg


def test_write_ready_snapshot_bakes_attention_and_default_decoding(tmp_path, monkeypatch) -> None:
    from omegaconf import OmegaConf

    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    monkeypatch.setattr(
        "parakeet_transcribe.modelstore.extracted_dir_for",
        lambda spec: tmp_path / "extracted" / spec.key,
    )
    extracted = _write_extracted_dir(tmp_path)
    (extracted / "model_config.yaml").write_text(
        "encoder:\n"
        "  self_attention_model: rel_pos\n"
        "  att_context_size:\n  - -1\n  - -1\n"
        "decoding:\n"
        "  strategy: greedy_batch\n"
    )
    saved: dict = {}

    def fake_save(tensors: dict, dest: Path) -> None:
        saved[dest.name] = tensors
        dest.write_bytes(b"ready")

    decoding_cfg = OmegaConf.create(
        {"strategy": "greedy_batch", "greedy": {"use_cuda_graph_decoder": False}}
    )
    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._save_safetensors_atomic", side_effect=fake_save),
    ):
        write_ready_snapshot(PARAKEET_V3, FakeLiveModel({"encoder.weight": FakeTensor("encoder.weight")}), decoding_cfg)

    ready = OmegaConf.load(str(ready_config_path_for(extracted)))
    assert ready.encoder.self_attention_model == "rel_pos_local_attn"
    assert list(ready.encoder.att_context_size) == [256, 256]
    assert ready.decoding.strategy == "greedy_batch"
    assert ready.decoding.greedy.use_cuda_graph_decoder is False
    # FP16 weights are saved to the local mirror first, then the bind mount.
    assert saved["model_weights_ready_fp16.safetensors"]["encoder.weight"].dtype == "fp16"
    assert ready_weights_path_for(extracted).read_bytes() == b"ready"
    assert (
        tmp_path / "local" / PARAKEET_V3.key / "model_weights_ready_fp16.safetensors"
    ).read_bytes() == b"ready"
    assert ready_snapshot_available(extracted) is True


def test_write_ready_snapshot_is_idempotent(tmp_path, monkeypatch) -> None:
    from omegaconf import OmegaConf

    monkeypatch.setattr(
        "parakeet_transcribe.modelstore.extracted_dir_for",
        lambda spec: tmp_path / "extracted" / spec.key,
    )
    extracted = _write_extracted_dir(tmp_path)
    _write_ready_config(extracted)
    ready_weights_path_for(extracted).write_bytes(b"existing")
    with patch("parakeet_transcribe.modelstore._save_safetensors_atomic") as save:
        write_ready_snapshot(PARAKEET_V3, FakeLiveModel({}), OmegaConf.create({"strategy": "greedy_batch"}))
    save.assert_not_called()
    assert ready_weights_path_for(extracted).read_bytes() == b"existing"


def test_restore_ready_model_skips_fp32_upcast(tmp_path, fake_model_class, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path)
    _write_ready_config(extracted)
    ready_weights_path_for(extracted).write_bytes(b"ready")
    seen: dict = {}

    def fake_load_weights(weights_path: Path, use_safetensors: bool) -> dict:
        seen["path"] = weights_path
        seen["safetensors"] = use_safetensors
        return {"w": FakeTensor("w")}

    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._load_weights_on_cuda", side_effect=fake_load_weights),
        patch("parakeet_transcribe.modelstore._spawn_background_fast_cache") as spawn,
    ):
        model = restore_ready_model(PARAKEET_V3, fake_model_class, extracted)

    assert model is fake_model_class.instance
    assert seen == {"path": ready_weights_path_for(extracted), "safetensors": True}
    # No FP32 upcast on the ready path: weights stay FP16, only buffers move to CUDA.
    assert model.to_calls == [("device:cuda",)]
    # Construction uses the ready config: local attention is baked at build time.
    assert fake_model_class.construct_args[0]["config"].encoder.self_attention_model == "rel_pos_local_attn"
    spawn.assert_called_once_with(PARAKEET_V3, extracted)


def test_restore_ready_model_prefers_fresh_local_mirror(tmp_path, fake_model_class, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path)
    _write_ready_config(extracted)
    ready_weights_path_for(extracted).write_bytes(b"ready-bind")
    local_dir = local_weights_dir(PARAKEET_V3)
    local_dir.mkdir(parents=True)
    local_ready = local_dir / "model_weights_ready_fp16.safetensors"
    local_ready.write_bytes(b"ready-local")
    # Keep the mirror newer than the bind-mounted source.
    os.utime(local_ready, (time.time() + 60, time.time() + 60))
    seen: dict = {}

    def fake_load_weights(weights_path: Path, use_safetensors: bool) -> dict:
        seen["path"] = weights_path
        return {"w": FakeTensor("w")}

    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._load_weights_on_cuda", side_effect=fake_load_weights),
        patch("parakeet_transcribe.modelstore._spawn_background_fast_cache") as spawn,
    ):
        restore_ready_model(PARAKEET_V3, fake_model_class, extracted)

    assert seen["path"] == local_ready
    spawn.assert_not_called()


def test_restore_ready_model_raises_when_weights_missing(tmp_path, fake_model_class, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path)
    _write_ready_config(extracted)
    with patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()):
        with pytest.raises(RuntimeError, match="Ready-state weights"):
            restore_ready_model(PARAKEET_V3, fake_model_class, extracted)


def test_restore_ready_model_tolerates_attention_bias_artifacts(tmp_path, fake_model_class, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path)
    _write_ready_config(extracted)
    ready_weights_path_for(extracted).write_bytes(b"ready")
    # NeMo's change_attention_model adds randomly-initialized bias keys that a
    # from-config construction does not produce; the ready path drops them.
    fake_model_class.instance.unexpected_keys = [
        "encoder.layers.0.self_attn.linear_q.bias",
        "encoder.layers.3.self_attn.linear_out.bias",
    ]

    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._load_weights_on_cuda", return_value={"w": FakeTensor("w")}),
        patch("parakeet_transcribe.modelstore._spawn_background_fast_cache"),
    ):
        model = restore_ready_model(PARAKEET_V3, fake_model_class, extracted)

    assert model is fake_model_class.instance
    assert model.state_calls[0]["strict"] is False
    assert model.state_calls[0]["assign"] is True


def test_restore_ready_model_rejects_unknown_key_mismatches(tmp_path, fake_model_class, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path)
    _write_ready_config(extracted)
    ready_weights_path_for(extracted).write_bytes(b"ready")
    fake_model_class.instance.unexpected_keys = ["decoder.project.weight"]
    fake_model_class.instance.missing_keys = ["joint.linear.weight"]

    with (
        patch("parakeet_transcribe.modelstore._import_torch", return_value=_fake_torch()),
        patch("parakeet_transcribe.modelstore._load_weights_on_cuda", return_value={"w": FakeTensor("w")}),
        patch("parakeet_transcribe.modelstore._spawn_background_fast_cache"),
    ):
        with pytest.raises(RuntimeError, match="key layout mismatch"):
            restore_ready_model(PARAKEET_V3, fake_model_class, extracted)


def test_ensure_fast_caches_mirrors_ready_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PARAKEET_LOCAL_WEIGHTS_DIR", str(tmp_path / "local"))
    extracted = _write_extracted_dir(tmp_path, with_safetensors=True)
    local_dir = local_weights_dir(PARAKEET_V3)
    local_dir.mkdir(parents=True)
    local_safe = local_dir / "model_weights_fp16.safetensors"
    local_safe.write_bytes(b"local")
    # Fresh FP16 mirror: only the ready-state weights need mirroring.
    os.utime(local_safe, (time.time() + 60, time.time() + 60))
    ready_weights_path_for(extracted).write_bytes(b"ready")
    copied: list = []

    def fake_copy(src: Path, dest: Path) -> None:
        copied.append(dest.name)
        dest.write_bytes(b"copied")

    with patch("parakeet_transcribe.modelstore._copy_file_atomic", side_effect=fake_copy):
        _ensure_fast_caches(PARAKEET_V3, extracted)

    assert copied == ["model_weights_ready_fp16.safetensors"]


def test_prewarm_local_caches_builds_caches_for_extracted_models(tmp_path) -> None:
    extracted = _write_extracted_dir(tmp_path)
    with (
        patch("parakeet_transcribe.modelstore.ensure_extracted", return_value=extracted),
        patch("parakeet_transcribe.modelstore._ensure_fast_caches") as ensure,
    ):
        assert prewarm_local_caches(PARAKEET_V3) is True
    ensure.assert_called_once_with(PARAKEET_V3, extracted)


def test_prewarm_local_caches_reports_missing_extraction(tmp_path) -> None:
    with patch("parakeet_transcribe.modelstore.ensure_extracted", return_value=None):
        assert prewarm_local_caches(PARAKEET_V3) is False


