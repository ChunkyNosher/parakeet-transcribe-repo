"""Tests for persistent NeMo checkpoint pre-extraction (modelstore)."""

import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from parakeet_transcribe.models import PARAKEET_V3, SORTFORMER
from parakeet_transcribe.modelstore import (
    _extract_members,
    _nemo_archive_path,
    ensure_extracted,
    extract_after_load,
    extract_nemo,
    extracted_dir_for,
    nemo_filename_for,
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
