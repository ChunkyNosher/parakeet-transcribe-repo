"""Persistent pre-extraction of NeMo checkpoints for faster cold loads.

NeMo's ``SaveRestoreConnector`` unpacks the ``.nemo`` tar archive into a
temporary directory on every ``restore_from`` and deletes it afterwards. For a
Parakeet 0.6B checkpoint that tar is ~2.5 GB, so every cold start pays the
decompression cost. NeMo natively supports restoring straight from an already
unpacked directory: when ``SaveRestoreConnector.model_extracted_dir`` points at
a directory containing ``model_config.yaml`` + ``model_weights.ckpt``, the
restore path skips tar extraction entirely.

This module resolves the cached ``.nemo`` file (via the Hugging Face cache that
``from_pretrained`` already populates) and unpacks it once into a persistent
directory under the model cache. Loads thereafter restore from that directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

from .types import ModelSpec, TranscriptionError

logger = logging.getLogger(__name__)

_NEMO_CONFIG_YAML = "model_config.yaml"
_NEMO_WEIGHTS_CKPT = "model_weights.ckpt"

# Files inside a .nemo archive that are safe to extract. The archive is the
# same artifact NeMo itself extracts during restore_from().
_ALLOWED_SUFFIXES = {".yaml", ".yml", ".json", ".ckpt", ".txt", ".vocab", ".model", ".nemo"}


def _model_cache_root() -> Path:
    """The HF cache parent that ``HF_HOME`` / ``HF_HUB_CACHE`` point into."""
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or "model_cache/huggingface"
    return Path(cache).expanduser().resolve().parent


def extracted_dir_for(spec: ModelSpec) -> Path:
    """Persistent unpacked-checkpoint directory for a model spec.

    Lives beside the HF hub cache (``<cache parent>/extracted/<spec.key>``) so
    it is covered by the same bind mount (``docker-data/model_cache``) and
    survives container restarts.
    """
    return _model_cache_root() / "extracted" / spec.key


def nemo_filename_for(spec: ModelSpec) -> str:
    """HF filename of the ``.nemo`` archive for a model id like ``org/name``."""
    return spec.model_id.rsplit("/", 1)[-1] + ".nemo"


def _nemo_archive_path(spec: ModelSpec) -> Path | None:
    """Locate the cached ``.nemo`` archive on disk (no network access)."""
    path = try_to_load_from_cache(repo_id=spec.model_id, filename=nemo_filename_for(spec))
    if path is None or str(path).endswith(".no_exist"):
        return None
    return Path(path)


def _extract_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        if member.isdir():
            continue
        name = Path(member.name).name.lower()
        if not any(name.endswith(suffix) for suffix in _ALLOWED_SUFFIXES):
            continue
        members.append(member)
    return members


def extract_nemo(spec: ModelSpec, archive: Path, dest_dir: Path) -> bool:
    """Unpack a ``.nemo`` archive into ``dest_dir`` (atomic via temp dir).

    Returns True when the archive was unpacked, False when it was skipped
    (destination already complete). Raises on corrupt/unreadable archives.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / _NEMO_WEIGHTS_CKPT
    if marker.is_file() and (dest_dir / _NEMO_CONFIG_YAML).is_file():
        return False

    with tempfile.TemporaryDirectory(prefix="nemo-extract-", dir=str(dest_dir.parent)) as tmp:
        tmp_dir = Path(tmp)
        try:
            with tarfile.open(str(archive), "r:") as tar:
                members = _extract_members(tar)
                tar.extractall(str(tmp_dir), members=members, filter="data")
        except tarfile.ReadError:
            # Some NeMo releases ship gzip-compressed archives.
            with tarfile.open(str(archive), "r:gz") as tar:
                members = _extract_members(tar)
                tar.extractall(str(tmp_dir), members=members, filter="data")
        if not (tmp_dir / _NEMO_WEIGHTS_CKPT).is_file() or not (tmp_dir / _NEMO_CONFIG_YAML).is_file():
            raise TranscriptionError(f"Checkpoint {archive.name} is missing model_config.yaml/model_weights.ckpt.")

        # Atomic swap into place so concurrent readers never see a partial dir.
        backup = dest_dir.with_name(dest_dir.name + ".stale")
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        if dest_dir.exists():
            dest_dir.rename(backup)
        try:
            tmp_dir.rename(dest_dir)
            shutil.rmtree(backup, ignore_errors=True)
        except OSError:  # pragma: no cover - cross-device/cross-fs edge case
            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in tmp_dir.iterdir():
                shutil.move(str(item), str(dest_dir))
            shutil.rmtree(backup, ignore_errors=True)
    return True


def ensure_extracted(spec: ModelSpec) -> Path | None:
    """Return a persistent extracted directory for ``spec`` if one is usable.

    Extracts the cached ``.nemo`` archive on first use (a one-time cost) and
    returns the directory for all later loads. Returns None when the archive is
    not cached yet (first-ever download) or extraction fails; callers must fall
    back to ``from_pretrained`` in that case.
    """
    dest = extracted_dir_for(spec)
    if (dest / _NEMO_WEIGHTS_CKPT).is_file() and (dest / _NEMO_CONFIG_YAML).is_file():
        return dest
    archive = _nemo_archive_path(spec)
    if archive is None or not archive.is_file():
        return None
    try:
        extract_nemo(spec, archive, dest)
        return dest
    except Exception as exc:  # pragma: no cover - corrupt cache edge case
        logger.warning("Persistent checkpoint extraction failed for %s: %s", spec.model_id, exc)
        return None


def extract_after_load(spec: ModelSpec) -> None:
    """Best-effort background-style extraction after a fresh ``from_pretrained``.

    Runs right after the first load so the *next* cold start skips tar
    decompression. Never raises; failures only slow the next load.
    """
    dest = extracted_dir_for(spec)
    if (dest / _NEMO_WEIGHTS_CKPT).is_file() and (dest / _NEMO_CONFIG_YAML).is_file():
        return
    archive = _nemo_archive_path(spec)
    if archive is None or not archive.is_file():
        return
    try:
        extract_nemo(spec, archive, dest)
        logger.info("Pre-extracted %s for faster future loads at %s", spec.model_id, dest)
    except Exception as exc:  # pragma: no cover - corrupt cache edge case
        logger.warning("Post-load extraction failed for %s: %s", spec.model_id, exc)
