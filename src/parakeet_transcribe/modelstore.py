"""Persistent pre-extraction + fast restore of NeMo checkpoints.

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

On top of that, ``restore_extracted_model`` implements a faster restore from
the unpacked directory than NeMo's stock ``restore_from``:

- weight I/O runs on a background thread straight onto the GPU, overlapped
  with model construction (``from_config_dict``) on the calling thread;
- the loaded tensors are assigned directly as parameters (``assign=True``),
  so the CPU never stages the full checkpoint and no separate ``.to(cuda)``
  pass over random init weights is needed;
- after the first successful restore a one-time background conversion stores
  the weights as FP16 safetensors (~1.2 GB for the 0.6B models, unpickled),
  which later loads read instead of the 2.5 GB pickle checkpoint.

Any failure in the fast path raises so callers can fall back to NeMo's stock
``restore_from`` (which itself falls back to ``from_pretrained``).
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from huggingface_hub import try_to_load_from_cache

from .types import ModelSpec, TranscriptionError

logger = logging.getLogger(__name__)

_NEMO_CONFIG_YAML = "model_config.yaml"
_NEMO_WEIGHTS_CKPT = "model_weights.ckpt"
_NEMO_WEIGHTS_SAFETENSORS = "model_weights_fp16.safetensors"

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


# ---------------------------------------------------------------------------
# Fast restore from an extracted checkpoint directory
# ---------------------------------------------------------------------------


def safetensors_path_for(extracted: Path) -> Path:
    """FP16 safetensors weights file derived from ``model_weights.ckpt``."""
    return extracted / _NEMO_WEIGHTS_SAFETENSORS


def local_weights_dir(spec: ModelSpec) -> Path:
    """Container-local (non-bind-mounted) cache dir for restored weights.

    Docker Desktop bind mounts read at ~200 MB/s while the container's own
    overlay filesystem reads at several GB/s, so a local mirror of the FP16
    safetensors makes repeated cold loads dramatically faster. The mirror is
    ephemeral by design; the bind-mounted copy remains the persistent source.
    """
    root = os.environ.get("PARAKEET_LOCAL_WEIGHTS_DIR", "/tmp/parakeet-fast-weights")
    return Path(root).expanduser() / spec.key


def _local_safetensors_path(spec: ModelSpec) -> Path:
    return local_weights_dir(spec) / _NEMO_WEIGHTS_SAFETENSORS


def _local_mirror_fresh(spec: ModelSpec, extracted: Path) -> bool:
    """True when the container-local mirror exists and is not stale."""
    local_safe = _local_safetensors_path(spec)
    if not local_safe.is_file():
        return False
    local_mtime = local_safe.stat().st_mtime
    for source in (extracted / _NEMO_WEIGHTS_CKPT, safetensors_path_for(extracted)):
        if source.is_file() and source.stat().st_mtime > local_mtime:
            return False
    return True


def _import_torch() -> Any:
    import torch

    return torch


def _load_weights_on_cuda(weights_path: Path, use_safetensors: bool) -> Any:
    """Load checkpoint tensors directly onto the GPU (no CPU staging copy)."""
    if use_safetensors:
        from safetensors.torch import load_file

        return load_file(str(weights_path), device="cuda")
    torch = _import_torch()
    # Mirrors NeMo's SaveRestoreConnector._load_state_dict_from_disk call; the
    # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD env var (set in compose.yaml for the
    # legacy 1.1B checkpoint) is honored by torch.load itself.
    return torch.load(str(weights_path), map_location=torch.device("cuda"), weights_only=True)


def _unwrap_state_dict(state: Any) -> dict:
    """Accept either a raw state dict or a Lightning-style wrapper."""
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise ValueError("Checkpoint does not contain a usable state dict.")


def _load_model_config(extracted: Path) -> Any:
    from omegaconf import OmegaConf

    conf = OmegaConf.load(str(extracted / _NEMO_CONFIG_YAML))
    # .nemo archives sometimes store the config under a top-level `model` key.
    if "model" in conf:
        conf = conf.model
    OmegaConf.set_struct(conf, True)
    return conf


def _construct_model_from_config(model_cls: Any, config: Any, extracted: Path) -> Any:
    """Instantiate the model the same way NeMo's restore path does.

    NeMo resolves ``nemo:``-prefixed artifact paths (tokenizer files) through
    ``AppState.nemo_file_folder`` and changes the working directory to the
    unpacked checkpoint folder while constructing the model. Replicate both so
    relative/artifact config entries resolve identically.
    """
    cwd = os.getcwd()
    model_cls._set_model_restore_state(is_being_restored=True, folder=str(extracted))
    try:
        os.chdir(extracted)
        return model_cls.from_config_dict(config=config)
    finally:
        os.chdir(cwd)


def restore_extracted_model(spec: ModelSpec, model_cls: Any, extracted: Path) -> Any:
    """Restore a model from an extracted checkpoint directory, fast.

    Overlaps weight I/O (background thread, straight to CUDA) with model
    construction (calling thread), then assigns the loaded tensors as the
    model's parameters. Raises on any problem so callers can fall back to
    NeMo's stock ``restore_from``.
    """
    torch = _import_torch()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for fast checkpoint restore.")
    # The loader thread touches CUDA immediately; create the context here so
    # the thread never races first-time context initialization.
    torch.cuda.init()

    local_safe = _local_safetensors_path(spec)
    bind_safe = safetensors_path_for(extracted)
    if _local_mirror_fresh(spec, extracted):
        weights_path, use_safetensors, weights_source = local_safe, True, "local fp16 safetensors"
    elif bind_safe.is_file():
        weights_path, use_safetensors, weights_source = bind_safe, True, "fp16 safetensors"
    else:
        weights_path, use_safetensors, weights_source = (
            extracted / _NEMO_WEIGHTS_CKPT,
            False,
            "fp32 checkpoint",
        )
    from_local_mirror = weights_path is local_safe
    print(
        f"Restoring {spec.model_id} from extracted checkpoint ({weights_source})",
        flush=True,
    )

    state_box: list[Any] = []
    load_error: list[BaseException] = []
    weights_elapsed_box: list[float] = []

    def _load_weights() -> None:
        weights_started = time.perf_counter()
        try:
            state_box.append(_load_weights_on_cuda(weights_path, use_safetensors))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            load_error.append(exc)
        finally:
            weights_elapsed_box.append(time.perf_counter() - weights_started)

    weights_thread = threading.Thread(
        target=_load_weights, name="parakeet-weights-load", daemon=True
    )
    weights_thread.start()

    construct_started = time.perf_counter()
    construct_error: BaseException | None = None
    instance: Any = None
    try:
        config = _load_model_config(extracted)
        instance = _construct_model_from_config(model_cls, config, extracted)
    except BaseException as exc:  # noqa: BLE001 - always join the loader thread
        construct_error = exc
    construct_elapsed = time.perf_counter() - construct_started

    weights_thread.join()
    load_started = time.perf_counter()
    try:
        if construct_error is not None:
            raise construct_error
        if load_error:
            raise load_error[0]
        state = _unwrap_state_dict(state_box[0])
        # assign=True makes the already-GPU tensors the parameters directly:
        # no CPU staging copy and no copy into randomly initialized weights.
        instance.load_state_dict(state, strict=True, assign=True)
        # Move any remaining CPU-resident buffers/attributes (parameters are
        # already on CUDA after the assignment).
        instance.to(torch.device("cuda"))
        if use_safetensors:
            # Stored FP16 keeps disk reads minimal; the standard load sequence
            # rebuilds attention/decoding in FP32 before its final FP16 cast.
            instance.to(torch.float32)
    finally:
        model_cls._set_model_restore_state(is_being_restored=False)
    load_elapsed = time.perf_counter() - load_started

    total_elapsed = time.perf_counter() - started
    print(
        f"Restored {spec.model_id} in {total_elapsed:.2f}s "
        f"(weights {weights_elapsed_box[0]:.2f}s overlapped with construction "
        f"{construct_elapsed:.2f}s; state assign {load_elapsed:.2f}s)",
        flush=True,
    )
    if not from_local_mirror:
        _spawn_background_fast_cache(spec, extracted)
    return instance


# ---------------------------------------------------------------------------
# One-time FP16 safetensors conversion + container-local mirror
# ---------------------------------------------------------------------------


def _fp16_state_dict_from_ckpt(ckpt: Path) -> dict:
    torch = _import_torch()
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    state = _unwrap_state_dict(state)
    # Cast in place so each FP32 tensor is released as soon as converted.
    for key in list(state.keys()):
        tensor = state[key]
        if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
            state[key] = tensor.to(torch.float16)
    return state


def convert_ckpt_to_fp16_safetensors(extracted: Path) -> bool:
    """Convert ``model_weights.ckpt`` into FP16 safetensors beside it.

    Returns True when the file was written, False when it already existed or
    the source checkpoint is missing. Raises on unreadable/corrupt input.
    """
    dest = safetensors_path_for(extracted)
    if dest.is_file():
        return False
    ckpt = extracted / _NEMO_WEIGHTS_CKPT
    if not ckpt.is_file():
        return False
    state = _fp16_state_dict_from_ckpt(ckpt)
    _save_safetensors_atomic(state, dest)
    return True


def _save_safetensors_atomic(tensors: dict, dest: Path) -> None:
    from safetensors.torch import save_file

    tmp_file = dest.with_name(dest.name + ".tmp")
    try:
        save_file(tensors, str(tmp_file))
        os.replace(tmp_file, dest)
    finally:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)


def _copy_file_atomic(src: Path, dest: Path) -> None:
    tmp_file = dest.with_name(dest.name + ".tmp")
    try:
        shutil.copyfile(src, tmp_file)
        os.replace(tmp_file, dest)
    finally:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)


def _ensure_fast_caches(spec: ModelSpec, extracted: Path) -> None:
    """Build the persistent FP16 file and the container-local mirror.

    Runs on a background thread after any restore that did not already read
    from the local mirror, so the next cold load uses the fastest source.
    """
    bind_safe = safetensors_path_for(extracted)
    local_dir = local_weights_dir(spec)
    local_safe = local_dir / _NEMO_WEIGHTS_SAFETENSORS

    if not bind_safe.is_file():
        ckpt = extracted / _NEMO_WEIGHTS_CKPT
        if not ckpt.is_file():
            return
        state = _fp16_state_dict_from_ckpt(ckpt)
        # Write the local mirror first (fast disk), then the persistent copy.
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            _save_safetensors_atomic(state, local_safe)
        except OSError as exc:
            logger.warning("Local weights mirror unavailable for %s: %s", spec.model_id, exc)
        _save_safetensors_atomic(state, bind_safe)
        print(
            f"Converted {spec.model_id} weights to FP16 safetensors "
            f"for faster future loads ({bind_safe}).",
            flush=True,
        )
        return

    if not _local_mirror_fresh(spec, extracted):
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            _copy_file_atomic(bind_safe, local_safe)
            print(
                f"Mirrored {spec.model_id} FP16 weights to container-local storage "
                f"({local_safe}) for faster future loads.",
                flush=True,
            )
        except OSError as exc:
            logger.warning("Local weights mirror unavailable for %s: %s", spec.model_id, exc)


def _spawn_background_fast_cache(spec: ModelSpec, extracted: Path) -> None:
    """Fire-and-forget cache building so the next load uses the fast path."""

    def _work() -> None:
        try:
            _ensure_fast_caches(spec, extracted)
        except Exception as exc:  # pragma: no cover - corrupt cache edge case
            logger.warning("Background fast-cache build failed for %s: %s", spec.model_id, exc)

    threading.Thread(target=_work, name="parakeet-fast-cache", daemon=True).start()
