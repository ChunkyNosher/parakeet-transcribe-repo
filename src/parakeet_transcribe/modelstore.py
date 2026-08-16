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

On top of the fast restore, ``write_ready_snapshot`` / ``restore_ready_model``
persist and reload a *ready-state* snapshot: the model config with long-form
local attention and the default greedy decoding baked in (NeMo's
``ConformerEncoder`` constructs ``rel_pos_local_attn`` + ``att_context_size``
natively, so no post-hoc ``change_attention_model`` rebuild is needed), plus
FP16 weights captured *after* the load-time reconfiguration. Cold loads from
the snapshot skip the attention/decoding rebuilds and the FP32↔FP16 double
cast; the state assign validates the key layout (tolerating only NeMo's
randomly-initialized attention-bias artifacts from ``change_attention_model``)
and any other mismatch falls back to the standard restore.

Any failure in a fast path raises so callers can fall back to NeMo's stock
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
_NEMO_READY_CONFIG_YAML = "model_config_ready.yaml"
_NEMO_READY_WEIGHTS_SAFETENSORS = "model_weights_ready_fp16.safetensors"

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


def ready_config_path_for(extracted: Path) -> Path:
    """Ready-state config: original config with local attention + default decoding baked in."""
    return extracted / _NEMO_READY_CONFIG_YAML


def ready_weights_path_for(extracted: Path) -> Path:
    """FP16 weights captured after the load-time attention/decoding reconfiguration."""
    return extracted / _NEMO_READY_WEIGHTS_SAFETENSORS


def ready_snapshot_available(extracted: Path) -> bool:
    """True when both ready-state snapshot files exist for an extracted checkpoint."""
    return ready_config_path_for(extracted).is_file() and ready_weights_path_for(extracted).is_file()


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


def _local_ready_safetensors_path(spec: ModelSpec) -> Path:
    return local_weights_dir(spec) / _NEMO_READY_WEIGHTS_SAFETENSORS


def _mirror_file_fresh(local: Path, sources: tuple[Path, ...]) -> bool:
    """True when a container-local mirror file exists and is not stale."""
    if not local.is_file():
        return False
    local_mtime = local.stat().st_mtime
    return all(
        not source.is_file() or source.stat().st_mtime <= local_mtime
        for source in sources
    )


def _local_mirror_fresh(spec: ModelSpec, extracted: Path) -> bool:
    return _mirror_file_fresh(
        _local_safetensors_path(spec),
        (extracted / _NEMO_WEIGHTS_CKPT, safetensors_path_for(extracted)),
    )


def _local_ready_fresh(spec: ModelSpec, extracted: Path) -> bool:
    return _mirror_file_fresh(
        _local_ready_safetensors_path(spec),
        (ready_weights_path_for(extracted),),
    )


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


# NeMo's change_attention_model() rebuilds attention modules without honoring
# the checkpoint's `use_bias: false`, silently adding randomly-initialized
# linear_q/k/v/out bias parameters. Config-based construction cannot reproduce
# them (and their values are re-randomized on every standard-path load), so
# the ready-state restore drops exactly these keys and nothing else.
_ATTENTION_BIAS_ARTIFACT_SUFFIXES = (
    "self_attn.linear_q.bias",
    "self_attn.linear_k.bias",
    "self_attn.linear_v.bias",
    "self_attn.linear_out.bias",
)


def _is_attention_bias_artifact(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in _ATTENTION_BIAS_ARTIFACT_SUFFIXES)


def _unwrap_state_dict(state: Any) -> dict:
    """Accept either a raw state dict or a Lightning-style wrapper."""
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise ValueError("Checkpoint does not contain a usable state dict.")


def _load_config_file(path: Path) -> Any:
    from omegaconf import OmegaConf

    conf = OmegaConf.load(str(path))
    # .nemo archives sometimes store the config under a top-level `model` key.
    if "model" in conf:
        conf = conf.model
    OmegaConf.set_struct(conf, True)
    return conf


def _load_model_config(extracted: Path) -> Any:
    return _load_config_file(extracted / _NEMO_CONFIG_YAML)


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


def _restore_with_overlap(
    spec: ModelSpec,
    model_cls: Any,
    extracted: Path,
    config: Any,
    weights_path: Path,
    use_safetensors: bool,
    weights_source: str,
    from_local_mirror: bool,
    upcast_fp32: bool,
    allow_attention_bias_artifacts: bool = False,
) -> tuple[Any, bool]:
    """Shared fast-restore core used by the standard and ready-state paths.

    Overlaps weight I/O (background thread, straight to CUDA) with model
    construction from ``config`` (calling thread), then assigns the loaded
    tensors as the model's parameters. Returns ``(instance, from_local_mirror)``.
    Raises on any problem so callers can fall back.

    With ``allow_attention_bias_artifacts`` the state assign tolerates — and
    drops — the randomly-initialized attention bias keys NeMo's
    ``change_attention_model`` adds to ``use_bias: false`` checkpoints (see
    ``_is_attention_bias_artifact``); any other mismatch still raises.
    """
    torch = _import_torch()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for fast checkpoint restore.")
    # The loader thread touches CUDA immediately; create the context here so
    # the thread never races first-time context initialization.
    torch.cuda.init()

    print(f"Restoring {spec.model_id} from {weights_source}", flush=True)

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
        if allow_attention_bias_artifacts:
            result = instance.load_state_dict(state, strict=False, assign=True)
            missing = list(result.missing_keys)
            dropped = [key for key in result.unexpected_keys if _is_attention_bias_artifact(key)]
            unexpected_extra = [
                key for key in result.unexpected_keys if not _is_attention_bias_artifact(key)
            ]
            if missing or unexpected_extra:
                raise RuntimeError(
                    "Ready-state key layout mismatch "
                    f"({len(missing)} missing, {len(unexpected_extra)} unexpected); "
                    f"first missing: {missing[:2]}, first unexpected: {unexpected_extra[:2]}"
                )
            if dropped:
                print(
                    f"Dropped {len(dropped)} change_attention_model bias artifacts "
                    f"from the ready-state snapshot for {spec.model_id}.",
                    flush=True,
                )
        else:
            # strict=True doubles as the fast-path validation: any key-layout
            # mismatch raises and the caller falls back.
            instance.load_state_dict(state, strict=True, assign=True)
        # Move any remaining CPU-resident buffers/attributes (parameters are
        # already on CUDA after the assignment).
        instance.to(torch.device("cuda"))
        if upcast_fp32:
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
    return instance, from_local_mirror


def restore_extracted_model(spec: ModelSpec, model_cls: Any, extracted: Path) -> Any:
    """Restore a model from an extracted checkpoint directory, fast.

    Weight source priority: the container-local FP16 mirror, the bind-mounted
    FP16 safetensors, then the raw FP32 checkpoint. Raises on any problem so
    callers can fall back to NeMo's stock ``restore_from``.
    """
    local_safe = _local_safetensors_path(spec)
    bind_safe = safetensors_path_for(extracted)
    if _local_mirror_fresh(spec, extracted):
        weights_path, use_safetensors = local_safe, True
        weights_source, from_local_mirror = "extracted checkpoint (local fp16 safetensors)", True
    elif bind_safe.is_file():
        weights_path, use_safetensors = bind_safe, True
        weights_source, from_local_mirror = "extracted checkpoint (fp16 safetensors)", False
    else:
        weights_path, use_safetensors = extracted / _NEMO_WEIGHTS_CKPT, False
        weights_source, from_local_mirror = "extracted checkpoint (fp32 checkpoint)", False

    config = _load_model_config(extracted)
    instance, from_local_mirror = _restore_with_overlap(
        spec,
        model_cls,
        extracted,
        config,
        weights_path=weights_path,
        use_safetensors=use_safetensors,
        weights_source=weights_source,
        from_local_mirror=from_local_mirror,
        upcast_fp32=use_safetensors,
    )
    if not from_local_mirror:
        _spawn_background_fast_cache(spec, extracted)
    return instance


def restore_ready_model(spec: ModelSpec, model_cls: Any, extracted: Path) -> Any:
    """Restore a model from its ready-state snapshot.

    The snapshot config has long-form local attention and the default greedy
    decoding baked in, and its FP16 weights were captured after the load-time
    reconfiguration — so this skips the attention/decoding rebuilds and the
    FP32↔FP16 double cast entirely. Raises on any problem (including key
    mismatches caught by the strict state assign) so callers fall back to the
    standard restore.
    """
    ready_cfg = ready_config_path_for(extracted)
    if not ready_cfg.is_file():
        raise RuntimeError("Ready-state config is missing.")
    local_ready = _local_ready_safetensors_path(spec)
    bind_ready = ready_weights_path_for(extracted)
    if _local_ready_fresh(spec, extracted):
        weights_path = local_ready
        weights_source, from_local_mirror = "ready-state snapshot (local fp16 safetensors)", True
    elif bind_ready.is_file():
        weights_path = bind_ready
        weights_source, from_local_mirror = "ready-state snapshot (fp16 safetensors)", False
    else:
        raise RuntimeError("Ready-state weights are missing.")

    config = _load_config_file(ready_cfg)
    instance, from_local_mirror = _restore_with_overlap(
        spec,
        model_cls,
        extracted,
        config,
        weights_path=weights_path,
        use_safetensors=True,
        weights_source=weights_source,
        from_local_mirror=from_local_mirror,
        upcast_fp32=False,
        allow_attention_bias_artifacts=True,
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


def _write_text_atomic(text: str, dest: Path) -> None:
    tmp_file = dest.with_name(dest.name + ".tmp")
    try:
        tmp_file.write_text(text, encoding="utf-8")
        os.replace(tmp_file, dest)
    finally:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)


def write_ready_snapshot(spec: ModelSpec, model: Any, decoding_cfg: Any) -> None:
    """Persist the ready-state snapshot after a standard load.

    Saves ``model_config_ready.yaml`` (the extracted config with local
    long-form attention and the default greedy decoding baked in — NeMo's
    ``ConformerEncoder`` constructs ``rel_pos_local_attn`` natively from
    config, so no ``change_attention_model`` rebuild is needed on reload) and
    ``model_weights_ready_fp16.safetensors`` (weights captured after the
    load-time reconfiguration and FP16 cast). Idempotent: files that already
    exist are left untouched. Best-effort by contract; callers run this on a
    background thread.
    """
    from omegaconf import OmegaConf, open_dict

    extracted = extracted_dir_for(spec)
    if not (extracted / _NEMO_CONFIG_YAML).is_file():
        return
    ready_cfg = ready_config_path_for(extracted)
    ready_bind = ready_weights_path_for(extracted)
    if ready_cfg.is_file() and ready_bind.is_file():
        return

    state: dict | None = None
    if not ready_bind.is_file():
        torch = _import_torch()
        # CPU copies up front: a concurrent park()/unload() swaps parameter
        # devices in place, and these private tensors stay valid regardless.
        state = {
            key: value.detach().to("cpu")
            for key, value in model.state_dict().items()
        }
        state = {
            key: (
                value.to(torch.float16)
                if value.is_floating_point() and value.dtype != torch.float16
                else value
            )
            for key, value in state.items()
        }

    if not ready_cfg.is_file():
        config = _load_config_file(extracted / _NEMO_CONFIG_YAML)
        with open_dict(config):
            if config.encoder is not None:
                config.encoder.self_attention_model = "rel_pos_local_attn"
                config.encoder.att_context_size = [256, 256]
        config.decoding = decoding_cfg
        _write_text_atomic(OmegaConf.to_yaml(config), ready_cfg)

    if state is not None:
        local_dir = local_weights_dir(spec)
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            local_ready = local_dir / _NEMO_READY_WEIGHTS_SAFETENSORS
            _save_safetensors_atomic(state, local_ready)
            _save_safetensors_atomic(state, ready_bind)
            # The bind copy is written second; keep the mirror "fresh" so the
            # next load does not pay bind-mount read speed once.
            _mark_fresher_than(ready_bind, local_ready)
        except OSError as exc:
            logger.warning("Local ready-weights mirror unavailable for %s: %s", spec.model_id, exc)
            _save_safetensors_atomic(state, ready_bind)
        print(
            f"Saved ready-state snapshot for {spec.model_id}; future cold loads "
            "skip the attention/decoding rebuilds.",
            flush=True,
        )


def _copy_file_atomic(src: Path, dest: Path) -> None:
    tmp_file = dest.with_name(dest.name + ".tmp")
    try:
        shutil.copyfile(src, tmp_file)
        os.replace(tmp_file, dest)
    finally:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)


def _mark_fresher_than(reference: Path, target: Path) -> None:
    """Bump ``target``'s mtime just past ``reference``'s.

    Writers that save the local mirror first and the bind-mounted copy second
    would otherwise leave the mirror looking stale (bind mtime is newer) and
    cost the next load a slow bind-mount read.
    """
    try:
        reference_time = reference.stat().st_mtime
        os.utime(target, (reference_time + 1.0, reference_time + 1.0))
    except OSError:  # pragma: no cover - cosmetic freshness hint only
        pass


def _ensure_fast_caches(spec: ModelSpec, extracted: Path) -> None:
    """Build the persistent FP16 file and the container-local mirrors.

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
            _save_safetensors_atomic(state, bind_safe)
            # Keep the mirror "fresh" relative to the later-written bind copy.
            _mark_fresher_than(bind_safe, local_safe)
        except OSError as exc:
            logger.warning("Local weights mirror unavailable for %s: %s", spec.model_id, exc)
            _save_safetensors_atomic(state, bind_safe)
        print(
            f"Converted {spec.model_id} weights to FP16 safetensors "
            f"for faster future loads ({bind_safe}).",
            flush=True,
        )
    elif not _local_mirror_fresh(spec, extracted):
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

    ready_bind = ready_weights_path_for(extracted)
    if ready_bind.is_file() and not _local_ready_fresh(spec, extracted):
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            _copy_file_atomic(ready_bind, local_dir / _NEMO_READY_WEIGHTS_SAFETENSORS)
            print(
                f"Mirrored {spec.model_id} ready-state weights to container-local "
                "storage for faster future loads.",
                flush=True,
            )
        except OSError as exc:
            logger.warning("Local ready-weights mirror unavailable for %s: %s", spec.model_id, exc)


def prewarm_local_caches(spec: ModelSpec) -> bool:
    """Disk-only pre-warm of one model's caches (no model load, no CUDA).

    Ensures the persistent extraction, the one-time FP16 safetensors
    conversion, and the container-local mirrors exist before the first
    request, so a cold load reads weights at container-local disk speed
    instead of bind-mount speed. Returns True when an extracted checkpoint
    is available for the model.
    """
    extracted = ensure_extracted(spec)
    if extracted is None:
        return False
    try:
        _ensure_fast_caches(spec, extracted)
    except Exception as exc:  # pragma: no cover - corrupt cache edge case
        logger.warning("Cache pre-warm failed for %s: %s", spec.model_id, exc)
    return True


def _spawn_background_fast_cache(spec: ModelSpec, extracted: Path) -> None:
    """Fire-and-forget cache building so the next load uses the fast path."""

    def _work() -> None:
        try:
            _ensure_fast_caches(spec, extracted)
        except Exception as exc:  # pragma: no cover - corrupt cache edge case
            logger.warning("Background fast-cache build failed for %s: %s", spec.model_id, exc)

    threading.Thread(target=_work, name="parakeet-fast-cache", daemon=True).start()
