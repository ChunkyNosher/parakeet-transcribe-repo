import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional


_BOOTSTRAP_STATE: Optional[Dict[str, Path]] = None


def _reconfigure_standard_streams() -> None:
    """Prefer UTF-8 stdout/stderr when the host supports reconfigure()."""

    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(stdout_reconfigure):
        try:
            stdout_reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if callable(stderr_reconfigure):
        try:
            stderr_reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _configure_threading_environment() -> None:
    """Pin shared runtime defaults before heavyweight ML imports."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _resolve_app_root(app_root: Optional[Path] = None) -> Path:
    if app_root is not None:
        return app_root.absolute()
    return Path(__file__).resolve().parent.parent.absolute()


def bootstrap_environment(app_root: Optional[Path] = None, verbose: bool = True) -> Dict[str, Path]:
    """Configure project-local cache and temp paths for Windows-safe runtime use."""

    global _BOOTSTRAP_STATE

    if _BOOTSTRAP_STATE is not None and app_root is None:
        return _BOOTSTRAP_STATE

    _reconfigure_standard_streams()
    _configure_threading_environment()

    resolved_root = _resolve_app_root(app_root)
    cache_dir = resolved_root / "model_cache"
    temp_dir = cache_dir / "tmp"
    torch_cache_dir = cache_dir / "torch"
    hf_cache_dir = cache_dir / "huggingface"
    nemo_cache_dir = cache_dir / "nemo"
    gradio_uploads_dir = cache_dir / "gradio_uploads"

    for path in (cache_dir, temp_dir, torch_cache_dir, hf_cache_dir, nemo_cache_dir, gradio_uploads_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_HOME"] = str(torch_cache_dir)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["NEMO_CACHE_DIR"] = str(nemo_cache_dir)
    os.environ["TMPDIR"] = str(temp_dir)

    if sys.platform == "win32":
        os.environ["TEMP"] = str(temp_dir)
        os.environ["TMP"] = str(temp_dir)

    tempfile.tempdir = str(temp_dir)
    actual_temp = tempfile.gettempdir()

    if verbose:
        if actual_temp != str(temp_dir):
            print(f"WARNING: tempfile.gettempdir() returned {actual_temp}")
            print(f"Expected: {temp_dir}")
            print("This may cause file locking issues!")
        else:
            print(f"Temp directory verified: {temp_dir}")

        print("Using tensor-based transcription helpers for Windows-safe temp handling")

    state = {
        "app_root": resolved_root,
        "cache_dir": cache_dir,
        "temp_dir": temp_dir,
        "torch_cache_dir": torch_cache_dir,
        "hf_cache_dir": hf_cache_dir,
        "nemo_cache_dir": nemo_cache_dir,
        "gradio_uploads_dir": gradio_uploads_dir,
    }

    if app_root is None:
        _BOOTSTRAP_STATE = state

    return state


_BOOTSTRAP_STATE = bootstrap_environment()
APP_ROOT = _BOOTSTRAP_STATE["app_root"]
CACHE_DIR = _BOOTSTRAP_STATE["cache_dir"]
TEMP_DIR = _BOOTSTRAP_STATE["temp_dir"]
TORCH_CACHE_DIR = _BOOTSTRAP_STATE["torch_cache_dir"]
HF_CACHE_DIR = _BOOTSTRAP_STATE["hf_cache_dir"]
NEMO_CACHE_DIR = _BOOTSTRAP_STATE["nemo_cache_dir"]
GRADIO_UPLOADS_DIR = _BOOTSTRAP_STATE["gradio_uploads_dir"]


def get_script_dir() -> Path:
    """Return the repository root used by the app entrypoints."""

    return APP_ROOT


__all__ = [
    "APP_ROOT",
    "CACHE_DIR",
    "GRADIO_UPLOADS_DIR",
    "HF_CACHE_DIR",
    "NEMO_CACHE_DIR",
    "TEMP_DIR",
    "TORCH_CACHE_DIR",
    "bootstrap_environment",
    "get_script_dir",
]