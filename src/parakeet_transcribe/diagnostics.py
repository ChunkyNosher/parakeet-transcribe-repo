from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _linux_triton_compiler_ready() -> tuple[bool, str]:
    """Linux torch pulls Triton, which JIT-builds helpers and needs a C compiler."""

    if sys.platform != "linux":
        return True, ""
    try:
        import triton  # noqa: F401
    except ImportError:
        return True, ""
    cc = os.environ.get("CC")
    if cc and (Path(cc).exists() or shutil.which(cc)):
        return True, f"- OK Triton C compiler: CC={cc}"
    for tool in ("gcc", "cc", "clang"):
        found = shutil.which(tool)
        if found:
            return True, f"- OK Triton C compiler: {found}"
    return (
        False,
        "- MISSING Triton C compiler: install build-essential/gcc or set CC "
        "(required by Linux PyTorch Triton helpers; rebuild the Docker image if using Compose)",
    )


def doctor_report() -> tuple[bool, str]:
    lines: list[str] = ["# Parakeet Transcribe diagnostics", ""]
    ready = True
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        lines.append(f"- {'OK' if found else 'MISSING'} {tool}: {found or 'not on PATH'}")
        ready = ready and found is not None
    try:
        import torch

        cuda = torch.cuda.is_available()
        lines.append(f"- {'OK' if cuda else 'MISSING'} Torch: {torch.__version__}")
        lines.append(f"- {'OK' if cuda else 'MISSING'} CUDA build: {torch.version.cuda or 'CPU-only'}")
        if cuda:
            lines.append(f"- OK GPU: {torch.cuda.get_device_name(0)}")
        else:
            lines.append("- ACTION Install the CUDA PyTorch wheel with `uv sync` from this project.")
        ready = ready and cuda
    except ImportError:
        lines.append("- MISSING Torch: run `uv sync`")
        ready = False
    compiler_ok, compiler_line = _linux_triton_compiler_ready()
    if compiler_line:
        lines.append(compiler_line)
    ready = ready and compiler_ok
    cache = Path("model_cache/huggingface")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        lines.append(f"- OK model cache: {cache.resolve()}")
    except OSError as exc:
        lines.append(f"- MISSING model cache: {exc}")
        ready = False
    return ready, "\n".join(lines)
