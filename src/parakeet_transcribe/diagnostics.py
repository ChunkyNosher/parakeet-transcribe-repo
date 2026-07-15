from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def inference_runtime_supported() -> tuple[bool, str]:
    """Inference is supported in Linux environments (Docker Compose GPU container)."""

    if sys.platform == "win32" and os.environ.get("PARAKEET_ALLOW_NATIVE_WINDOWS", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return (
            False,
            "- ACTION Native Windows inference is not supported. Use Docker Compose "
            "(`docker compose up --build`) for the NeMo Linux GPU runtime.",
        )
    return True, f"- OK inference platform: {sys.platform}"


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

    platform_ok, platform_line = inference_runtime_supported()
    lines.append(platform_line)
    ready = ready and platform_ok

    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        lines.append(f"- {'OK' if found else 'MISSING'} {tool}: {found or 'not on PATH'}")
        ready = ready and found is not None

    if sys.platform == "linux":
        try:
            import nemo.collections.asr as nemo_asr  # noqa: F401

            lines.append("- OK NeMo ASR import")
        except ImportError:
            lines.append("- MISSING NeMo ASR: rebuild the Docker image (`uv sync` with nemo_toolkit[asr])")
            ready = False
        try:
            import torch

            cuda = torch.cuda.is_available()
            lines.append(f"- {'OK' if cuda else 'MISSING'} Torch: {torch.__version__}")
            lines.append(f"- {'OK' if cuda else 'MISSING'} CUDA build: {torch.version.cuda or 'CPU-only'}")
            if cuda:
                lines.append(f"- OK GPU: {torch.cuda.get_device_name(0)}")
            else:
                lines.append("- ACTION CUDA is required inside the Linux GPU container.")
            ready = ready and cuda
        except ImportError:
            lines.append("- MISSING Torch: rebuild the Docker image so NeMo/PyTorch install correctly.")
            ready = False
        compiler_ok, compiler_line = _linux_triton_compiler_ready()
        if compiler_line:
            lines.append(compiler_line)
        ready = ready and compiler_ok
        try:
            import cuda  # noqa: F401

            lines.append("- OK cuda-python (NeMo CUDA-graph while-loops)")
        except ImportError:
            try:
                import cuda.bindings  # noqa: F401

                lines.append("- OK cuda-python (NeMo CUDA-graph while-loops)")
            except ImportError:
                lines.append(
                    "- MISSING CUDA Python bindings: rebuild the Docker image with "
                    "nemo_toolkit[asr,cu13] so NeMo can enable CUDA-graph while-loop decoding"
                )
                ready = False
    else:
        lines.append("- SKIP NeMo/CUDA checks on this host (inference runs in Docker).")

    cache = Path("model_cache/huggingface")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        lines.append(f"- OK model cache: {cache.resolve()}")
    except OSError as exc:
        lines.append(f"- MISSING model cache: {exc}")
        ready = False
    return ready, "\n".join(lines)
