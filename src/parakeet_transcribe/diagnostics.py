from __future__ import annotations

import shutil
from pathlib import Path


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
    cache = Path("model_cache/huggingface")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        lines.append(f"- OK model cache: {cache.resolve()}")
    except OSError as exc:
        lines.append(f"- MISSING model cache: {exc}")
        ready = False
    return ready, "\n".join(lines)
