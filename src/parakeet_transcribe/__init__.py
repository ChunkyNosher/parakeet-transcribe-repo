"""NVIDIA NeMo local transcription application (Docker Linux GPU)."""

import os
from pathlib import Path

__version__ = "1.0.0"

# Configure HF cache before NeMo / Hugging Face downloads. Native Windows
# environments commonly deny writes to the inherited user-level cache.
APP_ROOT = Path(__file__).resolve().parents[2]
HF_HOME = APP_ROOT / "model_cache" / "huggingface"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
