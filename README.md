# Parakeet Transcribe

Local file transcription for Windows using NVIDIA ASR checkpoints only.

## What it runs

- **NVIDIA Parakeet TDT 0.6B v3** is the default: 25 European languages, automatic language detection, punctuation, and word/segment timestamps.
- **NVIDIA Nemotron 3.5 ASR Streaming 0.6B** is optional: broader language coverage and automatic language detection, but no timestamped subtitle exports.

This is an offline file-transcription app. It intentionally does not ship NeMo, diarization, translation, Riva/NIM, or live microphone streaming.

## Setup

Install Python 3.12, current NVIDIA drivers, [uv](https://docs.astral.sh/uv/), and FFmpeg/FFprobe on `PATH`.

```powershell
uv sync --extra dev
uv run parakeet-transcribe doctor
uv run parakeet-transcribe
```

The lock file records the full dependency graph. The project explicitly installs the CUDA 13.0 Windows PyTorch wheel; `doctor` fails clearly if CUDA is unavailable instead of silently using CPU inference.

Models download on first use into `model_cache/huggingface`. Every completed run is saved below `outputs/` with individual artifacts and a ZIP bundle.

## Development

```powershell
uv run pytest
uv run ruff check .
```

## Model notices

See the NVIDIA model cards for [Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for licensing, language coverage, and limitations.
