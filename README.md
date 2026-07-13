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

## Docker Compose

Docker Desktop on Windows must be configured for Linux containers with NVIDIA GPU support. The Compose service reserves one GPU, publishes Gradio only to `127.0.0.1:7860`, and stores downloads plus generated files under `docker-data/` on the host.

```powershell
docker compose up --build
```

Open `http://127.0.0.1:7860`. First model use downloads into `docker-data/model_cache`; exports are written to `docker-data/outputs` and remain available through the UI. Stop the service with `docker compose down`.

## Model notices

See the NVIDIA model cards for [Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for licensing, language coverage, and limitations.
