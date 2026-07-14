# Parakeet Transcribe

Local file transcription for Windows using NVIDIA ASR checkpoints only.

## What it runs

- **NVIDIA Parakeet TDT 0.6B v3** is the default: 25 European languages, automatic language detection, punctuation, and word/segment timestamps.
- **NVIDIA Nemotron 3.5 ASR Streaming 0.6B** is optional: broader language coverage and automatic language detection, but no timestamped subtitle exports.

This is a local transcription app. It intentionally does not ship NeMo, Riva/NIM, live microphone streaming, or cloud ASR API wrappers.

Optional local extras (off by default):

- Speaker diarization via CPU MFCC clustering aligned to Parakeet word timestamps
- Extractive summary + chapters from pause gaps
- Regex PII redaction and light clean formatting

## Media and YouTube input

The file picker accepts any uploaded file. FFmpeg determines whether it contains a decodable audio stream, then converts it to canonical mono 16 kHz audio. This covers common containers and codecs including M4A/AAC, MP3, WAV, FLAC, OGG, MP4, MOV, MKV, AVI, and WebM, plus any other format supported by the installed FFmpeg build.

Paste one YouTube video URL into **YouTube video URL** and choose **Transcribe YouTube** to download its best available audio stream and run it through the same local NVIDIA ASR pipeline. It does not process playlists. Make sure you have permission to download and transcribe the video, and note that YouTube may occasionally require an updated `yt-dlp` dependency when it changes its delivery mechanisms.

## Setup

Install Python 3.12, current NVIDIA drivers, [uv](https://docs.astral.sh/uv/), and FFmpeg/FFprobe on `PATH`.

```powershell
uv sync --extra dev
uv run parakeet-transcribe doctor
uv run parakeet-transcribe
```

The lock file records the full dependency graph. The project explicitly installs the CUDA 13.0 Windows PyTorch wheel; `doctor` fails clearly if CUDA is unavailable instead of silently using CPU inference.

Models download on first use into `model_cache/huggingface`. Every completed run is saved below `outputs/` (or `PARAKEET_OUTPUT_DIR`) with individual artifacts and a ZIP bundle that excludes scratch audio under `.work` / `.youtube`.

### Throughput knobs

- **Chunk batch size** (UI, 1–16): higher values use more VRAM and usually raise GPU utilization. Default is 2. If CUDA OOM occurs, the service automatically retries with smaller chunks/batches.
- Keep the model loaded between files in a session; only use **Unload model** when you need VRAM back.
- Optional: set `PARAKEET_TORCH_COMPILE=1` to enable `torch.compile` on the generate path (extra warmup; off by default).

## Development

```powershell
uv run pytest
uv run ruff check .
```

## Docker Compose

Docker Desktop on Windows must be configured for Linux containers with NVIDIA GPU support. The Compose service reserves one GPU, publishes Gradio only to `127.0.0.1:7860`, and stores downloads plus generated files under `docker-data/` on the host. The image installs `build-essential` because Linux PyTorch pulls Triton, which needs a C compiler to JIT-build CUDA helpers at inference time.

```powershell
docker compose up --build
```

Open `http://127.0.0.1:7860`. First model use downloads into `docker-data/model_cache`; exports are written to `docker-data/outputs` (`PARAKEET_OUTPUT_DIR=/data/outputs`) and are served through Gradio via `allowed_paths`. Stop the service with `docker compose down`.

## Model notices

See the NVIDIA model cards for [Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for licensing, language coverage, and limitations.
