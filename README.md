# Parakeet Transcribe

Local file transcription using **NVIDIA NeMo** ASR (Parakeet / Nemotron) inside a Linux GPU Docker container.

## What it runs

- **NVIDIA Parakeet TDT 0.6B v3** is the default: 25 European languages, automatic language detection, punctuation, and word/segment timestamps via NeMo `transcribe(..., timestamps=True)`.
- **NVIDIA Parakeet TDT 0.6B v2** is the English-only alternative with the family's best English WER and the same timestamp exports.
- **NVIDIA Parakeet TDT 1.1B** is the larger English-only checkpoint (higher VRAM) with the same timestamp exports, but lowercase unpunctuated output. This legacy checkpoint requires `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, which `compose.yaml` already sets.
- **NVIDIA Nemotron 3.5 ASR Streaming 0.6B** is optional: broader language coverage and automatic language detection, but no timestamped subtitle exports.

NeMo-backed features in this app:

- Long-form transcription with FastConformer **local attention** (`rel_pos_local_attn`)
- **GPU-PB keyterm boosting** (phrase list + boost strength)
- Greedy batch decoding with **CUDA graph decoder**
- Silence-aware **chunking only as CUDA OOM fallback**

This app intentionally does **not** ship Riva/NIM, live microphone streaming, dual Transformers+NeMo backends, or cloud ASR API wrappers. Native Windows `uv run` inference is unsupported; use Docker Compose.

Optional local extras (off by default):

- Speaker diarization via CPU MFCC clustering aligned to Parakeet word timestamps
- Extractive summary + chapters from pause gaps
- Regex PII redaction and light clean formatting

## Media and YouTube input

The file picker accepts any uploaded file. FFmpeg determines whether it contains a decodable audio stream, then converts it to canonical mono 16 kHz audio. This covers common containers and codecs including M4A/AAC, MP3, WAV, FLAC, OGG, MP4, MOV, MKV, AVI, and WebM, plus any other format supported by the installed FFmpeg build.

Paste one YouTube video URL into **YouTube video URL** and choose **Transcribe YouTube** to download its best available audio stream and run it through the same local NeMo pipeline. It does not process playlists. Make sure you have permission to download and transcribe the video, and note that YouTube may occasionally require an updated `yt-dlp` dependency when it changes its delivery mechanisms.

## Setup (Docker Compose — supported path)

Docker Desktop on Windows must be configured for Linux containers with NVIDIA GPU support. The Compose service reserves one GPU, publishes Gradio only to `127.0.0.1:7860`, and stores downloads plus generated files under `docker-data/` on the host. The image installs `build-essential` and `libsndfile1` because NeMo/PyTorch need them at runtime.

```powershell
docker compose up --build --watch
```

`--watch` is the day-to-day loop: `./src` is bind-mounted into the container, so Python edits sync from the host and Compose restarts the app process automatically. You do **not** need `--build` for normal code patches.

Rebuild the image only when dependencies or the Dockerfile change (`pyproject.toml`, `uv.lock`, system packages). Rebuilds reuse a local BuildKit cache at `docker-data/build-cache` (apt + `uv` download caches and layer metadata). The first build is still slow; later dependency rebuilds should be much faster. That cache directory is gitignored with the rest of `docker-data/`.

Without watch, after editing `src/` you can still skip a rebuild and just restart:

```powershell
docker compose restart
```

Open `http://127.0.0.1:7860`. Model weights download once into the host bind mount `docker-data/model_cache` (not baked into the image). On container start the app warms the default Parakeet model into VRAM in the background so the first transcription skips that cold load; use **Unload model** if you need the memory back. Exports are written to `docker-data/outputs` (`PARAKEET_OUTPUT_DIR=/data/outputs`) and are served through Gradio via `allowed_paths`. Stop the service with `docker compose down`.


Inside the container you can also run:

```bash
parakeet-transcribe doctor
```

### Throughput knobs

- **Keyterms** + **boost strength**: NeMo GPU-PB shallow fusion for proper nouns / rare phrases.
- **Chunk batch size** (UI, 1–16): used when long-form local attention OOMs and the service falls back to chunked transcription. Default is 2.
- Keep the model loaded between files in a session; only use **Unload model** when you need VRAM back.

## Host development (tests / lint only)

Install Python 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg on `PATH` for local unit tests. NeMo is a Linux-only dependency and is not required for mocked host tests. Gradio is pinned to 5.x because NeMo’s Transformers pin requires `huggingface-hub<1.0`, which is incompatible with Gradio 6.

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run parakeet-transcribe doctor
```

`doctor` / `run` on native Windows report that inference must use Docker Compose. Do not expect CUDA ASR on the Windows host.

## Model notices

See the NVIDIA model cards for [Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [Parakeet v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [Parakeet 1.1B](https://huggingface.co/nvidia/parakeet-tdt-1.1b), and [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for licensing, language coverage, and limitations.
