# Parakeet Transcribe

Local file transcription using **NVIDIA NeMo** ASR (Parakeet / Nemotron) inside a Linux GPU Docker container.

## What it runs

- **NVIDIA Parakeet TDT 0.6B v3** is the default: 25 European languages, automatic language detection, punctuation, and word/segment timestamps via NeMo `transcribe(..., timestamps=True)`.
- **NVIDIA Parakeet TDT 0.6B v2** is the English-only alternative with the family's best English WER and the same timestamp exports.
- **NVIDIA Parakeet TDT 1.1B** is the larger English-only checkpoint (higher VRAM) with the same timestamp exports. It was trained on normalized text, so its tokenizer has no punctuation or capitalization tokens — the app restores these automatically via the `1-800-BAD-CODE/punctuation_fullstop_truecase_english` ONNX model (CPU), rebuilding sentence-based SRT/VTT cues from the ASR word timestamps. NeMo's own `PunctuationCapitalizationModel` was removed in NeMo ≥2.5, which is why a standalone model is used. This legacy checkpoint also requires `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, which `compose.yaml` already sets.
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

Open `http://127.0.0.1:7860`. Model weights download once into the host bind mount `docker-data/model_cache` (not baked into the image). Models are **not** loaded at startup: the first transcription request loads the selected model into VRAM. After 3 minutes of inactivity (`PARAKEET_IDLE_UNLOAD_SECONDS`, default `180`; `0` keeps it in VRAM) the model is **parked in system RAM** — VRAM is freed and the next request revives it in seconds — unless `PARAKEET_IDLE_PARK=0` drops it entirely. Use **Unload model** to free VRAM *and* RAM immediately. Cold loads are layered: checkpoints are pre-extracted once into `docker-data/model_cache/extracted`, converted once to FP16 safetensors (mirrored to fast container-local storage at startup via `PARAKEET_CACHE_PREWARM`), and after the first successful load a ready-state snapshot (local-attention config + post-reconfiguration FP16 weights) is saved so later cold loads skip NeMo's attention/decoding rebuilds and the tar decompression entirely. Triton's JIT cache is persisted under `docker-data/model_cache/triton` so JIT compilation is paid once, not per container. Exports are written to `docker-data/outputs` (`PARAKEET_OUTPUT_DIR=/data/outputs`) and are served through Gradio via `allowed_paths`. Stop the service with `docker compose down`.


Inside the container you can also run:

```bash
parakeet-transcribe doctor
```

### Throughput knobs

- **Keyterms** + **boost strength**: NeMo GPU-PB shallow fusion for proper nouns / rare phrases.
- **Chunk batch size** (UI, 1–16): used when long-form local attention OOMs and the service falls back to chunked transcription. Default is 2.
- **Idle park**: after `PARAKEET_IDLE_UNLOAD_SECONDS` (default 180) of inactivity the model moves from VRAM to system RAM (freeing VRAM; ~1.3 GB RAM for a 0.6B FP16 model) and the next request revives it in ~1-3s. `0` disables idle eviction (model stays in VRAM); `PARAKEET_IDLE_PARK=0` drops the model entirely instead of parking (frees all memory, next request pays a cold load).
- **Cold-load caches**: a disk-only startup pre-warm (`PARAKEET_CACHE_PREWARM`, default on; extra models via `PARAKEET_PREWARM_MODELS`) extracts checkpoints, builds the one-time FP16 safetensors, and mirrors weights to container-local storage before the first request. After the first successful load of a model, a ready-state snapshot makes later cold loads skip the attention/decoding rebuilds. The first-ever load of each model is the only slow one.
- **Inference mode override** (`PARAKEET_FORCE_INFERENCE_MODE=auto|offline|streaming`): `auto` (default) routes audio >30s to NeMo buffered streaming and shorter audio to offline long-form. `offline` / `streaming` force one path for A/B testing; the runtime records the effective mode in each result's `runtime.inference_mode`.

## Host development (tests / lint only)

Install Python 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg on `PATH` for local unit tests. NeMo is a Linux-only dependency and is not required for mocked host tests. Gradio is pinned to 5.x because NeMo’s Transformers pin requires `huggingface-hub<1.0`, which is incompatible with Gradio 6.

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run parakeet-transcribe doctor
```

`doctor` / `run` on native Windows report that inference must use Docker Compose. Do not expect CUDA ASR on the Windows host.

### A/B quality harness (YouTube corpus)

`scripts/ab_quality.py` compares transcription quality and first-load timing across
inference modes by driving the running container's Gradio API with YouTube URLs (the
container downloads and transcribes them through the app's own YouTube pipeline). It
restarts the container with a temporary Compose override per mode and restores the
default configuration when it finishes.

```powershell
# Corpus: one YouTube URL per line in docker-data/ab-corpus/urls.txt (or pass --urls).
# Include at least one video under ~30s and one longer so both inference paths run.
uv run python scripts/ab_quality.py --mode ready --mode standard
uv run python scripts/ab_quality.py --mode offline --mode streaming --urls "https://youtu.be/..."
uv run python scripts/ab_quality.py --dry-run   # connect + print the plan, no transcription
```

The report (also saved under `docker-data/ab-results/`) shows per-video word counts,
mean word confidence, segment counts, effective inference mode, elapsed time, first-load
time from the container logs, and unified text diffs between modes. Optional
`--reference-dir` with `<video_id>.txt` files adds approximate WER. `--keep-mode`
leaves the container in the last tested mode instead of restoring defaults.

## Model notices

See the NVIDIA model cards for [Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [Parakeet v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [Parakeet 1.1B](https://huggingface.co/nvidia/parakeet-tdt-1.1b), and [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for licensing, language coverage, and limitations.
