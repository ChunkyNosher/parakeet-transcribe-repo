# NeMo App Guide

## Purpose

`transcribe_nemo_app.py` is the NeMo side of the split two-app architecture. It keeps the direct NeMo audio path focused on the one model that is currently active in that app:

- `nvidia/parakeet-tdt-0.6b-v3`

The split NeMo app keeps the features that already fit the Parakeet path well: local `.nemo` artifacts, long-audio chunking, word-level timestamps, inverse text normalization, batch uploads, and TXT/SRT/CSV exports.

## Supported Model Scope

| Status | Model | Backend | Local artifact | Notes |
| --- | --- | --- | --- | --- |
| Supported in this app | `nvidia/parakeet-tdt-0.6b-v3` | NeMo | `local_models/parakeet-0.6b-v3.nemo` | Multilingual, word-level timestamps, chunking, best current fit for the split NeMo entrypoint |

## Outside This App

These models are not part of `transcribe_nemo_app.py`:

| Model | Current split status | Where it belongs |
| --- | --- | --- |
| `ibm-granite/granite-4.0-1b-speech` | Supported elsewhere | Use `transcribe_transformers_app.py` |
| `CohereLabs/cohere-transcribe-03-2026` | Supported elsewhere | Use `transcribe_transformers_app.py` |
| `mistralai/Voxtral-Mini-3B-2507` | Supported elsewhere | Use `transcribe_transformers_app.py` |
| `mistralai/Voxtral-Small-24B-2507` | Supported elsewhere | Use `transcribe_transformers_app.py` |
| `Qwen/Qwen3-ASR-1.7B` | Deferred from the split apps | Remains outside the phase-1 Transformers app because it needs the separate `qwen-asr` runtime path |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` | Deferred from the split apps | Realtime runtime is not exposed by the current Transformers stack in this repo |

## Setup

The current repository environment is still shared across both apps. Use the existing `environment.yml` plus `requirements.txt`, or the already configured `nvidia-asr` environment.

Prepare the NeMo artifact with the app-specific setup script:

```bash
python setup_local_models_nemo.py --status
python setup_local_models_nemo.py --download
```

What the setup script does:

- Saves `nvidia/parakeet-tdt-0.6b-v3` as `local_models/parakeet-0.6b-v3.nemo`
- Keeps the setup scope aligned with the active NeMo app only
- Avoids presenting Transformers, Qwen, or Voxtral Realtime as part of the NeMo setup flow

## Launch

Run the app from the repository root:

```bash
python transcribe_nemo_app.py
```

The app launches a local Gradio server on `http://127.0.0.1:7860` and opens a browser window automatically. Keep the terminal open while the UI is running.

## What the App Supports

- Audio inputs: WAV, MP3, FLAC, M4A, OGG, AAC, WMA
- Video inputs: MP4, AVI, MKV, MOV, WEBM, FLV, M4V
- Word-level timestamps when requested
- Long-audio chunking with overlap
- Batch uploads
- Inverse text normalization when `nemo_text_processing` is installed
- TXT, SRT, and CSV export files plus downloadable logs

## Artifact and Cache Layout

The split NeMo app uses the shared project-local cache bootstrap in `app_shared/env_bootstrap.py`.

| Location | Purpose |
| --- | --- |
| `local_models/parakeet-0.6b-v3.nemo` | Preferred offline NeMo artifact |
| `model_cache/torch` | Torch cache |
| `model_cache/huggingface` | Hugging Face cache |
| `model_cache/nemo` | NeMo cache |
| `model_cache/tmp` | Project temp directory used to avoid Windows temp-file problems |
| `model_cache/gradio_uploads` | Cached copies of uploaded files |
| `logs/transcription` | Saved transcription logs |
| `logs/error` | Saved error logs |

TXT, SRT, and CSV exports are written to the current working directory used when the app is launched. When you launch from the repo root, those files appear in the repo root.

If the local `.nemo` artifact is missing, the app can still fall back to the configured Hugging Face model source on first use.

## Validation

Use the app-specific health check before troubleshooting the UI:

```bash
python repo_healthcheck_nemo.py
```

This health check verifies:

- The NeMo-side Python imports required by the split app
- The app-specific setup script and shared directories
- Import of `transcribe_nemo_app.py`
- That the active model registry stays NeMo-only