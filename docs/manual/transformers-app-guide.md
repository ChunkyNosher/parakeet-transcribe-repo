# Transformers App Guide

## Purpose

`transcribe_transformers_app.py` is the strict Transformers side of the split two-app architecture. It is intentionally limited to the phase-1 Transformers model set and does not present Qwen or Voxtral Realtime as supported choices.

## Supported Model Scope

| Status | Model | Backend | Local artifact | Notes |
| --- | --- | --- | --- | --- |
| Supported in this app | `ibm-granite/granite-4.0-1b-speech` | Transformers Granite adapter | `local_models/granite-4.0-1b-speech` | Smallest supported phase-1 model and safest first test |
| Supported in this app | `mistralai/Voxtral-Mini-3B-2507` | Transformers Voxtral adapter | `local_models/Voxtral-Mini-3B-2507` | Smallest supported Voxtral option |
| Supported in this app | `CohereLabs/cohere-transcribe-03-2026` | Transformers speech-seq2seq adapter | `local_models/cohere-transcribe-03-2026` | Uses `trust_remote_code=True`; HF access may be gated |
| Supported in this app | `mistralai/Voxtral-Small-24B-2507` | Transformers Voxtral adapter | `local_models/Voxtral-Small-24B-2507` | Large offline option; workstation-class VRAM is expected |

## Deferred Models

These models are intentionally not supported by the split Transformers app today:

| Model | Status | Reason |
| --- | --- | --- |
| `Qwen/Qwen3-ASR-1.7B` | Deferred from the split app | Requires the separate `qwen-asr` runtime path rather than the strict Transformers phase-1 stack |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` | Deferred from the split app | The required realtime runtime is not exposed by the current Transformers build used in this repo |

The NeMo Parakeet path is also intentionally out of scope here; use `transcribe_nemo_app.py` for that backend.

## Setup

The current repository environment is still shared across both apps. Use the existing `environment.yml` plus `requirements.txt`, or the already configured `nvidia-asr` environment.

Prepare supported local snapshots with the app-specific setup script:

```bash
python setup_local_models_transformers.py --status
python setup_local_models_transformers.py --download all
```

Download just one supported artifact when needed:

```bash
python setup_local_models_transformers.py --download granite-4.0-1b-speech
python setup_local_models_transformers.py --download voxtral-mini-3b-2507
python setup_local_models_transformers.py --download cohere-transcribe-03-2026
python setup_local_models_transformers.py --download voxtral-small-24b-2507
```

What the setup script does:

- Snapshots only the phase-1 supported Transformers models
- Leaves Qwen and Voxtral Realtime out of the selectable setup flow
- Writes artifacts into `local_models/` so the app can prefer them before any Hugging Face fallback

## Launch

Run the app from the repository root:

```bash
python transcribe_transformers_app.py
```

The app launches a local Gradio server on `http://127.0.0.1:7860` and opens a browser window automatically. Run only one of the two split apps at a time because both default to the same local port.

## What the App Supports

- Audio inputs: WAV, MP3, FLAC, M4A, OGG, AAC, WMA
- Video inputs: MP4, AVI, MKV, MOV, WEBM, FLV, M4V
- Batch uploads
- TXT, SRT, and CSV exports
- Granite and Cohere batching within an adapter request
- Voxtral single-file inference per request

## Current Limitations

- The phase-1 Transformers adapters do not currently produce aligned timestamps
- TXT, SRT, and CSV files are still generated, but SRT and CSV timing falls back to whole-file estimates instead of word timing
- Cohere requires `trust_remote_code=True`
- Voxtral Small 24B is practical only on very large-memory GPUs or with aggressive offloading

## Artifact and Cache Layout

The split Transformers app uses the shared project-local cache bootstrap in `app_shared/env_bootstrap.py`.

| Location | Purpose |
| --- | --- |
| `local_models/granite-4.0-1b-speech` | Preferred Granite local snapshot |
| `local_models/Voxtral-Mini-3B-2507` | Preferred Voxtral Mini local snapshot |
| `local_models/cohere-transcribe-03-2026` | Preferred Cohere local snapshot |
| `local_models/Voxtral-Small-24B-2507` | Preferred Voxtral Small local snapshot |
| `model_cache/torch` | Torch cache |
| `model_cache/huggingface` | Hugging Face cache |
| `model_cache/tmp` | Project temp directory used to avoid Windows temp-file problems |
| `model_cache/gradio_uploads` | Cached copies of uploaded files |
| `logs/transcription` | Saved transcription logs |
| `logs/error` | Saved error logs |

TXT, SRT, and CSV exports are written to the current working directory used when the app is launched. When you launch from the repo root, those files appear in the repo root.

If a supported local snapshot is missing, the app falls back to the configured Hugging Face model source when access is available.

## Validation

Use the app-specific health check before troubleshooting the UI:

```bash
python repo_healthcheck_transformers.py
```

This health check verifies:

- The Transformers-side Python imports required by the split app
- The required Transformers runtime classes for Granite, Cohere, and Voxtral
- The app-specific setup script and shared directories
- Import of `transcribe_transformers_app.py`
- That Qwen and Voxtral Realtime stay deferred rather than listed as active choices