# Split App Model Support Matrix

This matrix describes the intended two-app architecture in this repository. `transcribe_ui.py` still exists as the older mixed entrypoint, but the split surfaces below are the current setup and documentation target.

## Active Split-App Support

| Model | NeMo app | Transformers app | Backend | Local artifact | Key notes |
| --- | --- | --- | --- | --- | --- |
| `nvidia/parakeet-tdt-0.6b-v3` | Supported | No | NeMo | `local_models/parakeet-0.6b-v3.nemo` | Word-level timestamps, long-audio chunking, ITN support |
| `ibm-granite/granite-4.0-1b-speech` | No | Supported | Transformers Granite adapter | `local_models/granite-4.0-1b-speech` | Smallest supported phase-1 Transformers model |
| `mistralai/Voxtral-Mini-3B-2507` | No | Supported | Transformers Voxtral adapter | `local_models/Voxtral-Mini-3B-2507` | Smallest supported Voxtral phase-1 model |
| `CohereLabs/cohere-transcribe-03-2026` | No | Supported | Transformers speech-seq2seq adapter | `local_models/cohere-transcribe-03-2026` | Requires `trust_remote_code=True`; HF access may be gated |
| `mistralai/Voxtral-Small-24B-2507` | No | Supported | Transformers Voxtral adapter | `local_models/Voxtral-Small-24B-2507` | Large offline model with very high VRAM demand |

## Deferred From the Split Apps

| Model | NeMo app | Transformers app | Status | Reason |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-ASR-1.7B` | No | Deferred | Not part of phase 1 | Requires the separate `qwen-asr` runtime path rather than the strict Transformers stack |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` | No | Deferred | Not part of phase 1 | Required realtime runtime is not exposed by the current Transformers build in this repo |

## App Routing

Use this quick routing rule when choosing an entrypoint:

| Need | Use |
| --- | --- |
| Word-level timestamps, NeMo direct-audio chunking, ITN | `transcribe_nemo_app.py` |
| Granite, Cohere, or supported offline Voxtral models | `transcribe_transformers_app.py` |
| Qwen runtime experiments | Not covered by the split apps yet |
| Voxtral Realtime experiments | Not covered by the split apps yet |

## Export and Artifact Notes

| Surface | Behavior |
| --- | --- |
| Exports from both apps | TXT, SRT, and CSV are generated when saving is enabled |
| NeMo timestamps | True model timestamps are available for the active Parakeet path |
| Transformers timestamps | Current phase-1 adapters do not emit aligned timestamps; export timing falls back to whole-file estimates |
| Preferred offline storage | `local_models/` |
| Shared caches | `model_cache/torch`, `model_cache/huggingface`, `model_cache/nemo`, `model_cache/tmp`, `model_cache/gradio_uploads` |
| Shared logs | `logs/transcription` and `logs/error` |