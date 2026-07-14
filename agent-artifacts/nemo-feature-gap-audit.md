# NeMo feature gap audit

Audit date: 2026-07-14. Scope: in-process NeMo ASR Gradio app vs NeMo ASR docs (timestamps, word boosting, confidence, streaming, LM fusion).

## Implemented

| Feature | Where |
|--------|--------|
| `ASRModel.from_pretrained` + CUDA FP16 | `backend.py` |
| Local attention long-form + optional subsampling chunking factor | `backend.py` |
| Greedy batch + CUDA graph decoder | `backend.py` |
| GPU-PB `boosting_tree` / `boosting_tree_alpha` | `backend.py` + UI |
| Word timestamps via `transcribe(..., timestamps=True)` | `backend.py` |
| **Native segment timestamps** (`timestamp['segment']`) | `backend.py` → `TranscriptResult.segments` (primary cue source) |
| App silence chunking as CUDA OOM fallback only | `service.py` |
| Model warm-up / unload | `service.py`, `__main__.py` |
| `cuda-python` dependency + doctor check | `pyproject.toml`, `diagnostics.py` |

## Honest / deferred in product

| Feature | Notes |
|--------|--------|
| Language forcing UI | Control is **auto-only** until a real NeMo locale API is wired (`service.py` still discards language). |
| Word / token confidence (`confidence_cfg`) | Documented in NeMo confidence tutorial; not wired. Optional follow-up. |
| Beam / `malsd_batch` + NGPU-LM | Requires LM artifacts; slower. Out of scope for this app. |
| Buffered / cache-aware / mic streaming | Explicitly out of scope (AGENTS.md). Nemotron “streaming” checkpoint is used as offline batch ASR. |
| NeMo MSDD / Sortformer diarization | Local MFCC clustering only; NeMo diarization spike rejected. |
| CTC-WS context biasing | For hybrid CTC models; GPU-PB is the TDT path. |
| App `segments_from_words` packer | Intentionally **not** used for SRT/VTT while validating NeMo native cues. Packer helpers remain in `chunking.py` for OOM audio split / historical tests only. |

## Explicitly out of scope

- Live microphone streaming UI
- Riva / NIM serving
- Dual Transformers + NeMo backends
- Cloud ASR API wrappers
- Native Windows inference
