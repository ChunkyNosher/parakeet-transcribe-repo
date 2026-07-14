# NeMo feature gap audit

Audit date: 2026-07-14 (refreshed). Scope: in-process NeMo ASR Gradio app vs NeMo ASR/diarization docs and commercial first-class STT features (Deepgram Nova, Mistral Voxtral, AssemblyAI).

## Inference backend

Sole ASR path is **in-process NVIDIA NeMo** inside Docker Compose Linux GPU. Native Windows inference is unsupported.

```
Gradio → TranscriptionService → NeMoASRBackend (ASRModel.transcribe)
                              → OOM-only silence chunking
                              → optional Sortformer GPU diarization (MFCC fallback)
                              → optional summary / PII / clean format
```

Models: `nvidia/parakeet-tdt-0.6b-v3` (timestamps), `nvidia/nemotron-3.5-asr-streaming-0.6b` (offline RNNT, no timestamps).

## Implemented

| Feature | Where |
|--------|--------|
| `ASRModel.from_pretrained` + CUDA FP16 | `backend.py` |
| Local attention long-form + optional subsampling chunking factor | `backend.py` |
| Greedy batch + CUDA graph decoder | `backend.py` |
| GPU-PB `boosting_tree` / `boosting_tree_alpha` | `backend.py` + UI |
| Word timestamps via `transcribe(..., timestamps=True)` | `backend.py` |
| **Native segment timestamps** (`timestamp['segment']`) | `backend.py` → `TranscriptResult.segments` |
| Word confidence via `ConfidenceConfig` + `return_hypotheses` | `backend.py` → `WordTimestamp.confidence` → JSON |
| App silence chunking as CUDA OOM fallback only | `service.py` |
| Model warm-up / unload | `service.py`, `__main__.py` |
| `cuda-python` dependency + doctor check | `pyproject.toml`, `diagnostics.py` |
| **Sortformer GPU diarization** (ASR unload → diarize → unload Sortformer; MFCC fallback) | `diarization.py` + `service.py` |

## Commercial STT features vs NeMo vs this app

| Feature | Deepgram / Voxtral / Assembly | NeMo counterpart | App status |
|---------|-------------------------------|------------------|------------|
| Transcription | Core | Parakeet / Nemotron | Done |
| Keyterms / context bias | `keyterm` / `context_bias` / `keyterms_prompt` | GPU-PB (wired); CTC-WS / Flashlight unused | Done (GPU-PB) |
| Word / segment timestamps | Yes | `transcribe(..., timestamps=True)` | Done (Parakeet); Nemotron none |
| Speaker diarization | `diarize` / `speaker_labels` | Sortformer / ClusteringDiarizer / Multitalker | Done (Sortformer preferred; MFCC fallback) |
| Word confidence | Yes | `ConfidenceConfig` + hypotheses | Done |
| Language hint / detect | `language` / detect | Model auto LID; Canary / AmberNet | Partial — auto only; UI language discarded |
| Smart format / numerals / ITN | Deepgram `smart_format` | Model PnC + optional `nemo_text_processing` | Partial — model PnC + light regex; no ITN |
| Utterances / paragraphs | Deepgram `utterances` | Segment timestamps / VAD | Partial — native NeMo segments |
| Streaming / realtime | Flux / Voxtral Realtime | Cache-aware / Nemotron streaming | Out of scope (Nemotron used offline) |
| Long-form | Batch limits | Local attention + buffered chunking | Done (local attn); app chunk = OOM only |
| Beam + LM fusion | Rare as API | NGPU-LM / `malsd_batch` | Missing (intentional) |
| Translation / AST | Separate products | Canary multi-task | Missing (different model) |
| VAD | Often internal | MarbleNet | Missing as first-class |
| PII / redaction | Cloud policy | None in ASR core | App regex only |
| Topics / sentiment / intents / entities / free-form prompt | Cloud / LLM layers | No NeMo ASR twin | N/A as NeMo ASR (app has extractive summary) |

## Honest / deferred in product

| Feature | Notes |
|--------|--------|
| Language forcing UI | Control is **auto-only** until a real NeMo locale API is wired (`service.py` still discards language). |
| Beam / `malsd_batch` + NGPU-LM | Requires LM artifacts; slower. Out of scope for this app. |
| Buffered / cache-aware / mic streaming | Explicitly out of scope (AGENTS.md). Nemotron “streaming” checkpoint is used as offline batch ASR. |
| Cascaded ClusteringDiarizer / MSDD / Multitalker Parakeet | Heavier pipelines; Sortformer post-pass is the commercial-parity path. |
| CTC-WS context biasing | For hybrid CTC models; GPU-PB is the TDT path. |
| ITN (`nemo_text_processing`) | Deepgram smart-format numerals/dates analogue; not wired. |
| Canary ASR+translation | Different checkpoint; not a Parakeet toggle. |
| App `segments_from_words` packer | Intentionally **not** used for SRT/VTT while validating NeMo native cues. |

## GPU diarization (Sortformer) — VRAM / align plan

- **Model:** `nvidia/diar_sortformer_4spk-v1` (`SortformerEncLabelModel`), offline E2E, typically 4 speakers max.
- **API:** `diar_model.diarize(audio=[path], batch_size=1)` → `(begin_s, end_s, speaker_index)` segments.
- **VRAM:** Service unloads ASR before Sortformer load; Sortformer is unloaded after labeling so the next file can reload ASR. On Sortformer failure / missing CUDA / import error → CPU MFCC fallback (same `diarize_transcript` API).
- **Align:** Overlap Sortformer RTTM-style segments onto Parakeet word timestamps; majority-vote speakers onto native NeMo cue segments (boundaries unchanged).

Sources: [NeMo diarization intro](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html), [HF Sortformer](https://huggingface.co/nvidia/diar_sortformer_4spk-v1).

## Explicitly out of scope

- Live microphone streaming UI
- Riva / NIM serving
- Dual Transformers + NeMo backends
- Cloud ASR API wrappers
- Native Windows inference

## Not NeMo ASR features

Do not treat as “missing NeMo”: topics, sentiment, intents, entity dashboards, Assembly-style free-text `prompt`, hosted speaker identification by name.
