# Diarization spike result (Phase 2A)

**Decision:** go with a **CPU-only MFCC + k-means** post-pass (`diarization.py`).

## Why not pyannote / NeMo MSDD first
- pyannote speaker-diarization models are gated (HF token), heavy, and compete for GPU VRAM with ASR.
- NeMo is explicitly out of product scope for this app.
- Local clustering keeps ASR resident, needs no extra CUDA model, and works on Windows + Docker with existing `librosa` / `numpy` deps.

## Memory plan
- Diarization runs **after** ASR decode on CPU using the already-normalized mono WAV samples in memory.
- No ASR unload required for the default path.
- Quality is below commercial Deepgram/AssemblyAI/Mistral `diarize`; UI copy states that.

## Follow-ups
- Optional future: swap backend behind the same `diarize_transcript` API if a stronger local model is approved.
