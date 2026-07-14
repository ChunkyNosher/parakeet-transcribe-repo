# Diarization spike result (Phase 2A → Sortformer upgrade)

**Current decision:** prefer **NeMo Sortformer on CUDA** behind `diarize_transcript`, with **CPU MFCC + k-means** fallback.

## History
Phase 2A initially chose MFCC-only (pyannote gated; NeMo diarization deferred for VRAM/scope). Commercial parity revisit wired Sortformer.

## VRAM / align plan
- After ASR finishes, `service.py` passes `release_vram=backend.unload` so Parakeet/Nemotron leave GPU before Sortformer loads (`nvidia/diar_sortformer_4spk-v1`).
- Sortformer `diarize()` → RTTM-style segments → overlap-align onto Parakeet word timestamps; native NeMo cue boundaries kept, speakers majority-voted.
- Sortformer unloaded after labeling; next file reloads ASR via `backend.load()`.
- On import/CUDA/empty-output failure → same MFCC path as before.

## Follow-ups
- Optional: Streaming Sortformer for long sessions; Multitalker Parakeet for overlap; ClusteringDiarizer for >4 speakers.
