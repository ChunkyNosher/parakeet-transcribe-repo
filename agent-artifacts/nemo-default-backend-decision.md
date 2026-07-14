# NeMo default backend (Docker)

**Decision:** Use **NVIDIA NeMo** as the sole ASR backend. Supported inference path is **Docker Compose (Linux GPU)**. Native Windows inference is unsupported.

## Why NeMo now
- Decode-time **GPU-PB / boosting_tree** keyterm boosting for Parakeet TDT
- Native **local-attention** long-form (`rel_pos_local_attn`) with app chunking as OOM fallback only
- Greedy batch decoding with **CUDA graph decoder**
- Product accepted Docker-only runtime (Option A)

## Explicitly out of scope
- Live microphone streaming UI
- Riva / NIM serving stacks
- Dual Transformers + NeMo backends
- Flashlight / CTC-WS boosting (GPU-PB is the TDT path)

## Supersedes
- `transformers-vs-nemo-decision.md` (stay on Transformers)
- `keyterm-prompting-deferred.md` (keyterms deferred)
