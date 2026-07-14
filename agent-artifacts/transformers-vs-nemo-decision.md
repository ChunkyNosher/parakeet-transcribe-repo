# Transformers vs NeMo — backend decision

**Decision:** Stay on Transformers. Do **not** refactor this app to NeMo (full or dual) at this time.

## Why Transformers remains optimal here
- Working local stack: TDT/RNNT `generate()`, honest timestamps, chunking, CPU prefetch, OOM ladder, Gradio exports, optional MFCC diarization
- Slim Windows + Docker install (`uv`, torch cu130) vs NeMo’s Linux-first, fat dependency surface and pin fights
- Rewrite cost would throw away recent backend/service investment for one feature (keyterms)

## What NeMo uniquely offers
- Decode-time **GPU-PB / boosting_tree** keyterm boosting for TDT
- Native buffered / local-attention long-form APIs (this app already approximates via chunking)

## Implications
| Topic | Action |
|-------|--------|
| Keyterms | Deferred (NeMo-only for true bias) |
| Silence-in-SRT bug | Fixed in `segments_from_words` — backend-agnostic |
| Dual backend | Rejected — doubles maintenance for one feature |

Revisit only if product accepts a Docker-only NeMo experiment dedicated to keyterms.
