# Keyterms, segments, and backend choice

Synced from revised plan `keyterms_and_segments_665370f9`. Research complete; ready for implementation approval.

## Backend verdict

**Stay on Transformers.** Do not refactor to NeMo (full or dual) for this app. NeMo’s unique win is GPU-PB keyterm boosting; it does not justify Windows/Docker install cost, pin fights, or rewriting the working Gradio/chunking/export stack.

## Keyterms

Deferred. True decode-time bias is NeMo-only on NVIDIA ASR. No fake keyterm UI.

## Silence segments

Fix in `segments_from_words` (clamp → gap-split). Backend-agnostic; NeMo would not fix it alone.
