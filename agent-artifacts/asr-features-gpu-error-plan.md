# Parakeet Transcribe — Execution Plan

ASR export fix, GPU throughput, and high-value local ASR extras.

**Status:** implemented in repo (Phases 0–2)  
**Scope:** original document was research/plan; code landed in `src/parakeet_transcribe/`  
**Workspace:** `E:\Chunky's Master Folder\parakeet-transcribe-repo`  
**Research sources:** Track A (`84f28d6d`), Track B (`26b51273`), Track C (`0f30a616`)

---

## Objective

Ship three sequenced outcomes without breaking the local-only CUDA Transformers architecture (one ASR model in VRAM, Gradio batch files):

1. **Unblock the user immediately** — Gradio can serve completed run artifacts from `PARAKEET_OUTPUT_DIR` (Docker: `/data/outputs`); ZIP downloads stay small by excluding scratch dirs.
2. **Raise sustained GPU utilization** — reduce the 30–40W idle valleys between short `generate()` bursts via batching, prefetch/overlap, chunk/token policy, and optional compile — without changing model identity.
3. **Add commercial-parity extras that stay local** — speaker diarization first, then summarization/chapters, PII redaction, optional clean/smart formatting; word confidence only if the loaded model exposes scores.

**Success criteria (overall):**

- Long successful runs (multi-hour m4a) show downloadable ZIP/JSON/SRT/VTT in the UI with no red Gradio `InvalidPathError`.
- ZIP size is dominated by text/subtitle artifacts, not `.work` audio scratch.
- Measurable wall-clock improvement on a fixed long fixture at batch > 1, with OOM fallback still working.
- Diarization spike produces labeled segments without unloading the ASR path permanently or wrapping cloud APIs.
- README/architecture remain: local NVIDIA ASR, no NeMo/Riva/NIM/live-mic/cloud ASR wrappers in v1 of these phases.

---

## Findings

Cross-cutting facts from research, grounded in current code:

| Area | Finding | Code anchors |
|------|---------|--------------|
| Export UX bug | Transcription of large m4a **succeeded** (~4.6h); artifacts on disk under `docker-data/outputs/...`. Red Gradio Error panels are `InvalidPathError`: `/data/outputs` not in `launch(allowed_paths=...)`. Not OOM, decode, or YouTube. | `__main__.py` `launch(...)` has no `allowed_paths`; `compose.yaml` sets `PARAKEET_OUTPUT_DIR=/data/outputs`; `exports.create_run_directory` / `app._publish_results` return those paths to `gr.File` |
| ZIP bloat | `write_bundle` zips the entire `run_dir`, which includes `.work` (~971MB scratch for long media) and possibly `.youtube` | `exports.write_bundle` → `shutil.make_archive(..., root_dir=run_dir)`; `service.transcribe_files` uses `work_dir / ".work"` |
| Already shipped | Punctuation, lang detect, timestamps/SRT/VTT (Parakeet), rich TXT/JSON/CSV/ZIP exports | `backend.py`, `exports.py`, `models.py` capabilities |
| Throughput shape | 30–40W ↔ ~150W sawtooth is expected: short GPU `generate()` + CPU preprocess/decode gaps. Batch UI capped 1–4, no prefetch, no `max_new_tokens`, no `torch.compile`, model can stay loaded via `_get_backend` | `app.py` slider max 4; `service._transcribe_chunk_groups` serial; `backend.transcribe` `generate(**inputs)` only; default chunk 120s |
| Architecture constraint | Local CUDA Transformers, one model in VRAM, Gradio batch files. Defer custom vocab, live streaming UI, cloud API wrappers | README + Track B |

---

## Proposed plan

### Phase 0 — Fix Gradio export path (unblock user)

**Goal:** Completed runs are downloadable in Gradio; ZIP is usable.

| # | Task | Files | Done when |
|---|------|-------|-----------|
| 0.1 | Pass `allowed_paths` into `build_app().launch(...)` including resolved `PARAKEET_OUTPUT_DIR` (default `outputs`) and any other dirs Gradio must read for file components (e.g. temp upload roots if required by Gradio 6.x). Prefer absolute resolved paths. | `src/parakeet_transcribe/__main__.py`; optionally small helper in `exports.py` or `diagnostics.py` for path resolution | Docker + native: after a successful run, ZIP/JSON/SRT/VTT download without `InvalidPathError` |
| 0.2 | Exclude scratch from ZIP: omit `.work`, `.youtube`, and other non-artifact dirs/files when building the archive. Prefer explicit include of written artifacts + `manifest.json` over “zip whole tree then filter” if simpler and safer. | `src/parakeet_transcribe/exports.py` (`write_bundle`); tests in `tests/test_exports.py` | Long-run ZIP is megabytes-scale text/subs, not ~GB of wav/m4a scratch; existing artifact keys still present |
| 0.3 | Regression coverage: unit test that a run dir with a fat `.work` child produces a ZIP without those members; smoke/docs note for Docker (`PARAKEET_OUTPUT_DIR=/data/outputs`). | `tests/test_exports.py`; brief README note only if behavior is user-visible | `uv run pytest` green for export tests |

**Dependencies:** none — ship ASAP, independent of Phase 1/2.  
**Do not yet:** change batch defaults, add diarization, or refactor Gradio layout beyond path/ZIP fixes.

---

### Phase 1 — Throughput / GPU utilization

**Goal:** Higher sustained GPU duty cycle and lower wall-clock for long files, with safe OOM fallback.

**Constraint:** Keep one ASR checkpoint loaded; do not add a second heavy model into VRAM during the hot ASR loop.

| # | Task | Files | Done when |
|---|------|-------|-----------|
| 1.1 | **Raise batch cap** — lift hard ceiling past 4 (UI slider + `service.transcribe_files` validation). Keep default conservative (1 or 2). Document VRAM risk. | `app.py`, `service.py`, optionally `types.ModelSpec.default_batch_size` / README | Slider and service accept higher batch; OOM ladder still recovers |
| 1.2 | **Prefetch / overlap CPU+GPU** — while GPU runs batch *N*, prepare tensors (processor) for batch *N+1* on CPU thread/async queue; avoid idle gaps from serial prepare→generate→decode. Preserve cancel checks between groups. | `service.py` (`_transcribe_chunk_groups`); possibly `backend.py` split prepare vs generate | Profiler/nvidia-smi: shorter low-watt valleys; cancel still works between chunks |
| 1.3 | **Chunk length vs batch** — evaluate shorter chunks (e.g. 60s) + larger batch vs current 120s default; update OOM attempt ladder (`(120, batch)`, `(120, 1)`, `(60, 1)`) to match new defaults. | `service.py`, `chunking.py` (only if API needs new knobs) | Benchmark matrix recorded (fixture duration, batch, chunk_s, elapsed, peak VRAM); choose default with best stable throughput |
| 1.4 | **`max_new_tokens` / generation bounds** — stop unbounded decode cost from huge `max_length` defaults; set explicit generation kwargs appropriate for chunk duration. | `backend.py` (`model.generate`) | No quality regression on short/long fixtures; decode time capped for empty/noisy chunks |
| 1.5 | **Static pad + `torch.compile` (optional stretch)** — after 1.1–1.4 prove wins: static shapes where feasible, `torch.compile` on generate path; feature-flag or env gate for first ship. | `backend.py`; env in `compose.yaml` / docs | Compile path optional; cold-start cost documented; steady-state faster or flag off by default if flaky on Windows/Docker |
| 1.6 | **Keep model loaded** — already mostly true via `_get_backend`; ensure UI/docs don’t encourage unload between files in a batch; avoid accidental unload in new code. | `service.py`, `app.py` (docs copy only if needed) | Multi-file jobs reuse loaded weights; unload remains explicit button |

**Dependencies:** Phase 0 not required technically, but Phase 0 should land first so users can validate long-run downloads while benchmarking.  
**Do not yet:** diarization, LLM post-pass, custom vocab, streaming UI, cloud ASR.

**Suggested benchmark protocol (Phase 1):**

1. Fixed local fixture (≥30–60 min if available; else longest checked-in/synthetic).
2. Record: wall time, peak VRAM, rough GPU power sawtooth notes, batch size, chunk_seconds.
3. Baselines: current defaults (batch=1, 120s) vs raised batch vs prefetch vs shorter chunk+larger batch.
4. Confirm OOM recovery path still triggers gracefully under intentional pressure.

---

### Phase 2 — High-value ASR extras

**Goal:** Local commercial parity beyond raw ASR, without cloud wrappers.

Architecture rules for this phase:

- Post-ASR text features may use a **second local model**, but **not concurrently** with ASR in VRAM unless VRAM headroom is proven — default: unload or sequential stages.
- Gradio remains **batch file** UX; no live mic streaming UI.
- Prefer optional toggles so core transcription stays fast/simple.

#### 2A — Speaker diarization (first spike)

| # | Task | Files | Done when |
|---|------|-------|-----------|
| 2A.1 | Spike: pick a **local** diarization approach compatible with Windows + Docker CUDA (e.g. pyannote-style pipeline or lighter alternative). Document license, model download, VRAM, and whether ASR must unload. | New research note under `agent-artifacts/` or README section; deps in `pyproject.toml` only after spike succeeds | Written spike result: go / no-go with chosen library and memory plan |
| 2A.2 | Integrate post-ASR (or parallel-on-CPU if possible): map speaker labels onto word/segment timeline; extend `TranscriptResult` / JSON schema carefully (`schema_version` bump if breaking). | `types.py`, `service.py`, new module e.g. `diarization.py`, `exports.py`, `app.py` toggle | Exports include speaker-labeled segments; TXT/JSON usable; timestamps still honest (no fabricated times) |
| 2A.3 | Tests with short multi-speaker fixture or mocked diarization backend. | `tests/` | Unit tests for merge/label logic; optional smoke marked slow |

#### 2B — Text post-features (after 2A spike decision)

Priority order:

| # | Feature | Notes | Files (likely) |
|---|---------|-------|----------------|
| 2B.1 | **Summarization / chapters** via **local LLM** | Sequential after ASR; optional; chapter timestamps from existing segments | New `postprocess.py` or `summarize.py`; `app.py`; `exports.py` |
| 2B.2 | **PII text redaction** | Rule + optional NER; operate on transcript text/segments; keep original vs redacted export | `postprocess.py`; export variants |
| 2B.3 | **Optional clean/smart format** post-pass | Light rewrite for readability; must not invent timestamps; reversible/off by default | `postprocess.py`; `app.py` |
| 2B.4 | **Word confidence** | **Only if** Parakeet/Nemotron generate path exposes usable scores; otherwise skip — do not fake | `backend.py`, `types.WordTimestamp`, exports |

**Do not do yet (explicit deferrals from research):**

- Custom vocabulary / hotwords
- Live streaming transcription UI / microphone
- Wrapping cloud ASR APIs (Deepgram, Assembly, Whisper API, etc.)
- Shipping NeMo/Riva/NIM as default stack
- Fabricating SRT/VTT for Nemotron (existing product rule)

---

## Validation strategy

### Phase 0

- Re-run or use existing long m4a success under Docker: UI downloads ZIP/JSON/SRT/VTT with no red `InvalidPathError`.
- Inspect ZIP membership: no `.work/**`, no huge media blobs; `manifest.json` + expected stems present.
- `uv run pytest tests/test_exports.py` (+ any new path helper tests).
- Native Windows launch still works with default `outputs/`.

### Phase 1

- Benchmark matrix (above) before/after each win; keep notes in `agent-artifacts/` or PR description.
- Stress: batch at new max on longest practical file; confirm OOM fallback messages still appear when forced.
- Cancel mid-run still stops between chunk groups.
- `uv run pytest`; `uv run ruff check .`
- Optional: `uv run parakeet-transcribe doctor` before GPU runs.

### Phase 2

- Diarization: multi-speaker sample → labeled JSON/segments; single-speaker does not crash.
- Post-features: toggles off = identical core ASR outputs; toggles on produce additive files without breaking subtitle timing invariants.
- VRAM: never OOM from loading LLM/diarization while ASR still resident unless explicitly supported and tested.
- No new network calls to cloud ASR in the hot path.

### Release / regression gates

- Docker Compose path (`PARAKEET_OUTPUT_DIR=/data/outputs`) is the primary Gradio path regression for Phase 0.
- Parakeet timestamp honesty rule remains: no fabricated SRT/VTT.
- Concurrency stays `queue(default_concurrency_limit=1)`.

---

## Delegation map

| Owner slice | Owns | Does not own |
|-------------|------|--------------|
| **Implementer A — Export unblock** | Phase 0: `__main__.py` `allowed_paths`, `write_bundle` exclude list, export tests, minimal README/Docker note | Throughput, diarization, LLM |
| **Implementer B — GPU throughput** | Phase 1: batch cap, prefetch/overlap, chunk/batch policy, `max_new_tokens`, optional compile flag, benchmark notes | Gradio path allowlist (unless conflict), ASR feature product scope |
| **Implementer C — Diarization spike** | Phase 2A: library/license/VRAM spike doc, then integration + schema/export/UI toggle | Cloud APIs, streaming UI, Phase 1 micro-opts |
| **Implementer D — Text post-pass** | Phase 2B after 2A memory plan is known: local LLM summary/chapters, PII, smart format; confidence only if scores exist | Changing core `generate()` batching (coordinate with B) |
| **Reviewer / parent** | Sequence approval (0 → 1 → 2A → 2B), reject deferrals creep, confirm success criteria per phase | Day-to-day coding unless handoff requested |

**Suggested sequencing for handoff:** A alone first (hours). B parallelizable after A merges. C starts spike anytime after A (needs downloadable artifacts for UX, not for diarization math). D waits on C’s VRAM/unload plan.

---

## Out of scope (this plan)

- Implementing any of the above in this repo change set
- Custom vocab, live streaming UI, cloud ASR wrappers
- Replacing Transformers NVIDIA checkpoints with NeMo/Riva
- Fixing non-issues: OOM/decode/YouTube for the Gradio red-panel report (Track A ruled those out)
