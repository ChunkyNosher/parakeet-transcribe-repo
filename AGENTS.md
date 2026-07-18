## Learned User Preferences

- When implementing an attached plan, do not edit the plan file; use the existing todos (do not recreate them), mark them in progress, and finish all of them.
- Prefers plan/research passes for complex ASR, performance, and architecture questions before implementation; often interrupts implementation to request another deep research pass.
- Does not want research agents to blast parallel TinyFish web search/fetch calls; still wants web search available via other tools or more restrained usage.
- Often diagnoses transcription failures with log files under `logs/` plus UI screenshots.
- Interested in commercial-ASR-style capabilities (for example keyterm/proper-noun prompting and related studio features) on this local Parakeet app, and in steadier GPU utilization plus faster transcription.

## Learned Workspace Facts

- This repo is a local NVIDIA Parakeet transcription app (Gradio) that uses **NVIDIA NeMo** as the sole ASR backend; supported inference runtime is Docker Compose (Linux GPU). Native Windows inference is unsupported. Riva/NIM, live microphone streaming, dual Transformers backends, and cloud ASR API wrappers remain out of scope.
- Docker Compose runs Linux GPU containers; Hugging Face / NeMo model files live on a host bind mount (`./docker-data/model_cache` → `/data/model_cache` via `HF_HOME` / `HF_HUB_CACHE`), and exports go under `./docker-data/outputs`.
- The Docker image needs a C compiler (`build-essential` / gcc) because Linux PyTorch pulls Triton, which JIT-builds CUDA helpers at inference time; `libsndfile1` is also required for NeMo audio I/O.
- NeMo features in-app: local-attention long-form, GPU-PB keyterm boosting, greedy CUDA-graph decoding; app silence chunking is OOM fallback only.
- Model registry (`models.py`): parakeet-tdt-0.6b-v3 (default), parakeet-tdt-0.6b-v2 and parakeet-tdt-1.1b (English-only TDT), nemotron-3.5-asr-streaming-0.6b. The legacy 1.1b checkpoint (lowercase unpunctuated output) only loads with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, set in `compose.yaml`.
- There is a long-standing segmentation issue where silence-heavy segments end with a word that belongs at the start of the next sentence/segment, despite prior fix attempts.
