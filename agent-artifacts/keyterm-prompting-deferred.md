# Keyterm prompting — deferred

**Verdict:** Not practical on the current Transformers Parakeet/Nemotron stack.

## Why
- App decode path is Hugging Face Transformers `AutoModelForTDT` / `AutoModelForRNNT` greedy `generate()`.
- NVIDIA word boosting / keyterms for Parakeet TDT live in **NeMo GPU-PB** (`boosting_tree`), not Transformers 5.13.1.
- Commercial APIs (Deepgram `keyterm`, AssemblyAI `keyterms_prompt`, Mistral `context_bias`) are first-class serving features.

## What we will not ship (for now)
- Gradio “keyterms” UI that implies decode-time boost
- Post-hoc glossary marketed as keyterm prompting
- Experimental `sequence_bias` as a production “keyterms” feature without WER eval

## Revisit when
- Transformers gains NeMo-style phrase boosting, or
- Product explicitly accepts a Docker-only NeMo experiment (see `transformers-vs-nemo-decision.md`)
