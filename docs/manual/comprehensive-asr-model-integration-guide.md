# Comprehensive ASR Model Integration Guide: All Parakeet & Canary Variants

**Date:** 2025-12-10 | **Scope:** Complete model selection and integration for Gradio UI | **Severity:** Enhancement

---

## Executive Summary

Your current transcription interface uses outdated models (Parakeet-0.6B-v2, Canary-Qwen-2.5B). **Six newer models are available** with different performance profiles to suit various use cases:

### Model Ecosystem Overview

**Parakeet Family (Fast, English ASR):**
- **Parakeet-TDT-0.6B-v3**: 600M params, 25 languages, auto-detection ⭐ Recommended for most
- **Parakeet-TDT-1.1B**: 1.1B params, English only, **best accuracy** (1.5% WER)

**Canary Family (Multilingual ASR + Translation):**
- **Canary-1B-v2**: 1B params, 25 languages, translator, encoder-decoder ⭐ Best balance
- **Canary-1B-Flash**: 883M params, 4 languages (En/De/Es/Fr), ultra-fast (1000+ RTFx), lightweight
- **Canary-180M-Flash**: 182M params, 4 languages, ultra-lightweight (1200+ RTFx), mobile-friendly
- **Canary-Qwen-2.5B**: 2.5B params, English only, hybrid LLM, complex SALM dependency (legacy)

**Quick Selection Guide:**
- **Need best English accuracy?** → **Parakeet-TDT-1.1B** (1.5% WER)
- **Need multilingual + auto-detect?** → **Parakeet-TDT-0.6B-v3** (25 languages)
- **Need translation?** → **Canary-1B-v2** (25 languages with AST)
- **Need speed for 4 languages?** → **Canary-1B-Flash** (1000+ RTFx, 883M params)
- **Need mobile/edge device?** → **Canary-180M-Flash** (1200+ RTFx, 182M params, runs on phones)

## Complete Model Specifications

| Model | Params | Languages | Key Feature | WER (En) | RTFx | VRAM | Use Case |
|-------|--------|-----------|-------------|----------|------|------|----------|
| **Parakeet-TDT-0.6B-v3** | 600M | 25 EU | Auto-detect, multilingual | ~1.7% | 3,380× | 3-4 GB | Default choice - balanced |
| **Parakeet-TDT-1.1B** | 1.1B | English only | Best accuracy | **1.5%** ✅ | 1,336× | 5-6 GB | Maximum accuracy (English) |
| **Canary-1B-v2** | 1B | 25 EU | Multilingual + translation | 1.88% | ~200× | 4-5 GB | ASR + translation |
| **Canary-1B-Flash** | 883M | 4 (En/De/Es/Fr) | Ultra-fast, lightweight | 1.48% (En) | **1000×** | 2-3 GB | Speed-optimized, 4 lang |
| **Canary-180M-Flash** | 182M | 4 (En/De/Es/Fr) | Mobile-friendly, tiny | 1.87% (En) | **1200×** | 1-2 GB | Edge/mobile devices |
| **Canary-Qwen-2.5B** | 2.5B | English only | LLM hybrid (SALM) | 5.63% | 418× | 6-7 GB | Legacy - complex deps |

**Your GPU (RTX 4080 Laptop, 12GB VRAM):** Can run all models. Recommended: Parakeet-1.1B (best accuracy) or Canary-1B-v2 (best versatility).

<scope>
**Modify:**
- `transcribe_ui.py` (MODEL_CONFIGS dictionary, lines 40-75)
- `transcribe_ui.py` (Model selection radio button, lines 450-480)
- `transcribe_ui.py` (Validation and startup messages, lines 800-900)
- `setup_local_models.py` (Parakeet download logic, lines 20-50)

**Do NOT Modify:**
- `transcribe_audio()` function (works identically for all models)
- Gradio interface structure
- Batch processing logic
- Video extraction logic
- Timestamp generation logic
</scope>

---

## Issue #1: Parakeet-TDT-0.6B-v2 Limited to English Only

### Problem

Current model supports **only English**. Cannot transcribe other European languages (Spanish, French, German, Italian, etc.).

### Root Cause

**File:** `transcribe_ui.py`  
**Location:** Lines 40-51 (MODEL_CONFIGS dictionary)  
**Issue:** Model ID hardcoded to `nvidia/parakeet-tdt-0.6b-v2` (English-only). No language detection or multilingual support.

### Fix Required

Provide option to upgrade Parakeet to **v3 (multilingual) or 1.1B (best accuracy)**. Both maintain identical API (`ASRModel.from_pretrained()`). No code logic changes needed—only model IDs change.

**Recommended:** Add both as options in MODEL_CONFIGS and let users choose via UI radio button.

---

## Issue #2: Canary-Qwen-2.5B Uses Complex SALM Architecture

### Problem

Current Canary uses hybrid **SALM** (Speech-Augmented Language Model) with Qwen LLM decoder. Requires:
- Specialized `nemo.collections.speechlm2` import
- Complex error handling for SALM vs traditional ASR
- Limited flexibility for language extensions
- Higher VRAM usage (6-7 GB)
- Slower than alternatives (418 RTFx)

### Root Cause

**File:** `transcribe_ui.py`  
**Location:** Lines 43-59 (Canary config, SALM import)  
**Issue:** Model pinned to `nvidia/canary-qwen-2.5b` (English-only, SALM-dependent). SALM code (~100 lines) adds unnecessary complexity.

### Fix Required

Replace with **Canary-1B-v2** as primary option (simpler encoder-decoder, 25 languages, faster). Optionally add **Canary-1B-Flash** for speed-optimized users and **Canary-180M-Flash** for edge devices.

All three use standard `ASRModel.from_pretrained()` API. No SALM imports needed. Eliminates ~100 lines of SALM-specific code.

---

## Issue #3: No Speed-Optimized Options for Real-Time Applications

### Problem

Current models are either high-accuracy (slower) or multilingual (moderate speed). No ultra-lightweight option for real-time processing or mobile deployment.

### Root Cause

**File:** `transcribe_ui.py`  
**Location:** MODEL_CONFIGS dictionary  
**Issue:** Missing lightweight variants designed for real-time inference.

### Fix Required

Add **Canary-1B-Flash** (883M, 1000+ RTFx, 4 languages) and **Canary-180M-Flash** (182M, 1200+ RTFx, runs on mobile) as optional models for users prioritizing speed over language coverage.

---

## Complete Implementation: All Recommended Models

### Updated MODEL_CONFIGS

**Comprehensive Configuration with All Options:**

```python
MODEL_CONFIGS = {
    # ========== PARAKEET MODELS ==========
    "parakeet-v3": {
        "local_path": "local_models/parakeet.nemo",
        "max_batch_size": 32,
        "display_name": "Parakeet-TDT-0.6B v3 (Multilingual, Default)",
        "loading_method": "local",
        "architecture": "FastConformer-TDT",
        "parameters": "600M",
        "languages": 25,
        "wer": "~1.7%",
        "rtfx": "3,380×",
        "vram_gb": "3-4",
        "recommended_for": "Best all-around choice, 25 languages, auto-detection"
    },
    
    "parakeet-1.1b": {
        "hf_model_id": "nvidia/parakeet-tdt-1.1b",
        "max_batch_size": 24,
        "display_name": "Parakeet-TDT-1.1B (Maximum Accuracy, English)",
        "loading_method": "huggingface",
        "architecture": "FastConformer-TDT",
        "parameters": "1.1B",
        "languages": 1,
        "wer": "1.5%",
        "rtfx": "1,336×",
        "vram_gb": "5-6",
        "recommended_for": "Best English transcription accuracy available"
    },
    
    # ========== CANARY MODELS ==========
    "canary-1b-v2": {
        "hf_model_id": "nvidia/canary-1b-v2",
        "max_batch_size": 16,
        "display_name": "Canary-1B v2 (Multilingual + Translation)",
        "loading_method": "huggingface",
        "architecture": "FastConformer-Transformer",
        "parameters": "1B",
        "languages": 25,
        "wer": "1.88% (English)",
        "rtfx": "~200×",
        "vram_gb": "4-5",
        "recommended_for": "Multilingual ASR + speech-to-text translation",
        "additional_features": ["Speech Translation (AST)", "NeMo Forced Aligner timestamps"]
    },
    
    "canary-1b-flash": {
        "hf_model_id": "nvidia/canary-1b-flash",
        "max_batch_size": 32,
        "display_name": "Canary-1B Flash (Speed Optimized, 4 Languages)",
        "loading_method": "huggingface",
        "architecture": "FastConformer-Transformer",
        "parameters": "883M",
        "languages": 4,  # En, De, Es, Fr
        "wer": "1.48% (English)",
        "rtfx": "1000+",
        "vram_gb": "2-3",
        "recommended_for": "Real-time transcription, 4-language coverage",
        "additional_features": ["1000+ RTFx real-time", "Speech Translation", "Lightweight"]
    },
    
    "canary-180m-flash": {
        "hf_model_id": "nvidia/canary-180m-flash",
        "max_batch_size": 32,
        "display_name": "Canary-180M Flash (Mobile, Ultra-Lightweight)",
        "loading_method": "huggingface",
        "architecture": "FastConformer-Transformer",
        "parameters": "182M",
        "languages": 4,  # En, De, Es, Fr
        "wer": "1.87% (English)",
        "rtfx": "1200+",
        "vram_gb": "1-2",
        "recommended_for": "Edge devices, mobile phones, minimal VRAM",
        "additional_features": ["1200+ RTFx ultra-fast", "Fits on mobile", "Speech Translation"]
    }
}

# Legacy model (not recommended - included for backward compatibility)
LEGACY_MODEL_CONFIGS = {
    "canary-qwen": {
        "hf_model_id": "nvidia/canary-qwen-2.5b",
        "max_batch_size": 16,
        "display_name": "Canary-Qwen-2.5B (Legacy, English Only)",
        "loading_method": "huggingface",
        "deprecated": True,
        "reason": "Requires complex SALM import; Canary-1B-v2 recommended instead"
    }
}
```

### Updated Gradio UI Radio Button

**Current (limited):**
```python
model_selector = gr.Radio(
    choices=[
        "Parakeet-TDT-0.6B v2 (Fast)",
        "Canary-Qwen-2.5B (Accurate)"
    ],
    value="Parakeet-TDT-0.6B v2 (Fast)",
    label="Model Selection"
)
```

**Updated (all options):**
```python
model_selector = gr.Radio(
    choices=[
        "📊 Parakeet-TDT-0.6B v3 (Multilingual, Default) - Balanced accuracy & speed",
        "🎯 Parakeet-TDT-1.1B (Maximum Accuracy, English Only) - 1.5% WER",
        "🌍 Canary-1B v2 (Multilingual + Translation) - 25 languages with AST",
        "⚡ Canary-1B Flash (Speed Optimized) - 1000+ RTFx, 4 languages",
        "📱 Canary-180M Flash (Mobile/Edge) - 1200+ RTFx, tiny footprint"
    ],
    value="📊 Parakeet-TDT-0.6B v3 (Multilingual, Default) - Balanced accuracy & speed",
    label="Model Selection",
    info="Choose based on your priority: accuracy, speed, languages, or VRAM"
)
```

### Model Key Extraction for Backend

**Extract model key from radio button choice:**
```python
def get_model_key_from_choice(choice_text):
    """Extract model key from radio button choice text"""
    choice_map = {
        "Parakeet-TDT-0.6B v3": "parakeet-v3",
        "Parakeet-TDT-1.1B": "parakeet-1.1b",
        "Canary-1B v2": "canary-1b-v2",
        "Canary-1B Flash": "canary-1b-flash",
        "Canary-180M Flash": "canary-180m-flash"
    }
    for choice, key in choice_map.items():
        if choice in choice_text:
            return key
    return "parakeet-v3"  # Default
```

---

## Setup & Download Instructions

### Parakeet Models (Local .nemo File)

Both Parakeet versions download to same location, overwriting:

```python
# setup_local_models.py - update Parakeet download

# Option 1: v3 (Multilingual) - RECOMMENDED
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
output_path = "local_models/parakeet.nemo"
model.save_to(output_path)

# Option 2: 1.1B (English, best accuracy)
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
output_path = "local_models/parakeet.nemo"
model.save_to(output_path)
```

**Download sizes:**
- Parakeet-0.6B-v3: ~1.2 GB download → ~2.4 GB saved
- Parakeet-1.1B: ~2.2 GB download → ~4.5 GB saved

### Canary Models (HuggingFace Cache)

All three Canary models download automatically on first use (no setup script needed):

```python
# Automatic download on first transcription call
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b-v2")      # ~2.5 GB
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b-flash")   # ~1.8 GB
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-180m-flash") # ~0.5 GB
```

Cache location: `~/.cache/torch/NeMo/`

---

## Language Coverage Comparison

### Parakeet-TDT-0.6B-v3 & Canary-1B-v2: 25 European Languages
```
🇪🇸 Spanish     🇫🇷 French      🇩🇪 German      🇮🇹 Italian     🇵🇹 Portuguese
🇵🇱 Polish      🇳🇱 Dutch       🇸🇪 Swedish     🇩🇰 Danish      🇳🇴 Norwegian
🇫🇮 Finnish     🇨🇿 Czech       🇭🇺 Hungarian   🇷🇴 Romanian    🇬🇷 Greek
🇧🇬 Bulgarian   🇷🇸 Serbian     🇭🇷 Croatian    🇸🇰 Slovak      🇸🇮 Slovenian
🇱🇹 Lithuanian  🇱🇻 Latvian     🇪🇪 Estonian    🇲🇹 Maltese     🇬🇧 English
```

### Canary-1B-Flash & Canary-180M-Flash: 4 Languages
```
🇬🇧 English  🇩🇪 German  🇪🇸 Spanish  🇫🇷 French
```

---

## Feature Comparison Matrix

| Feature | Parakeet-0.6B-v3 | Parakeet-1.1B | Canary-1B-v2 | Canary-1B-Flash | Canary-180M-Flash |
|---------|------------------|---------------|--------------|-----------------|-------------------|
| ASR (Transcription) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Speech Translation | ❌ | ❌ | ✅ | ✅ | ✅ |
| Punctuation/Capitalization | ✅ | ✅ | ✅ | ✅ | ✅ |
| Word-Level Timestamps | ✅ | ✅ | ✅ | ✅ | ✅ |
| Auto Language Detection | ✅ | ❌ | ✅ | ❌ | ❌ |
| Languages | 25 | 1 | 25 | 4 | 4 |
| Batch Processing | ✅ | ✅ | ✅ | ✅ | ✅ |
| Video Support (FFmpeg) | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Performance & Resource Trade-offs

### For Your RTX 4080 Laptop (12GB VRAM)

| Priority | Recommended Model | Reason |
|----------|-------------------|--------|
| **Best Accuracy** | Parakeet-TDT-1.1B | 1.5% WER (English), uses 5-6 GB VRAM |
| **Best All-Around** | Parakeet-TDT-0.6B-v3 | 25 languages, auto-detect, 1.7% WER, 3-4 GB VRAM |
| **Multilingual + Translation** | Canary-1B-v2 | 25 languages, AST, 1.88% WER, 4-5 GB VRAM |
| **Maximum Speed** | Canary-1B-Flash | 1000+ RTFx, 2-3 GB VRAM, 4 languages |
| **Smallest Footprint** | Canary-180M-Flash | 1200+ RTFx, 1-2 GB VRAM, perfect for testing |

### Accuracy vs Speed Trade-off

```
Accuracy (Lower WER)          Speed (Higher RTFx)
         ↓                            ↓
    1.5% WER ┌─ Parakeet-1.1B (1,336× RTFx)
             │
    1.7% WER ├─ Parakeet-0.6B-v3 (3,380× RTFx)
             │
    1.88% WER├─ Canary-1B-v2 (200× RTFx) [+ Translation]
             │
    1.48% WER├─ Canary-1B-Flash (1000+ RTFx) [+ Translation]
             │
    1.87% WER└─ Canary-180M-Flash (1200+ RTFx) [+ Translation]
```

---

## Model Loading: Implementation Patterns

All models use the same API (no code changes needed for most logic):

```python
def load_model(model_key, show_progress=False):
    """Load any model using unified API"""
    
    if model_key not in models_cache:
        config = MODEL_CONFIGS[model_key]
        
        if config.get("loading_method") == "local":
            # Load .nemo file (Parakeet v3 or 1.1B)
            model_path = get_script_dir() / config["local_path"]
            models_cache[model_key] = nemo_asr.models.ASRModel.restore_from(str(model_path))
        else:
            # Load from HuggingFace (all Canary models)
            models_cache[model_key] = nemo_asr.models.ASRModel.from_pretrained(
                config["hf_model_id"]
            )
    
    return models_cache[model_key]
```

**Key point:** No SALM imports needed. All models (including new ones) work through standard `ASRModel` API.

---

## Migration Checklist

<acceptance_criteria>
**Code Changes:**
- [ ] Update MODEL_CONFIGS dictionary with all 5 recommended models
- [ ] Add model_key extraction function for radio button choices
- [ ] Update Gradio radio button with all 5 options + descriptive labels
- [ ] Update model selection logic to use extracted model_key
- [ ] Remove SALM import block entirely (no longer needed)
- [ ] Remove SALM_AVAILABLE variable and SALM validation code
- [ ] Update setup_local_models.py to explain v3 and 1.1B options
- [ ] Update all validation/startup messages to reference new models

**Testing:**
- [ ] Load Parakeet-0.6B-v3 (multilingual)
- [ ] Load Parakeet-1.1B (English only, best accuracy)
- [ ] Load Canary-1B-v2 (multilingual + translation)
- [ ] Load Canary-1B-Flash (speed-optimized)
- [ ] Load Canary-180M-Flash (lightweight)
- [ ] Test ASR with each model
- [ ] Test speech translation with Canary models
- [ ] Verify batch processing works with all models
- [ ] Verify timestamps work with all models
- [ ] Verify punctuation output works with all models
- [ ] No SALM-related errors in console

**Compatibility:**
- [ ] All models work with existing transcribe_audio() logic
- [ ] Batch processing unchanged
- [ ] Video extraction unchanged
- [ ] Timestamp generation unchanged
- [ ] Model caching (models_cache dict) works for all
- [ ] File downloads functional
</acceptance_criteria>

---

## Decision Tree: Which Model to Choose?

```
START
  │
  ├─ Need maximum accuracy?
  │  └─ YES → Parakeet-TDT-1.1B (1.5% WER, English only)
  │
  ├─ Need speech translation to other languages?
  │  └─ YES → Canary-1B-v2 (25 lang AST)
  │  └─ NO  → Continue below
  │
  ├─ Need more than 4 languages?
  │  └─ YES → Parakeet-TDT-0.6B-v3 (25 languages)
  │  └─ NO  → Continue below
  │
  ├─ Speed critical (real-time processing)?
  │  └─ YES → Canary-1B-Flash (1000+ RTFx)
  │  └─ NO  → Continue below
  │
  ├─ Running on edge device or mobile?
  │  └─ YES → Canary-180M-Flash (1200+ RTFx, 182M)
  │  └─ NO  → Continue below
  │
  └─ Default → Parakeet-TDT-0.6B-v3 (recommended for most users)
```

---

## Removing SALM Dependency (Optional Cleanup)

With Canary-1B-v2 and Flash variants, you can **remove all SALM-specific code**:

**Lines to remove from transcribe_ui.py:**
- SALM import try/except block (~10 lines)
- `SALM_AVAILABLE` variable definition (~2 lines)
- `validate_salm_availability()` function call (~1 line)
- `_format_canary_error()` function (~40 lines)
- SALM-specific error handling in `load_model()` (~80 lines)
- SALM validation in startup messages (~20 lines)

**Total lines removed:** ~150 lines of SALM-specific code

**Benefit:** Simpler, cleaner codebase. All models now use uniform `ASRModel` API.

---

## Deployment Notes

### VRAM Requirements on Your RTX 4080 Laptop

```
Total VRAM: 12 GB

Option A (Maximum Accuracy):
  Parakeet-1.1B loaded: 5-6 GB
  + Gradio/PyTorch overhead: 1-2 GB
  = 6-8 GB used → Comfortable (4 GB free)

Option B (Recommended Default):
  Parakeet-0.6B-v3 loaded: 3-4 GB
  + Gradio/PyTorch overhead: 1-2 GB
  = 4-6 GB used → Very comfortable (6-8 GB free)

Option C (Speed + Features):
  Canary-1B-v2 loaded: 4-5 GB
  + Gradio/PyTorch overhead: 1-2 GB
  = 5-7 GB used → Comfortable (5 GB free)

Option D (Ultra-Fast):
  Canary-1B-Flash loaded: 2-3 GB
  + Gradio/PyTorch overhead: 1-2 GB
  = 3-5 GB used → Very comfortable (7-9 GB free)
```

**Conclusion:** All models run comfortably on your GPU with room to spare.

---

## Backward Compatibility

- **Code level:** 100% compatible. `load_model()` works identically for all.
- **API level:** All use `ASRModel.from_pretrained()` and `.transcribe()`.
- **Gradio level:** No UI changes except radio button options.
- **Batch processing:** Unchanged. Works with all models.
- **Timestamps:** Unchanged. All models support word/segment-level.
- **Video handling:** Unchanged. FFmpeg extraction works for all.

---

**Priority:** High (Essential for modern functionality) | **Target:** Comprehensive model integration | **Estimated Complexity:** Medium (configuration + testing)
