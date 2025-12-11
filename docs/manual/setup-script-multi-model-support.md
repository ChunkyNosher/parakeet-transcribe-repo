# Setup Script Enhancement: Multi-Model Support with Unique File Naming

**Date:** 2025-12-11 | **Scope:** Extend setup_local_models.py to support downloading and storing all Parakeet AND Canary models as distinct .nemo files | **Severity:** Medium | **Status:** Solution with Implementation Details

---

## Problem Summary

Current `setup_local_models.py` has two critical limitations:

1. **Single Model Storage:** Both Parakeet-0.6B-v3 and Parakeet-1.1B models overwrite the same `local_models/parakeet.nemo` file. Users can only have ONE Parakeet model locally at a time.

2. **No Canary Support:** Canary models (Canary-1B, Canary-1B-v2) are not downloadable via setup script. They only download from HuggingFace on first use, making them slower on initial transcription and harder to manage for users who want offline-guaranteed models.

**User Impact:**
- Cannot store all 4 recommended models locally for instant switching
- First transcription with Canary models is slow (requires HuggingFace download)
- No way to pre-download models for air-gapped environments
- Model selection UI shows 5 options but only 2-3 can be stored locally

---

## Root Cause Analysis

**File:** `setup_local_models.py`  
**Location:** Lines 87-90 (save logic)  
**Issue:** Hard-coded output filename ignores model selection. All models save to same path:

```python
output_path = os.path.abspath("local_models/parakeet.nemo")
model.save_to(output_path)  # ALWAYS "parakeet.nemo" regardless of which model
```

**Secondary Issue:** Script only handles Parakeet models (lines 57-93). Canary download logic is missing entirely, despite `transcribe_ui.py` supporting them via HuggingFace.

---

## Solution Overview

Redesign `setup_local_models.py` to:

1. **Support 4 downloadable models** with unique filenames:
   - `parakeet-0.6b-v3.nemo`
   - `parakeet-1.1b.nemo`
   - `canary-1b.nemo`
   - `canary-1b-v2.nemo`

2. **Menu-driven selection** allowing users to:
   - Download one model at a time
   - Download multiple models in sequence
   - Download all models at once

3. **Intelligent naming** based on model selection:
   - Each model saves with unique, descriptive filename
   - Filename matches model ID for easy identification
   - No accidental overwrites

4. **Configuration update** to `transcribe_ui.py`:
   - Add loading paths for each model to `MODEL_CONFIGS`
   - Allow `load_model()` to load from local `.nemo` files OR HuggingFace
   - Switch loading method based on file existence

---

## Implementation Details

### Part 1: Extend MODEL_CONFIGS in transcribe_ui.py

**File:** `transcribe_ui.py`  
**Location:** Lines 40-75 (MODEL_CONFIGS dictionary)  
**What to Change:** Add `local_path` field to each model config and update loading method logic

**Current Parakeet-v3 config:**
```python
"parakeet-v3": {
    "local_path": "local_models/parakeet.nemo",
    "hf_model_id": "nvidia/parakeet-tdt-0.6b-v3",
    "loading_method": "local",  # Always tries local
    ...
}
```

**Update to:** Add fallback logic and unique filenames for all models:

```python
"parakeet-v3": {
    "local_path": "local_models/parakeet-0.6b-v3.nemo",  # CHANGED: unique name
    "hf_model_id": "nvidia/parakeet-tdt-0.6b-v3",
    "loading_method": "local_or_huggingface",  # NEW: try local, fallback to HF
    ...
},

"parakeet-1.1b": {
    "local_path": "local_models/parakeet-1.1b.nemo",  # NEW: unique name
    "hf_model_id": "nvidia/parakeet-tdt-1.1b",
    "loading_method": "local_or_huggingface",  # NEW: try local, fallback to HF
    ...
},

"canary-1b": {
    "local_path": "local_models/canary-1b.nemo",  # NEW: local storage option
    "hf_model_id": "nvidia/canary-1b",
    "loading_method": "local_or_huggingface",  # NEW: try local, fallback to HF
    ...
},

"canary-1b-v2": {
    "local_path": "local_models/canary-1b-v2.nemo",  # NEW: local storage option
    "hf_model_id": "nvidia/canary-1b-v2",
    "loading_method": "local_or_huggingface",  # NEW: try local, fallback to HF
    ...
}
```

### Part 2: Update load_model() Logic

**File:** `transcribe_ui.py`  
**Location:** Lines 200-280 (load_model function)  
**What to Change:** Implement fallback logic for "local_or_huggingface" loading method

**Current logic (lines ~210-220):**
```python
if config.get("loading_method") == "local":
    # Load .nemo file
    model_path = script_dir / config["local_path"]
    if not model_path.exists():
        raise FileNotFoundError(...)
    models_cache[model_name] = nemo_asr.models.ASRModel.restore_from(...)
else:
    # Load from HuggingFace
    models_cache[model_name] = nemo_asr.models.ASRModel.from_pretrained(...)
```

**Update to:** Try local first, fallback to HuggingFace:

```python
if config.get("loading_method") == "local_or_huggingface":
    # Try loading from local .nemo file first
    model_path = script_dir / config["local_path"]
    
    if model_path.exists():
        # Local file found - use it
        print(f"📦 Loading {config['display_name']} from local file...")
        try:
            models_cache[model_name] = nemo_asr.models.ASRModel.restore_from(
                str(model_path)
            )
            print(f"✓ Loaded from {model_path}")
            return models_cache[model_name]
        except Exception as e:
            # Local file exists but is corrupted
            print(f"⚠️  Local file corrupted, falling back to HuggingFace...")
            # Fall through to HuggingFace download below
    else:
        # Local file not found - will download from HuggingFace
        print(f"📦 Local .nemo file not found: {model_path}")
        print(f"   To download locally, run: python setup_local_models.py")
        print(f"   Downloading from HuggingFace instead...")
    
    # Load from HuggingFace (either no local file or local file was corrupted)
    hf_model_id = config["hf_model_id"]
    print(f"📦 Loading {config['display_name']} from HuggingFace...")
    models_cache[model_name] = nemo_asr.models.ASRModel.from_pretrained(
        hf_model_id
    )

elif config.get("loading_method") == "local":
    # Original behavior: strictly local only
    model_path = script_dir / config["local_path"]
    if not model_path.exists():
        raise FileNotFoundError(...)
    models_cache[model_name] = nemo_asr.models.ASRModel.restore_from(...)
```

**Key behavior:**
- If local .nemo exists → Load immediately (no internet needed)
- If local .nemo missing → Download from HuggingFace (automatic fallback)
- If local .nemo corrupted → Log warning, download from HuggingFace
- User can optionally run setup script to pre-download for offline use

### Part 3: Complete Redesign of setup_local_models.py

**File:** `setup_local_models.py`  
**What to Change:** Entire script architecture

**New structure:**

1. **Model definitions dictionary** (instead of hardcoded if/else):
   ```python
   MODELS_TO_DOWNLOAD = {
       "1": {
           "model_id": "nvidia/parakeet-tdt-0.6b-v3",
           "filename": "parakeet-0.6b-v3.nemo",
           "display_name": "Parakeet-TDT-0.6B-v3 (Multilingual)",
           "download_size": "~1.2 GB",
           "saved_size": "~2.4 GB",
           "description": "25 languages, auto-detection, 3,380× RTFx"
       },
       "2": {
           "model_id": "nvidia/parakeet-tdt-1.1b",
           "filename": "parakeet-1.1b.nemo",
           "display_name": "Parakeet-TDT-1.1B (Maximum Accuracy)",
           "download_size": "~2.2 GB",
           "saved_size": "~4.5 GB",
           "description": "English only, 1.5% WER (best accuracy)"
       },
       "3": {
           "model_id": "nvidia/canary-1b",
           "filename": "canary-1b.nemo",
           "display_name": "Canary-1B (Multilingual + Translation)",
           "download_size": "~2.5 GB",
           "saved_size": "~5.0 GB",
           "description": "25 languages, speech-to-text translation"
       },
       "4": {
           "model_id": "nvidia/canary-1b-v2",
           "filename": "canary-1b-v2.nemo",
           "display_name": "Canary-1B v2 (Multilingual + Translation)",
           "download_size": "~2.5 GB",
           "saved_size": "~5.0 GB",
           "description": "25 languages, improved speech translation"
       }
   }
   ```

2. **Model selection menu:**
   - Display all 4 models with descriptions and sizes
   - Allow selection of multiple models
   - Option to download all at once (batch mode)
   - Track which models have been downloaded

3. **Generic download function:**
   ```python
   def download_and_save_model(model_id, filename, display_name, download_size, saved_size):
       """Download and save ANY model as .nemo file."""
       print(f"📦 Downloading {display_name}...")
       print(f"   Size: {download_size} → {saved_size}")
       
       try:
           model = nemo_asr.models.ASRModel.from_pretrained(model_id)
           output_path = os.path.abspath(f"local_models/{filename}")
           model.save_to(output_path)
           
           # Verify
           if os.path.exists(output_path):
               file_size = os.path.getsize(output_path) / (1024**3)
               print(f"✓ {display_name} saved ({file_size:.2f} GB)")
               return True
           else:
               print(f"❌ File was not created")
               return False
       except Exception as e:
           print(f"❌ Error: {e}")
           return False
   ```

4. **Menu flow:**
   ```
   Main Menu:
   -----------
   1. Download Parakeet-0.6B-v3
   2. Download Parakeet-1.1B
   3. Download Canary-1B
   4. Download Canary-1B-v2
   5. Download All Models
   6. Check what's already downloaded
   0. Exit
   
   Selection: [user enters choice]
   ```

5. **Status reporting:**
   - Show which models are already downloaded (with sizes)
   - Show which models need to be downloaded
   - After download, verify file integrity
   - Show total disk space used

---

## Why This Approach is Better

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Parakeet Storage** | Single file (overwrite) | 2 unique files |
| **Canary Storage** | Not available | 2 unique files |
| **Total Models** | 1 can be stored | 4 can be stored |
| **User Choice** | Limited to 1 model | Can store all 4 |
| **First Run Speed** | Depends on model | Fast (if pre-downloaded) |
| **Setup Flexibility** | One-time script | Menu-driven, repeatable |
| **Fallback** | None (fails if local missing) | Auto-downloads from HF |
| **File Organization** | Confusing (one name) | Clear (unique names) |
| **Offline Ready** | Only 1 model | All 4 models possible |

---

## File Naming Scheme

**Rationale:** Model ID components → filename components

| Model | HF Model ID | Current Filename | Proposed Filename | Why |
|-------|-------------|------------------|-------------------|-----|
| Parakeet 0.6B v3 | nvidia/parakeet-tdt-0.6b-v3 | parakeet.nemo | parakeet-0.6b-v3.nemo | Version number in filename |
| Parakeet 1.1B | nvidia/parakeet-tdt-1.1b | parakeet.nemo | parakeet-1.1b.nemo | Size + version distinction |
| Canary 1B | nvidia/canary-1b | ❌ Not downloadable | canary-1b.nemo | HF model ID format |
| Canary 1B v2 | nvidia/canary-1b-v2 | ❌ Not downloadable | canary-1b-v2.nemo | Version number included |

**Benefits:**
- **Reversible:** Can determine model from filename
- **Collision-proof:** Each model has unique name
- **Intuitive:** Matches HuggingFace model ID naming
- **Sortable:** Alphabetical order groups by model family
- **Recognizable:** Easy to see which versions exist

---

## Load Order Implementation

**In load_model() function:**

When user selects "Canary-1B v2" in Gradio:
1. Check if `local_models/canary-1b-v2.nemo` exists
2. If YES → Load from local (fast, no internet)
3. If NO → Download from HuggingFace via `from_pretrained()` (slow, first time only)
4. Subsequent uses → Cached in `~/.cache/torch/NeMo/` (fast)

**User experience:**
```
First run (no setup):
- User clicks Transcribe with Canary-1B-v2
- File not found locally
- "Downloading from HuggingFace..." (2-3 min)
- Transcription completes
- Model cached for future use

After running setup script:
- User runs: python setup_local_models.py
- Select option 4: Download Canary-1B-v2
- Saves to canary-1b-v2.nemo
- Next transcription: Instant load from local file
```

---

## Integration Checklist

<acceptance_criteria>
- [ ] Update MODEL_CONFIGS in transcribe_ui.py with unique local paths for all 4 models
- [ ] Change loading_method to "local_or_huggingface" for all models
- [ ] Implement fallback logic in load_model(): try local first, then HuggingFace
- [ ] Rewrite setup_local_models.py with model definitions dictionary
- [ ] Create menu system allowing multiple model selection
- [ ] Implement generic download_and_save_model() function
- [ ] Add model status checking (what's already downloaded)
- [ ] Add file verification after download (check size, verify loadable)
- [ ] Support batch downloading (all models at once)
- [ ] Update console output with clear status messages
- [ ] Test loading from local .nemo files (all 4 models)
- [ ] Test fallback to HuggingFace when local file missing
- [ ] Test fallback when local file is corrupted
- [ ] Verify unique filenames prevent overwrites
- [ ] Ensure storage paths match between setup script and transcribe_ui.py
- [ ] Update user documentation with new setup instructions
- [ ] Test on Windows (path handling with backslashes)
- [ ] Test with limited disk space (graceful failure)
</acceptance_criteria>

---

## Key Code Patterns

### Pattern 1: Model Definition Dictionary
```python
MODELS = {
    "choice": {
        "model_id": "hf_id",
        "filename": "local_name.nemo",
        "display_name": "User-Readable Name",
        ...
    }
}
# Benefits: DRY principle, easy to add/remove models, iterable for menus
```

### Pattern 2: Fallback Loading
```python
# Try local first
if local_path.exists():
    load from local
else:
    # Fallback to HF
    download and load from HF
# Benefits: Fast when possible, reliable when not
```

### Pattern 3: Unique Filenames
```python
# Based on model ID components
filename = model_id.split('/')[-1] + ".nemo"
# Result: "parakeet-tdt-0.6b-v3" → "parakeet-tdt-0.6b-v3.nemo"
# (or normalize to: "parakeet-0.6b-v3.nemo")
# Benefits: No collisions, intuitive, reversible
```

---

## Database/Storage Design

**Directory structure after full setup:**
```
my-project/
├── transcribe_ui.py
├── setup_local_models.py
├── local_models/
│   ├── parakeet-0.6b-v3.nemo      (~2.4 GB)
│   ├── parakeet-1.1b.nemo         (~4.5 GB)
│   ├── canary-1b.nemo             (~5.0 GB)
│   ├── canary-1b-v2.nemo          (~5.0 GB)
│   ├── torch/                      (cache)
│   ├── huggingface/                (cache)
│   └── tmp/                        (temp)
└── model_cache/                    (custom cache from Part 3)
```

**Total disk space:** ~16.9 GB for all 4 models + overhead
**Your RTX 4080 has:** 12GB VRAM (sufficient for any single model)

---

## Error Handling Specifics

**In setup script:**
- If download fails mid-way: Partial .nemo file exists, script warns user, suggests retry
- If save fails: Clear error message with disk space check
- If file too small: Corruption detected, prompt to re-download

**In transcribe_ui.py:**
- If local file exists but corrupted: Log warning, auto-download from HF
- If HF download fails: Return clear error to user in Gradio UI
- If both fail: Return specific error message with troubleshooting steps

---

## Backward Compatibility

✅ **100% backward compatible:**
- Old `parakeet.nemo` files will still work (setup script won't overwrite if named differently)
- Existing transcriptions continue working
- Users can manually rename old files to new scheme if desired
- Fallback to HuggingFace ensures nothing breaks

---

**Priority:** Medium (Enhancement) | **Dependencies:** None | **Complexity:** Medium-High (Script redesign + load logic)
