# User Model Setup Guide

This guide explains how to set up the local NVIDIA NeMo ASR models for the `transcribe_ui.py` application.

## Overview

The transcription interface uses local `.nemo` model files instead of downloading from HuggingFace on each run. This provides:

- **100% offline operation** - No internet required after setup
- **Faster loading times** - No cache validation overhead
- **No re-download issues** - Immune to HuggingFace revision changes
- **Predictable behavior** - Same model every time

## Prerequisites

Before running the setup script, ensure you have:

1. **NVIDIA GPU** with CUDA support (minimum 8GB VRAM recommended)
2. **CUDA Toolkit** installed and working
3. **Python 3.8+** with the following packages:
   - PyTorch with CUDA support
   - NeMo Framework (`nemo_toolkit[asr]`)
   - Gradio
4. **Internet connection** for initial model download
5. **Disk space** - At least 7GB free for model files

## Quick Setup

Run the setup script to download and save models locally:

```bash
python setup_local_models.py
```

The script will:
1. Check for CUDA GPU availability
2. Create the `local_models/` directory
3. Download models from HuggingFace (one-time only)
4. Save models as `.nemo` files for offline use

**Estimated download time**: 5-15 minutes depending on internet speed  
**Total disk space required**: ~5.7 GB

## Manual Setup (Alternative)

If you prefer to set up models manually or the setup script fails:

### Step 1: Create the directory

```bash
mkdir -p local_models
```

### Step 2: Download and save Parakeet model

```python
import nemo.collections.asr as nemo_asr

# Download from HuggingFace
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

# Save to local file
model.save_to("local_models/parakeet.nemo")
```

### Step 3: Download and save Canary model

```python
import nemo.collections.asr as nemo_asr

# Download from HuggingFace
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-qwen-2.5b")

# Save to local file
model.save_to("local_models/canary.nemo")
```

## Expected File Structure

After setup, your directory should look like:

```
myEditingScripts/
├── transcribe_ui.py
├── setup_local_models.py
├── local_models/
│   ├── parakeet.nemo    (~600 MB)
│   └── canary.nemo      (~5.1 GB)
└── docs/
    └── manual/
        └── user-model-setup-guide.md
```

## Verifying Setup

To verify your setup is correct:

1. Check that both files exist:
   ```bash
   ls -lh local_models/
   ```
   
   Expected output:
   ```
   -rw-r--r-- 1 user user 600M ... parakeet.nemo
   -rw-r--r-- 1 user user 5.1G ... canary.nemo
   ```

2. Run the transcription interface:
   ```bash
   python transcribe_ui.py
   ```
   
   You should see:
   ```
   📦 Checking local model files...
      Parakeet-TDT-0.6B v2: ✅ Found
      Canary-Qwen-2.5B: ✅ Found
   ```

## Troubleshooting

### "LOCAL MODELS NOT FOUND" Error

**Problem**: The `local_models/` directory or `.nemo` files don't exist.

**Solution**: Run the setup script:
```bash
python setup_local_models.py
```

### Download Fails

**Problem**: Network error or timeout during download.

**Solutions**:
1. Check your internet connection
2. Try again later (HuggingFace may be experiencing issues)
3. Use a VPN if HuggingFace is blocked in your region
4. Download manually using the manual setup steps above

### "Out of Memory" Error

**Problem**: Not enough GPU memory to load the model.

**Solutions**:
1. Close other GPU-intensive applications
2. Use the smaller Parakeet model instead of Canary
3. Restart Python to clear GPU memory

### Model Loading Slow

**Problem**: Model takes a long time to load from `.nemo` file.

**Note**: This is expected behavior. Loading a 5GB model takes ~2-3 seconds. The first transcription of each session will have this overhead, but subsequent transcriptions reuse the cached model.

### Corrupted .nemo File

**Problem**: Model fails to load with cryptic error messages.

**Solution**: Delete and re-download the problematic model:
```bash
rm local_models/canary.nemo
python setup_local_models.py
```

## Model Information

### Parakeet-TDT-0.6B v2
- **File**: `local_models/parakeet.nemo`
- **Size**: ~600 MB
- **Speed**: ~40-60× real-time
- **Accuracy**: 93.32% (6.68% WER)
- **Best for**: Quick transcriptions, bulk processing

### Canary-Qwen-2.5B
- **File**: `local_models/canary.nemo`
- **Size**: ~5.1 GB
- **Speed**: ~50-100× real-time
- **Accuracy**: 94.37% (5.63% WER)
- **Best for**: Critical accuracy, technical content

## Updating Models

When NVIDIA releases new model versions:

1. Delete old model files:
   ```bash
   rm local_models/*.nemo
   ```

2. Update `MODEL_CONFIGS` in `transcribe_ui.py` and `setup_local_models.py` with new model names

3. Re-run setup:
   ```bash
   python setup_local_models.py
   ```

## Technical Details

### What's in a .nemo file?

A `.nemo` file is a tar.gz archive containing:
- `model_config.yaml` - Model configuration
- `model_weights.ckpt` - PyTorch checkpoint with model weights
- Tokenizer files (vocabulary, BPE models, etc.)
- Any other artifacts the model needs

### Why local files instead of HuggingFace cache?

The HuggingFace cache system uses revision hashes to track model versions. When a repository is updated (even minor README changes), the hash changes, causing NeMo to:
1. Detect the cached model as "outdated"
2. Delete the entire cache (~5GB)
3. Re-download from scratch

This creates an infinite re-download loop if the repository is frequently updated.

Using `restore_from()` with local `.nemo` files bypasses all HuggingFace lookups, providing:
- No revision hash validation
- No network requests
- Predictable, consistent behavior
- Professional production-ready deployment

---

**Last Updated**: December 2024  
**Compatible with**: NeMo Framework 1.x, 2.x
