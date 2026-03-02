# Integration Plan for Latest Frontier Audio Transcription Models

## Overview

This document outlines the integration plan for adding the latest frontier audio transcription AI models to your existing Gradio transcription application. The plan covers model discovery, integration points, and implementation details for each candidate model.

## Current Implementation Analysis

Your existing transcription system consists of two main components:

1. **setup_local_models.py**: Handles downloading and managing Nvidia NeMo models (Parakeet and Canary families)
2. **transcribe_ui.py**: Provides the Gradio interface for audio transcription with features like:
   - Tensor-based transcription to avoid Windows file locking
   - Chunked processing for long audio files
   - ITN (Inverse Text Normalization) support
   - Multiple output formats (SRT, CSV, TXT with timestamps)
   - GPU optimization with mixed precision

## Target Models for Integration

Based on research conducted in March 2026, the following frontier models should be considered for integration:

### 1. IBM Granite Speech 3.3 8B

**Model Location**: HuggingFace (ibm-granite/granite-speech-3.3-8b)

**Key Specifications**:
- Parameters: ~9 billion
- WER: 5.85% (tops HuggingFace Open ASR leaderboard)
- Languages: English, French, German, Spanish, plus English-to-Japanese/Mandarin translation
- License: Apache 2.0

**Integration Approach**:
- Requires different loading mechanism (not NeMo-based)
- Likely needs custom inference pipeline
- May require significant architectural changes to your existing tensor-based transcription method

### 2. Mistral Voxtral Transcribe 2

**Model Location**: Mistral AI (voxtral-transcribe-2)

**Key Specifications**:
- Sub-200ms latency for real-time applications
- At 480ms delay: 1-2% WER (near-offline accuracy)
- Natively multilingual
- On-device execution capability
- Built-in speaker diarization

**Integration Approach**:
- Mistral provides both batch and streaming variants
- Different API compared to NeMo models
- May need custom implementation for Gradio integration
- Could be added to MODEL_CONFIGS with new loading method

### 3. Whisper Large V3 / Distil-Whisper

**Model Location**: OpenAI Whisper (openai/whisper-large-v3, distil-whisper/distil-whisper-large-v3)

**Key Specifications**:
- Whisper Large V3: 1.55B parameters, 7.4% WER, 99+ languages
- Distil-Whisper: 6x faster inference, 756M parameters, English-only optimized
- Well-documented with multiple integration options
- Extensive community support

**Integration Approach**:
- Use HuggingFace Transformers library for loading
- Compatible with existing tensor-based transcription approach
- Add to MODEL_CONFIGS with whisper loading method
- Leverage existing output formatting (timestamps, etc.)

### 4. Deepgram Nova-3 (Commercial API)

**Model Location**: Deepgram API (deepgram.com)

**Key Specifications**:
- WER: 5.26% (best among commercial offerings)
- Latency: Sub-300ms for real-time
- Pricing: $0.0043/min (batch), $0.0077/min (streaming)
- Requires API key integration

**Integration Approach**:
- Add as optional cloud-based transcription backend
- Requires API key configuration
- Can be primary or fallback option
- Different implementation path (API calls vs local inference)

### 5. Kyutai STT

**Model Location**: HuggingFace (kyutai/stt-1b-en_fr, kyutai/stt-2.6b-en_fr)

**Key Specifications**:
- 1B and 2.6B parameter variants
- Supports English and French
- Real-time streaming: 400 audio streams on single H100
- Delayed Streams Modeling architecture

**Integration Approach**:
- Use HuggingFace Transformers library
- Similar approach to Whisper integration
- Add to MODEL_CONFIGS

## Implementation Priority

Based on complexity and impact, here's the recommended implementation order:

1. **Whisper Large V3 / Distil-Whisper** - Easiest integration, well-documented
2. **Kyutai STT** - Similar architecture to Whisper
3. **IBM Granite Speech 3.3 8B** - Higher accuracy gains
4. **Mistral Voxtral Transcribe 2** - Advanced features, different API
5. **Deepgram Nova-3** - Requires API key, cloud-based

## Integration Points

### 1. setup_local_models.py

Modify the `MODELS_TO_DOWNLOAD` dictionary to include new model entries:

```python
MODELS_TO_DOWNLOAD = {
    # Existing NeMo models...
    
    # New Whisper model
    "6": {
        "model_id": "openai/whisper-large-v3",
        "filename": "whisper-large-v3.bin",
        "display_name": "Whisper Large V3",
        "library": "transformers",  # Different from nemo
    },
    # Additional models...
}
```

### 2. transcribe_ui.py

Key integration points:

#### a) Model Configuration (MODEL_CONFIGS)

Add new model configurations:

```python
MODEL_CONFIGS = {
    # Existing NeMo models...
    
    # Whisper Large V3
    "whisper-large-v3": {
        "hf_model_id": "openai/whisper-large-v3",
        "library": "transformers",
        "display_name": "Whisper Large V3",
        "languages": 99,
        "wer": "7.4%",
        "vram_gb": "~10GB",
    },
    
    # Distil-Whisper (faster alternative)
    "distil-whisper-large-v3": {
        "hf_model_id": "distil-whisper/distil-whisper-large-v3",
        "library": "transformers",
        "display_name": "Distil-Whisper Large V3",
        "languages": 1,  # English
        "wer": "~7.5%",
        "vram_gb": "~5GB",
    },
}
```

#### b) Model Loading Function

Create a flexible model loading mechanism that supports different libraries:

```python
def load_asr_model(model_key: str, model_config: dict):
    library = model_config.get("library", "nemo")
    
    if library == "nemo":
        # Existing NeMo loading
        return nemo_asr.models.ASRModel.from_pretrained(...)
    elif library == "transformers":
        # New Transformers-based loading for Whisper/Kyutai
        return load_transformers_model(model_config)
    elif library == "mistral":
        # Mistral-specific loading
        return load_mistral_model(model_config)
    elif library == "api":
        # Deepgram API client
        return DeepgramClient(model_config["api_key"])
```

#### c) Transcription Function

Update `_transcribe_single_buffer` to handle different model types:

```python
def _transcribe_single_buffer(model, buffer, model_config, use_cuda):
    library = model_config.get("library", "nemo")
    
    if library == "nemo":
        # Existing NeMo transcription
        return model.transcribe(...)
    elif library == "transformers":
        # Whisper/Kyutai transcription
        return transcribe_with_transformers(model, buffer)
    elif library == "mistral":
        # Mistral transcription
        return transcribe_with_mistral(model, buffer)
```

## Technical Considerations

### 1. Audio Preprocessing

Different models have varying input requirements:
- NeMo models expect 16kHz mono
- Whisper expects 16kHz mono
- Mistral may have different requirements

Maintain existing audio preprocessing but add model-specific adjustments.

### 2. Output Format Compatibility

Ensure timestamp extraction works across all models:
- NeMo: Word-level timestamps with `timestamp` attribute
- Whisper: Segment-level timestamps
- Mistral: Model-specific timestamp format

Create unified timestamp extraction interface.

### 3. VRAM Management

Different models have different VRAM requirements:
- Distil-Whisper: ~5GB (good for lower-end GPUs)
- Whisper Large V3: ~10GB
- IBM Granite Speech 3.3 8B: ~9GB (higher)

Update batch size calculations accordingly.

### 4. Error Handling

Each model may throw different errors. Create model-specific error handling:

```python
def handle_transcription_error(e, model_key):
    if "whisper" in model_key:
        # Whisper-specific error handling
        pass
    elif "granite" in model_key:
        # Granite-specific error handling
        pass
```

## Testing Strategy

1. **Unit Tests**: Test individual model loading and transcription
2. **Integration Tests**: Test full pipeline with sample audio
3. **Performance Tests**: Measure latency and throughput for each model
4. **Comparison Tests**: Verify WER against benchmark datasets

## Resources

- IBM Granite Speech: https://huggingface.co/ibm-granite/granite-speech-3.3-8b
- Mistral Voxtral: https://mistral.ai/news/voxtral-transcribe-2
- Whisper: https://huggingface.co/openai/whisper-large-v3
- Distil-Whisper: https://huggingface.co/distil-whisper/distil-whisper-large-v3
- Kyutai STT: https://huggingface.co/kyutai/stt-1b-en_fr
- Deepgram: https://deepgram.com

## Quick Reference: Model Comparison Table

| Model | WER | Parameters | Languages | VRAM | License | Integration Difficulty |
|-------|-----|------------|-----------|------|---------|----------------------|
| Nvidia Parakeet (baseline) | 8.0% | 1.1B | 1 | ~4GB | CC-BY-4.0 | Existing |
| Nvidia Canary (baseline) | 5.63% | 2.5B | 25 | ~5GB | CC-BY-4.0 | Existing |
| IBM Granite Speech 3.3 8B | 5.85% | ~9B | 5+ | ~9GB | Apache 2.0 | Medium |
| Mistral Voxtral Transcribe 2 | 1-2% | N/A | Multi | N/A | Proprietary | Hard |
| Whisper Large V3 | 7.4% | 1.55B | 99+ | ~10GB | MIT | Easy |
| Distil-Whisper | ~7.5% | 756M | 1 | ~5GB | MIT | Easy |
| Kyutai STT 1B | Competitive | 1B | EN/FR | ~4GB | Open | Easy |
| Deepgram Nova-3 | 5.26% | N/A | 10+ | N/A | Commercial | Easy (API) |

## Implementation Checklist

- [ ] Add new model configurations to MODEL_CONFIGS
- [ ] Implement flexible model loading for different libraries (transformers, mistral, api)
- [ ] Update transcription functions to handle multiple model types
- [ ] Add model-specific audio preprocessing
- [ ] Implement unified timestamp extraction
- [ ] Update VRAM management and batch size calculations
- [ ] Add model-specific error handling
- [ ] Update UI to display new model options
- [ ] Test each model integration
- [ ] Verify output format compatibility
- [ ] Document new models in setup_local_models.py