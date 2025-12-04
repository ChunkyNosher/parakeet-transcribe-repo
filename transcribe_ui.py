import gradio as gr
import nemo.collections.asr as nemo_asr
import torch
import os
import time

# Global model cache to avoid reloading
models_cache = {}

def load_model(model_name):
    """Load model if not already in cache"""
    if model_name not in models_cache:
        print(f"Loading {model_name} model...")
        if model_name == "parakeet":
            models_cache[model_name] = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/parakeet-tdt-0.6b-v2"
            )
        else:  # canary
            models_cache[model_name] = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/canary-qwen-2.5b"
            )
        print(f"✓ {model_name} loaded successfully")
    return models_cache[model_name]

def transcribe_audio(audio_file, model_choice, save_to_file, include_timestamps):
    """
    Main transcription function
    
    Args:
        audio_file: Path to uploaded audio file
        model_choice: "Parakeet (Fast)" or "Canary-Qwen (Accurate)"
        save_to_file: Boolean - whether to save .txt file
        include_timestamps: Boolean - whether to generate timestamps
    
    Returns:
        status_message: HTML formatted status
        transcription_text: The actual transcription
        download_file: Path to saved file (if save_to_file=True)
    """
    
    if audio_file is None:
        return "⚠️ Please upload an audio file first", "", None
    
    try:
        # Determine which model to use
        model_key = "parakeet" if "Parakeet" in model_choice else "canary"
        
        # Start timing
        start_time = time.time()
        
        # Load model (uses cache if already loaded)
        model = load_model(model_key)
        load_time = time.time() - start_time
        
        # Get audio duration
        import librosa
        duration = librosa.get_duration(path=audio_file)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        # Transcribe
        inference_start = time.time()
        result = model.transcribe(
            [audio_file], 
            batch_size=1,
            timestamps=include_timestamps
        )
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        # Extract transcription
        transcription = result[0].text
        
        # Get GPU stats
        vram_used = torch.cuda.memory_allocated() / 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        
        # Calculate real-time factor
        rtfx = duration / inference_time if inference_time > 0 else 0
        
        # Format timestamps if requested
        timestamp_text = ""
        if include_timestamps and hasattr(result[0], 'words') and result[0].words:
            timestamp_text = "\n\n### Word-Level Timestamps (first 50 words):\n\n"
            for i, word in enumerate(result[0].words[:50]):
                timestamp_text += f"`{word.start:.2f}s - {word.end:.2f}s` → **{word.text}**\n\n"
            if len(result[0].words) > 50:
                timestamp_text += f"\n*...and {len(result[0].words) - 50} more words*"
        
        # Build status message
        status = f"""
### ✅ Transcription Complete!

**📊 Statistics:**
- **Model**: {model_choice}
- **GPU**: {gpu_name}
- **Audio Duration**: {minutes}m {seconds}s
- **Processing Time**: {total_time:.2f} seconds
- **Inference Time**: {inference_time:.2f} seconds
- **Model Load Time**: {load_time:.2f} seconds
- **Real-Time Factor**: {rtfx:.1f}× (processed {rtfx:.1f}× faster than real-time)
- **VRAM Used**: {vram_used:.2f} GB / 12.0 GB
- **Transcription Length**: {len(transcription)} characters ({len(transcription.split())} words)

---
"""
        
        # Save to file if requested
        output_file = None
        if save_to_file:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_file = f"{base_name}_transcription.txt"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Audio File: {os.path.basename(audio_file)}\n")
                f.write(f"Model: {model_choice}\n")
                f.write(f"Duration: {minutes}m {seconds}s\n")
                f.write(f"Processing Time: {total_time:.2f}s\n")
                f.write(f"\n{'='*60}\n")
                f.write(f"TRANSCRIPTION\n")
                f.write(f"{'='*60}\n\n")
                f.write(transcription)
                
                if timestamp_text:
                    f.write(f"\n\n{'='*60}\n")
                    f.write(f"WORD-LEVEL TIMESTAMPS\n")
                    f.write(f"{'='*60}\n\n")
                    for word in result[0].words:
                        f.write(f"{word.start:.2f}s - {word.end:.2f}s: {word.text}\n")
            
            status += f"\n💾 **Saved to**: `{output_file}`"
        
        return status, transcription + timestamp_text, output_file
        
    except Exception as e:
        error_msg = f"""
### ❌ Error During Transcription

**Error Type**: {type(e).__name__}

**Error Message**: {str(e)}

**Troubleshooting:**
1. Make sure the audio file is valid
2. Check that you have enough VRAM (you have 12GB)
3. Try a shorter audio file first
4. Restart the interface if models are stuck

If the error persists, please check the console output for more details.
"""
        return error_msg, "", None

def get_system_info():
    """Display system information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_free = (torch.cuda.get_device_properties(0).total_memory - 
                     torch.cuda.memory_allocated()) / 1024**3
        
        info = f"""
### 🖥️ System Information

**GPU**: {gpu_name}
**Total VRAM**: {vram_total:.1f} GB
**Available VRAM**: {vram_free:.1f} GB
**CUDA Version**: {torch.version.cuda}
**PyTorch Version**: {torch.__version__}
**NeMo Available**: ✅ Yes

**Status**: ✅ Ready for transcription
"""
    else:
        info = """
### ⚠️ No GPU Detected

CUDA is not available. Please check:
1. NVIDIA GPU drivers are installed
2. CUDA Toolkit is installed
3. PyTorch was installed with CUDA support
"""
    return info

# Create the Gradio interface
with gr.Blocks(title="🎙️ Local ASR Transcription") as app:
    
    # Header
    gr.Markdown("""
    # 🎙️ NVIDIA NeMo Local Audio Transcription
    ### Powered by Parakeet-TDT & Canary-Qwen on Your RTX 4080
    
    Transform your audio files into accurate text transcriptions using state-of-the-art AI models running entirely on your local GPU.
    """)
    
    # System info at the top
    with gr.Accordion("📊 System Information", open=False):
        system_info = gr.Markdown(get_system_info())
        refresh_btn = gr.Button("🔄 Refresh System Info", size="sm")
        refresh_btn.click(fn=get_system_info, outputs=system_info)
    
    gr.Markdown("---")
    
    # Main interface
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Audio")
            
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio File"
            )
            
            gr.Markdown("**Supported formats**: WAV, MP3, FLAC, M4A, OGG, AAC")
            
            gr.Markdown("### ⚙️ Settings")
            
            model_selector = gr.Radio(
                choices=[
                    "Parakeet-TDT-0.6B v2 (Fast)",
                    "Canary-Qwen-2.5B (Accurate)"
                ],
                value="Parakeet-TDT-0.6B v2 (Fast)",
                label="Model Selection",
                info="Parakeet: Faster processing | Canary: Higher accuracy"
            )
            
            save_checkbox = gr.Checkbox(
                label="💾 Save transcription to .txt file",
                value=True,
                info="Creates a text file in the same directory"
            )
            
            timestamp_checkbox = gr.Checkbox(
                label="⏱️ Include word-level timestamps",
                value=False,
                info="Shows when each word was spoken"
            )
            
            transcribe_btn = gr.Button(
                "🚀 Start Transcription",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ---
            ### 📖 Model Information
            
            **Parakeet-TDT-0.6B v2:**
            - Speed: ~40-60× real-time
            - Accuracy: 93.32% (6.68% WER)
            - Best for: Quick transcriptions, bulk processing
            
            **Canary-Qwen-2.5B:**
            - Speed: ~50-100× real-time
            - Accuracy: 94.37% (5.63% WER)  
            - Best for: Critical accuracy, technical content
            """)
        
        # Right column - Output
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Transcription Results")
            
            status_output = gr.Markdown(
                "Upload an audio file and click 'Start Transcription' to begin..."
            )
            
            # FIXED: Replaced show_copy_button with buttons parameter
            transcription_output = gr.Textbox(
                label="Transcription Text",
                lines=20,
                placeholder="Your transcription will appear here...",
                buttons=["copy"],  # NEW: Use buttons parameter instead
                show_label=True
            )
            
            file_output = gr.File(
                label="📥 Download Transcription File",
                visible=True
            )
    
    # Bottom section
    gr.Markdown("---")
    
    with gr.Accordion("❓ How to Use", open=False):
        gr.Markdown("""
        ### Quick Start:
        
        1. **Upload audio** (or record with microphone)
        2. **Select model** (Parakeet for speed, Canary for accuracy)
        3. **Click "Start Transcription"**
        4. **Copy or download** your transcription
        
        ### Tips:
        - First run loads models (~20-30 seconds)
        - Subsequent runs much faster
        - Processing time: ~30-60 seconds per hour of audio
        """)
    
    with gr.Accordion("🔒 Privacy & Performance", open=False):
        gr.Markdown("""
        ### Privacy:
        - ✅ 100% local processing
        - ✅ No internet required (after model download)
        - ✅ Your audio never leaves your computer
        
        ### Performance:
        - **GPU**: RTX 4080 Laptop (12GB VRAM)
        - **Speed**: 40-100× faster than real-time
        - **Example**: 1 hour audio → 30-60 seconds processing
        """)
    
    # Connect button
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_selector, save_checkbox, timestamp_checkbox],
        outputs=[status_output, transcription_output, file_output]
    )

# Launch
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting NVIDIA NeMo Transcription Interface")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"✅ CUDA: {torch.version.cuda}")
    else:
        print("⚠️  WARNING: No CUDA GPU detected!")
    
    print("="*80)
    print("\n🌐 Opening in browser at: http://127.0.0.1:7860")
    print("💡 Keep this terminal open while using the interface")
    print("🛑 Press Ctrl+C to stop\n")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )
