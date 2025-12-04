import gradio as gr
import nemo.collections.asr as nemo_asr
import torch
import os
import time
from pathlib import Path
from huggingface_hub import scan_cache_dir

# Global model cache to avoid reloading
models_cache = {}

# Video file extensions supported (librosa + FFmpeg backend handles audio extraction)
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.m4v'}

# Audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# Separator string for output formatting
SEPARATOR = '=' * 60

# Model configurations
MODEL_CONFIGS = {
    "parakeet": {
        "repo_id": "nvidia/parakeet-tdt-0.6b-v2",
        "max_batch_size": 32,  # Increased from 16
        "display_name": "Parakeet-TDT-0.6B v2"
    },
    "canary": {
        "repo_id": "nvidia/canary-qwen-2.5b", 
        "max_batch_size": 16,  # Increased from default
        "display_name": "Canary-Qwen-2.5B"
    }
}

def check_model_cached(model_name):
    """Check if a model is already cached in HuggingFace cache"""
    try:
        cache_info = scan_cache_dir()
        model_repo_id = MODEL_CONFIGS[model_name]["repo_id"]
        for repo in cache_info.repos:
            if model_repo_id in repo.repo_id:
                return True
        return False
    except Exception:
        return False

def get_dynamic_batch_size(duration, model_key):
    """Calculate optimal batch size based on audio duration and model"""
    max_batch = MODEL_CONFIGS[model_key]["max_batch_size"]
    
    if duration < 60:  # Under 1 minute
        return min(8, max_batch)
    elif duration < 300:  # Under 5 minutes
        return min(16, max_batch)
    elif duration < 900:  # Under 15 minutes
        return min(24, max_batch)
    else:  # Longer audio
        return max_batch

def setup_gpu_optimizations():
    """Enable GPU optimizations for better performance"""
    if torch.cuda.is_available():
        # Enable TF32 for matrix multiplication (Ampere+ GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cuDNN benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True
        print("✅ GPU optimizations enabled (TF32, cuDNN benchmark)")

def load_model(model_name, show_progress=False):
    """Load model if not already in cache
    
    Args:
        model_name: "parakeet" or "canary"
        show_progress: Whether to show loading progress (for startup)
    """
    if model_name not in models_cache:
        config = MODEL_CONFIGS[model_name]
        is_cached = check_model_cached(model_name)
        
        if is_cached:
            print(f"📦 Loading {config['display_name']} from cache...")
        else:
            print(f"⬇️ Downloading {config['display_name']} (first time, please wait)...")
        
        start_time = time.time()
        models_cache[model_name] = nemo_asr.models.ASRModel.from_pretrained(
            config["repo_id"]
        )
        load_time = time.time() - start_time
        
        print(f"✓ {config['display_name']} loaded in {load_time:.1f}s")
    return models_cache[model_name]

def transcribe_audio(audio_files, model_choice, save_to_file, include_timestamps):
    """
    Main transcription function with batch processing, video support, and GPU optimization
    
    Args:
        audio_files: Path to uploaded audio/video file OR list of paths for batch processing
        model_choice: "Parakeet-TDT-0.6B v2 (Fast)" or "Canary-Qwen-2.5B (Accurate)"
        save_to_file: Boolean - whether to save .txt file
        include_timestamps: Boolean - whether to generate timestamps
    
    Returns:
        status_message: HTML formatted status
        transcription_text: The actual transcription(s)
        download_file: Path to saved file (if save_to_file=True)
    """
    
    if audio_files is None:
        return "⚠️ Please upload an audio or video file first", "", None
    
    try:
        import librosa
        
        # Handle both single file and multiple files
        # Gradio gr.File with file_count="multiple" returns list of file objects
        # Each file object has a .name attribute with the path
        if audio_files is None or (isinstance(audio_files, list) and len(audio_files) == 0):
            return "⚠️ Please upload an audio or video file first", "", None
        
        # Convert file objects to paths
        if isinstance(audio_files, str):
            file_list = [audio_files]
        elif isinstance(audio_files, list):
            # Handle list of file objects or strings
            file_list = []
            for f in audio_files:
                if hasattr(f, 'name'):
                    file_list.append(f.name)
                else:
                    file_list.append(str(f))
        elif hasattr(audio_files, 'name'):
            file_list = [audio_files.name]
        else:
            file_list = [str(audio_files)]
        
        is_batch = len(file_list) > 1
        
        # Determine which model to use
        model_key = "parakeet" if "Parakeet" in model_choice else "canary"
        
        # Start timing
        start_time = time.time()
        
        # Load model (uses cache if already loaded)
        model = load_model(model_key)
        load_time = time.time() - start_time
        
        # Process files and detect video files
        processed_files = []
        file_info = []
        total_duration = 0
        video_count = 0
        
        for file_path in file_list:
            file_ext = os.path.splitext(file_path)[1].lower()
            is_video = file_ext in VIDEO_EXTENSIONS
            
            if is_video:
                video_count += 1
                print(f"🎬 Extracting audio from video: {os.path.basename(file_path)}")
            
            # librosa.get_duration handles both audio and video files (via FFmpeg)
            try:
                duration = librosa.get_duration(path=file_path)
            except Exception as e:
                # Handle case where video has no audio track
                if is_video:
                    return f"❌ Video file '{os.path.basename(file_path)}' appears to have no audio track or cannot be processed.\n\nError: {str(e)}", "", None
                raise
            
            total_duration += duration
            processed_files.append(file_path)
            file_info.append({
                "path": file_path,
                "name": os.path.basename(file_path),
                "duration": duration,
                "is_video": is_video
            })
        
        # Calculate optimal batch size based on total duration
        batch_size = get_dynamic_batch_size(total_duration, model_key)
        
        # Prepare status update for video processing
        video_status = ""
        if video_count > 0:
            video_status = f"🎬 Extracted audio from {video_count} video file(s)\n"
        
        # Transcribe with mixed precision (FP16) for GPU acceleration
        inference_start = time.time()
        
        if torch.cuda.is_available():
            # Use mixed precision (FP16) for faster inference on CUDA
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = model.transcribe(
                    processed_files, 
                    batch_size=batch_size,
                    timestamps=include_timestamps
                )
        else:
            # CPU fallback - no autocast
            result = model.transcribe(
                processed_files, 
                batch_size=batch_size,
                timestamps=include_timestamps
            )
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        # Get GPU stats
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
        else:
            vram_used = 0
            gpu_name = "CPU"
        
        # Calculate real-time factor
        rtfx = total_duration / inference_time if inference_time > 0 else 0
        
        # Build output based on single vs batch processing
        if is_batch:
            # Batch processing output
            all_transcriptions = []
            per_file_stats = []
            
            for i, (res, info) in enumerate(zip(result, file_info)):
                transcription = res.text
                all_transcriptions.append(transcription)
                
                file_duration = info["duration"]
                file_mins = int(file_duration // 60)
                file_secs = int(file_duration % 60)
                file_type = "🎬 Video" if info["is_video"] else "🎵 Audio"
                
                per_file_stats.append(
                    f"**{i+1}. {info['name']}** ({file_type})\n"
                    f"   - Duration: {file_mins}m {file_secs}s\n"
                    f"   - Words: {len(transcription.split())}"
                )
            
            total_mins = int(total_duration // 60)
            total_secs = int(total_duration % 60)
            total_words = sum(len(t.split()) for t in all_transcriptions)
            
            status = f"""
### ✅ Batch Transcription Complete!

{video_status}**📊 Overall Statistics:**
- **Files Processed**: {len(file_list)}
- **Model**: {model_choice}
- **GPU**: {gpu_name}
- **Total Audio Duration**: {total_mins}m {total_secs}s
- **Processing Time**: {total_time:.2f} seconds
- **Inference Time**: {inference_time:.2f} seconds
- **Batch Size Used**: {batch_size}
- **Real-Time Factor**: {rtfx:.1f}× (processed {rtfx:.1f}× faster than real-time)
- **VRAM Used**: {vram_used:.2f} GB
- **Total Words**: {total_words}

**📁 Per-File Statistics:**
{chr(10).join(per_file_stats)}

---
"""
            # Combine transcriptions with file headers
            combined_transcription = ""
            for i, (info, trans) in enumerate(zip(file_info, all_transcriptions)):
                combined_transcription += f"\n{SEPARATOR}\n"
                combined_transcription += f"FILE {i+1}: {info['name']}\n"
                combined_transcription += f"{SEPARATOR}\n\n"
                combined_transcription += trans + "\n"
            
            transcription_output = combined_transcription
            
        else:
            # Single file processing (original behavior)
            transcription = result[0].text
            info = file_info[0]
            
            minutes = int(info["duration"] // 60)
            seconds = int(info["duration"] % 60)
            
            # Format timestamps if requested
            timestamp_text = ""
            if include_timestamps and hasattr(result[0], 'words') and result[0].words:
                timestamp_text = "\n\n### Word-Level Timestamps (first 50 words):\n\n"
                for i, word in enumerate(result[0].words[:50]):
                    timestamp_text += f"`{word.start:.2f}s - {word.end:.2f}s` → **{word.text}**\n\n"
                if len(result[0].words) > 50:
                    timestamp_text += f"\n*...and {len(result[0].words) - 50} more words*"
            
            file_type_msg = "🎬 Video file detected - audio extracted automatically\n" if info["is_video"] else ""
            
            status = f"""
### ✅ Transcription Complete!

{file_type_msg}**📊 Statistics:**
- **Model**: {model_choice}
- **GPU**: {gpu_name}
- **Audio Duration**: {minutes}m {seconds}s
- **Processing Time**: {total_time:.2f} seconds
- **Inference Time**: {inference_time:.2f} seconds
- **Model Load Time**: {load_time:.2f} seconds
- **Batch Size Used**: {batch_size}
- **Real-Time Factor**: {rtfx:.1f}× (processed {rtfx:.1f}× faster than real-time)
- **VRAM Used**: {vram_used:.2f} GB
- **Transcription Length**: {len(transcription)} characters ({len(transcription.split())} words)

---
"""
            transcription_output = transcription + timestamp_text
        
        # Save to file if requested
        output_file = None
        if save_to_file:
            if is_batch:
                output_file = f"batch_transcription_{len(file_list)}_files.txt"
            else:
                base_name = os.path.splitext(os.path.basename(file_list[0]))[0]
                output_file = f"{base_name}_transcription.txt"
            
            with open(output_file, "w", encoding="utf-8") as f:
                if is_batch:
                    f.write(f"Batch Transcription - {len(file_list)} files\n")
                    f.write(f"Model: {model_choice}\n")
                    f.write(f"Total Duration: {int(total_duration // 60)}m {int(total_duration % 60)}s\n")
                    f.write(f"Processing Time: {total_time:.2f}s\n")
                    f.write(f"\n{SEPARATOR}\n")
                    
                    for i, (info, trans) in enumerate(zip(file_info, all_transcriptions)):
                        f.write(f"\nFILE {i+1}: {info['name']}\n")
                        f.write(f"Duration: {int(info['duration'] // 60)}m {int(info['duration'] % 60)}s\n")
                        f.write(f"{SEPARATOR}\n\n")
                        f.write(trans)
                        f.write("\n")
                else:
                    info = file_info[0]
                    f.write(f"Audio File: {info['name']}\n")
                    f.write(f"Model: {model_choice}\n")
                    f.write(f"Duration: {int(info['duration'] // 60)}m {int(info['duration'] % 60)}s\n")
                    f.write(f"Processing Time: {total_time:.2f}s\n")
                    f.write(f"\n{SEPARATOR}\n")
                    f.write(f"TRANSCRIPTION\n")
                    f.write(f"{SEPARATOR}\n\n")
                    f.write(transcription)
                    
                    if include_timestamps and hasattr(result[0], 'words') and result[0].words:
                        f.write(f"\n\n{SEPARATOR}\n")
                        f.write(f"WORD-LEVEL TIMESTAMPS\n")
                        f.write(f"{SEPARATOR}\n\n")
                        for word in result[0].words:
                            f.write(f"{word.start:.2f}s - {word.end:.2f}s: {word.text}\n")
            
            status += f"\n💾 **Saved to**: `{output_file}`"
        
        return status, transcription_output, output_file
        
    except Exception as e:
        # Get actual VRAM info for error message
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_info = f"you have {vram_total:.0f}GB"
        else:
            vram_info = "no GPU detected"
        
        error_msg = f"""
### ❌ Error During Transcription

**Error Type**: {type(e).__name__}

**Error Message**: {str(e)}

**Troubleshooting:**
1. Make sure the audio/video file is valid
2. Check that you have enough VRAM ({vram_info})
3. Try a shorter audio file first
4. Restart the interface if models are stuck
5. For video files, ensure FFmpeg is installed

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

def get_privacy_performance_info():
    """Generate dynamic privacy & performance information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"**GPU**: {gpu_name} ({vram_total:.0f}GB VRAM)"
    else:
        gpu_info = "**GPU**: CPU Mode (No CUDA GPU detected)"
    
    return f"""
### Privacy:
- ✅ 100% local processing
- ✅ No internet required (after model download)
- ✅ Your audio never leaves your computer

### Performance Optimizations:
- {gpu_info}
- **Speed**: 40-100× faster than real-time
- **Example**: 1 hour audio → 30-60 seconds processing
- **Mixed Precision**: FP16 inference for 1.5-2.5× speedup
- **TF32 Tensor Cores**: Enabled for matrix operations
- **Dynamic Batch Sizing**: Optimized based on audio duration
"""

# Create the Gradio interface
with gr.Blocks(title="🎙️ Local ASR Transcription") as app:
    
    # Header
    gr.Markdown("""
    # 🎙️ NVIDIA NeMo Local Audio Transcription
    ### Powered by Parakeet-TDT & Canary-Qwen on Your Local GPU
    
    Transform your audio and video files into accurate text transcriptions using state-of-the-art AI models running entirely on your local GPU.
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
            gr.Markdown("### 📂 Upload Audio/Video Files")
            
            # Use gr.File for multiple file uploads (audio and video)
            audio_input = gr.File(
                file_count="multiple",
                file_types=[
                    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma",  # Audio
                    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".m4v"   # Video
                ],
                label="Audio/Video Files (select multiple)"
            )
            
            gr.Markdown("""
**Supported audio formats**: WAV, MP3, FLAC, M4A, OGG, AAC, WMA

**Supported video formats**: MP4, AVI, MKV, MOV, WEBM, FLV, M4V
*(Audio will be automatically extracted from video files)*

💡 **Tip**: Select multiple files for batch processing!
""")
            
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
        
        1. **Upload files** (audio or video - select multiple for batch processing)
        2. **Select model** (Parakeet for speed, Canary for accuracy)
        3. **Click "Start Transcription"**
        4. **Copy or download** your transcription
        
        ### Features:
        - 🎵 **Audio Support**: WAV, MP3, FLAC, M4A, OGG, AAC, WMA
        - 🎬 **Video Support**: MP4, AVI, MKV, MOV, WEBM, FLV, M4V
        - 📦 **Batch Processing**: Upload multiple files at once
        - ⚡ **GPU Optimized**: Uses FP16 mixed precision for faster processing
        
        ### Tips:
        - First run downloads models (~20-30 seconds with progress)
        - Subsequent runs are much faster (models cached)
        - Processing time: ~30-60 seconds per hour of audio
        - Video files: Audio is automatically extracted
        """)
    
    with gr.Accordion("🔒 Privacy & Performance", open=False):
        gr.Markdown(get_privacy_performance_info())
    
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
        
        # Enable GPU optimizations (TF32, cuDNN)
        setup_gpu_optimizations()
    else:
        print("⚠️  WARNING: No CUDA GPU detected!")
    
    print("="*80)
    
    # Check model cache status
    print("\n📦 Checking model cache...")
    for model_name, config in MODEL_CONFIGS.items():
        is_cached = check_model_cached(model_name)
        status = "✅ Cached" if is_cached else "⬇️ Will download on first use"
        print(f"   {config['display_name']}: {status}")
    
    print("\n" + "="*80)
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
