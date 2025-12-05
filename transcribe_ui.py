import gradio as gr
import nemo.collections.asr as nemo_asr
import torch
import os
import sys
import time
from pathlib import Path

# Try to import SALM module for Canary model
# SALM (Speech-Aware Language Model) is required for Canary-Qwen-2.5B
# This requires NeMo 2.6+ with speechlm2 collection installed
try:
    from nemo.collections.speechlm2.models import SALM
    SALM_AVAILABLE = True
    SALM_IMPORT_ERROR = None
except ImportError as e:
    SALM = None
    SALM_AVAILABLE = False
    SALM_IMPORT_ERROR = str(e)

# Global model cache to avoid reloading
models_cache = {}

# Video file extensions supported (librosa + FFmpeg backend handles audio extraction)
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.m4v'}

# Audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# Separator string for output formatting
SEPARATOR = '=' * 60

# Canary model revision - pinned to prevent re-downloads when HuggingFace repo updates
# Update this hash if you want to use a newer version of Canary
CANARY_PINNED_REVISION = "2399591399e4e6438fa7804f2f1f1660"

# Model configurations
# Note: Canary uses SALM architecture that cannot be saved as .nemo file
# Parakeet: Loads from local .nemo file (offline)
# Canary: Loads from HuggingFace cache with pinned revision (first use downloads, then cached)
MODEL_CONFIGS = {
    "parakeet": {
        "local_path": "local_models/parakeet.nemo",
        "max_batch_size": 32,  # Increased from 16
        "display_name": "Parakeet-TDT-0.6B v2",
        "loading_method": "local"  # Load from local .nemo file
    },
    "canary": {
        "hf_model_id": "nvidia/canary-qwen-2.5b",
        "revision": CANARY_PINNED_REVISION,
        "max_batch_size": 16,  # Increased from default
        "display_name": "Canary-Qwen-2.5B",
        "loading_method": "huggingface"  # Load from HuggingFace using SALM
    }
}

def get_script_dir():
    """Get the directory where the script is located"""
    return Path(__file__).parent.absolute()

def validate_local_models():
    """
    Validate that Parakeet local .nemo model file exists before launching UI.
    
    Note: Canary uses SALM architecture and loads from HuggingFace cache,
    so it doesn't require a local .nemo file.
    
    Returns:
        bool: True if Parakeet model exists, False otherwise
    """
    script_dir = get_script_dir()
    local_models_dir = script_dir / "local_models"
    
    # Check if local_models directory exists
    if not local_models_dir.exists():
        print("\n" + "="*80)
        print("❌ LOCAL MODELS NOT FOUND!")
        print("="*80)
        print(f"\nThe 'local_models/' directory does not exist.")
        print(f"Expected location: {local_models_dir}")
        print("\nRequired file:")
        print(f"  • {local_models_dir / 'parakeet.nemo'}")
        print("\nNote: Canary loads from HuggingFace cache (no local file needed)")
        print("\nThe Parakeet file must be created once using the setup script.")
        print("Please run: python setup_local_models.py")
        print("\nSee docs/manual/user-model-setup-guide.md for instructions.")
        print("="*80 + "\n")
        return False
    
    # Only check for Parakeet .nemo file (Canary uses HuggingFace)
    parakeet_config = MODEL_CONFIGS["parakeet"]
    parakeet_path = script_dir / parakeet_config["local_path"]
    
    if not parakeet_path.exists():
        print("\n" + "="*80)
        print("❌ PARAKEET MODEL NOT FOUND!")
        print("="*80)
        print(f"\nMissing model file:")
        print(f"  • {parakeet_path} ({parakeet_config['display_name']})")
        print("\nNote: Canary loads from HuggingFace cache (no local file needed)")
        print("\nThe Parakeet file must be created once using the setup script.")
        print("Please run: python setup_local_models.py")
        print("\nSee docs/manual/user-model-setup-guide.md for instructions.")
        print("="*80 + "\n")
        return False
    
    return True

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

def _format_model_error(title, model_path, display_name, problem_msg, solution_msg, original_error=None):
    """Format a model loading error message with consistent styling.
    
    Args:
        title: Error title (e.g., "MODEL FILE NOT FOUND!")
        model_path: Path to the model file
        display_name: Human-readable model name
        problem_msg: Description of what went wrong
        solution_msg: How to fix the issue
        original_error: Original exception message if wrapping an error
    
    Returns:
        Formatted error message string
    """
    error_lines = [
        f"\n{'='*80}",
        f"❌ {title}",
        f"{'='*80}",
        "",
        f"File path: {model_path}",
        f"Model: {display_name}",
    ]
    
    if original_error:
        error_lines.append(f"Original error: {str(original_error)}")
    
    error_lines.extend([
        "",
        problem_msg,
        solution_msg,
        "",
        "See docs/manual/user-model-setup-guide.md for instructions.",
        f"{'='*80}",
    ])
    
    return "\n".join(error_lines)


def load_model(model_name, show_progress=False):
    """Load model using the appropriate method based on model type.
    
    - Parakeet: Loads from local .nemo file using restore_from()
    - Canary: Loads from HuggingFace using SALM.from_pretrained() with pinned revision
    
    Canary uses SALM (Speech-Aware Language Model) architecture that cannot be 
    packaged as .nemo files. It loads from HuggingFace cache with a pinned revision
    to prevent re-downloads.
    
    Args:
        model_name: "parakeet" or "canary"
        show_progress: Whether to show loading progress (for startup)
    
    Returns:
        The loaded ASR model
        
    Raises:
        FileNotFoundError: If the Parakeet .nemo file doesn't exist
        ConnectionError: If Canary download fails due to network issues
        OSError: If Canary download fails due to disk space issues
    """
    if model_name not in models_cache:
        config = MODEL_CONFIGS[model_name]
        script_dir = get_script_dir()
        
        start_time = time.time()
        
        if config.get("loading_method") == "local":
            # Parakeet: Load from local .nemo file
            model_path = script_dir / config["local_path"]
            
            # Check if .nemo file exists
            if not model_path.exists():
                raise FileNotFoundError(_format_model_error(
                    title="MODEL FILE NOT FOUND!",
                    model_path=model_path,
                    display_name=config['display_name'],
                    problem_msg="The .nemo file must be created once using the setup script.",
                    solution_msg="Please run: python setup_local_models.py"
                ))
            
            print(f"📦 Loading {config['display_name']} from local file...")
            
            try:
                models_cache[model_name] = nemo_asr.models.ASRModel.restore_from(
                    str(model_path)
                )
            except FileNotFoundError as e:
                # Re-raise with more helpful message
                raise FileNotFoundError(_format_model_error(
                    title="FAILED TO LOAD MODEL!",
                    model_path=model_path,
                    display_name=config['display_name'],
                    problem_msg="The .nemo file may be corrupted or incomplete.",
                    solution_msg="Please recreate it using: python setup_local_models.py",
                    original_error=e
                ))
        else:
            # Canary: Load from HuggingFace using SALM with pinned revision
            # SALM (Speech-Aware Language Model) architecture cannot be saved as .nemo
            
            # Check if SALM module is available
            if not SALM_AVAILABLE:
                raise ImportError(_format_canary_error(
                    title="SALM MODULE NOT AVAILABLE!",
                    display_name=config['display_name'],
                    problem_msg="The nemo.collections.speechlm2 module is not installed.",
                    solution_msg="Please upgrade NeMo to version 2.6.0 or later: pip install nemo_toolkit[all]>=2.6.0",
                    original_error=None
                ))
            
            hf_model_id = config["hf_model_id"]
            revision = config.get("revision")
            
            print(f"📦 Loading {config['display_name']} from HuggingFace cache...")
            if revision:
                print(f"   Using pinned revision: {revision[:12]}...")
            print("   (First load downloads ~5GB, subsequent loads use cache)")
            
            try:
                # Use SALM.from_pretrained for Canary-Qwen-2.5B
                # The revision parameter pins to a specific commit to prevent re-downloads
                models_cache[model_name] = SALM.from_pretrained(
                    hf_model_id,
                    revision=revision
                )
            except ConnectionError as e:
                # Network error during download
                raise ConnectionError(_format_canary_error(
                    title="NETWORK ERROR LOADING CANARY!",
                    display_name=config['display_name'],
                    problem_msg="Failed to connect to HuggingFace to download the model.",
                    solution_msg="Please check your internet connection and try again.",
                    original_error=e
                ))
            except OSError as e:
                # Disk space or file system error
                error_str = str(e).lower()
                if "no space" in error_str or "disk" in error_str:
                    raise OSError(_format_canary_error(
                        title="DISK SPACE ERROR LOADING CANARY!",
                        display_name=config['display_name'],
                        problem_msg="Insufficient disk space to download the model (~5GB required).",
                        solution_msg="Please free up disk space and try again.",
                        original_error=e
                    ))
                raise OSError(_format_canary_error(
                    title="FILE SYSTEM ERROR LOADING CANARY!",
                    display_name=config['display_name'],
                    problem_msg="A file system error occurred while loading the model.",
                    solution_msg="Please check file permissions and try again.",
                    original_error=e
                ))
            except Exception as e:
                # Generic error handler for other issues
                raise RuntimeError(_format_canary_error(
                    title="ERROR LOADING CANARY!",
                    display_name=config['display_name'],
                    problem_msg=f"An unexpected error occurred: {type(e).__name__}",
                    solution_msg="Try clearing the cache and retrying. If the problem persists, check the console for details.",
                    original_error=e
                ))
        
        load_time = time.time() - start_time
        print(f"✓ {config['display_name']} loaded in {load_time:.1f}s")
    return models_cache[model_name]


def _format_canary_error(title, display_name, problem_msg, solution_msg, original_error=None):
    """Format a Canary model loading error message with consistent styling.
    
    Args:
        title: Error title (e.g., "NETWORK ERROR LOADING CANARY!")
        display_name: Human-readable model name
        problem_msg: Description of what went wrong
        solution_msg: How to fix the issue
        original_error: Original exception message if wrapping an error
    
    Returns:
        Formatted error message string
    """
    error_lines = [
        f"\n{'='*80}",
        f"❌ {title}",
        f"{'='*80}",
        "",
        f"Model: {display_name}",
        "",
        "Note: Canary uses SALM architecture and loads from HuggingFace cache.",
        "      First load requires internet connection (~5GB download).",
        "      Subsequent loads work offline (cached).",
    ]
    
    if original_error:
        error_lines.append(f"\nOriginal error: {str(original_error)}")
    
    error_lines.extend([
        "",
        f"Problem: {problem_msg}",
        f"Solution: {solution_msg}",
        "",
        "Troubleshooting steps:",
        "  1. Check your internet connection",
        "  2. Ensure you have at least 10GB free disk space",
        "  3. Try clearing cache: Remove ~/.cache/torch/NeMo folder",
        "  4. If problem persists, check console output for details",
        "",
        "See docs/manual/canary-hybrid-loading-fix.md for more information.",
        f"{'='*80}",
    ])
    
    return "\n".join(error_lines)


def validate_salm_availability():
    """
    Validate SALM module availability at startup and print warning if not available.
    
    This gives users immediate feedback about Canary availability without preventing
    the script from starting (Parakeet should still work).
    
    Returns:
        bool: True if SALM is available, False otherwise
    """
    if not SALM_AVAILABLE:
        print("\n" + "="*80)
        print("⚠️  WARNING: SALM MODULE NOT AVAILABLE")
        print("="*80)
        print("\nCanary model will NOT work in this session.")
        print("Parakeet model will work normally.")
        print("\nTo enable Canary-Qwen-2.5B support:")
        print("  1. Upgrade NeMo to version 2.6.0 or later:")
        print("     pip install --upgrade \"nemo_toolkit[all]\"")
        print("  2. Restart this application")
        if SALM_IMPORT_ERROR:
            print(f"\nImport error details: {SALM_IMPORT_ERROR}")
        print("\nSee docs/manual/user-model-setup-guide.md for more information.")
        print("="*80 + "\n")
        return False
    return True


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
    salm_status = "✅ Yes (Canary available)" if SALM_AVAILABLE else "❌ No (install nemo_toolkit[all])"
    
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
**SALM Available**: {salm_status}

**Status**: ✅ Ready for transcription
"""
    else:
        info = f"""
### ⚠️ No GPU Detected

CUDA is not available. Please check:
1. NVIDIA GPU drivers are installed
2. CUDA Toolkit is installed
3. PyTorch was installed with CUDA support

**SALM Available**: {salm_status}
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
- ✅ All transcription processing happens locally on your machine
- ✅ Your audio never leaves your computer
- ✅ Parakeet: 100% offline (loads from local .nemo file)
- ✅ Canary: Offline after first load (cached from HuggingFace)

### Model Storage:
- **Parakeet**: `local_models/parakeet.nemo` (~2.4GB)
- **Canary**: HuggingFace cache (~5.3GB) - uses SALM architecture
- Note: Canary cannot be saved as .nemo due to SALM model architecture

### Performance Optimizations:
- {gpu_info}
- **Speed**: 40-100× faster than real-time
- **Example**: 1 hour audio → 30-60 seconds processing
- **Mixed Precision**: FP16 inference for 1.5-2.5× speedup
- **TF32 Tensor Cores**: Enabled for matrix operations
- **Dynamic Batch Sizing**: Optimized based on audio duration
- **Memory Caching**: Models stay in RAM after first load
"""

# Create the Gradio interface
with gr.Blocks(title="🎙️ Local ASR Transcription") as app:
    
    # Header
    gr.Markdown("""
    # 🎙️ NVIDIA NeMo Local Audio Transcription
    ### 100% Offline - Powered by Local Parakeet-TDT & Canary-Qwen Models
    
    Transform your audio and video files into accurate text transcriptions using state-of-the-art AI models stored locally on your system. No internet required.
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
            - Loads from: `local_models/parakeet.nemo` (instant, offline)
            - Requirements: Base NeMo installation
            
            **Canary-Qwen-2.5B:**
            - Speed: ~50-100× real-time
            - Accuracy: 94.37% (5.63% WER)  
            - Best for: Critical accuracy, technical content
            - Loads from: HuggingFace cache (first use downloads ~5GB, then cached)
            - Note: Uses SALM architecture (cannot be saved as .nemo file)
            - **Requirements**: NeMo 2.6+ with speechlm2 collection
            - Install: `pip install --upgrade "nemo_toolkit[all]"`
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
        - 🔒 **Privacy First**: All processing happens locally on your machine
        
        ### Model Loading:
        - **Parakeet**: Loads from local `.nemo` file (instant, works offline)
        - **Canary**: Loads from HuggingFace cache (first use downloads ~5GB, then works offline)
        - Both models stay in memory after first load (instant subsequent use)
        
        ### Requirements:
        - **Parakeet**: Works with base NeMo installation
        - **Canary**: Requires NeMo 2.6+ with speechlm2 collection
          - Install: `pip install --upgrade "nemo_toolkit[all]"`
          - First use downloads ~5GB from HuggingFace
          - Subsequent uses load from cache (offline)
        
        ### Tips:
        - First transcription loads model into memory (~2-3 seconds for Parakeet, ~10-15s for Canary first time)
        - Subsequent transcriptions reuse cached model (instant)
        - Processing time: ~30-60 seconds per hour of audio
        - Video files: Audio is automatically extracted
        
        ### Setup:
        - Run `python setup_local_models.py` to set up Parakeet
        - Canary downloads automatically on first use (no setup needed)
        - See `docs/manual/user-model-setup-guide.md` for instructions
        
        ### Troubleshooting SALM Errors (Canary):
        If you see "SALM MODULE NOT AVAILABLE" when using Canary:
        1. Upgrade NeMo: `pip install --upgrade "nemo_toolkit[all]"`
        2. Restart this application
        3. Check console for detailed error messages
        4. See `docs/manual/canary-hybrid-loading-fix.md` for more help
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
    
    # Check SALM availability for Canary model
    validate_salm_availability()
    
    # Validate models - Parakeet from local, Canary from HuggingFace
    print("\n📦 Checking model availability...")
    script_dir = get_script_dir()
    
    # Check Parakeet (local .nemo file)
    parakeet_config = MODEL_CONFIGS["parakeet"]
    parakeet_path = script_dir / parakeet_config["local_path"]
    if parakeet_path.exists():
        size_mb = parakeet_path.stat().st_size / (1024 * 1024)
        print(f"   {parakeet_config['display_name']}: ✅ Found (local .nemo)")
        print(f"      Path: {parakeet_path}")
        print(f"      Size: {size_mb:.1f} MB")
    else:
        print(f"   {parakeet_config['display_name']}: ❌ Missing")
    
    # Show Canary info (loads from HuggingFace)
    canary_config = MODEL_CONFIGS["canary"]
    if SALM_AVAILABLE:
        print(f"   {canary_config['display_name']}: 📡 Loads from HuggingFace cache")
        print(f"      Model ID: {canary_config['hf_model_id']}")
        print(f"      Revision: {canary_config['revision'][:12]}...")
        print(f"      Note: Downloads ~5GB on first use, then cached offline")
    else:
        print(f"   {canary_config['display_name']}: ⚠️ Unavailable (SALM not installed)")
        print(f"      Requires: pip install --upgrade \"nemo_toolkit[all]\"")
    
    if not validate_local_models():
        sys.exit(1)
    
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
