# ============================================================================
# WINDOWS TEMP FILE FIX - Configure cache directories BEFORE importing libraries
# ============================================================================
# This prevents WinError 32 (file in use) errors on Windows caused by:
# - Antivirus software scanning %TEMP% folder
# - Cloud sync services (OneDrive, Dropbox, Google Drive) monitoring temp files
# - Windows Search/Indexing service holding file handles
#
# By setting TORCH_HOME, HF_HOME, and TMPDIR BEFORE importing NeMo/PyTorch,
# all model downloads and extractions go to a project-local cache directory
# that Windows services typically don't monitor.
# ============================================================================

import os
import sys
from pathlib import Path

# Get script directory and create cache directory structure
_script_dir = Path(__file__).parent.absolute()
_cache_dir = _script_dir / "model_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)

# Set cache environment variables BEFORE importing torch/NeMo
# This affects all subsequent model downloads and extractions
os.environ["TORCH_HOME"] = str(_cache_dir / "torch")
os.environ["HF_HOME"] = str(_cache_dir / "huggingface")
os.environ["NEMO_CACHE_DIR"] = str(_cache_dir / "nemo")

# Set TMPDIR to use our cache instead of Windows %TEMP%
# This is the CRITICAL line that prevents file lock issues
os.environ["TMPDIR"] = str(_cache_dir / "tmp")
if sys.platform == "win32":
    # Windows uses different env vars for temp directory
    os.environ["TEMP"] = str(_cache_dir / "tmp")
    os.environ["TMP"] = str(_cache_dir / "tmp")

# Create temp directory
_temp_dir = _cache_dir / "tmp"
_temp_dir.mkdir(parents=True, exist_ok=True)

# Create other cache subdirectories
(_cache_dir / "torch").mkdir(parents=True, exist_ok=True)
(_cache_dir / "huggingface").mkdir(parents=True, exist_ok=True)
(_cache_dir / "nemo").mkdir(parents=True, exist_ok=True)

# Store cache_dir for error messages later
CACHE_DIR = _cache_dir

# ============================================================================
# NOW import libraries (after cache directories are configured)
# ============================================================================

import gradio as gr
import nemo.collections.asr as nemo_asr
import torch
import time
import gc

# Global model cache to avoid reloading
models_cache = {}

# Video file extensions supported (librosa + FFmpeg backend handles audio extraction)
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.m4v'}

# Audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# Separator string for output formatting
SEPARATOR = '=' * 60

# Model configurations
# All models use standard ASRModel.from_pretrained() API (no SALM required)
# Parakeet models: Can load from local .nemo file OR HuggingFace
# Canary models: Load from HuggingFace or local .nemo file if downloaded
#
# loading_method options:
#   - "local": ONLY load from local .nemo file (fails if missing)
#   - "huggingface": ONLY load from HuggingFace
#   - "local_or_huggingface": Try local first, fallback to HuggingFace
MODEL_CONFIGS = {
    # ========== PARAKEET MODELS ==========
    "parakeet-v3": {
        "local_path": "local_models/parakeet-0.6b-v3.nemo",  # Unique filename
        "hf_model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "max_batch_size": 32,
        "display_name": "Parakeet-TDT-0.6B v3",
        "loading_method": "local_or_huggingface",  # Try local, fallback to HF
        "architecture": "FastConformer-TDT",
        "parameters": "600M",
        "languages": 25,
        "wer": "~1.7%",
        "rtfx": "3,380×",
        "vram_gb": "3-4",
        "recommended_for": "Best all-around choice, 25 languages, auto-detection"
    },
    
    "parakeet-1.1b": {
        "local_path": "local_models/parakeet-1.1b.nemo",  # Unique filename
        "hf_model_id": "nvidia/parakeet-tdt-1.1b",
        "max_batch_size": 24,
        "display_name": "Parakeet-TDT-1.1B",
        "loading_method": "local_or_huggingface",  # Try local, fallback to HF
        "architecture": "FastConformer-TDT",
        "parameters": "1.1B",
        "languages": 1,
        "wer": "1.5%",
        "rtfx": "1,336×",
        "vram_gb": "5-6",
        "recommended_for": "Best English transcription accuracy available"
    },
    
    # ========== CANARY MODELS ==========
    "canary-1b": {
        "local_path": "local_models/canary-1b.nemo",  # Unique filename
        "hf_model_id": "nvidia/canary-1b",
        "max_batch_size": 16,
        "display_name": "Canary-1B",
        "loading_method": "local_or_huggingface",  # Try local, fallback to HF
        "architecture": "FastConformer-Transformer",
        "parameters": "1B",
        "languages": 25,
        "wer": "~1.9% (English)",
        "rtfx": "~200×",
        "vram_gb": "4-5",
        "recommended_for": "Multilingual ASR + speech-to-text translation",
        "additional_features": ["Speech Translation (AST)", "NeMo Forced Aligner timestamps"]
    },
    
    "canary-1b-v2": {
        "local_path": "local_models/canary-1b-v2.nemo",  # Unique filename
        "hf_model_id": "nvidia/canary-1b-v2",
        "max_batch_size": 16,
        "display_name": "Canary-1B v2",
        "loading_method": "local_or_huggingface",  # Try local, fallback to HF
        "architecture": "FastConformer-Transformer",
        "parameters": "1B",
        "languages": 25,
        "wer": "1.88% (English)",
        "rtfx": "~200×",
        "vram_gb": "4-5",
        "recommended_for": "Multilingual ASR + speech-to-text translation (improved)",
        "additional_features": ["Speech Translation (AST)", "NeMo Forced Aligner timestamps"]
    }
}

def get_model_key_from_choice(choice_text):
    """Extract model key from radio button choice text
    
    Args:
        choice_text: The display text from the radio button selection
    
    Returns:
        Model key string for accessing MODEL_CONFIGS
    """
    choice_map = {
        "Parakeet-TDT-0.6B v3": "parakeet-v3",
        "Parakeet-TDT-1.1B": "parakeet-1.1b",
        "Canary-1B v2": "canary-1b-v2",
        "Canary-1B": "canary-1b",
        # Legacy support
        "Parakeet-TDT-0.6B v2": "parakeet-v3",  # Map old to new multilingual version
        # Legacy SALM model mapping: Canary-Qwen-2.5B was a 2.5B param SALM-based model
        # We map it to Canary-1B-v2 (1B param, standard encoder-decoder architecture)
        # Trade-off: Slightly different model size, but gains multilingual support + simpler architecture
        "Canary-Qwen-2.5B": "canary-1b-v2"
    }
    
    # Try exact matches first
    for choice, key in choice_map.items():
        if choice in choice_text:
            return key
    
    # Default to parakeet-v3 if no match
    return "parakeet-v3"

def get_script_dir():
    """Get the directory where the script is located"""
    return Path(__file__).parent.absolute()

def validate_local_models():
    """
    Validate local model setup and provide informational messages.
    
    With "local_or_huggingface" loading method, local .nemo files are OPTIONAL.
    Models will automatically download from HuggingFace if local files are missing.
    
    Returns:
        bool: Always returns True (no longer blocks startup)
    """
    script_dir = get_script_dir()
    local_models_dir = script_dir / "local_models"
    
    print("\n" + "="*80)
    print("📦 Model Availability Check")
    print("="*80)
    
    # Check if local_models directory exists
    if not local_models_dir.exists():
        print(f"\n📁 Local models directory not found: {local_models_dir}")
        print("   This is OK - models will download from HuggingFace on first use.")
        print("   To download models locally, run: python setup_local_models.py")
    else:
        print(f"\n📁 Local models directory: {local_models_dir}")
    
    # Check each model's local availability
    local_found = []
    hf_fallback = []
    
    for model_key, config in MODEL_CONFIGS.items():
        local_path = config.get("local_path")
        if local_path:
            full_path = script_dir / local_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                local_found.append(f"   ✅ {config['display_name']}: {full_path.name} ({size_mb:.1f} MB)")
            else:
                hf_fallback.append(f"   📥 {config['display_name']}: Will download from HuggingFace ({config['hf_model_id']})")
        else:
            hf_fallback.append(f"   📥 {config['display_name']}: HuggingFace only ({config['hf_model_id']})")
    
    if local_found:
        print("\n📦 Local .nemo files found (instant loading):")
        for msg in local_found:
            print(msg)
    
    if hf_fallback:
        print("\n📡 Models to download on first use:")
        for msg in hf_fallback:
            print(msg)
    
    print("\n💡 Tip: Run 'python setup_local_models.py' to download all models locally")
    print("="*80 + "\n")
    
    # Always return True - no longer blocks startup
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


def _load_from_huggingface_with_retry(hf_model_id, config, max_retries=3):
    """Load model from HuggingFace with retry logic for Windows file lock issues.
    
    Args:
        hf_model_id: HuggingFace model ID (e.g., "nvidia/parakeet-tdt-0.6b-v3")
        config: Model configuration dict
        max_retries: Maximum number of retry attempts for file lock errors
    
    Returns:
        Loaded ASR model
        
    Raises:
        PermissionError, ConnectionError, OSError, RuntimeError on failure
    """
    base_delay = 0.2  # 200ms base delay for retries
    
    for attempt in range(max_retries):
        try:
            return nemo_asr.models.ASRModel.from_pretrained(hf_model_id)
            
        except PermissionError as e:
            error_str = str(e)
            is_file_lock = "WinError 32" in error_str or "being used by another process" in error_str
            
            if is_file_lock and attempt < max_retries - 1:
                # File lock detected - retry with backoff
                delay = base_delay * (attempt + 1)  # 0.2s, 0.4s, 0.6s
                print(f"⏳ File lock detected (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
                
                # Force garbage collection and cache cleanup before retry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                time.sleep(delay)
                continue
            
            elif is_file_lock:
                # Final retry failed - provide detailed diagnostics
                raise PermissionError(
                    f"\n{'='*80}\n"
                    f"❌ FILE LOCK ERROR (PERSISTED AFTER {max_retries} RETRIES)\n"
                    f"{'='*80}\n\n"
                    f"Model: {config['display_name']}\n"
                    f"HuggingFace ID: {hf_model_id}\n\n"
                    f"The model extraction cannot complete due to persistent file locks.\n"
                    f"This typically happens when Windows services hold file handles open.\n\n"
                    f"🔒 Likely Causes:\n"
                    f"  1. Windows Defender or antivirus scanning temp files\n"
                    f"  2. OneDrive, Google Drive, or Dropbox syncing the cache folder\n"
                    f"  3. Windows Search or Windows Indexing Service\n"
                    f"  4. Another Python process accessing the same models\n\n"
                    f"💡 Solutions (in order of likelihood to work):\n"
                    f"  1. Pause OneDrive/Dropbox/Google Drive temporarily\n"
                    f"  2. Add cache directory to antivirus exclusions:\n"
                    f"     {CACHE_DIR}\n"
                    f"  3. Restart your computer\n"
                    f"  4. Run as Administrator\n"
                    f"  5. Disable Windows Search indexing\n\n"
                    f"⚙️ Cache Location:\n"
                    f"  {CACHE_DIR}\n\n"
                    f"If this persists, try manually deleting the cache directory.\n"
                    f"{'='*80}"
                )
            else:
                # Different PermissionError (not file lock)
                raise PermissionError(
                    f"\n{'='*80}\n"
                    f"❌ PERMISSION ERROR!\n"
                    f"{'='*80}\n\n"
                    f"Model: {config['display_name']}\n"
                    f"Error: {error_str}\n\n"
                    f"The process does not have permission to access the cache directory.\n"
                    f"Try running as Administrator or checking directory permissions.\n"
                    f"Cache location: {CACHE_DIR}\n"
                    f"{'='*80}"
                )
    
    # Should never reach here, but just in case
    raise RuntimeError(f"Failed to load model after {max_retries} attempts")


def load_model(model_name, show_progress=False):
    """Load model using the appropriate method based on model type.
    
    All models now use standard ASRModel API (no SALM required).
    
    Loading methods:
    - "local": ONLY load from local .nemo file (fails if missing)
    - "huggingface": ONLY load from HuggingFace
    - "local_or_huggingface": Try local first, fallback to HuggingFace
    
    Args:
        model_name: Model key from MODEL_CONFIGS (e.g., "parakeet-v3", "canary-1b-v2")
        show_progress: Whether to show loading progress (for startup)
    
    Returns:
        The loaded ASR model
        
    Raises:
        FileNotFoundError: If the local .nemo file doesn't exist (for "local" method)
        ConnectionError: If HuggingFace download fails due to network issues
        OSError: If download fails due to disk space issues
        PermissionError: If file locks prevent model extraction
    """
    if model_name not in models_cache:
        config = MODEL_CONFIGS[model_name]
        script_dir = get_script_dir()
        loading_method = config.get("loading_method", "huggingface")
        
        start_time = time.time()
        
        # ============================================================
        # LOCAL_OR_HUGGINGFACE: Try local first, fallback to HuggingFace
        # ============================================================
        if loading_method == "local_or_huggingface":
            local_path = config.get("local_path")
            model_path = script_dir / local_path if local_path else None
            
            # Try loading from local .nemo file first
            if model_path and model_path.exists():
                print(f"📦 Loading {config['display_name']} from local file...")
                print(f"   Path: {model_path}")
                
                try:
                    models_cache[model_name] = nemo_asr.models.ASRModel.restore_from(
                        str(model_path)
                    )
                    load_time = time.time() - start_time
                    print(f"✓ {config['display_name']} loaded from local file in {load_time:.1f}s")
                    return models_cache[model_name]
                    
                except PermissionError as e:
                    error_str = str(e)
                    if "WinError 32" in error_str or "being used by another process" in error_str:
                        print(f"⚠️  Local file locked, falling back to HuggingFace...")
                        # Fall through to HuggingFace download
                    else:
                        raise
                        
                except Exception as e:
                    # Local file exists but is corrupted or invalid
                    print(f"⚠️  Local file corrupted or invalid: {e}")
                    print(f"   Falling back to HuggingFace download...")
                    # Fall through to HuggingFace download
            else:
                # Local file not found - inform user
                print(f"📦 Loading {config['display_name']} from HuggingFace...")
                if model_path:
                    print(f"   Local .nemo file not found: {model_path}")
                print(f"   To download locally, run: python setup_local_models.py")
            
            # Load from HuggingFace (either no local file, or local file failed)
            hf_model_id = config["hf_model_id"]
            print(f"   Model ID: {hf_model_id}")
            print("   (First load downloads model, subsequent loads use cache)")
            
            try:
                models_cache[model_name] = _load_from_huggingface_with_retry(
                    hf_model_id, config, max_retries=3
                )
            except ConnectionError as e:
                raise ConnectionError(
                    f"\n{'='*80}\n"
                    f"❌ NETWORK ERROR LOADING MODEL!\n"
                    f"{'='*80}\n\n"
                    f"Model: {config['display_name']}\n"
                    f"Failed to connect to HuggingFace to download the model.\n\n"
                    f"Solution: Please check your internet connection and try again.\n"
                    f"Original error: {str(e)}\n"
                    f"{'='*80}"
                )
            except OSError as e:
                error_str = str(e).lower()
                if "no space" in error_str or "disk" in error_str:
                    raise OSError(
                        f"\n{'='*80}\n"
                        f"❌ DISK SPACE ERROR!\n"
                        f"{'='*80}\n\n"
                        f"Model: {config['display_name']}\n"
                        f"Insufficient disk space to download the model.\n\n"
                        f"Solution: Please free up disk space and try again.\n"
                        f"Original error: {str(e)}\n"
                        f"{'='*80}"
                    )
                raise
        
        # ============================================================
        # LOCAL: Strictly load from local .nemo file only
        # ============================================================
        elif loading_method == "local":
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
            except PermissionError as e:
                error_str = str(e)
                if "WinError 32" in error_str or "being used by another process" in error_str:
                    problem = (
                        "Windows temp file cleanup failed (WinError 32). This is often "
                        "caused by antivirus software (especially Windows Defender), "
                        "cloud sync services (OneDrive, Dropbox, Google Drive), or "
                        "file indexing services monitoring the temp folder."
                    )
                    solution = (
                        "Troubleshooting steps:\n"
                        f"1) The cache is now at: {CACHE_DIR}\n"
                        "2) Add this folder to antivirus exclusions\n"
                        "3) Pause cloud sync or exclude this folder\n"
                        "4) Close other applications accessing temp files\n"
                        "5) Restart your computer and try again"
                    )
                    raise PermissionError(_format_model_error(
                        title="WINDOWS FILE LOCK ERROR!",
                        model_path=model_path,
                        display_name=config['display_name'],
                        problem_msg=problem,
                        solution_msg=solution,
                        original_error=e
                    ))
                else:
                    raise
            except FileNotFoundError as e:
                raise FileNotFoundError(_format_model_error(
                    title="FAILED TO LOAD MODEL!",
                    model_path=model_path,
                    display_name=config['display_name'],
                    problem_msg="The .nemo file may be corrupted or incomplete.",
                    solution_msg="Please recreate it using: python setup_local_models.py",
                    original_error=e
                ))
        
        # ============================================================
        # HUGGINGFACE: Load from HuggingFace only
        # ============================================================
        else:
            hf_model_id = config["hf_model_id"]
            
            print(f"📦 Loading {config['display_name']} from HuggingFace...")
            print(f"   Model ID: {hf_model_id}")
            print("   (First load downloads model, subsequent loads use cache)")
            
            try:
                models_cache[model_name] = _load_from_huggingface_with_retry(
                    hf_model_id, config, max_retries=3
                )
            except ConnectionError as e:
                raise ConnectionError(
                    f"\n{'='*80}\n"
                    f"❌ NETWORK ERROR LOADING MODEL!\n"
                    f"{'='*80}\n\n"
                    f"Model: {config['display_name']}\n"
                    f"Failed to connect to HuggingFace to download the model.\n\n"
                    f"Solution: Please check your internet connection and try again.\n"
                    f"Original error: {str(e)}\n"
                    f"{'='*80}"
                )
            except OSError as e:
                error_str = str(e).lower()
                if "no space" in error_str or "disk" in error_str:
                    raise OSError(
                        f"\n{'='*80}\n"
                        f"❌ DISK SPACE ERROR!\n"
                        f"{'='*80}\n\n"
                        f"Model: {config['display_name']}\n"
                        f"Insufficient disk space to download the model.\n\n"
                        f"Solution: Please free up disk space and try again.\n"
                        f"Original error: {str(e)}\n"
                        f"{'='*80}"
                    )
                raise OSError(
                    f"\n{'='*80}\n"
                    f"❌ FILE SYSTEM ERROR!\n"
                    f"{'='*80}\n\n"
                    f"Model: {config['display_name']}\n"
                    f"A file system error occurred while loading the model.\n\n"
                    f"Solution: Please check file permissions and try again.\n"
                    f"Cache location: {CACHE_DIR}\n"
                    f"Original error: {str(e)}\n"
                    f"{'='*80}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"❌ ERROR LOADING MODEL!\n"
                    f"{'='*80}\n\n"
                    f"Model: {config['display_name']}\n"
                    f"An unexpected error occurred: {type(e).__name__}\n\n"
                    f"Solution: Try clearing the cache and retrying.\n"
                    f"Cache location: {CACHE_DIR}\n"
                    f"Original error: {str(e)}\n"
                    f"{'='*80}"
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
        
        # Determine which model to use - extract key from choice text
        model_key = get_model_key_from_choice(model_choice)
        
        # Start timing
        start_time = time.time()
        
        # Load model with explicit error handling for Gradio display
        # This ensures any model loading errors appear in the UI status box
        try:
            model = load_model(model_key)
        except PermissionError as e:
            # File lock or permission error - display in Gradio status
            error_msg = str(e)
            print(f"GRADIO_ERROR (PermissionError): {error_msg}")
            return f"### ❌ Model Loading Error\n\n{error_msg}", "", None
        except ConnectionError as e:
            error_msg = str(e)
            print(f"GRADIO_ERROR (ConnectionError): {error_msg}")
            return f"### ❌ Network Error\n\n{error_msg}", "", None
        except FileNotFoundError as e:
            error_msg = str(e)
            print(f"GRADIO_ERROR (FileNotFoundError): {error_msg}")
            return f"### ❌ File Not Found\n\n{error_msg}", "", None
        except OSError as e:
            error_msg = str(e)
            print(f"GRADIO_ERROR (OSError): {error_msg}")
            return f"### ❌ File System Error\n\n{error_msg}", "", None
        except RuntimeError as e:
            error_msg = str(e)
            print(f"GRADIO_ERROR (RuntimeError): {error_msg}")
            return f"### ❌ Runtime Error\n\n{error_msg}", "", None
        except Exception as e:
            error_msg = (
                f"{'='*80}\n"
                f"❌ UNEXPECTED ERROR LOADING MODEL\n"
                f"{'='*80}\n\n"
                f"Type: {type(e).__name__}\n"
                f"Message: {str(e)}\n\n"
                f"Please check the console for more details.\n"
                f"{'='*80}"
            )
            print(f"GRADIO_ERROR (Unexpected): {error_msg}")
            return f"### ❌ Unexpected Error\n\n{error_msg}", "", None
        
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
**Models Available**: All Parakeet and Canary models supported
"""
    else:
        info = f"""
### ⚠️ No GPU Detected

CUDA is not available. Please check:
1. NVIDIA GPU drivers are installed
2. CUDA Toolkit is installed
3. PyTorch was installed with CUDA support

**NeMo Available**: Check console output
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
- ✅ All models: Try local .nemo first, then HuggingFace fallback
- ✅ Once downloaded, models work offline from cache

### Model Storage:
- **Local .nemo files**: `local_models/` directory (run `python setup_local_models.py`)
  - `parakeet-0.6b-v3.nemo` (~2.4GB)
  - `parakeet-1.1b.nemo` (~4.5GB)
  - `canary-1b.nemo` (~5.0GB)
  - `canary-1b-v2.nemo` (~5.0GB)
- **HuggingFace Cache**: `model_cache/` directory (auto-downloads if .nemo missing)

### Cache Location:
- **Custom cache**: `{CACHE_DIR}` (prevents Windows temp file issues)
- This directory is used for all model downloads and extraction

### Performance Optimizations:
- {gpu_info}
- **Speed**: 200-3,380× faster than real-time (model dependent)
- **Example**: 1 hour audio → 10-60 seconds processing
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
    ### 100% Offline - Powered by Parakeet & Canary ASR Models
    
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
                    "📊 Parakeet-TDT-0.6B v3 (Multilingual, Default) - 25 languages, auto-detect",
                    "🎯 Parakeet-TDT-1.1B (Maximum Accuracy) - 1.5% WER, English only",
                    "🌍 Canary-1B v2 (Multilingual + Translation) - 25 languages with AST",
                    "🌐 Canary-1B (Multilingual) - 25 languages, standard ASR"
                ],
                value="📊 Parakeet-TDT-0.6B v3 (Multilingual, Default) - 25 languages, auto-detect",
                label="Model Selection",
                info="Choose based on your priority: accuracy, speed, languages, or features"
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
            
            **Parakeet-TDT-0.6B v3** (Recommended):
            - Languages: 25 European languages with auto-detection
            - Speed: 3,380× real-time (ultra-fast)
            - Accuracy: ~1.7% WER
            - VRAM: 3-4 GB
            - Loads from: `local_models/parakeet-0.6b-v3.nemo` OR HuggingFace
            
            **Parakeet-TDT-1.1B** (Best Accuracy):
            - Languages: English only
            - Speed: 1,336× real-time
            - Accuracy: **1.5% WER** (best available)
            - VRAM: 5-6 GB
            - Loads from: `local_models/parakeet-1.1b.nemo` OR HuggingFace
            
            **Canary-1B v2** (Multilingual + Translation):
            - Languages: 25 European languages
            - Speed: ~200× real-time
            - Accuracy: 1.88% WER (English)
            - VRAM: 4-5 GB
            - Features: **Speech Translation** (AST)
            - Loads from: `local_models/canary-1b-v2.nemo` OR HuggingFace
            
            **Canary-1B** (Standard):
            - Languages: 25 European languages
            - Speed: ~200× real-time
            - Accuracy: ~1.9% WER (English)
            - VRAM: 4-5 GB
            - Loads from: `local_models/canary-1b.nemo` OR HuggingFace
            
            💡 **Tip**: Run `python setup_local_models.py` to download models locally for faster loading.
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
        2. **Select model** (choose based on your needs: speed, accuracy, or language support)
        3. **Click "Start Transcription"**
        4. **Copy or download** your transcription
        
        ### Features:
        - 🎵 **Audio Support**: WAV, MP3, FLAC, M4A, OGG, AAC, WMA
        - 🎬 **Video Support**: MP4, AVI, MKV, MOV, WEBM, FLV, M4V
        - 📦 **Batch Processing**: Upload multiple files at once
        - ⚡ **GPU Optimized**: Uses FP16 mixed precision for faster processing
        - 🔒 **Privacy First**: All processing happens locally on your machine
        - 🌍 **Multilingual**: Support for 25 European languages (model dependent)
        
        ### Model Loading:
        - **All Models**: Try local `.nemo` file first, then HuggingFace fallback
        - Local files load instantly (no internet required)
        - HuggingFace downloads are cached for future offline use
        - All models stay in memory after first load (instant subsequent use)
        
        ### Tips:
        - First transcription loads model into memory (~2-15 seconds depending on model)
        - Subsequent transcriptions reuse cached model (instant)
        - Processing time: 10-60 seconds per hour of audio (model dependent)
        - Video files: Audio is automatically extracted via FFmpeg
        - Choose Parakeet-1.1B for best English accuracy
        - Choose Parakeet-v3 or Canary for multilingual support
        - Canary models support speech translation features
        
        ### Setup (Optional but Recommended):
        - Run `python setup_local_models.py` to download models locally
        - You can download all 4 models or select specific ones
        - Local `.nemo` files load faster than HuggingFace downloads
        - See `docs/manual/user-model-setup-guide.md` for instructions
        """)
    
    with gr.Accordion("🔒 Privacy & Performance", open=False):
        gr.Markdown(get_privacy_performance_info())
    
    # Connect button - queue=True ensures proper callback execution
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_selector, save_checkbox, timestamp_checkbox],
        outputs=[status_output, transcription_output, file_output],
        queue=True
    )

# Launch
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting NVIDIA NeMo Transcription Interface")
    print("="*80)
    
    # Show cache directory info
    print(f"\n📁 Cache Directory: {CACHE_DIR}")
    print("   (Used for model downloads and extraction - prevents Windows temp file issues)")
    
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"✅ CUDA: {torch.version.cuda}")
        
        # Enable GPU optimizations (TF32, cuDNN)
        setup_gpu_optimizations()
    else:
        print("\n⚠️  WARNING: No CUDA GPU detected!")
    
    print("="*80)
    
    # Validate models - uses new validation that shows local vs HuggingFace status
    validate_local_models()
    
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

