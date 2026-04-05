from __future__ import annotations

import gc
import importlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app_shared.env_bootstrap import CACHE_DIR, bootstrap_environment, get_script_dir

bootstrap_environment(verbose=True)

import gradio as gr
import torch

from app_shared import (
    LoadedModelHandle,
    LogCapture,
    OutputFilesConfig,
    ResultProcessingContext,
    SimpleHypothesis,
    TranscriptionStats,
    configure_output_timing,
    extract_timestamps,
    format_as_csv,
    format_as_srt,
    format_as_txt_with_timestamps,
    load_audio_to_numpy,
    normalize_file_list,
    process_audio_files,
    save_logs,
)
from app_shared.transcription_flow import (
    _generate_and_save_output_files,
    _make_error_response,
    _process_batch_transcription,
    _process_single_transcription,
)


def _import_dependency(module_name: str, package_name: Optional[str] = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        dependency_name = package_name or module_name
        raise ImportError(
            f"Missing dependency '{dependency_name}'. "
            "Use the configured project environment and install the repo requirements. "
            f"Original error: {exc}"
        ) from exc


def _require_nemo_asr() -> Any:
    return _import_dependency("nemo.collections.asr", "nemo-toolkit[asr]")


def _dependency_is_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


DEFAULT_CHUNK_DURATION_SEC = 120
CHUNK_OVERLAP_SEC = 2
DEFAULT_LONG_AUDIO_THRESHOLD_SEC = DEFAULT_CHUNK_DURATION_SEC + 30
DEFAULT_SILENCE_THRESHOLD_SEC = 0.5
DEFAULT_MAX_WORD_DURATION_SEC = 2.0
ITN_MODE_CHOICES = ["per_chunk", "final_pass", "both", "disabled"]
DEFAULT_ITN_MODE = "per_chunk"

chunk_duration_sec = DEFAULT_CHUNK_DURATION_SEC
long_audio_threshold_sec = DEFAULT_LONG_AUDIO_THRESHOLD_SEC
itn_mode = DEFAULT_ITN_MODE
auto_unload_after_transcription = False

models_cache: Dict[str, LoadedModelHandle] = {}
log_capture = LogCapture()


MODEL_DISPLAY_ORDER = ["parakeet-v3"]
DEFAULT_MODEL_KEY = "parakeet-v3"

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "parakeet-v3": {
        "backend": "nemo",
        "choice_label": "NVIDIA Parakeet 0.6B-v3 :: NeMo, 25 languages, timestamps",
        "display_name": "NVIDIA Parakeet 0.6B-v3",
        "hf_model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "local_path": "local_models/parakeet-0.6b-v3.nemo",
        "loading_method": "local_or_huggingface",
        "max_batch_size": 32,
        "architecture": "FastConformer-TDT",
        "parameters": "600M",
        "languages": 25,
        "wer": "~1.7%",
        "rtfx": "3,380x",
        "vram_gb": "3-4",
        "recommended_for": "Fast local baseline with word-level timestamps",
        "supports_timestamps": True,
        "supports_chunking": True,
        "supports_local_setup": True,
        "summary": "NeMo backend, offline after first download, multilingual with timestamps",
    }
}


ITN_NORMALIZER: Optional[Any] = None
ITN_AVAILABLE = False

try:
    from nemo_text_processing.inverse_text_normalization import InverseNormalizer  # type: ignore[reportUnusedImport]

    ITN_AVAILABLE = True
    print("ITN available: numbers can be normalized to digits")
except ImportError:
    print("ITN not installed: numbers will remain as words")


def _get_itn_normalizer(language: str = "en") -> Optional[Any]:
    global ITN_NORMALIZER

    if not ITN_AVAILABLE:
        return None

    if ITN_NORMALIZER is None:
        try:
            from nemo_text_processing.inverse_text_normalization import InverseNormalizer

            print(f"   Initializing ITN normalizer for '{language}'")
            ITN_NORMALIZER = InverseNormalizer(lang=language, cache_dir=str(CACHE_DIR / "itn"))
        except Exception as exc:
            print(f"   Failed to initialize ITN: {exc}")
            return None

    return ITN_NORMALIZER


def _split_text_into_word_chunks(text: str, max_words: int = 50) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    for index in range(0, len(words), max_words):
        chunk = " ".join(words[index:index + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text]


def _normalize_chunks_with_fallback(normalizer: Any, chunks: List[str]) -> List[str]:
    try:
        return normalizer.normalize_list(chunks, verbose=False)
    except Exception:
        normalized: List[str] = []
        for chunk in chunks:
            try:
                normalized.append(normalizer.normalize(chunk, verbose=False))
            except Exception:
                normalized.append(chunk)
        return normalized


def _try_itn_sentence_splitting(normalizer: Any, text: str) -> Tuple[bool, str, str]:
    try:
        sentences = normalizer.split_text_into_sentences(text)
        if sentences:
            normalized = _normalize_chunks_with_fallback(normalizer, sentences)
            return True, " ".join(normalized), f"sentence splitting: {len(sentences)} sentences"
    except Exception as exc:
        print(f"   ITN sentence splitting failed: {exc}")
    return False, text, ""


def _try_itn_regex_splitting(normalizer: Any, text: str) -> Tuple[bool, str, str]:
    import re

    try:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if len(sentences) > 1:
            normalized = _normalize_chunks_with_fallback(normalizer, sentences)
            return True, " ".join(normalized), f"regex splitting: {len(sentences)} sentences"
    except Exception as exc:
        print(f"   ITN regex splitting failed: {exc}")
    return False, text, ""


def _try_itn_chunk_splitting(normalizer: Any, text: str) -> Tuple[bool, str, str]:
    try:
        chunks = _split_text_into_word_chunks(text, max_words=50)
        if len(chunks) > 1:
            normalized = _normalize_chunks_with_fallback(normalizer, chunks)
            return True, " ".join(normalized), f"chunk splitting: {len(chunks)} chunks"
        result = normalizer.normalize(text, verbose=False)
        return True, result, "single text"
    except Exception as exc:
        print(f"   ITN normalization failed completely: {exc}")
    return False, text, ""


def _is_itn_applicable(normalizer: Any, text: str) -> bool:
    return normalizer is not None and bool(text) and bool(text.strip())


def apply_inverse_text_normalization(text: str, language: str = "en") -> str:
    normalizer = _get_itn_normalizer(language)
    if not _is_itn_applicable(normalizer, text):
        return text

    for strategy in (
        _try_itn_sentence_splitting,
        _try_itn_regex_splitting,
        _try_itn_chunk_splitting,
    ):
        success, result, message = strategy(normalizer, text)
        if success:
            print(f"   ITN applied ({message})")
            return result

    return text


def apply_itn_to_segment(text: str, language: str = "en") -> str:
    normalizer = _get_itn_normalizer(language)
    if not _is_itn_applicable(normalizer, text):
        return text

    word_count = len(text.split())
    if word_count <= 50:
        try:
            return normalizer.normalize(text.strip(), verbose=False)
        except Exception as exc:
            print(f"   ITN direct normalization failed: {exc}")

    chunks = _split_text_into_word_chunks(text, max_words=50)
    if len(chunks) <= 1 and word_count <= 50:
        return text

    try:
        normalized = normalizer.normalize_list(chunks, verbose=False)
        print(f"   ITN applied (chunked: {len(chunks)} chunks, {word_count} words)")
        return " ".join(normalized)
    except Exception as exc:
        print(f"   ITN batch failed ({exc}), trying individually")
        normalized = []
        for chunk in chunks:
            try:
                normalized.append(normalizer.normalize(chunk, verbose=False))
            except Exception:
                normalized.append(chunk)
        if normalized:
            print(f"   ITN applied (individual: {len(chunks)} chunks)")
        return " ".join(normalized)


def _artifact_exists(path: Path) -> bool:
    return path.exists() and (path.is_file() or any(path.iterdir()) if path.is_dir() else True)


def get_model_key_from_choice(choice_text: str) -> str:
    for model_key in MODEL_DISPLAY_ORDER:
        if MODEL_CONFIGS[model_key]["choice_label"] == choice_text:
            return model_key
    return DEFAULT_MODEL_KEY


def get_model_choice_labels() -> List[str]:
    return [MODEL_CONFIGS[key]["choice_label"] for key in MODEL_DISPLAY_ORDER]


def get_default_model_choice() -> str:
    return MODEL_CONFIGS[DEFAULT_MODEL_KEY]["choice_label"]


def _check_model_local_availability(script_dir: Path, config: Dict[str, Any]) -> Tuple[bool, str]:
    local_path = config.get("local_path")
    display_name = config["display_name"]
    hf_model_id = config["hf_model_id"]

    if not local_path:
        return False, f"   {display_name}: Hugging Face only ({hf_model_id})"

    full_path = script_dir / local_path
    if _artifact_exists(full_path):
        if full_path.is_dir():
            item_count = len(list(full_path.iterdir()))
            return True, f"   {display_name}: {full_path.name}/ ({item_count} items)"
        size_mb = full_path.stat().st_size / (1024 * 1024)
        return True, f"   {display_name}: {full_path.name} ({size_mb:.1f} MB)"

    return False, f"   {display_name}: will download from Hugging Face ({hf_model_id})"


def validate_local_models() -> None:
    script_dir = get_script_dir()
    local_models_dir = script_dir / "local_models"

    print("\n" + "=" * 80)
    print("NeMo model availability")
    print("=" * 80)

    if not local_models_dir.exists():
        print(f"\nLocal models directory not found: {local_models_dir}")
        print("Models will download from Hugging Face on first use.")
    else:
        print(f"\nLocal models directory: {local_models_dir}")

    for model_key in MODEL_DISPLAY_ORDER:
        _is_local, message = _check_model_local_availability(script_dir, MODEL_CONFIGS[model_key])
        print(message)

    print("=" * 80 + "\n")


def setup_gpu_optimizations() -> None:
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("GPU optimizations enabled (TF32, cuDNN benchmark)")


def _format_network_error(display_name: str, error: Exception) -> str:
    return (
        f"Model: {display_name}\n"
        "Failed to connect to Hugging Face to download the model.\n\n"
        "Check your internet connection and try again.\n"
        f"Original error: {error}"
    )


def _format_disk_space_error(display_name: str, error: Exception) -> str:
    return (
        f"Model: {display_name}\n"
        "Insufficient disk space to download the model.\n\n"
        "Free up disk space and try again.\n"
        f"Original error: {error}"
    )


def _format_filesystem_error(display_name: str, error: Exception) -> str:
    return (
        f"Model: {display_name}\n"
        "A file system error occurred while loading the model.\n\n"
        f"Cache location: {CACHE_DIR}\n"
        f"Original error: {error}"
    )


def _format_file_lock_error(display_name: str, model_source: str, max_retries: int) -> str:
    return (
        f"File lock error persisted after {max_retries} retries.\n\n"
        f"Model: {display_name}\n"
        f"Source: {model_source}\n\n"
        "Windows services may be holding model files open.\n"
        "Pause cloud sync, add the cache directory to antivirus exclusions, or retry after a reboot.\n\n"
        f"Cache location: {CACHE_DIR}"
    )


def _format_permission_error(display_name: str, error_str: str) -> str:
    return (
        f"Model: {display_name}\n"
        f"Error: {error_str}\n\n"
        "The process does not have permission to access the cache directory.\n"
        f"Cache location: {CACHE_DIR}"
    )


def _is_file_lock_error(error_str: str) -> bool:
    return "WinError 32" in error_str or "being used by another process" in error_str


def _handle_retry_delay(attempt: int, base_delay: float, max_retries: int) -> bool:
    if attempt >= max_retries - 1:
        return False

    delay = base_delay * (attempt + 1)
    print(f"   Retry {attempt + 1}/{max_retries}, waiting {delay:.1f}s")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(delay)
    return True


def _override_model_dataloader_config(model: Any) -> None:
    try:
        from omegaconf import OmegaConf

        OmegaConf.set_struct(model.cfg, False)
        modified: List[str] = []

        for dataset_name in ("train_ds", "validation_ds", "test_ds"):
            if hasattr(model.cfg, dataset_name):
                dataset_cfg = getattr(model.cfg, dataset_name)
                if hasattr(dataset_cfg, "num_workers"):
                    old_value = dataset_cfg.num_workers
                    dataset_cfg.num_workers = 0
                    modified.append(f"{dataset_name}: {old_value} -> 0")

        OmegaConf.set_struct(model.cfg, True)

        if modified:
            print(f"   Dataloader workers overridden: {', '.join(modified)}")
    except Exception as exc:
        print(f"   Could not override model config (non-fatal): {exc}")


def _load_from_huggingface_with_retry(hf_model_id: str, config: Dict[str, Any], max_retries: int = 3) -> Any:
    nemo_asr = _require_nemo_asr()
    base_delay = 0.5

    for attempt in range(max_retries):
        try:
            return nemo_asr.models.ASRModel.from_pretrained(hf_model_id)
        except PermissionError as exc:
            error_str = str(exc)
            if _is_file_lock_error(error_str):
                if _handle_retry_delay(attempt, base_delay, max_retries):
                    continue
                raise PermissionError(_format_file_lock_error(config["display_name"], hf_model_id, max_retries))
            raise PermissionError(_format_permission_error(config["display_name"], error_str))

    raise RuntimeError(f"Failed to load model after {max_retries} attempts")


def _load_with_retry(restore_path: Path, config: Dict[str, Any], max_retries: int = 3) -> Any:
    nemo_asr = _require_nemo_asr()
    base_delay = 0.5

    for attempt in range(max_retries):
        try:
            return nemo_asr.models.ASRModel.restore_from(restore_path=str(restore_path))
        except PermissionError as exc:
            error_str = str(exc)
            if _is_file_lock_error(error_str):
                if _handle_retry_delay(attempt, base_delay, max_retries):
                    continue
                raise PermissionError(_format_file_lock_error(config["display_name"], str(restore_path), max_retries))
            raise
        except Exception:
            if _handle_retry_delay(attempt, base_delay, max_retries):
                continue
            raise

    raise RuntimeError(f"Failed to restore local model after {max_retries} attempts")


def _build_loaded_handle(model_key: str, runtime: Any, config: Dict[str, Any], source: str) -> LoadedModelHandle:
    return LoadedModelHandle(
        model_key=model_key,
        backend="nemo",
        runtime=runtime,
        source=source,
        config=config,
        supports_timestamps=bool(config.get("supports_timestamps", False)),
        supports_chunking=bool(config.get("supports_chunking", False)),
    )


def _get_model_runtime(model: Any) -> Any:
    if isinstance(model, LoadedModelHandle):
        return model.runtime
    return model


def _move_runtime_to_device(model: Any, device: str) -> Any:
    runtime = _get_model_runtime(model)

    if device == "cpu" and hasattr(runtime, "cpu"):
        runtime.cpu()
        return model

    if hasattr(runtime, "to"):
        runtime = runtime.to(device)
        if isinstance(model, LoadedModelHandle):
            model.runtime = runtime
            return model
        return runtime

    return model


def _unload_cached_models(model_name: str) -> None:
    model_keys = [key for key in models_cache.keys() if key != model_name]
    if not model_keys or not torch.cuda.is_available():
        return

    for old_model_key in model_keys:
        try:
            old_model = models_cache[old_model_key]
            print(f"Unloading {old_model_key} to free VRAM for {model_name}")
            _move_runtime_to_device(old_model, "cpu")
            del models_cache[old_model_key]
            del old_model
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as exc:
            print(f"   Failed to unload {old_model_key}: {exc}")


def _load_model_huggingface(config: Dict[str, Any]) -> Tuple[Any, str]:
    hf_model_id = config["hf_model_id"]
    print(f"Loading {config['display_name']} from Hugging Face")
    print(f"   Model ID: {hf_model_id}")
    print("   First load downloads the model, later loads use the cache")

    try:
        model = _load_from_huggingface_with_retry(hf_model_id, config, max_retries=3)
        _override_model_dataloader_config(model)
        return model, hf_model_id
    except ConnectionError as exc:
        raise ConnectionError(_format_network_error(config["display_name"], exc))
    except OSError as exc:
        error_str = str(exc).lower()
        if "no space" in error_str or "disk" in error_str:
            raise OSError(_format_disk_space_error(config["display_name"], exc))
        raise OSError(_format_filesystem_error(config["display_name"], exc))


def _load_model_local_or_huggingface(model_name: str, config: Dict[str, Any], script_dir: Path) -> Tuple[Any, str]:
    local_path = config.get("local_path")
    model_path = script_dir / local_path if local_path else None

    if model_path is not None and model_path.exists():
        print(f"Loading {config['display_name']} from local file")
        print(f"   Path: {model_path}")
        try:
            model = _load_with_retry(model_path, config, max_retries=3)
            _override_model_dataloader_config(model)
            return model, str(model_path)
        except PermissionError as exc:
            if _is_file_lock_error(str(exc)):
                print("   Local file is locked, falling back to Hugging Face")
            else:
                raise
        except Exception as exc:
            print(f"   Local model load failed ({exc}), falling back to Hugging Face")

    if model_path is not None:
        print(f"Local model artifact not found: {model_path}")
    return _load_model_huggingface(config)


def _load_model_local_only(config: Dict[str, Any], script_dir: Path) -> Tuple[Any, str]:
    model_path = script_dir / config["local_path"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model artifact not found: {model_path}. "
            "Place the .nemo file under local_models/ or switch to local_or_huggingface loading."
        )

    model = _load_with_retry(model_path, config, max_retries=3)
    _override_model_dataloader_config(model)
    return model, str(model_path)


def _move_model_to_cuda(model: LoadedModelHandle) -> LoadedModelHandle:
    if not torch.cuda.is_available():
        return model

    try:
        moved = _move_runtime_to_device(model, "cuda")
        if isinstance(moved, LoadedModelHandle):
            return moved
    except Exception as exc:
        print(f"   Could not move model to CUDA: {exc}")
    return model


def load_model(model_name: str, show_progress: bool = False) -> LoadedModelHandle:
    del show_progress

    config = MODEL_CONFIGS[model_name]
    if model_name in models_cache:
        cached_model = models_cache[model_name]
        if _get_model_runtime(cached_model) is not None:
            return cached_model
        del models_cache[model_name]

    _unload_cached_models(model_name)
    script_dir = get_script_dir()
    loading_method = config.get("loading_method", "huggingface")

    start_time = time.time()
    if loading_method == "local":
        runtime, source = _load_model_local_only(config, script_dir)
    elif loading_method == "local_or_huggingface":
        runtime, source = _load_model_local_or_huggingface(model_name, config, script_dir)
    else:
        runtime, source = _load_model_huggingface(config)

    handle = _build_loaded_handle(model_name, runtime, config, source)
    handle = _move_model_to_cuda(handle)
    models_cache[model_name] = handle

    load_time = time.time() - start_time
    print(f"{config['display_name']} loaded in {load_time:.1f}s")
    return handle


def unload_all_models() -> str:
    global models_cache

    if not models_cache:
        return "No models currently loaded in memory"

    unloaded: List[str] = []
    initial_vram = 0.0
    final_vram = 0.0

    if torch.cuda.is_available():
        initial_vram = torch.cuda.memory_allocated() / 1024**3

    for model_key in list(models_cache.keys()):
        try:
            model = models_cache[model_key]
            _move_runtime_to_device(model, "cpu")
            del models_cache[model_key]
            del model
            unloaded.append(model_key)
        except Exception as exc:
            print(f"Failed to unload {model_key}: {exc}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_vram = torch.cuda.memory_allocated() / 1024**3

    freed_vram = initial_vram - final_vram
    if unloaded:
        return f"Unloaded: {', '.join(unloaded)}\nFreed about {freed_vram:.1f} GB VRAM"
    return "No models were successfully unloaded"


def set_auto_unload(enabled: bool) -> str:
    global auto_unload_after_transcription
    auto_unload_after_transcription = enabled
    return f"Auto-unload {'enabled' if enabled else 'disabled'}"


def _clear_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _execute_transcription(runtime: Any, transcribe_kwargs: Dict[str, Any], use_cuda: bool) -> Any:
    if use_cuda and torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return runtime.transcribe(**transcribe_kwargs)
    return runtime.transcribe(**transcribe_kwargs)


def _transcribe_single_buffer(runtime: Any, buffer: Any, use_cuda: bool) -> Any:
    base_kwargs: Dict[str, Any] = {
        "audio": [buffer],
        "batch_size": 1,
        "verbose": False,
    }
    mode_attempts: List[Dict[str, Any]] = [
        {"return_hypotheses": True, "timestamps": True},
        {"return_hypotheses": True, "timestamps": False},
        {"return_hypotheses": False, "timestamps": False},
    ]

    last_error: Optional[Exception] = None
    for index, mode_kwargs in enumerate(mode_attempts, start=1):
        try:
            return _execute_transcription(runtime, {**base_kwargs, **mode_kwargs}, use_cuda)
        except Exception as exc:
            last_error = exc
            if index < len(mode_attempts):
                print(f"   Chunk mode {index} failed: {type(exc).__name__}: {exc}; trying fallback")
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Chunk transcription failed without a captured exception")


def _unwrap_transcription_item(item: Any) -> Any:
    if isinstance(item, list):
        return _unwrap_transcription_item(item[0]) if item else item
    if isinstance(item, tuple):
        for candidate in reversed(item):
            if hasattr(candidate, "text") or isinstance(candidate, (str, dict)):
                return candidate
            if isinstance(candidate, list) and candidate:
                return _unwrap_transcription_item(candidate[0])
        return item[-1] if item else item
    return item


def _extract_hypothesis_text(hypothesis: Any) -> str:
    if hasattr(hypothesis, "text"):
        return str(hypothesis.text)
    if isinstance(hypothesis, tuple):
        for item in hypothesis:
            if hasattr(item, "text"):
                return str(item.text)
            if isinstance(item, str):
                return item
        return ""
    if isinstance(hypothesis, dict):
        for key in ("text", "pred_text", "prediction", "answer"):
            if hypothesis.get(key) is not None:
                return str(hypothesis[key])
        return ""
    if isinstance(hypothesis, list):
        return _extract_hypothesis_text(hypothesis[0]) if hypothesis else ""
    if isinstance(hypothesis, str):
        return hypothesis
    return str(hypothesis)


def _adjust_chunk_timestamps(
    chunk_word_timestamps: List[Dict[str, Any]],
    left_context_duration: float,
    chunk_start_time: float,
) -> List[Dict[str, Any]]:
    adjusted_timestamps: List[Dict[str, Any]] = []

    for timestamp in chunk_word_timestamps:
        raw_start = float(timestamp.get("start", 0.0))
        raw_end = float(timestamp.get("end", 0.0))

        if raw_end <= left_context_duration:
            continue
        if raw_start < left_context_duration:
            raw_start = left_context_duration

        adjusted_timestamp: Dict[str, Any] = {
            "start": max(0.0, raw_start - left_context_duration + chunk_start_time),
            "end": max(0.0, raw_end - left_context_duration + chunk_start_time),
        }
        adjusted_timestamp["end"] = max(float(adjusted_timestamp["start"]), float(adjusted_timestamp["end"]))

        for key in ("word", "segment", "char", "text"):
            if key in timestamp:
                adjusted_timestamp[key] = timestamp[key]

        adjusted_timestamps.append(adjusted_timestamp)

    return adjusted_timestamps


def _create_chunk_fallback_timestamp(chunk_start_time: float, chunk_end_time: float, chunk_text: str) -> Dict[str, Any]:
    return {
        "start": chunk_start_time,
        "end": chunk_end_time,
        "text": chunk_text,
    }


def _process_single_chunk(
    runtime: Any,
    buffer: Any,
    use_cuda: bool,
    apply_itn_per_chunk: bool,
    chunk_start_time: float,
    chunk_end_time: float,
    left_context_duration: float,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    result = _transcribe_single_buffer(runtime, buffer, use_cuda)
    if not result:
        print("   Chunk returned empty transcription output")
        return None, []

    hypothesis = _unwrap_transcription_item(result[0])
    chunk_timestamps, _timestamp_level = extract_timestamps(hypothesis, include_timestamps=True)

    if chunk_timestamps:
        adjusted = _adjust_chunk_timestamps(chunk_timestamps, left_context_duration, chunk_start_time)
        if not adjusted:
            return None, []

        words: List[str] = []
        for timestamp in adjusted:
            word = timestamp.get("word", timestamp.get("text", timestamp.get("segment", timestamp.get("char", ""))))
            if word:
                words.append(str(word))

        chunk_text = " ".join(words).strip()
        if not chunk_text:
            return None, []

        if apply_itn_per_chunk:
            chunk_text = apply_itn_to_segment(chunk_text)

        return chunk_text, adjusted

    chunk_text = _extract_hypothesis_text(hypothesis).strip()
    if not chunk_text:
        return None, []

    if apply_itn_per_chunk:
        chunk_text = apply_itn_to_segment(chunk_text)

    fallback_timestamp = _create_chunk_fallback_timestamp(chunk_start_time, chunk_end_time, chunk_text)
    return chunk_text, [fallback_timestamp]


def _transcribe_long_audio_chunked(
    runtime: Any,
    audio_array: Any,
    sample_rate: int = 16000,
    use_cuda: bool = True,
    chunk_size_override: Optional[int] = None,
    apply_itn_per_chunk: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    effective_chunk_duration = chunk_size_override or chunk_duration_sec
    chunk_samples = int(effective_chunk_duration * sample_rate)
    context_samples = int(CHUNK_OVERLAP_SEC * sample_rate)
    total_samples = len(audio_array)
    total_duration = total_samples / sample_rate

    print(f"   Chunked transcription: {total_duration:.1f}s audio -> {effective_chunk_duration}s chunks")

    transcriptions: List[str] = []
    merged_timestamps: List[Dict[str, Any]] = []
    failed_chunks = 0
    position = 0
    chunk_number = 0

    while position < total_samples:
        chunk_number += 1
        start = max(0, position - context_samples)
        end = min(total_samples, position + chunk_samples + context_samples)
        buffer = audio_array[start:end]

        chunk_start_time = position / sample_rate
        chunk_end_time = min((position + chunk_samples) / sample_rate, total_duration)
        left_context_duration = (position - start) / sample_rate if position > start else 0.0

        print(
            f"   Chunk {chunk_number}: {chunk_start_time:.1f}s - {chunk_end_time:.1f}s "
            f"({len(buffer) / sample_rate:.1f}s with context)"
        )

        _clear_vram()
        try:
            chunk_text, timestamps = _process_single_chunk(
                runtime,
                buffer,
                use_cuda,
                apply_itn_per_chunk,
                chunk_start_time,
                chunk_end_time,
                left_context_duration,
            )
            if chunk_text:
                transcriptions.append(chunk_text)
                merged_timestamps.extend(timestamps)
        except Exception as exc:
            failed_chunks += 1
            print(f"   Chunk {chunk_number} failed: {type(exc).__name__}: {exc}")

        position += chunk_samples
        _clear_vram()

    print(f"   Processed {chunk_number} chunks")
    if not transcriptions:
        raise RuntimeError(
            f"Chunked transcription failed for all {chunk_number} chunks. "
            f"Failed chunks: {failed_chunks}."
        )

    full_transcription = " ".join(transcriptions)
    while "  " in full_transcription:
        full_transcription = full_transcription.replace("  ", " ")

    return full_transcription.strip(), merged_timestamps


def _load_audio_files_to_memory(files: List[str]) -> List[Tuple[Any, float]]:
    print(f"   Loading {len(files)} audio file(s) into memory")
    audio_data: List[Tuple[Any, float]] = []

    for file_path in files:
        audio_array, sample_rate = load_audio_to_numpy(file_path, target_sr=16000)
        duration_sec = len(audio_array) / sample_rate
        audio_data.append((audio_array, duration_sec))
        print(f"      {Path(file_path).name}: {duration_sec:.1f}s")

    return audio_data


def _transcribe_chunked_files(
    runtime: Any,
    audio_data: List[Tuple[Any, float]],
    use_cuda: bool,
    chunk_size_override: Optional[int],
    threshold: float,
    apply_itn_per_chunk: bool = False,
) -> Tuple[List[Any], Dict[int, List[Dict[str, Any]]]]:
    print(f"   Long audio detected (>{threshold}s) - using chunked transcription")
    results: List[Any] = []
    chunk_timestamps_map: Dict[int, List[Dict[str, Any]]] = {}

    for index, (audio_np, duration) in enumerate(audio_data):
        if duration > threshold:
            text, chunk_timestamps = _transcribe_long_audio_chunked(
                runtime,
                audio_np,
                sample_rate=16000,
                use_cuda=use_cuda,
                chunk_size_override=chunk_size_override,
                apply_itn_per_chunk=apply_itn_per_chunk,
            )
            chunk_timestamps_map[index] = chunk_timestamps
            results.append(SimpleHypothesis(text=text, chunk_timestamps=chunk_timestamps))
            continue

        short_result = _transcribe_single_buffer(runtime, audio_np, use_cuda)
        if short_result:
            results.extend(short_result)
        _clear_vram()

    return results, chunk_timestamps_map


def _transcribe_short_audio_batch(
    runtime: Any,
    audio_arrays: List[Any],
    batch_size: int,
    use_cuda: bool,
    max_retries: int,
    base_delay: float,
) -> Any:
    print("   Audio loaded into memory, starting transcription")
    base_kwargs: Dict[str, Any] = {
        "audio": audio_arrays,
        "batch_size": batch_size,
        "verbose": True,
    }
    mode_attempts: List[Dict[str, Any]] = [
        {"return_hypotheses": True, "timestamps": True},
        {"return_hypotheses": True, "timestamps": False},
        {"return_hypotheses": False, "timestamps": False},
    ]

    last_error: Optional[Exception] = None
    for mode_index, mode_kwargs in enumerate(mode_attempts, start=1):
        transcribe_kwargs = {**base_kwargs, **mode_kwargs}
        for attempt in range(max_retries):
            try:
                return _execute_transcription(runtime, transcribe_kwargs, use_cuda)
            except Exception as exc:
                last_error = exc
                _clear_vram()
                if _handle_retry_delay(attempt, base_delay, max_retries):
                    continue
                break

        if mode_index < len(mode_attempts):
            print(f"   Batch mode {mode_index} failed, trying fallback mode")
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Batch transcription failed without a captured exception")


def _transcribe_with_retry(
    model: LoadedModelHandle,
    files: List[str],
    batch_size: int,
    use_cuda: bool = True,
    max_retries: int = 3,
    chunk_size_override: Optional[int] = None,
    apply_itn: bool = False,
) -> Tuple[List[Any], Dict[int, List[Dict[str, Any]]]]:
    audio_data = _load_audio_files_to_memory(files)
    effective_threshold = long_audio_threshold_sec

    if any(duration > effective_threshold for _, duration in audio_data):
        return _transcribe_chunked_files(
            model.runtime,
            audio_data,
            use_cuda,
            chunk_size_override,
            effective_threshold,
            apply_itn_per_chunk=apply_itn,
        )

    max_batch_size = int(model.config.get("max_batch_size", batch_size))
    effective_batch_size = max(1, min(batch_size, max_batch_size))
    audio_arrays = [audio for audio, _duration in audio_data]
    result = _transcribe_short_audio_batch(
        model.runtime,
        audio_arrays,
        effective_batch_size,
        use_cuda,
        max_retries,
        base_delay=0.5,
    )
    return result, {}


def _load_model_for_transcription(
    model_key: str,
    log_capture_obj: LogCapture,
) -> Tuple[Optional[LoadedModelHandle], Optional[Tuple[Any, ...]]]:
    try:
        return load_model(model_key), None
    except PermissionError as exc:
        error_message = str(exc)
        error_type = "permission_file_lock" if _is_file_lock_error(error_message) else "permission"
        return None, _make_error_response(error_type, error_message, log_capture_obj)
    except ConnectionError as exc:
        return None, _make_error_response("network", str(exc), log_capture_obj)
    except FileNotFoundError as exc:
        return None, _make_error_response("file_not_found", str(exc), log_capture_obj)
    except OSError as exc:
        return None, _make_error_response("filesystem", str(exc), log_capture_obj)
    except ImportError as exc:
        return None, _make_error_response("runtime", str(exc), log_capture_obj)
    except RuntimeError as exc:
        return None, _make_error_response("runtime", str(exc), log_capture_obj)
    except Exception as exc:
        error_message = f"Type: {type(exc).__name__}\nMessage: {exc}"
        return None, _make_error_response("generic", error_message, log_capture_obj)


def _run_transcription(
    model: LoadedModelHandle,
    processed_files: List[str],
    batch_size: int,
    chunk_size: int,
    apply_itn: bool,
    log_capture_obj: LogCapture,
) -> Tuple[Optional[List[Any]], Dict[int, List[Dict[str, Any]]], Optional[Tuple[Any, ...]]]:
    try:
        result, chunk_timestamps_map = _transcribe_with_retry(
            model=model,
            files=processed_files,
            batch_size=batch_size,
            use_cuda=torch.cuda.is_available(),
            max_retries=3,
            chunk_size_override=chunk_size,
            apply_itn=apply_itn,
        )
        return result, chunk_timestamps_map, None
    except PermissionError as exc:
        error_str = str(exc)
        error_type = "transcription_file_lock" if _is_file_lock_error(error_str) else "permission"
        return None, {}, _make_error_response(error_type, error_str, log_capture_obj)
    except Exception as exc:
        error_message = f"Error Type: {type(exc).__name__}\n\nDetails: {exc}"
        return None, {}, _make_error_response("transcription", error_message, log_capture_obj)


def _get_gpu_stats() -> Tuple[float, str]:
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        return vram_used, gpu_name
    return 0.0, "CPU"


def _format_preview_output(
    transcription: str,
    timestamps: List[Dict[str, Any]],
    timestamp_level: str,
    output_format: str,
) -> str:
    if output_format == "srt":
        return format_as_srt(transcription, timestamps, timestamp_level)
    if output_format == "csv":
        return format_as_csv(transcription, timestamps, timestamp_level)
    if timestamps:
        return format_as_txt_with_timestamps(transcription, timestamps, timestamp_level)
    return transcription


def transcribe_audio(
    audio_files: Any,
    model_choice: str,
    save_to_file: bool,
    include_timestamps: bool,
    output_format: str = "txt",
    apply_itn: bool = True,
    chunk_size: int = DEFAULT_CHUNK_DURATION_SEC,
    batch_size: int = 1,
    max_word_duration: float = DEFAULT_MAX_WORD_DURATION_SEC,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD_SEC,
    itn_mode_choice: str = DEFAULT_ITN_MODE,
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    log_capture.start()
    print("\n" + "=" * 60)
    print(f"Transcription started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    global chunk_duration_sec, long_audio_threshold_sec, itn_mode
    chunk_duration_sec = chunk_size
    long_audio_threshold_sec = chunk_size + 30
    itn_mode = itn_mode_choice
    configure_output_timing(silence_threshold=silence_threshold, max_word_duration=max_word_duration)

    apply_itn_per_chunk = itn_mode_choice in ("per_chunk", "both")
    apply_itn_final = itn_mode_choice in ("final_pass", "both")
    itn_enabled = apply_itn and itn_mode_choice != "disabled"

    file_list = normalize_file_list(audio_files)
    if not file_list:
        log_capture.stop()
        return "Please upload an audio or video file first.", "", None, None, None, None

    try:
        is_batch = len(file_list) > 1
        print(f"Files: {len(file_list)}")
        print(f"Model: {model_choice}")
        print(f"Preview format: {output_format.upper()}")
        print(f"ITN mode: {itn_mode_choice} ({'enabled' if itn_enabled else 'disabled'})")
        print(f"Timestamps: {'enabled' if include_timestamps else 'disabled'}")
        print(f"Chunk size: {chunk_size}s")
        print(f"Silence threshold: {silence_threshold}s")

        model_key = get_model_key_from_choice(model_choice)
        start_time = time.time()

        model, error_response = _load_model_for_transcription(model_key, log_capture)
        if error_response is not None:
            return error_response
        assert model is not None

        load_time = time.time() - start_time
        processed_files, file_info, total_duration, video_count = process_audio_files(file_list)
        video_status = f"Video files processed: {video_count}\n" if video_count > 0 else ""

        inference_start = time.time()
        result, chunk_timestamps_map, error_response = _run_transcription(
            model,
            processed_files,
            batch_size,
            chunk_size,
            apply_itn_per_chunk and itn_enabled,
            log_capture,
        )
        if error_response is not None:
            return error_response
        assert result is not None

        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        vram_used, gpu_name = _get_gpu_stats()
        rtfx = total_duration / inference_time if inference_time > 0 else 0.0

        stats = TranscriptionStats(
            model_choice=model_choice,
            gpu_name=gpu_name,
            total_duration=total_duration,
            total_time=total_time,
            inference_time=inference_time,
            load_time=load_time,
            chunk_size=chunk_size,
            rtfx=rtfx,
            vram_used=vram_used,
            apply_itn=itn_enabled,
        )
        context = ResultProcessingContext(
            stats=stats,
            file_list=file_list,
            file_info=file_info,
            include_timestamps=include_timestamps,
            video_status=video_status,
            load_time=load_time,
            apply_itn_final=apply_itn_final and itn_enabled,
            had_itn_per_chunk=apply_itn_per_chunk and itn_enabled,
            text_normalizer=apply_inverse_text_normalization if itn_enabled else None,
            itn_available=ITN_AVAILABLE,
        )

        if is_batch:
            (
                status,
                transcription_output,
                timestamps,
                timestamp_level,
                all_transcriptions,
                all_timestamps,
            ) = _process_batch_transcription(result, chunk_timestamps_map, context)
        else:
            (
                status,
                transcription_output,
                timestamps,
                timestamp_level,
                error_response,
            ) = _process_single_transcription(result, chunk_timestamps_map, log_capture, context)
            if error_response is not None:
                return error_response
            assert status is not None
            assert transcription_output is not None
            file_output_transcription = transcription_output
            transcription_output = _format_preview_output(
                file_output_transcription if not timestamps else file_output_transcription.replace("\r\n", "\n"),
                timestamps,
                timestamp_level,
                output_format,
            )
            all_transcriptions = None
            all_timestamps = None

        output_config = OutputFilesConfig(
            file_list=file_list,
            file_info=file_info,
            is_batch=is_batch,
            include_timestamps=include_timestamps,
            model_choice=model_choice,
            total_duration=total_duration,
            total_time=total_time,
            apply_itn=itn_enabled,
            transcription=None if is_batch else file_output_transcription,
            timestamps=None if is_batch else timestamps,
            timestamp_level=timestamp_level,
            all_transcriptions=all_transcriptions if is_batch else None,
            all_timestamps=all_timestamps if is_batch else None,
            itn_available=ITN_AVAILABLE,
        )
        txt_file, srt_file, csv_file, status_suffix = _generate_and_save_output_files(save_to_file, output_config)
        status = (status or "") + status_suffix

        print("\n" + "=" * 60)
        print(f"Transcription complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        if auto_unload_after_transcription:
            print("\nAuto-unloading model to free VRAM")
            print(unload_all_models())

        logs = log_capture.stop()
        log_file = save_logs(logs, "transcription")
        return status, transcription_output or "", txt_file, srt_file, csv_file, log_file
    except OSError as exc:
        return _make_error_response("filesystem", str(exc), log_capture)
    except Exception as exc:
        vram_info = "no GPU detected"
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            vram_info = f"about {total_vram:.0f}GB available"

        error_message = (
            "### Error During Transcription\n\n"
            f"**Error Type**: {type(exc).__name__}\n\n"
            f"**Error Message**: {exc}\n\n"
            "**Troubleshooting:**\n"
            "1. Make sure the audio or video file is valid\n"
            f"2. Check that you have enough VRAM ({vram_info})\n"
            "3. Try a shorter audio file first\n"
            "4. Restart the interface if the runtime appears stuck\n"
            "5. For video files, ensure FFmpeg is installed\n"
        )
        logs = log_capture.stop()
        log_file = save_logs(logs, "error")
        return error_message, "", None, None, None, log_file


def get_system_info() -> str:
    nemo_available = _dependency_is_available("nemo.collections.asr")
    configured_models = len(MODEL_CONFIGS)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        return f"""
### System Information

**GPU**: {gpu_name}
**Total VRAM**: {total_vram:.1f} GB
**Available VRAM**: {free_vram:.1f} GB
**CUDA Version**: {torch.version.cuda}
**PyTorch Version**: {torch.__version__}
**NeMo Runtime**: {'Ready' if nemo_available else 'Missing'}
**Configured Models**: {configured_models}/{configured_models}

**Status**: Ready for NeMo transcription
"""

    return f"""
### CPU Mode

CUDA is not available.

**NeMo Runtime**: {'Ready' if nemo_available else 'Missing'}
**Configured Models**: {configured_models}/{configured_models}

The app can still run on CPU, but transcription will be slower.
"""


def get_model_info() -> str:
    config = MODEL_CONFIGS[DEFAULT_MODEL_KEY]
    local_path = get_script_dir() / config["local_path"]
    local_status = "Found locally" if local_path.exists() else "Will download on first use"

    return f"""
### Active NeMo Model

**{config['display_name']}**
- Backend: NeMo
- Architecture: {config['architecture']}
- Languages: {config['languages']}
- Word-level timestamps: Yes
- Long-audio chunking: Yes
- Recommended VRAM: {config['vram_gb']} GB
- Local artifact: `{config['local_path']}` ({local_status})
- Hugging Face fallback: `{config['hf_model_id']}`

This app intentionally exposes only the active Parakeet NeMo path.
"""


def get_privacy_performance_info() -> str:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"**GPU**: {gpu_name} ({total_vram:.0f} GB VRAM)"
    else:
        gpu_info = "**GPU**: CPU mode"

    return f"""
### Privacy

- All transcription runs locally on your machine.
- Audio files stay on your computer.
- Local artifacts under `local_models/` are preferred when present.
- Downloads are cached under `model_cache/` for later offline reuse.

### Runtime Layout

- **Backend**: NeMo ASR only
- **Cache directory**: `{CACHE_DIR}`
- **Long audio**: direct-audio chunking with overlap
- **Exports**: TXT, SRT, CSV, plus log download

### Performance Notes

- {gpu_info}
- Mixed precision is used on CUDA where available.
- Models stay cached in memory until manually unloaded or auto-unload runs.
- Increase chunk size only if you have spare VRAM.
"""


with gr.Blocks(title="Local NeMo Transcription") as app:
    gr.Markdown(
        """
# Local NeMo / Parakeet Transcription
### NeMo-focused offline transcription for audio and video files

This entrypoint keeps the existing NeMo direct-audio path, chunking, timestamps, ITN support, batch uploads, exports, and downloadable logs, without exposing Transformers or qwen-asr model choices.
"""
    )

    with gr.Accordion("System Information", open=False):
        system_info = gr.Markdown(get_system_info())
        refresh_system_button = gr.Button("Refresh System Info", size="sm")
        refresh_system_button.click(fn=get_system_info, outputs=system_info)

    with gr.Accordion("VRAM Management", open=False):
        gr.Markdown(
            """
Unload the cached NeMo model when you are done transcribing if you want to free GPU memory for other applications.
"""
        )
        with gr.Row():
            unload_button = gr.Button("Unload All Models", size="sm", variant="secondary")
            auto_unload_checkbox = gr.Checkbox(
                label="Auto-unload after transcription",
                value=False,
                info="Automatically free VRAM after each transcription finishes",
            )
        unload_status = gr.Markdown("")
        unload_button.click(fn=unload_all_models, outputs=unload_status)
        auto_unload_checkbox.change(fn=set_auto_unload, inputs=auto_unload_checkbox, outputs=unload_status)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Audio or Video Files")
            audio_input = gr.File(
                file_count="multiple",
                file_types=[
                    ".wav",
                    ".mp3",
                    ".flac",
                    ".m4a",
                    ".ogg",
                    ".aac",
                    ".wma",
                    ".mp4",
                    ".avi",
                    ".mkv",
                    ".mov",
                    ".webm",
                    ".flv",
                    ".m4v",
                ],
                label="Audio or video files",
            )

            gr.Markdown(
                """
**Supported audio**: WAV, MP3, FLAC, M4A, OGG, AAC, WMA

**Supported video**: MP4, AVI, MKV, MOV, WEBM, FLV, M4V

Audio is extracted from uploaded video files automatically. Select multiple files for batch transcription.
"""
            )

            gr.Markdown("### Settings")
            model_selector = gr.Radio(
                choices=get_model_choice_labels(),
                value=get_default_model_choice(),
                label="NeMo model",
                info="This app intentionally exposes only the active NeMo model registry.",
            )

            save_checkbox = gr.Checkbox(
                label="Save transcription files",
                value=True,
                info="Creates TXT, SRT, and CSV files in the current directory",
            )

            output_format = gr.Dropdown(
                choices=["txt", "srt", "csv"],
                value="txt",
                label="Preview format",
                info="Changes the single-file preview. Downloads always include TXT, SRT, and CSV.",
            )

            timestamp_checkbox = gr.Checkbox(
                label="Include timestamps",
                value=True,
                info="Use timestamps in previews and exports when the model provides them",
            )

            itn_checkbox = gr.Checkbox(
                label="Convert spoken numbers to digits (ITN)",
                value=True,
                info="Converts phrases like 'twenty twenty two' to '2022' when nemo_text_processing is available",
            )

            with gr.Accordion("Advanced Settings", open=False):
                chunk_size_slider = gr.Slider(
                    minimum=30,
                    maximum=1200,
                    value=DEFAULT_CHUNK_DURATION_SEC,
                    step=10,
                    label="Chunk size (seconds)",
                    info="Larger chunks improve throughput but need more VRAM",
                )

                batch_size_slider = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=1,
                    step=1,
                    label="Batch size",
                    info="Controls how many short files are transcribed together",
                )

                max_word_duration_slider = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=DEFAULT_MAX_WORD_DURATION_SEC,
                    step=0.1,
                    label="Max word duration (seconds)",
                    info="Caps unusually long word spans in subtitle segmentation",
                )

                silence_threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=3.0,
                    value=DEFAULT_SILENCE_THRESHOLD_SEC,
                    step=0.1,
                    label="Silence threshold (seconds)",
                    info="Ends subtitle segments when a pause exceeds this duration",
                )

                itn_mode_dropdown = gr.Dropdown(
                    choices=ITN_MODE_CHOICES,
                    value=DEFAULT_ITN_MODE,
                    label="ITN mode",
                    info="per_chunk is safest for long recordings; final_pass runs after the full transcript is assembled",
                )

                gr.Markdown(
                    f"""
**Chunk size guidance**
- 60-90s: safest on smaller GPUs
- 120-180s: good default for mid-range GPUs
- 300s+: better throughput when VRAM is available

**ITN availability**: {'Installed' if ITN_AVAILABLE else 'Not installed'}
"""
                )

            transcribe_button = gr.Button("Start Transcription", variant="primary", size="lg")

            with gr.Accordion("Model Details", open=False):
                model_info = gr.Markdown(get_model_info())
                refresh_model_button = gr.Button("Refresh Model Status", size="sm")
                refresh_model_button.click(fn=get_model_info, outputs=model_info)

        with gr.Column(scale=2):
            gr.Markdown("### Transcription Results")
            status_output = gr.Markdown("Upload one or more files and start transcription.")
            transcription_output = gr.Textbox(
                label="Transcription Preview",
                lines=20,
                placeholder="Transcription output will appear here...",
                buttons=["copy"],
                show_label=True,
            )

            gr.Markdown("### Downloads")
            with gr.Row():
                txt_file_output = gr.File(label="TXT", visible=True)
                srt_file_output = gr.File(label="SRT", visible=True)
                csv_file_output = gr.File(label="CSV", visible=True)

            log_file_output = gr.File(label="Processing Logs", visible=True)

    gr.Markdown("---")

    with gr.Accordion("How to Use", open=False):
        gr.Markdown(
            """
1. Upload audio or video files. Multiple files are processed as a batch.
2. Keep the default Parakeet NeMo model selected.
3. Adjust chunking, timestamps, or ITN settings if needed.
4. Start transcription and download the generated files.

This app keeps the NeMo-only path focused: local cache bootstrap, direct-audio transcription, long-audio chunking, timestamps, exports, and log downloads.
"""
        )

    with gr.Accordion("Privacy and Performance", open=False):
        gr.Markdown(get_privacy_performance_info())

    transcribe_button.click(
        fn=transcribe_audio,
        inputs=[
            audio_input,
            model_selector,
            save_checkbox,
            timestamp_checkbox,
            output_format,
            itn_checkbox,
            chunk_size_slider,
            batch_size_slider,
            max_word_duration_slider,
            silence_threshold_slider,
            itn_mode_dropdown,
        ],
        outputs=[
            status_output,
            transcription_output,
            txt_file_output,
            srt_file_output,
            csv_file_output,
            log_file_output,
        ],
        queue=False,
    )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Starting Local NeMo Transcription Interface")
    print("=" * 80)
    print(f"\nCache directory: {CACHE_DIR}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
        setup_gpu_optimizations()
    else:
        print("No CUDA GPU detected")

    validate_local_models()
    print("Opening browser at http://127.0.0.1:7860")
    print("Keep this terminal open while using the interface\n")

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )