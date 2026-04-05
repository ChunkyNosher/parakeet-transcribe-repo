from __future__ import annotations

from app_shared.env_bootstrap import CACHE_DIR, bootstrap_environment, get_script_dir

bootstrap_environment(verbose=False)

import gc
import importlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gradio as gr
import torch

from app_shared.file_pipeline import normalize_file_list, process_audio_files, load_audio_to_numpy
from app_shared.logging_utils import log_capture, save_logs
from app_shared.output_formats import SEPARATOR, save_output_files
from app_shared.result_types import LoadedModelHandle, OutputFilesConfig, SimpleHypothesis
from app_shared.transcription_flow import _make_error_response, validate_transcription_result


TIMESTAMP_EXPORT_NOTE = (
    "Current phase 1 Transformers adapters do not emit aligned timestamps. "
    "TXT, SRT, and CSV exports are still created, but subtitle/spreadsheet timing falls back "
    "to whole-file estimates rather than word timing."
)

MODEL_DISPLAY_ORDER = [
    "granite-4.0-1b-speech",
    "voxtral-mini-3b-2507",
    "cohere-transcribe-03-2026",
    "voxtral-small-24b-2507",
]

DEFAULT_MODEL_KEY = "granite-4.0-1b-speech"

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "granite-4.0-1b-speech": {
        "backend": "transformers_granite",
        "choice_label": "ibm-granite/granite-4.0-1b-speech :: compact Transformers speech model",
        "display_name": "ibm-granite/granite-4.0-1b-speech",
        "hf_model_id": "ibm-granite/granite-4.0-1b-speech",
        "local_path": "local_models/granite-4.0-1b-speech",
        "loading_method": "local_or_huggingface",
        "architecture": "Granite Speech",
        "parameters": "1B",
        "languages": 6,
        "wer": "5.52 open ASR leaderboard",
        "rtfx": "Fast",
        "vram_gb": "Resource-friendly",
        "recommended_for": "Safest first choice in this Transformers-only app",
        "supports_timestamps": False,
        "supports_batching": True,
        "max_batch_size": 8,
        "max_new_tokens": 256,
        "summary": "Compact local speech transcription through the Granite Transformers adapter.",
    },
    "voxtral-mini-3b-2507": {
        "backend": "transformers_voxtral",
        "choice_label": "mistralai/Voxtral-Mini-3B-2507 :: offline Voxtral transcription",
        "display_name": "mistralai/Voxtral-Mini-3B-2507",
        "hf_model_id": "mistralai/Voxtral-Mini-3B-2507",
        "local_path": "local_models/Voxtral-Mini-3B-2507",
        "loading_method": "local_or_huggingface",
        "architecture": "Voxtral audio-text model",
        "parameters": "3B",
        "languages": 8,
        "wer": "Strong offline ASR",
        "rtfx": "Hardware dependent",
        "vram_gb": "9-10",
        "recommended_for": "Smallest locally runnable Voxtral option in phase 1",
        "supports_timestamps": False,
        "supports_batching": False,
        "max_batch_size": 1,
        "max_new_tokens": 500,
        "summary": "Offline transcription via Transformers plus the local Voxtral processor stack.",
    },
    "cohere-transcribe-03-2026": {
        "backend": "transformers_cohere",
        "choice_label": "CohereLabs/cohere-transcribe-03-2026 :: gated Cohere ASR",
        "display_name": "CohereLabs/cohere-transcribe-03-2026",
        "hf_model_id": "CohereLabs/cohere-transcribe-03-2026",
        "local_path": "local_models/cohere-transcribe-03-2026",
        "loading_method": "local_or_huggingface",
        "architecture": "Cohere ASR",
        "parameters": "2B",
        "languages": 14,
        "wer": "Best-in-class (model card)",
        "rtfx": "Fast",
        "vram_gb": "Model dependent",
        "recommended_for": "High-accuracy transcription when local artifacts or HF access are approved",
        "supports_timestamps": False,
        "supports_batching": True,
        "max_batch_size": 8,
        "max_new_tokens": 256,
        "default_language": "en",
        "trust_remote_code": True,
        "summary": "Transformers speech-seq2seq adapter using Cohere's local remote-code package.",
        "warning": (
            "This adapter loads with trust_remote_code=True. Only use vetted local artifacts or an approved "
            "Hugging Face source."
        ),
    },
    "voxtral-small-24b-2507": {
        "backend": "transformers_voxtral",
        "choice_label": "mistralai/Voxtral-Small-24B-2507 :: large offline Voxtral transcription",
        "display_name": "mistralai/Voxtral-Small-24B-2507",
        "hf_model_id": "mistralai/Voxtral-Small-24B-2507",
        "local_path": "local_models/Voxtral-Small-24B-2507",
        "loading_method": "local_or_huggingface",
        "architecture": "Voxtral audio-text model",
        "parameters": "24B",
        "languages": 8,
        "wer": "State-of-the-art (model card)",
        "rtfx": "Hardware dependent",
        "vram_gb": "~55",
        "recommended_for": "Highest-capacity Voxtral offline transcription if the machine can carry it",
        "supports_timestamps": False,
        "supports_batching": False,
        "max_batch_size": 1,
        "max_new_tokens": 500,
        "summary": "Offline transcription via the larger Voxtral adapter path.",
        "warning": "Expect roughly 55 GB of VRAM or aggressive offloading. This is not a realistic fit for most consumer GPUs.",
    },
}

DEFERRED_MODELS: Dict[str, Dict[str, str]] = {
    "voxtral-mini-4b-realtime-2602": {
        "display_name": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "reason": "Deferred. The required realtime runtime is not exposed by the current Transformers stack in this repo.",
    },
    "qwen3-asr-1.7b": {
        "display_name": "Qwen/Qwen3-ASR-1.7B",
        "reason": "Deferred from this strict Transformers app. Qwen remains outside phase 1 because it needs its own qwen-asr runtime path.",
    },
}

models_cache: Dict[str, LoadedModelHandle] = {}
auto_unload_after_transcription = False


def _import_dependency(module_name: str, package_name: Optional[str] = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        dependency_name = package_name or module_name
        raise ImportError(
            f"Missing dependency '{dependency_name}'. Use the configured project environment and install the repo requirements. "
            f"Original error: {exc}"
        ) from exc


def _dependency_is_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _transformers_attr_available(attr_name: str) -> bool:
    try:
        transformers = importlib.import_module("transformers")
    except Exception:
        return False
    return hasattr(transformers, attr_name)


def _require_transformers_attr(attr_name: str) -> Any:
    transformers = _import_dependency("transformers", "transformers==4.57.6")
    if not hasattr(transformers, attr_name):
        raise ImportError(
            f"Transformers does not expose '{attr_name}' in the active environment. "
            "Use transformers==4.57.6 for the supported local backend mix."
        )
    return getattr(transformers, attr_name)


def _preferred_torch_dtype() -> Any:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _prepare_pretrained_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"torch_dtype": _preferred_torch_dtype()}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["low_cpu_mem_usage"] = True
    return kwargs


def _artifact_exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        try:
            next(path.iterdir())
            return True
        except StopIteration:
            return False
    return True


def _resolve_preferred_source(script_dir: Path, config: Dict[str, Any]) -> Tuple[str, bool]:
    local_path = config.get("local_path")
    if local_path:
        candidate = script_dir / local_path
        if _artifact_exists(candidate):
            return str(candidate), True
        if config.get("loading_method") == "local":
            raise FileNotFoundError(f"Local model artifact not found: {candidate}")

    remote_source = config.get("hf_model_id")
    if not remote_source:
        raise FileNotFoundError(f"No remote source configured for {config['display_name']}")
    return str(remote_source), False


def _load_with_local_fallback(
    script_dir: Path,
    config: Dict[str, Any],
    load_from_source: Callable[[str], LoadedModelHandle],
) -> LoadedModelHandle:
    source, is_local = _resolve_preferred_source(script_dir, config)
    try:
        return load_from_source(source)
    except Exception as exc:
        if not is_local or config.get("loading_method") != "local_or_huggingface":
            raise
        remote_source = str(config["hf_model_id"])
        print(f"⚠️ Local artifact for {config['display_name']} failed to load: {exc}")
        print(f"   Falling back to Hugging Face source: {remote_source}")
        return load_from_source(remote_source)


def _get_runtime_device(runtime: Any) -> str:
    device = getattr(runtime, "device", None)
    if device is not None:
        return str(device)

    parameters = getattr(runtime, "parameters", None)
    if callable(parameters):
        try:
            first_param = next(parameters())
            return str(first_param.device)
        except StopIteration:
            pass
        except Exception:
            pass

    return "cuda" if torch.cuda.is_available() else "cpu"


def _batch_to_runtime(batch: Any, runtime: Any, use_dtype: bool = False) -> Any:
    if not hasattr(batch, "to"):
        return batch
    device = _get_runtime_device(runtime)
    if use_dtype and hasattr(runtime, "dtype"):
        return batch.to(device, dtype=runtime.dtype)
    return batch.to(device)


def _normalize_text_list(decoded: Any) -> List[str]:
    if isinstance(decoded, list):
        return [str(item).strip() for item in decoded]
    return [str(decoded).strip()]


def _move_runtime_to_device(model: Any, device: str) -> Any:
    runtime = model.runtime if isinstance(model, LoadedModelHandle) else model

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


def _build_loaded_handle(
    model_key: str,
    backend: str,
    runtime: Any,
    processor: Any,
    source: str,
    config: Dict[str, Any],
) -> LoadedModelHandle:
    return LoadedModelHandle(
        model_key=model_key,
        backend=backend,
        runtime=runtime,
        processor=processor,
        source=source,
        config=config,
        supports_timestamps=bool(config.get("supports_timestamps", False)),
        supports_chunking=False,
        default_language=config.get("default_language"),
        warning=config.get("warning"),
    )


def _load_voxtral_backend(model_key: str, config: Dict[str, Any], script_dir: Path) -> LoadedModelHandle:
    AutoProcessor = _require_transformers_attr("AutoProcessor")
    VoxtralForConditionalGeneration = _require_transformers_attr("VoxtralForConditionalGeneration")
    load_kwargs = _prepare_pretrained_kwargs()

    def load_from_source(source: str) -> LoadedModelHandle:
        processor = AutoProcessor.from_pretrained(source)
        runtime = VoxtralForConditionalGeneration.from_pretrained(source, **load_kwargs)
        return _build_loaded_handle(model_key, config["backend"], runtime, processor, source, config)

    return _load_with_local_fallback(script_dir, config, load_from_source)


def _load_granite_backend(model_key: str, config: Dict[str, Any], script_dir: Path) -> LoadedModelHandle:
    AutoProcessor = _require_transformers_attr("AutoProcessor")
    GraniteSpeechForConditionalGeneration = _require_transformers_attr("GraniteSpeechForConditionalGeneration")
    load_kwargs = _prepare_pretrained_kwargs()

    def load_from_source(source: str) -> LoadedModelHandle:
        processor = AutoProcessor.from_pretrained(source)
        runtime = GraniteSpeechForConditionalGeneration.from_pretrained(source, **load_kwargs)
        return _build_loaded_handle(model_key, config["backend"], runtime, processor, source, config)

    return _load_with_local_fallback(script_dir, config, load_from_source)


def _load_cohere_backend(model_key: str, config: Dict[str, Any], script_dir: Path) -> LoadedModelHandle:
    AutoProcessor = _require_transformers_attr("AutoProcessor")
    AutoModelForSpeechSeq2Seq = _require_transformers_attr("AutoModelForSpeechSeq2Seq")
    load_kwargs = _prepare_pretrained_kwargs()
    trust_remote_code = bool(config.get("trust_remote_code", False))

    def load_from_source(source: str) -> LoadedModelHandle:
        print("ℹ️ Cohere adapter uses trust_remote_code=True for the local model package.")
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=trust_remote_code)
        runtime = AutoModelForSpeechSeq2Seq.from_pretrained(
            source,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
        return _build_loaded_handle(model_key, config["backend"], runtime, processor, source, config)

    return _load_with_local_fallback(script_dir, config, load_from_source)


def _load_transformers_backend(model_key: str, config: Dict[str, Any], script_dir: Path) -> LoadedModelHandle:
    backend = config["backend"]
    if backend == "transformers_voxtral":
        return _load_voxtral_backend(model_key, config, script_dir)
    if backend == "transformers_granite":
        return _load_granite_backend(model_key, config, script_dir)
    if backend == "transformers_cohere":
        return _load_cohere_backend(model_key, config, script_dir)
    raise RuntimeError(f"Unsupported Transformers backend: {backend}")


def get_model_key_from_choice(choice_text: str) -> str:
    for model_key in MODEL_DISPLAY_ORDER:
        if MODEL_CONFIGS[model_key]["choice_label"] == choice_text:
            return model_key
    return DEFAULT_MODEL_KEY


def get_model_choice_labels() -> List[str]:
    return [MODEL_CONFIGS[key]["choice_label"] for key in MODEL_DISPLAY_ORDER]


def get_default_model_choice() -> str:
    return MODEL_CONFIGS[DEFAULT_MODEL_KEY]["choice_label"]


def setup_gpu_optimizations() -> None:
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _unload_cached_models(model_name: str) -> None:
    for old_model_key in list(models_cache.keys()):
        if old_model_key == model_name:
            continue
        try:
            old_model = models_cache[old_model_key]
            print(f"🔄 Unloading {old_model_key} to free memory for {model_name}...")
            _move_runtime_to_device(old_model, "cpu")
            del models_cache[old_model_key]
            del old_model
        except Exception as exc:
            print(f"   ⚠️ Failed to unload {old_model_key}: {exc}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_all_models() -> str:
    if not models_cache:
        return "ℹ️ No Transformers models are currently loaded"

    initial_vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    unloaded: List[str] = []

    for model_key in list(models_cache.keys()):
        try:
            model = models_cache[model_key]
            _move_runtime_to_device(model, "cpu")
            del models_cache[model_key]
            del model
            unloaded.append(model_key)
        except Exception as exc:
            print(f"⚠️ Failed to unload {model_key}: {exc}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_vram = torch.cuda.memory_allocated() / 1024**3
    else:
        final_vram = 0.0

    freed = max(initial_vram - final_vram, 0.0)
    if not unloaded:
        return "⚠️ No models were successfully unloaded"
    return f"✅ Unloaded: {', '.join(unloaded)}\n💾 Freed ~{freed:.1f} GB VRAM"


def set_auto_unload(enabled: bool) -> str:
    global auto_unload_after_transcription
    auto_unload_after_transcription = enabled
    return f"Auto-unload {'enabled' if enabled else 'disabled'}"


def load_model(model_name: str, show_progress: bool = False) -> LoadedModelHandle:
    del show_progress

    if model_name in DEFERRED_MODELS:
        raise NotImplementedError(DEFERRED_MODELS[model_name]["reason"])
    if model_name not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model key: {model_name}")
    if model_name in models_cache:
        print(f"✅ Reusing cached model: {model_name}")
        return models_cache[model_name]

    _unload_cached_models(model_name)
    setup_gpu_optimizations()

    config = MODEL_CONFIGS[model_name]
    script_dir = get_script_dir()

    print(f"📦 Loading {config['display_name']}")
    print(f"   Backend: {config['backend']}")
    print(f"   Preferred local path: {config['local_path']}")
    if config.get("warning"):
        print(f"   ⚠️ {config['warning']}")

    handle = _load_transformers_backend(model_name, config, script_dir)
    models_cache[model_name] = handle
    return handle


def _load_model_for_transcription(model_key: str) -> Tuple[Optional[LoadedModelHandle], Optional[Tuple[Any, ...]]]:
    try:
        return load_model(model_key), None
    except PermissionError as exc:
        error_type = "permission_file_lock" if ("WinError 32" in str(exc) or "being used by another process" in str(exc)) else "permission"
        return None, _make_error_response(error_type, str(exc), log_capture)
    except ConnectionError as exc:
        return None, _make_error_response("network", str(exc), log_capture)
    except FileNotFoundError as exc:
        return None, _make_error_response("file_not_found", str(exc), log_capture)
    except OSError as exc:
        return None, _make_error_response("filesystem", str(exc), log_capture)
    except (ImportError, NotImplementedError, RuntimeError, ValueError, KeyError) as exc:
        return None, _make_error_response("runtime", str(exc), log_capture)
    except Exception as exc:
        detail = f"Type: {type(exc).__name__}\nMessage: {exc}"
        return None, _make_error_response("generic", detail, log_capture)


def _transcribe_transformers_granite(handle: LoadedModelHandle, files: List[str]) -> List[Any]:
    assert handle.processor is not None
    audio_arrays = [load_audio_to_numpy(file_path, target_sr=16000)[0] for file_path in files]
    batch_audio: Any = audio_arrays if len(audio_arrays) > 1 else audio_arrays[0]
    inputs = handle.processor(audio=batch_audio, return_tensors="pt", padding=len(audio_arrays) > 1)
    inputs = _batch_to_runtime(inputs, handle.runtime)
    generated_ids = handle.runtime.generate(
        **inputs,
        max_new_tokens=int(handle.config.get("max_new_tokens", 256)),
        do_sample=False,
    )
    texts = handle.processor.batch_decode(generated_ids, skip_special_tokens=True)
    return [SimpleHypothesis(text=text) for text in _normalize_text_list(texts)]


def _transcribe_transformers_cohere(handle: LoadedModelHandle, files: List[str]) -> List[Any]:
    assert handle.processor is not None
    audio_arrays = [load_audio_to_numpy(file_path, target_sr=16000)[0] for file_path in files]
    batch_audio: Any = audio_arrays if len(audio_arrays) > 1 else audio_arrays[0]
    language = str(handle.default_language or "en")
    inputs = handle.processor(
        batch_audio,
        sampling_rate=16000,
        return_tensors="pt",
        language=language,
        punctuation=True,
    )
    audio_chunk_index = inputs.get("audio_chunk_index") if hasattr(inputs, "get") else None
    inputs = _batch_to_runtime(inputs, handle.runtime, use_dtype=True)
    generated_ids = handle.runtime.generate(
        **inputs,
        max_new_tokens=int(handle.config.get("max_new_tokens", 256)),
    )
    decoded = handle.processor.decode(
        generated_ids,
        skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index,
        language=language,
    )
    return [SimpleHypothesis(text=text) for text in _normalize_text_list(decoded)]


def _transcribe_transformers_voxtral(handle: LoadedModelHandle, files: List[str]) -> List[Any]:
    assert handle.processor is not None
    results: List[Any] = []
    for file_path in files:
        request_kwargs: Dict[str, Any] = {
            "audio": file_path,
            "model_id": handle.config["hf_model_id"],
        }
        if handle.default_language:
            request_kwargs["language"] = handle.default_language
        inputs = handle.processor.apply_transcription_request(**request_kwargs)
        inputs = _batch_to_runtime(inputs, handle.runtime, use_dtype=True)
        generated_ids = handle.runtime.generate(
            **inputs,
            max_new_tokens=int(handle.config.get("max_new_tokens", 500)),
            do_sample=False,
            temperature=0.0,
        )
        prompt_tokens = inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else 0
        decoded = handle.processor.batch_decode(generated_ids[:, prompt_tokens:], skip_special_tokens=True)
        texts = _normalize_text_list(decoded)
        results.append(SimpleHypothesis(text=texts[0] if texts else ""))
    return results


def _iter_file_batches(files: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(files), batch_size):
        yield files[start:start + batch_size]


def _transcribe_backend_files(handle: LoadedModelHandle, files: List[str], requested_batch_size: int) -> Tuple[List[Any], int]:
    backend = handle.backend
    max_batch_size = max(1, int(handle.config.get("max_batch_size", 1)))
    effective_batch_size = max(1, min(int(requested_batch_size), max_batch_size))

    if backend == "transformers_voxtral":
        effective_batch_size = 1

    all_results: List[Any] = []
    total_batches = (len(files) + effective_batch_size - 1) // effective_batch_size
    for batch_index, batch_files in enumerate(_iter_file_batches(files, effective_batch_size), start=1):
        print(f"🧠 Running {backend} batch {batch_index}/{total_batches} with {len(batch_files)} file(s)")

        if backend == "transformers_granite":
            batch_results = _transcribe_transformers_granite(handle, batch_files)
        elif backend == "transformers_cohere":
            batch_results = _transcribe_transformers_cohere(handle, batch_files)
        elif backend == "transformers_voxtral":
            batch_results = _transcribe_transformers_voxtral(handle, batch_files)
        else:
            raise NotImplementedError(f"Unsupported backend: {backend}")

        all_results.extend(batch_results)

    return all_results, effective_batch_size


def _run_transcription(
    model: LoadedModelHandle,
    processed_files: List[str],
    batch_size: int,
) -> Tuple[Optional[List[Any]], int, Optional[Tuple[Any, ...]]]:
    try:
        result, effective_batch_size = _transcribe_backend_files(model, processed_files, batch_size)
        return result, effective_batch_size, None
    except PermissionError as exc:
        error_type = "transcription_file_lock" if ("WinError 32" in str(exc) or "being used by another process" in str(exc)) else "permission"
        return None, 0, _make_error_response(error_type, str(exc), log_capture)
    except OSError as exc:
        return None, 0, _make_error_response("filesystem", str(exc), log_capture)
    except Exception as exc:
        return None, 0, _make_error_response("transcription", str(exc), log_capture)


def _get_gpu_stats() -> Tuple[float, str]:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.get_device_name(0)
    return 0.0, "CPU"


def _format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"


def _format_model_notes(config: Dict[str, Any]) -> str:
    notes = [
        f"- **Timestamps**: {'Available' if config.get('supports_timestamps') else 'Not currently available from this adapter'}",
        f"- **Batch Requests**: {'Multiple files per adapter call' if config.get('supports_batching') else 'One file at a time'}",
        f"- **Summary**: {config['summary']}",
    ]
    if config.get("warning"):
        notes.append(f"- **Warning**: {config['warning']}")
    if config.get("trust_remote_code"):
        notes.append("- **Remote Code**: Cohere is loaded with trust_remote_code=True from the selected model source")
    return "\n".join(notes)


def _build_batch_preview(file_info: List[Dict[str, Any]], transcriptions: List[str]) -> str:
    sections: List[str] = []
    for index, (info, transcription) in enumerate(zip(file_info, transcriptions), start=1):
        sections.append(
            f"{SEPARATOR}\nFILE {index}: {info['name']}\n{SEPARATOR}\n\n{transcription}"
        )
    return "\n\n".join(sections)


def _generate_output_files(save_to_file: bool, config: OutputFilesConfig) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    if not save_to_file:
        return None, None, None, ""

    if config.is_batch:
        base_filename = f"batch_transcription_{len(config.file_list)}_files"
    else:
        base_filename = f"{Path(config.file_list[0]).stem}_transcription"

    txt_file, srt_file, csv_file = save_output_files(base_filename, config)
    return txt_file, srt_file, csv_file, f"\n\n💾 **Files saved**: `{base_filename}.[txt/srt/csv]`"


def _format_single_status(
    info: Dict[str, Any],
    model_choice: str,
    config: Dict[str, Any],
    gpu_name: str,
    load_time: float,
    inference_time: float,
    total_time: float,
    rtfx: float,
    vram_used: float,
    transcription: str,
    effective_batch_size: int,
    video_count: int,
) -> str:
    video_status = f"🎬 Extracted audio from {video_count} video file(s)\n\n" if video_count > 0 else ""
    return f"""
### ✅ Transcription Complete!

{video_status}**📊 Run Summary:**
- **Model**: {model_choice}
- **GPU**: {gpu_name}
- **Audio Duration**: {_format_duration(info['duration'])}
- **Processing Time**: {total_time:.2f} seconds
- **Inference Time**: {inference_time:.2f} seconds
- **Model Load Time**: {load_time:.2f} seconds
- **Adapter Batch Size**: {effective_batch_size} file(s) per request
- **Real-Time Factor**: {rtfx:.1f}×
- **VRAM Used**: {vram_used:.2f} GB
- **Transcription Length**: {len(transcription)} characters ({len(transcription.split())} words)
- **Exports**: TXT, SRT, CSV available

**Capability Notes:**
- {TIMESTAMP_EXPORT_NOTE}
{_format_model_notes(config)}
"""


def _format_batch_status(
    file_info: List[Dict[str, Any]],
    model_choice: str,
    config: Dict[str, Any],
    gpu_name: str,
    total_duration: float,
    inference_time: float,
    total_time: float,
    rtfx: float,
    vram_used: float,
    effective_batch_size: int,
    transcriptions: List[str],
    errors: List[str],
    video_count: int,
) -> str:
    success_count = len(file_info) - len(errors)
    word_total = sum(len(item.split()) for item in transcriptions if not item.startswith("[Transcription failed:"))
    file_lines: List[str] = []
    for info, transcription in zip(file_info, transcriptions):
        if transcription.startswith("[Transcription failed:"):
            file_lines.append(f"- **{info['name']}**: failed")
        else:
            file_lines.append(
                f"- **{info['name']}**: {_format_duration(info['duration'])}, {len(transcription.split())} words"
            )

    error_block = ""
    if errors:
        error_block = "\n\n**Failed files:**\n" + "\n".join(f"- {item}" for item in errors)

    video_status = f"🎬 Extracted audio from {video_count} video file(s)\n\n" if video_count > 0 else ""
    file_summary = "\n".join(file_lines)
    return f"""
### ✅ Batch Transcription Complete!

{video_status}**📊 Overall Summary:**
- **Files Processed**: {len(file_info)} ({success_count} successful, {len(errors)} failed)
- **Model**: {model_choice}
- **GPU**: {gpu_name}
- **Total Audio Duration**: {_format_duration(total_duration)}
- **Processing Time**: {total_time:.2f} seconds
- **Inference Time**: {inference_time:.2f} seconds
- **Adapter Batch Size**: {effective_batch_size} file(s) per request
- **Real-Time Factor**: {rtfx:.1f}×
- **VRAM Used**: {vram_used:.2f} GB
- **Total Words**: {word_total}
- **Exports**: TXT, SRT, CSV available

**Capability Notes:**
- {TIMESTAMP_EXPORT_NOTE}
{_format_model_notes(config)}

**Per-file summary:**
{file_summary}{error_block}
"""


def transcribe_audio(
    audio_files: Any,
    model_choice: str,
    save_to_file: bool,
    batch_size: int,
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    log_capture.start()
    print(f"\n{'=' * 60}")
    print(f"🎙️ Transformers transcription started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    file_list = normalize_file_list(audio_files)
    if not file_list:
        log_capture.stop()
        return "⚠️ Please upload an audio or video file first", "", None, None, None, None

    try:
        model_key = get_model_key_from_choice(model_choice)
        config = MODEL_CONFIGS[model_key]
        start_time = time.time()

        print(f"📁 Files: {len(file_list)}")
        print(f"📊 Model: {model_choice}")
        print(f"📦 Requested adapter batch size: {batch_size}")

        model, error_response = _load_model_for_transcription(model_key)
        if error_response is not None:
            return error_response
        assert model is not None

        load_time = time.time() - start_time
        processed_files, file_info, total_duration, video_count = process_audio_files(file_list)

        inference_start = time.time()
        result, effective_batch_size, error_response = _run_transcription(model, processed_files, batch_size)
        if error_response is not None:
            return error_response
        assert result is not None

        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        vram_used, gpu_name = _get_gpu_stats()
        rtfx = total_duration / inference_time if inference_time > 0 else 0.0

        print(f"✅ Transcription finished in {inference_time:.2f}s for {total_duration:.2f}s of audio")

        is_batch = len(file_info) > 1
        if is_batch:
            all_transcriptions: List[str] = []
            all_timestamps: List[Tuple[List[Dict[str, Any]], str]] = []
            per_file_errors: List[str] = []

            for index, info in enumerate(file_info):
                success, transcription, error_msg = validate_transcription_result(result, index)
                if success:
                    all_transcriptions.append(transcription)
                    all_timestamps.append(([], "none"))
                else:
                    all_transcriptions.append(f"[Transcription failed: {error_msg}]")
                    all_timestamps.append(([], "none"))
                    per_file_errors.append(f"{info['name']}: {error_msg}")

            status = _format_batch_status(
                file_info=file_info,
                model_choice=model_choice,
                config=config,
                gpu_name=gpu_name,
                total_duration=total_duration,
                inference_time=inference_time,
                total_time=total_time,
                rtfx=rtfx,
                vram_used=vram_used,
                effective_batch_size=effective_batch_size,
                transcriptions=all_transcriptions,
                errors=per_file_errors,
                video_count=video_count,
            )
            transcription_output = _build_batch_preview(file_info, all_transcriptions)
            output_config = OutputFilesConfig(
                file_list=file_list,
                file_info=file_info,
                is_batch=True,
                include_timestamps=False,
                model_choice=model_choice,
                total_duration=total_duration,
                total_time=total_time,
                apply_itn=False,
                all_transcriptions=all_transcriptions,
                all_timestamps=all_timestamps,
            )
        else:
            success, transcription, error_msg = validate_transcription_result(result, 0)
            if not success:
                return _make_error_response("validation", error_msg, log_capture)

            info = file_info[0]
            status = _format_single_status(
                info=info,
                model_choice=model_choice,
                config=config,
                gpu_name=gpu_name,
                load_time=load_time,
                inference_time=inference_time,
                total_time=total_time,
                rtfx=rtfx,
                vram_used=vram_used,
                transcription=transcription,
                effective_batch_size=effective_batch_size,
                video_count=video_count,
            )
            transcription_output = transcription
            output_config = OutputFilesConfig(
                file_list=file_list,
                file_info=file_info,
                is_batch=False,
                include_timestamps=False,
                model_choice=model_choice,
                total_duration=total_duration,
                total_time=total_time,
                apply_itn=False,
                transcription=transcription,
                timestamps=[],
                timestamp_level="none",
            )

        txt_file, srt_file, csv_file, status_suffix = _generate_output_files(save_to_file, output_config)
        status = status + status_suffix

        if auto_unload_after_transcription:
            print("🗑️ Auto-unloading models after transcription...")
            print(unload_all_models())

        logs = log_capture.stop()
        log_file = save_logs(logs, "transcription")
        return status, transcription_output, txt_file, srt_file, csv_file, log_file

    except Exception as exc:
        vram_info = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB detected"
            if torch.cuda.is_available()
            else "no CUDA GPU detected"
        )
        error_msg = f"""
### ❌ Error During Transcription

**Error Type**: {type(exc).__name__}

**Error Message**: {exc}

**Troubleshooting:**
1. Check that the selected model is actually supported by this Transformers app.
2. Confirm you have enough VRAM or system RAM ({vram_info}).
3. Try a shorter file or a smaller model such as Granite 1B.
4. For Cohere, confirm the local package or HF source is trusted and accessible.
5. For Voxtral Small 24B, expect heavy hardware pressure and possible offloading.
"""
        logs = log_capture.stop()
        log_file = save_logs(logs, "error")
        return error_msg, "", None, None, None, log_file


def _check_model_local_availability(script_dir: Path, config: Dict[str, Any]) -> str:
    local_path = script_dir / config["local_path"]
    if _artifact_exists(local_path):
        return f"- ✅ {config['display_name']}: local artifact present at {local_path.name}"
    return f"- 📥 {config['display_name']}: will use Hugging Face on first load if access is available"


def validate_local_models() -> None:
    script_dir = get_script_dir()
    print("\n" + "=" * 80)
    print("📦 Transformers App Model Availability")
    print("=" * 80)
    for model_key in MODEL_DISPLAY_ORDER:
        print(_check_model_local_availability(script_dir, MODEL_CONFIGS[model_key]))
    print("\nDeferred models:")
    for item in DEFERRED_MODELS.values():
        print(f"- {item['display_name']}: {item['reason']}")
    print("=" * 80 + "\n")


def get_system_info() -> str:
    transformers_ready = _dependency_is_available("transformers")
    voxtral_runtime_ready = _transformers_attr_available("VoxtralForConditionalGeneration")
    granite_runtime_ready = _transformers_attr_available("GraniteSpeechForConditionalGeneration")
    cohere_runtime_ready = _transformers_attr_available("AutoModelForSpeechSeq2Seq")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        cuda_version = torch.version.cuda
        pytorch_version = torch.__version__
        return f"""
### 🖥️ System Information

**GPU**: {gpu_name}
**Total VRAM**: {total_vram:.1f} GB
**Available VRAM**: {free_vram:.1f} GB
**CUDA Version**: {cuda_version}
**PyTorch Version**: {pytorch_version}
**Transformers Runtime**: {'✅ Ready' if transformers_ready else '⚠️ Missing'}
**Voxtral Class Support**: {'✅ Ready' if voxtral_runtime_ready else '⚠️ Missing in current Transformers build'}
**Granite Class Support**: {'✅ Ready' if granite_runtime_ready else '⚠️ Missing in current Transformers build'}
**Cohere Seq2Seq Support**: {'✅ Ready' if cohere_runtime_ready else '⚠️ Missing in current Transformers build'}
**Supported Models in This App**: 4
**Deferred Models**: 2 (Qwen, Voxtral Realtime)
"""

    return f"""
### ⚠️ No CUDA GPU Detected

This Transformers app can still import, but these models are intended for GPU-backed inference.

**Transformers Runtime**: {'✅ Ready' if transformers_ready else '⚠️ Missing'}
**Voxtral Class Support**: {'✅ Ready' if voxtral_runtime_ready else '⚠️ Missing'}
**Granite Class Support**: {'✅ Ready' if granite_runtime_ready else '⚠️ Missing'}
**Cohere Seq2Seq Support**: {'✅ Ready' if cohere_runtime_ready else '⚠️ Missing'}
**Supported Models in This App**: 4
**Deferred Models**: 2 (Qwen, Voxtral Realtime)
"""


def get_privacy_performance_info() -> str:
    if torch.cuda.is_available():
        gpu_info = f"**GPU**: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB VRAM)"
    else:
        gpu_info = "**GPU**: CPU-only fallback detected"

    return f"""
### Privacy
- ✅ All inference stays local to your machine once artifacts are available.
- ✅ Local artifacts under `local_models/` are preferred before any HF fallback.
- ✅ Model downloads and extraction use the project cache under `{CACHE_DIR}`.

### Performance and Scope
- {gpu_info}
- ✅ Batch uploads are supported.
- ✅ Audio and video files are supported.
- ⚠️ Phase 1 scope is strict Transformers only: Voxtral, Granite, and Cohere.
- ⚠️ Qwen is intentionally excluded from this app.
- ⚠️ Voxtral Mini 4B Realtime remains unavailable in the current Transformers runtime.

### Export Honesty
- {TIMESTAMP_EXPORT_NOTE}
- Cohere uses trust_remote_code from its local package or HF source.
- Voxtral Small 24B needs workstation-class VRAM to be practical.
"""


def get_model_details(choice_text: str) -> str:
    model_key = get_model_key_from_choice(choice_text)
    config = MODEL_CONFIGS[model_key]
    warning_block = f"\n- **Warning**: {config['warning']}" if config.get("warning") else ""
    trust_block = "\n- **Remote Code**: trust_remote_code=True is required for this Cohere adapter" if config.get("trust_remote_code") else ""
    batching_note = "Multiple files per adapter request" if config.get("supports_batching") else "One file per adapter request"

    return f"""
### 📖 Selected Model

- **Display Name**: {config['display_name']}
- **Architecture**: {config['architecture']}
- **Parameters**: {config['parameters']}
- **Languages**: {config['languages']}
- **Estimated VRAM**: {config['vram_gb']}
- **Recommended For**: {config['recommended_for']}
- **Timestamp Support**: Not currently wired in this app
- **Adapter Batching**: {batching_note}
- **Local Artifact Path**: {config['local_path']}
- **Summary**: {config['summary']}{warning_block}{trust_block}
"""


DEFERRED_MODELS_MARKDOWN = """
### Deferred or Unavailable in This App

- **Qwen/Qwen3-ASR-1.7B**: intentionally excluded from this phase-1 Transformers app because it depends on the separate qwen-asr runtime path.
- **mistralai/Voxtral-Mini-4B-Realtime-2602**: explicitly unavailable here because the required realtime runtime is not exposed by the current Transformers build.
"""


with gr.Blocks(title="Transformers Local ASR") as app:
    gr.Markdown(
        """
    # 🎙️ Local Offline ASR Transcription: Transformers App
    ### Strict Transformers phase 1 for Voxtral, Granite, and Cohere

    This entrypoint keeps the non-NeMo Transformers models separate from the legacy mixed app. It supports local artifacts when present and falls back to Hugging Face only when the configured source is available.
    """
    )

    with gr.Accordion("📊 System Information", open=False):
        system_info = gr.Markdown(get_system_info())
        refresh_btn = gr.Button("🔄 Refresh System Info", size="sm")
        refresh_btn.click(fn=get_system_info, outputs=system_info)

    with gr.Accordion("💾 Loaded Model Management", open=False):
        gr.Markdown(
            """
        Keep the currently loaded Transformers model in memory for faster repeat runs, or unload it to free VRAM for other work.
        """
        )
        with gr.Row():
            unload_btn = gr.Button("🗑️ Unload All Models", size="sm", variant="secondary")
            auto_unload_checkbox = gr.Checkbox(
                label="Auto-unload after transcription",
                value=False,
                info="Free model memory automatically after each run",
            )
        unload_status = gr.Markdown("")
        unload_btn.click(fn=unload_all_models, outputs=unload_status)
        auto_unload_checkbox.change(fn=set_auto_unload, inputs=auto_unload_checkbox, outputs=unload_status)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Audio or Video Files")
            audio_input = gr.File(
                file_count="multiple",
                file_types=[
                    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma",
                    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".m4v",
                ],
                label="Audio/Video Files",
            )

            gr.Markdown(
                """
            **Supported audio formats**: WAV, MP3, FLAC, M4A, OGG, AAC, WMA

            **Supported video formats**: MP4, AVI, MKV, MOV, WEBM, FLV, M4V
            *(Audio is extracted automatically for transcription.)*

            Batch uploads are supported.
            """
            )

            gr.Markdown("### ⚙️ Settings")
            model_selector = gr.Radio(
                choices=get_model_choice_labels(),
                value=get_default_model_choice(),
                label="Model Selection",
                info="Only models with active Transformers adapters are listed here.",
            )

            model_details = gr.Markdown(get_model_details(get_default_model_choice()))

            save_checkbox = gr.Checkbox(
                label="💾 Create TXT, SRT, and CSV exports",
                value=True,
                info="Exports are written to the current working directory and exposed for download below.",
            )

            batch_size_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=2,
                step=1,
                label="📦 Adapter Batch Size",
                info="Granite and Cohere can batch multiple files per request. Voxtral remains one file per request.",
            )

            gr.Markdown(
                f"""
            ### ⏱️ Timestamp and Export Notes

            - {TIMESTAMP_EXPORT_NOTE}
            - TXT, SRT, and CSV downloads still work for every supported model.
            - Cohere keeps the trust_remote_code requirement explicit.
            - Voxtral Small 24B keeps its hardware warning explicit.
            """
            )

            transcribe_btn = gr.Button("🚀 Start Transcription", variant="primary", size="lg")

            with gr.Accordion("🧭 Deferred Models", open=False):
                gr.Markdown(DEFERRED_MODELS_MARKDOWN)

        with gr.Column(scale=2):
            gr.Markdown("### 📝 Transcription Results")
            status_output = gr.Markdown("Upload files and click 'Start Transcription' to begin.")
            transcription_output = gr.Textbox(
                label="Transcription Preview",
                lines=20,
                placeholder="Transcription text will appear here.",
                buttons=["copy"],
                show_label=True,
            )

            gr.Markdown("### 📥 Download Files")
            with gr.Row():
                txt_file_output = gr.File(label="📄 TXT", visible=True)
                srt_file_output = gr.File(label="🎬 SRT", visible=True)
                csv_file_output = gr.File(label="📊 CSV", visible=True)

            log_file_output = gr.File(label="📋 Processing Logs", visible=True)

    gr.Markdown("---")

    with gr.Accordion("❓ How to Use", open=False):
        gr.Markdown(
            """
        1. Upload one or more audio/video files.
        2. Pick a supported Transformers model.
        3. Review the model-specific notes before starting.
        4. Start transcription and download TXT/SRT/CSV plus the processing log if needed.

        **Scope reminders**
        - This app does not include Qwen in phase 1.
        - Voxtral Mini 4B Realtime is intentionally listed only as deferred, not supported.
        - Long-form chunked timestamp extraction is not wired for these adapters yet.
        """
        )

    with gr.Accordion("🔒 Privacy and Performance", open=False):
        gr.Markdown(get_privacy_performance_info())

    model_selector.change(fn=get_model_details, inputs=model_selector, outputs=model_details)
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, model_selector, save_checkbox, batch_size_slider],
        outputs=[status_output, transcription_output, txt_file_output, srt_file_output, csv_file_output, log_file_output],
        queue=False,
    )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 Starting Transformers transcription interface")
    print("=" * 80)
    print(f"\n📁 Cache Directory: {CACHE_DIR}")
    print("   (Used for model downloads and extraction; configured for Windows-safe temp handling)")

    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"✅ CUDA: {torch.version.cuda}")
        setup_gpu_optimizations()
    else:
        print("\n⚠️ No CUDA GPU detected")

    validate_local_models()
    print("🌐 Opening in browser at: http://127.0.0.1:7860")
    print("💡 Keep this terminal open while using the interface")
    print("🛑 Press Ctrl+C to stop\n")

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )