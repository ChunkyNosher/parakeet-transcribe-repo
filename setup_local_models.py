#!/usr/bin/env python3
"""Download or prepare the requested local ASR model artifacts.

This script supports two local artifact strategies:
1. Save NeMo models as `.nemo` files.
2. Snapshot Hugging Face model repositories into `local_models/`.

The resulting artifacts can be used by transcribe_ui.py when the matching
backend runtime is available in the configured Python environment.

For the split-app workflow, prefer setup_local_models_nemo.py or
setup_local_models_transformers.py. This script remains the legacy mixed-app
artifact manager for transcribe_ui.py.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict


def _require_nemo_asr() -> Any:
    """Load NeMo only when a NeMo-backed model is requested."""
    try:
        return importlib.import_module("nemo.collections.asr")
    except Exception as exc:
        raise ImportError(
            "Missing dependency 'nemo-toolkit[asr]'. "
            "Use the configured project environment and install the repo requirements. "
            f"Original error: {exc}"
        ) from exc


def _require_snapshot_download() -> Any:
    """Load Hugging Face snapshot_download only when needed."""
    try:
        return importlib.import_module("huggingface_hub").snapshot_download
    except Exception as exc:
        raise ImportError(
            "Missing dependency 'huggingface_hub'. "
            "Install the repo requirements in the configured Python environment. "
            f"Original error: {exc}"
        ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = SCRIPT_DIR / "local_models"

MODELS_TO_DOWNLOAD: Dict[str, Dict[str, Any]] = {
    "1": {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "artifact_name": "parakeet-0.6b-v3.nemo",
        "artifact_kind": "file",
        "download_strategy": "nemo_save",
        "display_name": "NVIDIA Parakeet 0.6B-v3",
        "download_size": "~1.2 GB",
        "saved_size": "~2.4 GB",
        "min_size_gb": 1.5,
        "description": "NeMo export with multilingual ASR and timestamps",
        "recommended": True,
    },
    "2": {
        "model_id": "mistralai/Voxtral-Small-24B-2507",
        "artifact_name": "Voxtral-Small-24B-2507",
        "artifact_kind": "directory",
        "download_strategy": "snapshot",
        "display_name": "Mistral Voxtral Small 24B 2507",
        "download_size": "~45+ GB",
        "saved_size": "~45+ GB",
        "description": "Offline Voxtral transcription snapshot",
    },
    "3": {
        "model_id": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "artifact_name": "Voxtral-Mini-4B-Realtime-2602",
        "artifact_kind": "directory",
        "download_strategy": "snapshot",
        "display_name": "Mistral Voxtral Mini 4B Realtime 2602",
        "download_size": "~8+ GB",
        "saved_size": "~8+ GB",
        "description": "Realtime Voxtral snapshot; runtime support depends on local Transformers version",
    },
    "4": {
        "model_id": "mistralai/Voxtral-Mini-3B-2507",
        "artifact_name": "Voxtral-Mini-3B-2507",
        "artifact_kind": "directory",
        "download_strategy": "snapshot",
        "display_name": "Mistral Voxtral Mini 3B 2507",
        "download_size": "~6+ GB",
        "saved_size": "~6+ GB",
        "description": "Offline Voxtral mini snapshot",
    },
    "5": {
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "artifact_name": "Qwen3-ASR-1.7B",
        "artifact_kind": "directory",
        "download_strategy": "snapshot",
        "display_name": "Qwen Qwen3-ASR 1.7B",
        "download_size": "~4+ GB",
        "saved_size": "~4+ GB",
        "description": "Qwen ASR runtime snapshot",
    },
    "6": {
        "model_id": "CohereLabs/cohere-transcribe-03-2026",
        "artifact_name": "cohere-transcribe-03-2026",
        "artifact_kind": "directory",
        "download_strategy": "snapshot",
        "display_name": "Cohere Transcribe 03-2026",
        "download_size": "~6+ GB",
        "saved_size": "~6+ GB",
        "description": "Transformers model with trusted remote code",
    },
    "7": {
        "model_id": "ibm-granite/granite-4.0-1b-speech",
        "artifact_name": "granite-4.0-1b-speech",
        "artifact_kind": "directory",
        "download_strategy": "snapshot",
        "display_name": "IBM Granite 4.0 1B Speech",
        "download_size": "~3+ GB",
        "saved_size": "~3+ GB",
        "description": "Granite speech transcription snapshot",
    },
}

STATUS_OPTION = str(len(MODELS_TO_DOWNLOAD) + 1)
BATCH_OPTION = str(len(MODELS_TO_DOWNLOAD) + 2)


def create_local_models_directory() -> None:
    """Create local_models directory if it doesn't exist."""
    if not LOCAL_MODELS_DIR.exists():
        LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created {LOCAL_MODELS_DIR}")
    else:
        print(f"local_models directory already exists: {LOCAL_MODELS_DIR}")


def _artifact_path(model: Dict[str, Any]) -> Path:
    return LOCAL_MODELS_DIR / model["artifact_name"]


def _path_size_gb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 ** 3)
    total_bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return total_bytes / (1024 ** 3)


def _is_artifact_valid(path: Path, model: Dict[str, Any]) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return _path_size_gb(path) >= float(model.get("min_size_gb", 0.1))
    return any(path.rglob("*"))


def get_model_status() -> Dict[str, Dict[str, Any]]:
    """Check which requested model artifacts are already stored locally."""
    status: Dict[str, Dict[str, Any]] = {}
    for choice, model in MODELS_TO_DOWNLOAD.items():
        path = _artifact_path(model)
        exists = path.exists()
        size_gb = _path_size_gb(path) if exists else 0.0
        status[choice] = {
            "exists": exists,
            "size_gb": size_gb,
            "valid": _is_artifact_valid(path, model),
            "path": str(path),
        }
    return status


def display_model_status() -> None:
    """Display current status of all configured model artifacts."""
    print("\n" + "=" * 80)
    print("CURRENT MODEL STATUS")
    print("=" * 80)

    status = get_model_status()
    total_size = 0.0
    downloaded_count = 0

    for choice, model in MODELS_TO_DOWNLOAD.items():
        artifact_status = status[choice]
        print(f"\n  {choice}. {model['display_name']}")
        if artifact_status["exists"]:
            downloaded_count += 1
            total_size += artifact_status["size_gb"]
            validity = "OK" if artifact_status["valid"] else "Check contents"
            print(f"     Present ({artifact_status['size_gb']:.2f} GB, {validity})")
            print(f"     Path: {artifact_status['path']}")
        else:
            print("     Not downloaded")
            print(f"     Size: {model['download_size']} -> {model['saved_size']}")

    print("\n" + "-" * 40)
    print(f"Total: {downloaded_count}/{len(MODELS_TO_DOWNLOAD)} artifacts ({total_size:.2f} GB)")
    print("=" * 80 + "\n")


def _download_nemo_model(model: Dict[str, Any], output_path: Path) -> bool:
    nemo_asr = _require_nemo_asr()
    print("   Downloading NeMo model from Hugging Face...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model["model_id"])
    print(f"   Saving NeMo artifact to {output_path}...")
    asr_model.save_to(output_path)
    del asr_model
    return output_path.exists() and _is_artifact_valid(output_path, model)


def _download_snapshot_model(model: Dict[str, Any], output_path: Path) -> bool:
    snapshot_download = _require_snapshot_download()
    print("   Downloading repository snapshot from Hugging Face...")
    snapshot_download(
        repo_id=model["model_id"],
        local_dir=str(output_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return output_path.exists() and _is_artifact_valid(output_path, model)


def download_and_save_model(choice: str) -> bool:
    """Download or prepare a single local model artifact."""
    if choice not in MODELS_TO_DOWNLOAD:
        print(f"Invalid choice: {choice}")
        return False

    model = MODELS_TO_DOWNLOAD[choice]
    output_path = _artifact_path(model).resolve()

    print("\n" + "=" * 80)
    print(f"PREPARING: {model['display_name']}")
    print("=" * 80)
    print(f"   Model ID: {model['model_id']}")
    print(f"   Output: {output_path}")
    print(f"   Download size: {model['download_size']}")
    print(f"   Saved size: {model['saved_size']}")
    print(f"   Description: {model['description']}")

    try:
        strategy = model["download_strategy"]
        if strategy == "nemo_save":
            success = _download_nemo_model(model, output_path)
        elif strategy == "snapshot":
            success = _download_snapshot_model(model, output_path)
        else:
            raise RuntimeError(f"Unsupported download strategy: {strategy}")

        if not success:
            print(f"\nArtifact check failed for {model['display_name']}")
            return False

        size_gb = _path_size_gb(output_path)
        print(f"\nReady: {model['display_name']} ({size_gb:.2f} GB)")
        return True
    except Exception as exc:
        print(f"\nError preparing {model['display_name']}:")
        print(f"   {type(exc).__name__}: {exc}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def download_all_models() -> None:
    """Download all requested model artifacts in sequence."""
    print("\n" + "=" * 80)
    print("BATCH DOWNLOAD: ALL REQUESTED MODELS")
    print("=" * 80)
    print(f"\nThis will prepare all {len(MODELS_TO_DOWNLOAD)} requested model artifacts.")
    print("Disk requirements vary widely; Voxtral Small is the largest download.")

    confirm = input("\nProceed with batch download? (y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("\nBatch download cancelled")
        return

    results: Dict[str, bool] = {}
    for choice in MODELS_TO_DOWNLOAD:
        results[choice] = download_and_save_model(choice)

    print("\n" + "=" * 80)
    print("BATCH DOWNLOAD SUMMARY")
    print("=" * 80)
    success_count = sum(1 for value in results.values() if value)
    for choice, success in results.items():
        model = MODELS_TO_DOWNLOAD[choice]
        status = "Success" if success else "Failed"
        print(f"   {model['display_name']}: {status}")
    print(f"\n   Total: {success_count}/{len(MODELS_TO_DOWNLOAD)} artifacts prepared successfully")
    print("=" * 80 + "\n")


def display_menu() -> str:
    """Display the main menu and return the user selection."""
    print("\n" + "=" * 80)
    print("Local ASR Model Setup")
    print("=" * 80)
    print("\nSelect an option:\n")

    for choice, model in MODELS_TO_DOWNLOAD.items():
        rec = " [RECOMMENDED]" if model.get("recommended") else ""
        print(f"  {choice}. Download {model['display_name']}{rec}")
        print(f"     Size: {model['download_size']} -> {model['saved_size']}")
        print(f"     {model['description']}\n")

    print(f"  {BATCH_OPTION}. Download ALL requested artifacts")
    print(f"  {STATUS_OPTION}. Check what is already downloaded")
    print("  0. Exit\n")

    return input("Enter your choice: ").strip()


def main() -> None:
    print("\n" + "=" * 80)
    print("Local ASR Multi-Backend Setup Script")
    print("=" * 80)
    print("Split apps now have their own setup scripts:")
    print("- setup_local_models_nemo.py")
    print("- setup_local_models_transformers.py")
    print("\nThis script prepares the requested local ASR model artifacts.")
    print("NeMo models are saved as .nemo files; the others are stored as local repo snapshots.")
    print(f"Artifact destination: {LOCAL_MODELS_DIR}")

    create_local_models_directory()

    while True:
        choice = display_menu()

        if choice == "0":
            print("\nGoodbye")
            break
        if choice == BATCH_OPTION:
            download_all_models()
        elif choice == STATUS_OPTION:
            display_model_status()
        elif choice in MODELS_TO_DOWNLOAD:
            success = download_and_save_model(choice)
            print("\nDownload complete" if success else "\nDownload failed")
        else:
            print("\nInvalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\nUnexpected error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
