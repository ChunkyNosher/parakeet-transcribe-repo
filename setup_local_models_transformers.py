#!/usr/bin/env python3
"""Prepare phase-1 local model artifacts for transcribe_transformers_app.py.

This setup surface is intentionally limited to the currently supported
Transformers app scope: Granite, Cohere, Voxtral Mini 3B, and Voxtral Small 24B.
Qwen and Voxtral Realtime are not included here.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = SCRIPT_DIR / "local_models"

MODEL_ORDER = [
    "granite-4.0-1b-speech",
    "voxtral-mini-3b-2507",
    "cohere-transcribe-03-2026",
    "voxtral-small-24b-2507",
]

MODELS: Dict[str, Dict[str, Any]] = {
    "granite-4.0-1b-speech": {
        "model_id": "ibm-granite/granite-4.0-1b-speech",
        "artifact_name": "granite-4.0-1b-speech",
        "display_name": "IBM Granite 4.0 1B Speech",
        "download_size": "~3+ GB",
        "saved_size": "~3+ GB",
        "description": "Compact phase-1 Transformers speech model",
        "recommended": True,
    },
    "voxtral-mini-3b-2507": {
        "model_id": "mistralai/Voxtral-Mini-3B-2507",
        "artifact_name": "Voxtral-Mini-3B-2507",
        "display_name": "Mistral Voxtral Mini 3B 2507",
        "download_size": "~6+ GB",
        "saved_size": "~6+ GB",
        "description": "Offline Voxtral phase-1 artifact",
    },
    "cohere-transcribe-03-2026": {
        "model_id": "CohereLabs/cohere-transcribe-03-2026",
        "artifact_name": "cohere-transcribe-03-2026",
        "display_name": "Cohere Transcribe 03-2026",
        "download_size": "~6+ GB",
        "saved_size": "~6+ GB",
        "description": "Gated Cohere ASR package; requires approved HF access if downloaded remotely",
    },
    "voxtral-small-24b-2507": {
        "model_id": "mistralai/Voxtral-Small-24B-2507",
        "artifact_name": "Voxtral-Small-24B-2507",
        "display_name": "Mistral Voxtral Small 24B 2507",
        "download_size": "~45+ GB",
        "saved_size": "~45+ GB",
        "description": "Largest supported phase-1 Transformers artifact",
    },
}

DEFERRED_MODELS = [
    "Qwen/Qwen3-ASR-1.7B is intentionally excluded because it needs the separate qwen-asr runtime path.",
    "mistralai/Voxtral-Mini-4B-Realtime-2602 remains deferred because the realtime runtime is not exposed in the current Transformers stack.",
]


def _require_snapshot_download() -> Any:
    try:
        return importlib.import_module("huggingface_hub").snapshot_download
    except Exception as exc:
        raise ImportError(
            "Missing dependency 'huggingface_hub'. "
            "Install the repo requirements in the configured Python environment. "
            f"Original error: {exc}"
        ) from exc


def _artifact_path(model_key: str) -> Path:
    return LOCAL_MODELS_DIR / MODELS[model_key]["artifact_name"]


def _path_size_gb(path: Path) -> float:
    total_bytes = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return total_bytes / (1024 ** 3)


def _is_artifact_valid(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.rglob("*"))


def create_local_models_directory() -> None:
    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Artifact directory: {LOCAL_MODELS_DIR}")


def display_model_status() -> None:
    print("\n" + "=" * 80)
    print("TRANSFORMERS APP MODEL STATUS")
    print("=" * 80)

    for model_key in MODEL_ORDER:
        model = MODELS[model_key]
        artifact_path = _artifact_path(model_key)
        print(f"\n{model['display_name']}")
        print(f"Model key: {model_key}")
        print(f"Model ID: {model['model_id']}")
        print(f"Expected artifact: {artifact_path}")
        if artifact_path.exists():
            size_gb = _path_size_gb(artifact_path)
            validity = "OK" if _is_artifact_valid(artifact_path) else "Check contents"
            print(f"Status: Present ({size_gb:.2f} GB, {validity})")
        else:
            print("Status: Not downloaded")
            print(f"Expected download size: {model['download_size']}")

    print("\nDeferred from this script:")
    for message in DEFERRED_MODELS:
        print(f"- {message}")
    print("=" * 80 + "\n")


def download_model(model_key: str, force: bool = False) -> bool:
    model = MODELS[model_key]
    output_path = _artifact_path(model_key).resolve()

    if output_path.exists() and _is_artifact_valid(output_path) and not force:
        print(f"{model['display_name']} is already ready at {output_path}")
        print("Use --force to refresh the local snapshot.")
        return True

    print("\n" + "=" * 80)
    print(f"PREPARING: {model['display_name']}")
    print("=" * 80)
    print(f"Model ID: {model['model_id']}")
    print(f"Output: {output_path}")
    print(f"Download size: {model['download_size']}")
    print(f"Saved size: {model['saved_size']}")
    print(f"Description: {model['description']}")

    try:
        output_path.mkdir(parents=True, exist_ok=True)
        snapshot_download = _require_snapshot_download()
        snapshot_download(
            repo_id=model["model_id"],
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        if not _is_artifact_valid(output_path):
            print("Artifact validation failed after download.")
            return False

        print(f"Ready: {model['display_name']} ({_path_size_gb(output_path):.2f} GB)")
        return True
    except Exception as exc:
        print(f"Error preparing {model['display_name']}:")
        print(f"  {type(exc).__name__}: {exc}")
        return False


def download_all_models(force: bool = False) -> bool:
    print("\nPreparing all supported Transformers app artifacts...")
    results = {model_key: download_model(model_key, force=force) for model_key in MODEL_ORDER}
    success_count = sum(1 for value in results.values() if value)
    print(f"\nCompleted: {success_count}/{len(MODEL_ORDER)} supported artifacts ready")
    return success_count == len(MODEL_ORDER)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare supported Transformers app artifacts.")
    parser.add_argument("--status", action="store_true", help="Show whether the supported Transformers artifacts are present")
    parser.add_argument(
        "--download",
        choices=["all", *MODEL_ORDER],
        help="Download one supported model artifact or all supported artifacts",
    )
    parser.add_argument("--force", action="store_true", help="Refresh an artifact even if the existing snapshot looks valid")
    return parser.parse_args()


def _display_menu() -> str:
    print("\n" + "=" * 80)
    print("Split Transformers App Setup")
    print("=" * 80)
    for index, model_key in enumerate(MODEL_ORDER, start=1):
        model = MODELS[model_key]
        recommended = " [RECOMMENDED]" if model.get("recommended") else ""
        print(f"{index}. Prepare {model['display_name']}{recommended}")
        print(f"   {model['description']}")
    print(f"{len(MODEL_ORDER) + 1}. Prepare all supported Transformers artifacts")
    print(f"{len(MODEL_ORDER) + 2}. Check local artifact status")
    print("0. Exit")
    return input("Enter your choice: ").strip()


def main() -> int:
    args = _parse_args()

    print("\n" + "=" * 80)
    print("Transformers App Local Model Setup")
    print("=" * 80)
    print("This script prepares the phase-1 Transformers artifacts for transcribe_transformers_app.py.")
    print("Supported here: Granite, Cohere, Voxtral Mini 3B, and Voxtral Small 24B.")
    print("Excluded here: Qwen and Voxtral Realtime.")

    create_local_models_directory()

    if args.status:
        display_model_status()
        return 0

    if args.download:
        if args.download == "all":
            return 0 if download_all_models(force=args.force) else 1
        return 0 if download_model(args.download, force=args.force) else 1

    while True:
        choice = _display_menu()
        if choice == "0":
            print("Goodbye")
            return 0
        if choice == str(len(MODEL_ORDER) + 1):
            success = download_all_models(force=False)
            print("\nSetup complete" if success else "\nSetup finished with failures")
        elif choice == str(len(MODEL_ORDER) + 2):
            display_model_status()
        elif choice.isdigit() and 1 <= int(choice) <= len(MODEL_ORDER):
            model_key = MODEL_ORDER[int(choice) - 1]
            success = download_model(model_key, force=False)
            print("\nSetup complete" if success else "\nSetup failed")
        else:
            print("\nInvalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)