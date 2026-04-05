#!/usr/bin/env python3
"""Prepare local model artifacts for transcribe_nemo_app.py.

This setup surface is intentionally limited to the currently active NeMo app
scope: NVIDIA Parakeet 0.6B-v3 saved as a .nemo artifact.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = SCRIPT_DIR / "local_models"

ACTIVE_MODEL: Dict[str, Any] = {
    "model_id": "nvidia/parakeet-tdt-0.6b-v3",
    "artifact_name": "parakeet-0.6b-v3.nemo",
    "display_name": "NVIDIA Parakeet 0.6B-v3",
    "download_size": "~1.2 GB",
    "saved_size": "~2.4 GB",
    "min_size_gb": 1.5,
    "description": "NeMo export for the split NeMo app: multilingual ASR with timestamps",
}


def _require_nemo_asr() -> Any:
    try:
        return importlib.import_module("nemo.collections.asr")
    except Exception as exc:
        raise ImportError(
            "Missing dependency 'nemo-toolkit[asr]'. "
            "Use the configured project environment and install the repo requirements. "
            f"Original error: {exc}"
        ) from exc


def _artifact_path() -> Path:
    return LOCAL_MODELS_DIR / ACTIVE_MODEL["artifact_name"]


def _path_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def _is_artifact_valid(path: Path) -> bool:
    return path.exists() and path.is_file() and _path_size_gb(path) >= float(ACTIVE_MODEL["min_size_gb"])


def create_local_models_directory() -> None:
    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Artifact directory: {LOCAL_MODELS_DIR}")


def display_model_status() -> bool:
    artifact_path = _artifact_path()
    print("\n" + "=" * 80)
    print("NEMO APP MODEL STATUS")
    print("=" * 80)
    print(f"Model: {ACTIVE_MODEL['display_name']}")
    print(f"Model ID: {ACTIVE_MODEL['model_id']}")
    print(f"Expected artifact: {artifact_path}")

    if artifact_path.exists():
        size_gb = _path_size_gb(artifact_path)
        validity = "OK" if _is_artifact_valid(artifact_path) else "Check contents"
        print(f"Status: Present ({size_gb:.2f} GB, {validity})")
        print("Used by: transcribe_nemo_app.py")
        print("=" * 80 + "\n")
        return True

    print("Status: Not downloaded")
    print(f"Expected download size: {ACTIVE_MODEL['download_size']}")
    print(f"Expected saved size: {ACTIVE_MODEL['saved_size']}")
    print("Used by: transcribe_nemo_app.py")
    print("=" * 80 + "\n")
    return False


def download_and_save_model(force: bool = False) -> bool:
    create_local_models_directory()
    output_path = _artifact_path().resolve()

    if output_path.exists() and _is_artifact_valid(output_path) and not force:
        print(f"{ACTIVE_MODEL['display_name']} is already ready at {output_path}")
        print("Use --force to re-download the artifact.")
        return True

    if output_path.exists():
        output_path.unlink()

    print("\n" + "=" * 80)
    print(f"PREPARING: {ACTIVE_MODEL['display_name']}")
    print("=" * 80)
    print(f"Model ID: {ACTIVE_MODEL['model_id']}")
    print(f"Output: {output_path}")
    print(f"Download size: {ACTIVE_MODEL['download_size']}")
    print(f"Saved size: {ACTIVE_MODEL['saved_size']}")
    print(f"Description: {ACTIVE_MODEL['description']}")

    try:
        nemo_asr = _require_nemo_asr()
        print("Downloading NeMo model from Hugging Face...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(ACTIVE_MODEL["model_id"])
        print(f"Saving NeMo artifact to {output_path}...")
        asr_model.save_to(str(output_path))
        del asr_model

        if not _is_artifact_valid(output_path):
            print("Artifact validation failed after save.")
            return False

        print(f"Ready: {ACTIVE_MODEL['display_name']} ({_path_size_gb(output_path):.2f} GB)")
        return True
    except Exception as exc:
        print(f"Error preparing {ACTIVE_MODEL['display_name']}:")
        print(f"  {type(exc).__name__}: {exc}")
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the active NeMo app model artifact.")
    parser.add_argument("--status", action="store_true", help="Show whether the NeMo app artifact is already present")
    parser.add_argument("--download", action="store_true", help="Download or refresh the active NeMo app artifact")
    parser.add_argument("--force", action="store_true", help="Re-download even if the existing artifact looks valid")
    return parser.parse_args()


def _display_menu() -> str:
    print("\n" + "=" * 80)
    print("Split NeMo App Setup")
    print("=" * 80)
    print("1. Prepare NVIDIA Parakeet 0.6B-v3 for transcribe_nemo_app.py")
    print("2. Check local artifact status")
    print("0. Exit")
    return input("Enter your choice: ").strip()


def main() -> int:
    args = _parse_args()

    print("\n" + "=" * 80)
    print("NeMo App Local Model Setup")
    print("=" * 80)
    print("This script prepares the active NeMo artifact for transcribe_nemo_app.py.")
    print("The split NeMo app currently exposes only NVIDIA Parakeet 0.6B-v3.")

    if args.status:
        create_local_models_directory()
        display_model_status()
        return 0

    if args.download:
        return 0 if download_and_save_model(force=args.force) else 1

    create_local_models_directory()
    while True:
        choice = _display_menu()
        if choice == "0":
            print("Goodbye")
            return 0
        if choice == "1":
            success = download_and_save_model(force=False)
            print("\nSetup complete" if success else "\nSetup failed")
        elif choice == "2":
            display_model_status()
        else:
            print("\nInvalid choice. Please try again.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)