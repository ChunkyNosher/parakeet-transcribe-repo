#!/usr/bin/env python3
"""Dependency and import health check for the split Transformers app."""

from __future__ import annotations

import importlib
from pathlib import Path

REQUIRED_IMPORTS = [
    ("gradio", "gradio"),
    ("torch", "torch"),
    ("librosa", "librosa"),
    ("numpy", "numpy"),
    ("soundfile", "soundfile"),
    ("transformers", "transformers==4.57.6"),
    ("huggingface_hub", "huggingface-hub"),
    ("mistral_common", "mistral-common"),
]

REQUIRED_TRANSFORMERS_ATTRS = [
    "AutoProcessor",
    "AutoModelForSpeechSeq2Seq",
    "GraniteSpeechForConditionalGeneration",
    "VoxtralForConditionalGeneration",
]

REQUIRED_PATHS = [
    "transcribe_transformers_app.py",
    "setup_local_models_transformers.py",
    "local_models",
    "model_cache",
    "app_shared",
]

REQUIRED_DEFERRED_MODELS = {
    "qwen3-asr-1.7b",
    "voxtral-mini-4b-realtime-2602",
}


def _check_imports(module_specs: list[tuple[str, str]]) -> list[str]:
    failures: list[str] = []
    for module_name, package_name in module_specs:
        try:
            importlib.import_module(module_name)
            print(f"[OK] import {module_name}")
        except Exception as exc:
            print(f"[ERROR] import {module_name}: {type(exc).__name__}: {exc}")
            failures.append(package_name)
    return failures


def _check_paths(repo_root: Path) -> list[str]:
    failures: list[str] = []
    for relative_path in REQUIRED_PATHS:
        full_path = repo_root / relative_path
        if full_path.exists():
            print(f"[OK] path {relative_path}")
        else:
            print(f"[ERROR] missing path {relative_path}")
            failures.append(relative_path)
    return failures


def _check_transformers_attrs() -> list[str]:
    failures: list[str] = []
    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        print(f"[ERROR] import transformers: {type(exc).__name__}: {exc}")
        return ["transformers"]

    for attr_name in REQUIRED_TRANSFORMERS_ATTRS:
        if hasattr(transformers, attr_name):
            print(f"[OK] transformers.{attr_name}")
        else:
            print(f"[ERROR] transformers missing {attr_name}")
            failures.append(attr_name)
    return failures


def _check_entrypoint() -> list[str]:
    failures: list[str] = []

    try:
        module = importlib.import_module("transcribe_transformers_app")
        print("[OK] import transcribe_transformers_app")
    except Exception as exc:
        print(f"[ERROR] import transcribe_transformers_app: {type(exc).__name__}: {exc}")
        return ["transcribe_transformers_app"]

    if hasattr(module, "app"):
        print("[OK] transcribe_transformers_app exposes a Gradio app")
    else:
        print("[ERROR] transcribe_transformers_app is missing app")
        failures.append("transcribe_transformers_app.app")

    model_configs = getattr(module, "MODEL_CONFIGS", {})
    model_display_order = getattr(module, "MODEL_DISPLAY_ORDER", [])
    default_model_key = getattr(module, "DEFAULT_MODEL_KEY", None)
    deferred_models = getattr(module, "DEFERRED_MODELS", {})

    if not model_display_order:
        print("[ERROR] Transformers app model registry is empty")
        failures.append("MODEL_DISPLAY_ORDER")
    else:
        print(f"[OK] Transformers app exposes {len(model_display_order)} active model(s)")

    if default_model_key in model_configs:
        print(f"[OK] default model key {default_model_key}")
    else:
        print(f"[ERROR] default model key missing from registry: {default_model_key}")
        failures.append("DEFAULT_MODEL_KEY")

    invalid_backends = [
        key
        for key in model_display_order
        if model_configs.get(key, {}).get("backend") not in {
            "transformers_granite",
            "transformers_voxtral",
            "transformers_cohere",
        }
    ]
    if invalid_backends:
        print(f"[ERROR] unexpected backends in split Transformers app: {', '.join(invalid_backends)}")
        failures.append("MODEL_CONFIGS.backends")
    else:
        print("[OK] split Transformers app registry is limited to supported Transformers backends")

    accidentally_supported = REQUIRED_DEFERRED_MODELS.intersection(model_display_order)
    if accidentally_supported:
        joined = ", ".join(sorted(accidentally_supported))
        print(f"[ERROR] deferred models appear in the active display order: {joined}")
        failures.append("MODEL_DISPLAY_ORDER.deferred")
    else:
        print("[OK] deferred models are not exposed as active choices")

    missing_deferred = REQUIRED_DEFERRED_MODELS.difference(deferred_models)
    if missing_deferred:
        joined = ", ".join(sorted(missing_deferred))
        print(f"[ERROR] deferred model registry is missing: {joined}")
        failures.append("DEFERRED_MODELS")
    else:
        print("[OK] deferred model registry includes Qwen and Voxtral Realtime")

    return failures


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    print(f"Repo root: {repo_root}")

    path_failures = _check_paths(repo_root)
    import_failures = _check_imports(REQUIRED_IMPORTS)
    transformers_attr_failures = _check_transformers_attrs()
    entrypoint_failures = _check_entrypoint()

    total_failures = (
        len(path_failures)
        + len(import_failures)
        + len(transformers_attr_failures)
        + len(entrypoint_failures)
    )
    if total_failures:
        print(f"\nTransformers app health check failed with {total_failures} issue(s).")
        return 1

    print("\nTransformers app health check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())