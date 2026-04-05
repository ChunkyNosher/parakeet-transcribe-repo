#!/usr/bin/env python3
"""Dependency and import health check for the split NeMo app."""

from __future__ import annotations

import importlib
from pathlib import Path

REQUIRED_IMPORTS = [
    ("gradio", "gradio"),
    ("torch", "torch"),
    ("librosa", "librosa"),
    ("numpy", "numpy"),
    ("omegaconf", "omegaconf"),
    ("soundfile", "soundfile"),
    ("nemo.collections.asr", "nemo-toolkit[asr]"),
]

OPTIONAL_IMPORTS = [
    ("nemo_text_processing.inverse_text_normalization", "nemo-text-processing"),
]

REQUIRED_PATHS = [
    "transcribe_nemo_app.py",
    "setup_local_models_nemo.py",
    "local_models",
    "model_cache",
    "app_shared",
]


def _check_imports(module_specs: list[tuple[str, str]], required: bool) -> list[str]:
    failures: list[str] = []
    for module_name, package_name in module_specs:
        try:
            importlib.import_module(module_name)
            print(f"[OK] import {module_name}")
        except Exception as exc:
            level = "ERROR" if required else "WARN"
            print(f"[{level}] import {module_name}: {type(exc).__name__}: {exc}")
            if required:
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


def _check_entrypoint() -> list[str]:
    failures: list[str] = []

    try:
        module = importlib.import_module("transcribe_nemo_app")
        print("[OK] import transcribe_nemo_app")
    except Exception as exc:
        print(f"[ERROR] import transcribe_nemo_app: {type(exc).__name__}: {exc}")
        return ["transcribe_nemo_app"]

    if hasattr(module, "app"):
        print("[OK] transcribe_nemo_app exposes a Gradio app")
    else:
        print("[ERROR] transcribe_nemo_app is missing app")
        failures.append("transcribe_nemo_app.app")

    model_configs = getattr(module, "MODEL_CONFIGS", {})
    model_display_order = getattr(module, "MODEL_DISPLAY_ORDER", [])
    default_model_key = getattr(module, "DEFAULT_MODEL_KEY", None)

    if not model_display_order:
        print("[ERROR] NeMo app model registry is empty")
        failures.append("MODEL_DISPLAY_ORDER")
    else:
        print(f"[OK] NeMo app exposes {len(model_display_order)} active model(s)")

    if default_model_key in model_configs:
        print(f"[OK] default model key {default_model_key}")
    else:
        print(f"[ERROR] default model key missing from registry: {default_model_key}")
        failures.append("DEFAULT_MODEL_KEY")

    invalid_backends = [
        key for key in model_display_order if model_configs.get(key, {}).get("backend") != "nemo"
    ]
    if invalid_backends:
        print(f"[ERROR] unexpected non-NeMo backends in split NeMo app: {', '.join(invalid_backends)}")
        failures.append("MODEL_CONFIGS.backends")
    else:
        print("[OK] split NeMo app registry is NeMo-only")

    return failures


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    print(f"Repo root: {repo_root}")

    path_failures = _check_paths(repo_root)
    import_failures = _check_imports(REQUIRED_IMPORTS, required=True)
    _check_imports(OPTIONAL_IMPORTS, required=False)
    entrypoint_failures = _check_entrypoint()

    total_failures = len(path_failures) + len(import_failures) + len(entrypoint_failures)
    if total_failures:
        print(f"\nNeMo app health check failed with {total_failures} issue(s).")
        return 1

    print("\nNeMo app health check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())