#!/usr/bin/env python3
"""Quick integration/dependency health check for this repository."""

from __future__ import annotations

import importlib
from pathlib import Path

REQUIRED_IMPORTS = [
    "gradio",
    "torch",
    "nemo",
    "librosa",
    "numpy",
    "omegaconf",
]

OPTIONAL_IMPORTS = [
    "nemo_text_processing.inverse_text_normalization",
]

REQUIRED_PATHS = [
    "transcribe_ui.py",
    "setup_local_models.py",
    "test_manifest_fix.py",
    "local_models",
    "model_cache",
]


def _check_imports(module_names: list[str], required: bool) -> list[str]:
    failures: list[str] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            print(f"[OK] import {module_name}")
        except Exception as exc:
            level = "ERROR" if required else "WARN"
            print(f"[{level}] import {module_name}: {type(exc).__name__}: {exc}")
            if required:
                failures.append(module_name)
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


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    print(f"Repo root: {repo_root}")

    path_failures = _check_paths(repo_root)
    import_failures = _check_imports(REQUIRED_IMPORTS, required=True)
    _check_imports(OPTIONAL_IMPORTS, required=False)

    total_failures = len(path_failures) + len(import_failures)
    if total_failures:
        print(f"\nHealth check failed with {total_failures} issue(s).")
        return 1

    print("\nHealth check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
