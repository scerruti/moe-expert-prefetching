#!/usr/bin/env python3
"""Validate the Phase 1 Python environment and create required directory structure."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_PACKAGES = [
    ("torch", "2.14.0"),
    ("transformers", "5.17.0"),
    ("datasets", "5.0.1"),
    ("pyarrow", "25.0.1"),
    ("tqdm", "4.70.1"),
]


def get_required_requirements():
    return REQUIRED_PACKAGES


def ensure_directory(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    if not any(directory.iterdir()):
        (directory / "README.md").write_text(
            f"# {directory}\n\nThis directory was created for the Phase 1 environment setup issue.\n",
            encoding="utf-8",
        )
    return directory


def check_dependencies() -> list[str]:
    missing = []
    for package, expected_version in REQUIRED_PACKAGES:
        try:
            module = __import__(package)
            actual_version = getattr(module, "__version__", None)
            if actual_version is None:
                missing.append(f"{package} ({expected_version} required)")
            elif actual_version != expected_version:
                missing.append(f"{package} ({expected_version} required, found {actual_version})")
        except Exception:
            missing.append(f"{package} ({expected_version} required)")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1 Python environment setup.")
    parser.add_argument("--create-dirs", action="store_true", help="Create the requested directory structure.")
    args = parser.parse_args()

    if args.create_dirs:
        ensure_directory("data")
        ensure_directory("phase_1/scripts")

    missing = check_dependencies()
    if missing:
        print("Missing or mismatched dependencies:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("Environment is ready for Phase 1 data collection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
