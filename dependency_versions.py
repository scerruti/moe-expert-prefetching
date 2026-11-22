"""Read the repository's exact pins without importing scientific packages."""

from importlib import metadata
from pathlib import Path

REQUIREMENTS_PATH = Path(__file__).resolve().parent / "requirements.txt"


def pinned_requirements():
    pins = {}
    for line in REQUIREMENTS_PATH.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            package, pinned = (part.strip() for part in line.split("==", 1))
            if not package or not pinned or package in pins:
                raise ValueError(f"Invalid or duplicate exact pin: {line}")
            pins[package] = pinned
    if not pins:
        raise ValueError("requirements.txt contains no exact pins")
    return pins


def version_mismatches():
    mismatches = []
    for package, pinned in pinned_requirements().items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{package}=={pinned}: not installed")
            continue
        # CPU and CUDA wheel suffixes identify builds of the same public release.
        if installed is None or installed.split("+", 1)[0] != pinned:
            mismatches.append(
                f"{package}=={pinned}: found {installed or 'unknown version'}"
            )
    return mismatches
