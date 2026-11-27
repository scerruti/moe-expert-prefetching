#!/usr/bin/env python3
"""
The run manifest: manifest.json, written next to each collection's data (#134).

It records what the collection ran with, so anything that reads the data
(the determinism check, the window computation, Phase 2) takes the model and
settings from it instead of guessing. The collection writes it before any
data, with status "running", and adds the results at the end: "finished"
for a complete, valid collection, "failed" otherwise.
"""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
REPO = Path(__file__).resolve().parents[2]


class ManifestError(ValueError):
    """The manifest is missing, unfinished, or contradicts what it is used with."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def software_versions() -> Dict[str, Optional[str]]:
    """Python and the packages that affect routing; None for one that isn't installed."""
    return {"python": platform.python_version(),
            **{p: _version(p) for p in ("torch", "transformers", "accelerate", "triton")}}


def gpu_description() -> Optional[Dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        return None
    return {"name": torch.cuda.get_device_name(0),
            "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}


def code_state(repo: Path = REPO) -> Dict[str, Any]:
    """The commit, and whether the working tree differs from it. Refuses outside a git checkout."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    except (OSError, subprocess.CalledProcessError) as error:
        raise ManifestError(f"Can't record the commit: {repo} isn't a usable git checkout ({error})") from error
    return {"commit": commit, "uncommitted_changes": dirty}


def write_manifest(folder: Path, manifest: Dict[str, Any]) -> Path:
    """Write manifest.json whole: to a temporary file, then renamed over the old one."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / MANIFEST_NAME
    temporary = folder / f"{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, path)
    return path


def read_manifest(folder: Path) -> Optional[Dict[str, Any]]:
    """The folder's manifest, or None if it has none."""
    path = Path(folder) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def require_finished_manifest(folder: Path) -> Dict[str, Any]:
    """The folder's manifest, refusing a missing one or a run that didn't finish."""
    manifest = read_manifest(folder)
    if manifest is None:
        raise ManifestError(
            f"There's no {MANIFEST_NAME} in {folder}, so nothing says which model collected this "
            "data. Collections write one (#134); validate a collection's output folder.")
    status = manifest.get("status")
    if status != "finished":
        raise ManifestError(
            f"{Path(folder) / MANIFEST_NAME}: the run's status is {status!r}, not 'finished', so "
            "the collection is incomplete or failed. Its error, if any: "
            f"{manifest.get('error') or manifest.get('results', {}).get('errors') or 'none recorded'}")
    if not manifest.get("model"):
        raise ManifestError(f"{Path(folder) / MANIFEST_NAME} names no model")
    return manifest
