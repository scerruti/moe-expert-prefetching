#!/usr/bin/env python3
"""
Installed packages must match the pins in requirements.txt (#107).

Local and Colab runs only exercise the same code paths when they run the same
versions; e.g. Qwen2MoE's gate is nn.Linear in transformers 4.45 but a
different class with different output in 5.x.
"""

from importlib.metadata import version
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def pinned_requirements():
    pins = {}
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, pinned = line.split("==")
            pins[name.strip()] = pinned.strip()
    return pins


@pytest.mark.parametrize("package,pinned", sorted(pinned_requirements().items()))
def test_installed_version_matches_pin(package, pinned):
    """#107: installed version equals the pin (ignoring local suffixes like +cu126)."""
    assert version(package).split("+")[0] == pinned
