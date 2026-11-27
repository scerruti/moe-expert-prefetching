"""Settings that must be in the environment before torch, and so Triton, is imported (#133).

TRITON_DEFAULT_FP_FUSION: Triton 3.9 fuses multiplies and adds by default, and
warns that 3.10 will stop. Fusion changes results in the last bits, and
differences that small can flip near-tie routing decisions (#114). #140 found
the setting doesn't change Qwen's routing on the current code path, but setting
it makes it a recorded choice rather than a default that changes with a torch
upgrade. Triton reads it when it starts, so it has to be set first.

- Entry points that run the model call apply_run_settings() before importing
  torch: the root conftest.py (every pytest run) and the collection notebook.
- Code that runs the model calls require_run_settings(), which refuses unless
  an entry point applied them.

This module must not import torch, or anything that does.
"""

import os
import sys

RUN_SETTINGS = {"TRITON_DEFAULT_FP_FUSION": "1"}

# Set once the settings are applied before torch was imported. It is inherited
# by child processes, whose environment then already holds the settings.
APPLIED = "MOE_RUN_SETTINGS_APPLIED"

_EARLY_MODULES = ("torch", "triton")


class RunSettingsError(RuntimeError):
    """The run settings are missing, conflict, or came too late to take effect."""


def _conflicts():
    return [
        f"{name} is set to {os.environ[name]!r}; this project requires {value!r}"
        for name, value in RUN_SETTINGS.items()
        if name in os.environ and os.environ[name] != value
    ]


def _raise_conflicts(conflicts):
    raise RunSettingsError(
        "The environment conflicts with the project's run settings (#133):\n  "
        + "\n  ".join(conflicts)
        + "\nUnset these variables, or set them to the required values, and start again."
    )


def _in_place():
    return os.environ.get(APPLIED) == "1" and all(
        os.environ.get(name) == value for name, value in RUN_SETTINGS.items()
    )


def apply_run_settings():
    """Set the run settings. Call before importing torch; returns the settings applied."""
    conflicts = _conflicts()
    if conflicts:
        _raise_conflicts(conflicts)
    if _in_place():
        return dict(RUN_SETTINGS)  # applied earlier, before torch: a repeat call or a child process
    loaded = [name for name in _EARLY_MODULES if name in sys.modules]
    if loaded:
        raise RunSettingsError(
            f"{' and '.join(loaded)} was imported before the run settings were applied, so "
            "Triton may already have read its defaults. Call apply_run_settings() before "
            "importing torch. In a notebook: Runtime → Restart session, then run all."
        )
    os.environ.update(RUN_SETTINGS)
    os.environ[APPLIED] = "1"
    return dict(RUN_SETTINGS)


def require_run_settings():
    """Refuse unless an entry point applied the run settings before torch; returns them."""
    conflicts = _conflicts()
    if conflicts:
        _raise_conflicts(conflicts)
    if not _in_place():
        raise RunSettingsError(
            "The run settings were not applied. The entry point must call "
            "run_settings.apply_run_settings() before importing torch (#133)."
        )
    return dict(RUN_SETTINGS)
