#!/usr/bin/env python3
"""
The window computation on real data (#8's acceptance criteria).

- CI: a 10% sample of the trial (fixtures/trial_sample/), checked against figures
  from reference_root_changes.py, a separate script that shares no code with the
  validator.
- Locally: the full trial reproduces the documented figures and the committed
  window record. It needs the trial data (414 MB, not in git), so it's skipped,
  with the reason, where the data isn't present.
"""

import json
import os
from pathlib import Path

import pytest

from determinism_validation import compute_window

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SAMPLE = HERE / "fixtures" / "trial_sample"
QWEN = "qwen1.5-moe-a2.7b"
RECORD = REPO / "phase_1" / "near_tie_windows" / f"{QWEN}.json"
TRIAL = Path(os.environ.get("MOE_TRIAL_DIR", REPO / "data" / "phase_1_trial_38ef1da"))


def test_the_sample_matches_the_independent_reference():
    """reference_root_changes.py on the sample: 339 root changes, largest gap 0.00027835."""
    record = compute_window(SAMPLE, model=QWEN)  # the trial predates manifests (#134)
    assert record["root_changes"] == 339
    assert record["largest_root_gap"] == pytest.approx(0.00027835, abs=5e-9)
    assert record["window"] == pytest.approx(3 * record["largest_root_gap"])


def test_the_committed_window_record_matches_the_documented_figures():
    """phase_1/docs/determinism_check.md: 3,429 root changes, largest gap 0.00035036, window about 0.00105."""
    record = json.loads(RECORD.read_text())
    assert record["root_changes"] == 3429
    assert record["largest_root_gap"] == pytest.approx(0.00035036, abs=5e-9)
    assert record["window"] == pytest.approx(0.00105, abs=5e-6)
    assert record["comparison"] == "ordering 0 against each other ordering"
    assert (record["target_rate"], record["margin"], record["required_root_changes"]) == (0.01, 3.0, 99)


@pytest.mark.skipif(not (TRIAL / "determinism").is_dir(),
                    reason=f"needs the full trial data at {TRIAL} (414 MB, not in git); set MOE_TRIAL_DIR")
def test_the_full_trial_reproduces_the_window_record():
    record = json.loads(RECORD.read_text())
    recomputed = compute_window(TRIAL, model=QWEN)
    for field in ("root_changes", "largest_root_gap", "window", "pilot_files", "experts_per_token"):
        assert recomputed[field] == record[field], field
