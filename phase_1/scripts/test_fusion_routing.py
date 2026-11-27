"""
Regression: Triton's FP fusion setting doesn't change routing (#133, from #140).

#140 found that with fusion on and off, routing is bitwise identical: the one
kernel Triton compiles on this path is an outer product, which has no add to
fuse. This test repeats that check on a tiny Qwen2MoE, so it reruns on every
torch upgrade (a dependency change runs the GPU tests, per CONTRIBUTING.md).
If it fails, fusion now matters: decide on #124 or the upgrade's issue whether
to keep it on, and re-establish the near-tie window (#128).
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from routing_test_helpers import requires_gpu

PROBE = Path(__file__).resolve().parent / "fusion_probe.py"


def run_probe(out, fusion):
    env = {k: v for k, v in os.environ.items() if k not in ("TRITON_DEFAULT_FP_FUSION", "MOE_RUN_SETTINGS_APPLIED")}
    result = subprocess.run(
        [sys.executable, str(PROBE), "--fusion", fusion, "--out", str(out)],
        capture_output=True, text=True, timeout=600, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "report.json").read_text())
    routing = np.load(out / "routing.npz")
    return report, routing["expert_probs"], routing["selected_experts"]


@requires_gpu
def test_fusion_on_and_off_give_bitwise_identical_routing(tmp_path):
    on_report, on_probs, on_selected = run_probe(tmp_path / "fusion_on", "1")
    off_report, off_probs, off_selected = run_probe(tmp_path / "fusion_off", "0")

    assert (on_report["fusion"], off_report["fusion"]) == ("1", "0")
    # With no Triton kernel, the setting can't have been exercised, and identical routing proves nothing
    assert off_report["triton_kernels"], (
        "Triton compiled no kernels with fusion off, so this test can't show whether fusion matters. "
        "If the routing path no longer uses Triton, the setting can't affect it: update this test."
    )
    assert on_probs.dtype == off_probs.dtype and on_probs.shape == off_probs.shape
    assert on_probs.size > 0
    differing = (on_probs.view(np.uint8).reshape(len(on_probs), -1)
                 != off_probs.view(np.uint8).reshape(len(off_probs), -1)).any(axis=1)
    assert not differing.any(), (
        f"{int(differing.sum())} of {len(differing)} routing rows differ in probability bits "
        f"between fusion on and off (kernels: {off_report['triton_kernels']})"
    )
    np.testing.assert_array_equal(on_selected, off_selected)
