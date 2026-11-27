#!/usr/bin/env python3
"""
The collection notebook's wiring (#133, #134), checked from its source.

The notebook only runs on Colab, so these tests read its code cells: the run
settings come before torch, the window record is passed explicitly, the
manifest replaces provenance.json, and the superseded batch-variance analysis
no longer runs.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTEBOOK = REPO / "phase_1" / "notebooks" / "Phase1_Full_Collection_Colab.ipynb"


def code_cells():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def all_code():
    return "\n".join(code_cells())


def test_the_run_settings_are_applied_before_torch_is_imported():
    code = all_code()
    assert "apply_run_settings()" in code
    assert code.index("apply_run_settings()") < code.index("import torch")


def test_the_window_record_is_passed_explicitly_and_exists():
    code = all_code()
    assert "DeterminismValidator(output_dir, window=" in code
    settings = code_cells()[0]
    assert 'WINDOW_RECORD = "phase_1/near_tie_windows/qwen1.5-moe-a2.7b.json"' in settings
    assert (REPO / "phase_1" / "near_tie_windows" / "qwen1.5-moe-a2.7b.json").is_file()


def test_the_collection_is_told_its_ref_and_the_notebook_writes_no_provenance_file():
    code = all_code()
    assert "ref=REF" in code
    assert "provenance.json" not in code


def test_the_official_collection_runs_from_main():
    assert 'REF = "main"' in code_cells()[0]


def test_the_superseded_batch_variance_analysis_no_longer_runs():
    assert "batch_variance" not in all_code()
