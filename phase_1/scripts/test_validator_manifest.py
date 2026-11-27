#!/usr/bin/env python3
"""
The validator and the window computation read the model from the run manifest (#134).

- validate_all() refuses a collection with no manifest.json, one whose run
  didn't finish, or a window record for a different model.
- compute_window() takes the model from the pilot's manifest, or requires it
  to be named; it never assumes one.

CPU only, on the 10% trial sample (fixtures/trial_sample/), copied with a manifest.
"""

import json
import shutil
from pathlib import Path

import pytest

from determinism_validation import DeterminismValidator, compute_window
from run_manifest import MANIFEST_NAME, ManifestError

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "fixtures" / "trial_sample"
QWEN = "qwen1.5-moe-a2.7b"
QWEN_WINDOW = HERE.parent / "near_tie_windows" / f"{QWEN}.json"


def collection(tmp_path, manifest=None):
    """A copy of the trial sample's orderings, with the given manifest (None: no manifest)."""
    folder = tmp_path / "collection"
    shutil.copytree(SAMPLE / "determinism", folder / "determinism")
    if manifest is not None:
        (folder / MANIFEST_NAME).write_text(json.dumps(manifest))
    return folder


def finished(model=QWEN, **fields):
    return {"status": "finished", "model": model, **fields}


def test_a_collection_without_a_manifest_is_refused(tmp_path):
    """Forced failure: without manifest.json nothing says which model collected the data."""
    folder = collection(tmp_path)
    with pytest.raises(ManifestError, match=f"no {MANIFEST_NAME}"):
        DeterminismValidator(folder, window=QWEN_WINDOW).validate_all()


@pytest.mark.parametrize("status", ["running", "failed"])
def test_a_run_that_did_not_finish_is_refused(tmp_path, status):
    """Forced failure: a partial collection would give a report that looks complete."""
    folder = collection(tmp_path, {"status": status, "model": QWEN})
    with pytest.raises(ManifestError, match=f"status is '{status}'"):
        DeterminismValidator(folder, window=QWEN_WINDOW).validate_all()


def test_a_window_record_for_another_model_is_refused(tmp_path):
    """Forced failure: the wrong window would make every near-tie statistic meaningless."""
    folder = collection(tmp_path, finished(model="mixtral-8x7b"))
    with pytest.raises(ManifestError) as error:
        DeterminismValidator(folder, window=QWEN_WINDOW).validate_all()
    assert "mixtral-8x7b" in str(error.value) and QWEN in str(error.value)


def test_a_window_record_without_a_model_is_refused(tmp_path):
    folder = collection(tmp_path, finished())
    with pytest.raises(ManifestError, match="names no model"):
        DeterminismValidator(folder, window={"window": 0.001}).validate_all()


def test_the_report_names_the_manifest_model_and_keeps_the_manifest(tmp_path):
    folder = collection(tmp_path, finished(commit="abc1234"))
    validator = DeterminismValidator(folder, window=QWEN_WINDOW)
    report = validator.validate_all()

    assert report.model == QWEN
    assert report.near_tie.status == "ran"
    validator.save_report(report, tmp_path / "report.json")
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["model"] == QWEN
    assert saved["manifest"]["commit"] == "abc1234"


def test_without_a_window_the_report_is_incomplete_not_refused(tmp_path):
    folder = collection(tmp_path, finished())
    report = DeterminismValidator(folder).validate_all()
    assert report.model == QWEN
    assert report.near_tie.status == "not_run"
    assert report.status == "incomplete"


def test_compute_window_takes_the_model_from_the_pilot_manifest(tmp_path):
    folder = collection(tmp_path, finished(model="some-moe"))
    assert compute_window(folder)["model"] == "some-moe"


def test_compute_window_refuses_a_model_that_contradicts_the_manifest(tmp_path):
    folder = collection(tmp_path, finished(model="some-moe"))
    with pytest.raises(ManifestError, match="some-moe"):
        compute_window(folder, model=QWEN)


def test_compute_window_refuses_an_unfinished_pilot(tmp_path):
    folder = collection(tmp_path, {"status": "running", "model": QWEN})
    with pytest.raises(ManifestError, match="status is 'running'"):
        compute_window(folder)


def test_compute_window_without_a_manifest_needs_the_model_named(tmp_path):
    """Forced failure: no default model; a pilot from before manifests names it explicitly."""
    folder = collection(tmp_path)
    with pytest.raises(ManifestError, match="name the model"):
        compute_window(folder)
    assert compute_window(folder, model=QWEN)["model"] == QWEN
