#!/usr/bin/env python3
"""
The collection's run manifest and same-batch repeat (#134).

- Each collection writes manifest.json next to its data, before collecting
  anything (status "running"), and adds the results at the end ("finished",
  or "failed" with the error).
- The collection reruns ordering 0 as determinism/repeat_run_0_<dataset>.parquet,
  so the same-batch check can run.

CPU only: a recording pipeline stands in for the passes, and the toy Qwen2MoE
for the model where real parquet files are needed.
"""

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers

from full_pipeline import RUN_SEEDS, DETERMINISM_SUBSET_SEED, DETERMINISM_SUBSET_SIZE, FullPipeline
from run_manifest import MANIFEST_NAME
from test_collection_scope import CharTokenizer, recording_pipeline, synthetic_prompts
from routing_test_helpers import make_capture, make_toy_qwen

REPO = Path(__file__).resolve().parents[2]


def manifest_of(pipeline):
    return json.loads((pipeline.output_dir / MANIFEST_NAME).read_text())


def with_model(pipeline):
    """The recording pipeline skips load_model; give it a model to describe."""
    pipeline.model = SimpleNamespace(dtype=torch.float16)
    return pipeline


def test_manifest_records_every_setting_and_version(tmp_path):
    pipeline, _, _ = recording_pipeline(tmp_path, num_runs=3)
    pipeline.ref = "feature/example"
    with_model(pipeline)
    stats = pipeline.run_collection()

    manifest = manifest_of(pipeline)
    assert manifest["status"] == "finished"
    assert manifest["model"] == pipeline.model_name
    assert manifest["precision"] == "float16"
    assert manifest["batch_size"] == pipeline.batch_size
    assert manifest["num_orderings"] == 3
    assert manifest["prompt_limit"] is None
    assert manifest["seeds"] == {
        "orderings": RUN_SEEDS[:3],
        "determinism_subset": DETERMINISM_SUBSET_SEED,
    }
    assert manifest["determinism_subset_size"] == DETERMINISM_SUBSET_SIZE
    assert manifest["same_batch_repeat"] == {"ordering": 0, "seed": RUN_SEEDS[0]}
    assert manifest["run_settings"] == {"TRITON_DEFAULT_FP_FUSION": "1"}

    software = manifest["software"]
    assert software["python"] == platform.python_version()
    assert software["torch"] == torch.__version__
    assert software["transformers"] == transformers.__version__
    assert software["accelerate"] == metadata.version("accelerate")
    assert "triton" in software  # None where Triton isn't installed (CPU-only torch)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()
    assert manifest["code"]["commit"] == head
    assert isinstance(manifest["code"]["uncommitted_changes"], bool)
    assert manifest["code"]["ref"] == "feature/example"

    expected_gpu = None if not torch.cuda.is_available() else torch.cuda.get_device_name(0)
    assert (manifest["gpu"] or {}).get("name") == expected_gpu
    assert manifest["started_utc"] and manifest["finished_utc"]

    results = manifest["results"]
    assert results["prompt_passes"] == stats.total_prompts_processed
    assert results["routing_rows"] == stats.total_routing_events
    assert results["full_passes_completed"] == 2
    assert results["orderings_completed"] == 3
    assert results["repeats_completed"] == 1
    assert results["errors"] == []
    assert not list(pipeline.output_dir.glob("*.tmp"))  # written whole, by rename


def test_manifest_is_written_before_any_data(tmp_path):
    """A run that crashes partway still says what it was running with."""
    pipeline, _, _ = recording_pipeline(tmp_path, num_runs=2)
    with_model(pipeline)
    seen = []
    record = pipeline.run_single_pass

    def first_look(*args, **kwargs):
        if not seen:
            seen.append(manifest_of(pipeline))
        return record(*args, **kwargs)

    pipeline.run_single_pass = first_look
    pipeline.run_collection()

    assert seen[0]["status"] == "running"
    assert seen[0]["model"] == pipeline.model_name
    assert "results" not in seen[0]


def test_a_crash_marks_the_manifest_failed(tmp_path):
    """Forced failure: an exception mid-collection leaves status "failed" and the error."""
    pipeline, _, _ = recording_pipeline(tmp_path, num_runs=2)
    with_model(pipeline)

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash")

    pipeline.run_single_pass = crash
    with pytest.raises(RuntimeError, match="simulated crash"):
        pipeline.run_collection()

    manifest = manifest_of(pipeline)
    assert manifest["status"] == "failed"
    assert "simulated crash" in manifest["error"]


def test_a_failed_pass_marks_the_manifest_failed(tmp_path):
    """Forced failure: a pass that records an error finishes the run, but not as "finished"."""
    pipeline, _, _ = recording_pipeline(
        tmp_path, num_runs=2, fail_on=lambda c: c.dataset == "mbpp" and c.path.parent.name == "full")
    with_model(pipeline)
    pipeline.run_collection()

    manifest = manifest_of(pipeline)
    assert manifest["status"] == "failed"
    assert manifest["results"]["errors"]


def test_the_collection_reruns_ordering_0_for_the_same_batch_check(tmp_path):
    pipeline, _, calls = recording_pipeline(tmp_path, num_runs=3)
    with_model(pipeline)
    stats = pipeline.run_collection()

    for dataset in ("gsm8k", "mbpp"):
        first = next(c for c in calls if c.dataset == dataset and c.path.name == f"run_0_{dataset}.parquet")
        repeats = [c for c in calls if c.dataset == dataset and c.path.name.startswith("repeat_")]
        assert len(repeats) == 1
        repeat = repeats[0]
        assert repeat.path == pipeline.output_dir / "determinism" / f"repeat_run_0_{dataset}.parquet"
        # Same seed and the same prompts in the same order, so the batches are identical
        assert (repeat.run_id, repeat.seed) == (0, first.seed) == (0, RUN_SEEDS[0])
        assert repeat.prompts == first.prompts
    assert stats.repeats_completed == 1
    assert stats.validation_passed


def test_a_failed_repeat_fails_the_collection(tmp_path):
    """Forced failure: without the repeat, the same-batch check can't run."""
    pipeline, _, _ = recording_pipeline(
        tmp_path, num_runs=2, fail_on=lambda c: c.path.name.startswith("repeat_") and c.dataset == "gsm8k")
    with_model(pipeline)
    stats = pipeline.run_collection()

    assert stats.repeats_completed == 0
    assert not stats.validation_passed
    assert manifest_of(pipeline)["status"] == "failed"


def test_a_toy_collection_passes_the_same_batch_check(tmp_path):
    """End to end on CPU: real parquet files from the toy model, then the validator."""
    from determinism_validation import DeterminismValidator

    pipeline = FullPipeline(output_dir=tmp_path / "collection", num_runs=2, batch_size=4)
    pipeline.tokenizer = CharTokenizer()
    pipeline.capture = make_capture(make_toy_qwen())
    pipeline.model = pipeline.capture.model
    pipeline.load_model = lambda: None
    pipeline.load_datasets = lambda: (
        synthetic_prompts("gsm8k", {"train": 6, "test": 4}),
        synthetic_prompts("mbpp", {"train": 5, "test": 5}),
    )
    stats = pipeline.run_collection()
    assert stats.validation_passed, stats.errors

    determinism = pipeline.output_dir / "determinism"
    assert {p.name for p in determinism.glob("repeat_*.parquet")} == {
        "repeat_run_0_gsm8k.parquet", "repeat_run_0_mbpp.parquet"}
    window = {"model": pipeline.model_name, "window": 0.001}
    report = DeterminismValidator(pipeline.output_dir, window=window).validate_all()
    assert report.same_batch.status == "passed", report.summary
    assert report.same_batch.compared_rows > 0
    assert report.near_tie.status == "ran", report.summary
    assert report.status == "passed", report.summary
