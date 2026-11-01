#!/usr/bin/env python3
"""
What Phase 1 collects, and how each row is identified (#104, #109).

- Rows carry stable "<dataset>-<split>-<example>" sequence IDs (#104).
- Every split of GSM8K and MBPP is collected; nothing is held out (#109).
- The full datasets are collected once, plus a stratified 200-prompt
  determinism subset per dataset in num_runs orderings (#109).

CPU only: the toy Qwen2MoE stands in for the model, and a character-level
tokenizer for the Qwen tokenizer.
"""

import re
from collections import Counter
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch

from full_pipeline import FullPipeline, Prompt, determinism_subset
from routing_test_helpers import make_capture, make_toy_qwen


class CharTokenizer:
    """Character-level tokenizer within the toy model's 128-token vocabulary; pads left with 0."""

    def encode(self, text):
        return [ord(c) % 127 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def __call__(self, texts, **kwargs):
        encoded = [self.encode(t) for t in texts]
        width = max(len(e) for e in encoded)
        ids = [[0] * (width - len(e)) + e for e in encoded]
        mask = [[0] * (width - len(e)) + [1] * len(e) for e in encoded]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def toy_pipeline(**kwargs):
    pipeline = FullPipeline(**kwargs)
    pipeline.tokenizer = CharTokenizer()
    pipeline.capture = make_capture(make_toy_qwen())
    return pipeline


def test_rows_carry_prompt_sequence_ids(tmp_path):
    """#104: every parquet row carries its prompt's dataset-split-example ID."""
    prompts = [
        Prompt("gsm8k-test-00007", "What is 2 + 2?"),
        Prompt("mbpp-validation-602", "Write a function to add two numbers."),
        Prompt("gsm8k-train-00000", "Natalia sold clips."),
    ]
    output = tmp_path / "run.parquet"

    processed, rows = toy_pipeline(batch_size=2).run_single_pass(0, 42, prompts, "test", output)

    assert processed == len(prompts) and rows > 0
    ids = pq.read_table(output, columns=["sequence_id"]).column(0).to_pylist()
    assert set(ids) == {p.sequence_id for p in prompts}


def split_of(prompt):
    return prompt.sequence_id.split("-")[1]


def test_load_datasets_covers_every_split():
    """#109: GSM8K train + test and all four MBPP splits, each with a unique, traceable ID."""
    gsm8k, mbpp = FullPipeline().load_datasets()

    assert Counter(map(split_of, gsm8k)) == {"train": 7473, "test": 1319}
    assert Counter(map(split_of, mbpp)) == {"train": 374, "test": 500, "validation": 90, "prompt": 10}
    for prompts, pattern in ((gsm8k, r"gsm8k-(train|test)-\d{5}"), (mbpp, r"mbpp-[a-z]+-\d+")):
        ids = [p.sequence_id for p in prompts]
        assert len(set(ids)) == len(ids)
        assert all(re.fullmatch(pattern, i) for i in ids)
        assert all(isinstance(p.text, str) and p.text for p in prompts)


def synthetic_prompts(dataset, split_sizes):
    return [
        Prompt(f"{dataset}-{split}-{i:05d}", f"{dataset} {split} prompt {i}")
        for split, n in split_sizes.items()
        for i in range(n)
    ]


def test_determinism_subset_is_stratified_by_split():
    """#109: 200 prompts, shared across splits in proportion to their size."""
    gsm8k = synthetic_prompts("gsm8k", {"train": 7473, "test": 1319})
    mbpp = synthetic_prompts("mbpp", {"train": 374, "test": 500, "validation": 90, "prompt": 10})

    gsm8k_subset = determinism_subset(gsm8k)
    mbpp_subset = determinism_subset(mbpp)

    assert Counter(map(split_of, gsm8k_subset)) == {"train": 170, "test": 30}
    assert Counter(map(split_of, mbpp_subset)) == {"train": 77, "test": 103, "validation": 18, "prompt": 2}
    for subset, prompts in ((gsm8k_subset, gsm8k), (mbpp_subset, mbpp)):
        assert len(set(subset)) == 200 and set(subset) <= set(prompts)


def test_determinism_subset_is_reproducible():
    """#109: the same prompts and seed always give the same subset, whatever the input order."""
    prompts = synthetic_prompts("gsm8k", {"train": 500, "test": 100})
    assert determinism_subset(prompts) == determinism_subset(list(reversed(prompts)))
    assert determinism_subset(prompts, seed=1) != determinism_subset(prompts, seed=2)


def test_determinism_subset_of_small_dataset_is_everything():
    prompts = synthetic_prompts("mbpp", {"train": 30})
    assert set(determinism_subset(prompts)) == set(prompts)


def recording_pipeline(tmp_path, num_runs, fail_on=None):
    """FullPipeline whose model loading and passes are replaced by a recorder."""
    pipeline = FullPipeline(num_runs=num_runs, output_dir=tmp_path)
    datasets = (
        synthetic_prompts("gsm8k", {"train": 900, "test": 100}),
        synthetic_prompts("mbpp", {"train": 50, "test": 60}),
    )
    pipeline.load_model = lambda: None
    pipeline.load_datasets = lambda: datasets
    pipeline.capture = SimpleNamespace(cleanup=lambda: None)
    calls = []

    def run_single_pass(run_id, seed, prompts, dataset_name, output_path, label=None):
        calls.append(SimpleNamespace(run_id=run_id, seed=seed, prompts=list(prompts),
                                     dataset=dataset_name, path=output_path))
        if fail_on and fail_on(calls[-1]):
            pipeline.stats.errors.append(f"Run {run_id + 1}: simulated failure")
            return 0, 0
        return len(prompts), 10 * len(prompts)

    pipeline.run_single_pass = run_single_pass
    return pipeline, datasets, calls


def test_run_collection_runs_full_pass_once_then_subset_orderings(tmp_path):
    """#109: one full pass per dataset, then num_runs orderings of the same subset."""
    pipeline, (gsm8k, mbpp), calls = recording_pipeline(tmp_path, num_runs=3)
    stats = pipeline.run_collection()

    full = [c for c in calls if c.path.parent.name == "full"]
    subset = [c for c in calls if c.path.parent.name == "determinism"]
    assert len(calls) == 2 + 3 * 2

    assert {c.dataset: set(c.prompts) for c in full} == {"gsm8k": set(gsm8k), "mbpp": set(mbpp)}
    for dataset in ("gsm8k", "mbpp"):
        runs = [c for c in subset if c.dataset == dataset]
        assert [c.run_id for c in runs] == [0, 1, 2]
        assert len({c.seed for c in runs}) == 3
        assert all(set(c.prompts) == set(runs[0].prompts) for c in runs)
        assert len(runs[0].prompts) == min(200, len(gsm8k if dataset == "gsm8k" else mbpp))
        assert len({c.path for c in runs}) == 3

    assert stats.full_passes_completed == 2
    assert stats.runs_completed == 3
    assert stats.validation_passed


def test_failed_ordering_is_not_counted_as_completed(tmp_path):
    """A failed pass doesn't count toward runs_completed, and validation fails (#83)."""
    pipeline, _, _ = recording_pipeline(
        tmp_path, num_runs=3,
        fail_on=lambda c: c.dataset == "mbpp" and c.path.parent.name == "determinism" and c.run_id == 1,
    )
    stats = pipeline.run_collection()

    assert stats.full_passes_completed == 2
    assert stats.runs_completed == 2
    assert not stats.validation_passed


def test_pass_records_stage_timings(tmp_path):
    """#106: each pass adds a per-stage timing breakdown to the run's stats."""
    pipeline = toy_pipeline(batch_size=2)
    prompts = [Prompt(f"gsm8k-train-{i:05d}", f"Question number {i}?") for i in range(5)]

    pipeline.run_single_pass(0, 42, prompts, "test", tmp_path / "run.parquet")

    timings = pipeline.stats.timings
    assert {"tokenize", "forward", "extract", "convert", "write", "verify", "other"} <= set(timings)
    assert all(seconds >= 0 for seconds in timings.values())
    assert timings["forward"] > 0 and timings["extract"] > 0
