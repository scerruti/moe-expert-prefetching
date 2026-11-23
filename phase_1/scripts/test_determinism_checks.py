#!/usr/bin/env python3
"""
Determinism validation must fail on missing or duplicate data rather than
pass, and must scale to full runs (#105). The rules themselves are tested
in test_near_tie_check.py.
"""

import time

import pyarrow as pa
import pyarrow.parquet as pq

from determinism_validation import DeterminismValidator
from parquet_storage import ROUTING_SCHEMA
from routing_test_helpers import QWEN_TOP_K, TOY_EXPERTS, routing_row

WINDOW = {"window": 0.003}


def synthetic_probs(num_rows):
    """Distinct, valid probability vectors, one per row."""
    base = [0.30, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05]
    return [base[i % TOY_EXPERTS:] + base[:i % TOY_EXPERTS] for i in range(num_rows)]


def write_run(path, probs_per_row):
    """Write one run's parquet file; keys (sequence_id, position, layer) are unique per row."""
    rows = []
    for i, probs in enumerate(probs_per_row):
        top = sorted(range(len(probs)), key=lambda e: -probs[e])[:QWEN_TOP_K]
        rows.append(routing_row(
            sequence_id=f"s{i // 10}",
            token_id=i,
            token_str=str(i),
            token_position=i % 10,
            expert_probs=probs,
            selected_experts=top,
            selected_probs=[probs[e] for e in top],
        ))
    pq.write_table(pa.Table.from_pylist(rows, schema=ROUTING_SCHEMA), path)


def test_one_ordering_fails_rather_than_passing(tmp_path):
    """#105: one run has nothing to compare against; that must not count as a pass."""
    write_run(tmp_path / "run_0_gsm8k.parquet", synthetic_probs(100))
    result = DeterminismValidator(tmp_path, window=WINDOW).check_near_tie()
    assert result.status == "failed" and "at least 2" in result.problems[0]


def test_determinism_check_scales_to_large_runs(tmp_path):
    """#105: the cross-run comparison is vectorised, not O(N^2) iterrows filtering."""
    probs = synthetic_probs(10_000)
    write_run(tmp_path / "run_0_gsm8k.parquet", probs)
    write_run(tmp_path / "run_1_gsm8k.parquet", probs)

    start = time.perf_counter()
    result = DeterminismValidator(tmp_path, window=WINDOW).check_near_tie()
    elapsed = time.perf_counter() - start

    assert result.compared_decisions == 10_000
    assert elapsed < 2.0, f"check_near_tie took {elapsed:.1f}s for 10k rows"


def test_report_with_numpy_values_saves_as_json(tmp_path):
    """#63: numpy scalars and numpy dict keys in a report serialise to JSON."""
    import json

    import numpy as np

    from determinism_validation import ExpertActivationStats, ValidationReport

    report = ValidationReport(model="qwen1.5-moe-a2.7b", dataset="gsm8k")
    report.near_tie.largest_root_gap = np.float32(0.125)
    report.expert_stats = [ExpertActivationStats(
        layer_index=0,
        per_expert_count={np.int16(3): np.int64(7)},
        mean_activation=np.float32(0.5),
    )]
    path = tmp_path / "report.json"

    DeterminismValidator(tmp_path).save_report(report, path)

    saved = json.loads(path.read_text())
    assert saved["near_tie"]["largest_root_gap"] == 0.125
    assert saved["expert_stats"][0]["per_expert_count"] == {"3": 7}


# Probabilities with a clear gap (0.06) between the 4th and 5th expert.
CLEAR = [0.40, 0.25, 0.15, 0.10, 0.04, 0.03, 0.02, 0.01]


def write_rows(path, rows):
    """rows: (sequence_id, position, probs, selected_experts or None for top-k of probs)."""
    records = []
    for seq_id, pos, probs, selected in rows:
        top = selected or sorted(range(len(probs)), key=lambda e: -probs[e])[:QWEN_TOP_K]
        records.append(routing_row(
            sequence_id=seq_id, token_position=pos, expert_probs=probs,
            selected_experts=top, selected_probs=[probs[e] for e in top],
        ))
    pq.write_table(pa.Table.from_pylist(records, schema=ROUTING_SCHEMA), path)


def test_missing_rows_fail(tmp_path):
    """#105: a row present in one ordering but not another fails determinism."""
    write_rows(tmp_path / "run_0_gsm8k.parquet", [("s0", 0, CLEAR, None), ("s0", 1, CLEAR, None)])
    write_rows(tmp_path / "run_1_gsm8k.parquet", [("s0", 0, CLEAR, None)])

    result = DeterminismValidator(tmp_path, window=WINDOW).check_near_tie()

    assert result.status == "failed" and "1 rows not in both" in result.problems[0]


def test_duplicate_rows_fail(tmp_path):
    """#105: the same (sequence_id, position, layer) twice in a run fails."""
    write_rows(tmp_path / "run_0_gsm8k.parquet", [("s0", 0, CLEAR, None), ("s0", 0, CLEAR, None)])
    write_rows(tmp_path / "run_1_gsm8k.parquet", [("s0", 0, CLEAR, None)])

    result = DeterminismValidator(tmp_path, window=WINDOW).check_near_tie()
    assert result.status == "failed" and "duplicate" in result.problems[0]


def test_collection_layout_uses_determinism_and_full_directories(tmp_path):
    """#105: orderings come from determinism/; activation counts come from full/ only."""
    (tmp_path / "full").mkdir()
    (tmp_path / "determinism").mkdir()
    write_rows(tmp_path / "full" / "gsm8k.parquet", [("s0", 0, CLEAR, None), ("s9", 0, CLEAR, None)])
    for run in range(3):
        write_rows(tmp_path / "determinism" / f"run_{run}_gsm8k.parquet", [("s0", 0, CLEAR, None)])
    validator = DeterminismValidator(tmp_path, window=WINDOW)

    near_tie = validator.check_near_tie()
    activation = validator.analyze_expert_activation()

    assert near_tie.status == "ran" and near_tie.compared_decisions == 2  # run_1 and run_2 vs run_0
    assert activation[0].total_activations == 2 * QWEN_TOP_K  # the two full/ rows, not subset runs
