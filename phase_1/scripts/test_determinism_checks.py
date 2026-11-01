#!/usr/bin/env python3
"""
Determinism validation must fail on missing data and on real differences,
and must scale to full runs (#105).
"""

import time

import pyarrow as pa
import pyarrow.parquet as pq

from determinism_validation import DeterminismValidator
from parquet_storage import ROUTING_SCHEMA
from routing_test_helpers import QWEN_TOP_K, TOY_EXPERTS, routing_row


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


def test_determinism_check_fails_without_runs_to_compare(tmp_path):
    """#105: one run has nothing to compare against; that must not count as a pass."""
    write_run(tmp_path / "run_0_gsm8k.parquet", synthetic_probs(100))
    assert DeterminismValidator(tmp_path).check_determinism().passed is False


def test_determinism_check_fails_on_one_differing_row(tmp_path):
    """#105: a single row whose routing changed between runs is non-determinism."""
    probs = synthetic_probs(200)
    write_run(tmp_path / "run_0_gsm8k.parquet", probs)
    changed = [list(p) for p in probs]
    changed[17] = list(reversed(changed[17]))
    write_run(tmp_path / "run_1_gsm8k.parquet", changed)
    assert DeterminismValidator(tmp_path).check_determinism().passed is False


def test_determinism_check_scales_to_large_runs(tmp_path):
    """#105: the cross-run comparison is vectorised, not O(N^2) iterrows filtering."""
    probs = synthetic_probs(10_000)
    write_run(tmp_path / "run_0_gsm8k.parquet", probs)
    write_run(tmp_path / "run_1_gsm8k.parquet", probs)

    start = time.perf_counter()
    result = DeterminismValidator(tmp_path).check_determinism()
    elapsed = time.perf_counter() - start

    assert result.total_comparisons == 10_000
    assert elapsed < 2.0, f"check_determinism took {elapsed:.1f}s for 10k rows"
