#!/usr/bin/env python3
"""
The batch-variance analysis behind #114 (batch_variance.py).

Synthetic determinism runs (CPU): one ordering is a copy of the other except
for a near-tie flip at layer 0 and a large difference later on the same token.
"""

import pyarrow as pa
import pyarrow.parquet as pq

import batch_variance
from parquet_storage import ROUTING_SCHEMA
from routing_test_helpers import QWEN_TOP_K, routing_row

PROBS = [0.40, 0.25, 0.15, 0.10, 0.04, 0.03, 0.02, 0.01]
NEAR_TIE_A = [0.40, 0.25, 0.15, 0.095, 0.085, 0.01, 0.005, 0.005]
NEAR_TIE_B = [0.40, 0.25, 0.15, 0.085, 0.095, 0.01, 0.005, 0.005]
SHIFTED = [0.30, 0.35, 0.15, 0.10, 0.04, 0.03, 0.02, 0.01]  # differs by 0.1 from PROBS


def write_run(path, layer_probs):
    rows = []
    for layer, probs in enumerate(layer_probs):
        top = sorted(range(len(probs)), key=lambda e: -probs[e])[:QWEN_TOP_K]
        rows.append(routing_row(sequence_id="gsm8k-train-00000", token_position=0, layer_index=layer,
                                expert_probs=probs, selected_experts=top,
                                selected_probs=[probs[e] for e in top]))
    pq.write_table(pa.Table.from_pylist(rows, schema=ROUTING_SCHEMA), path)


def test_identical_runs_report_no_changes_and_no_nan(tmp_path, capsys):
    write_run(tmp_path / "run_0_gsm8k.parquet", [PROBS, PROBS, PROBS])
    write_run(tmp_path / "run_1_gsm8k.parquet", [PROBS, PROBS, PROBS])

    batch_variance.main(tmp_path)
    out = capsys.readouterr().out

    assert "flips 0.00%" in out and "bitwise 100.00%" in out
    assert "nan" not in out and "n/a (no such rows)" in out


def test_large_difference_after_a_flip_counts_as_cascade(tmp_path, capsys):
    write_run(tmp_path / "run_0_gsm8k.parquet", [NEAR_TIE_A, PROBS, PROBS])
    write_run(tmp_path / "run_1_gsm8k.parquet", [NEAR_TIE_B, PROBS, SHIFTED])

    batch_variance.main(tmp_path)
    out = capsys.readouterr().out

    assert "flip on the same token: 100.0%" in out
    assert "large deviations with no earlier flip: 0 rows" in out
