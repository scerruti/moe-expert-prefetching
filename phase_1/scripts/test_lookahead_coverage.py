#!/usr/bin/env python3
"""
The question-3 analysis (#114): over-prediction coverage and lookahead exposure.

Synthetic determinism runs (CPU), one token over 4 layers. The second ordering
has a near-tie flip at layer 0, and an unrelated, larger change at layer 3 that
cascades from it.
"""

import pyarrow as pa
import pyarrow.parquet as pq

import lookahead_coverage
from parquet_storage import ROUTING_SCHEMA
from routing_test_helpers import QWEN_TOP_K, routing_row

PROBS = [0.40, 0.25, 0.15, 0.10, 0.04, 0.03, 0.02, 0.01]
NEAR_TIE_A = [0.40, 0.25, 0.15, 0.095, 0.085, 0.01, 0.005, 0.005]
NEAR_TIE_B = [0.40, 0.25, 0.15, 0.085, 0.095, 0.01, 0.005, 0.005]
DIVERGED = [0.02, 0.03, 0.04, 0.01, 0.10, 0.15, 0.25, 0.40]  # top-4 = experts 4-7


def write_run(path, layer_probs):
    rows = []
    for layer, probs in enumerate(layer_probs):
        top = sorted(range(len(probs)), key=lambda e: -probs[e])[:QWEN_TOP_K]
        rows.append(routing_row(sequence_id="gsm8k-train-00000", token_position=0, layer_index=layer,
                                expert_probs=probs, selected_experts=top,
                                selected_probs=[probs[e] for e in top]))
    pq.write_table(pa.Table.from_pylist(rows, schema=ROUTING_SCHEMA), path)


def test_near_tie_flip_and_cascade_are_classified(tmp_path):
    write_run(tmp_path / "run_0_gsm8k.parquet", [NEAR_TIE_A, PROBS, PROBS, PROBS])
    write_run(tmp_path / "run_1_gsm8k.parquet", [NEAR_TIE_B, PROBS, PROBS, DIVERGED])

    rows = lookahead_coverage.comparisons(tmp_path).set_index("layer_index")

    assert rows.changed.tolist() == [True, False, False, True]
    assert (rows.first_flip == 0).all()

    # Layer 0: a near-tie, covered by predicting one extra expert
    assert rows.loc[0, "cov1"]
    # Layer 3: a real divergence, not covered by predicting one extra expert
    assert not rows.loc[3, "cov1"]

    # Lookahead: layer 0's change is unforeseeable for any H (its first flip is
    # at layer 0 itself); layer 3's is foreseeable for H <= 3, since the token
    # had already flipped at layer 0
    first, layer = rows.loc[3, "first_flip"], 3
    assert not (first > layer - 1) and not (first > layer - 3)
    assert first > layer - 4
