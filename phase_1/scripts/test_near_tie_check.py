#!/usr/bin/env python3
"""
The determinism check as decided in phase_1/docs/determinism_check.md (#128, #8):
root changes, a near-tie window computed from a pilot run, warnings rather
than failures for cross-batch changes, and a same-batch bitwise check.
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from run_manifest import write_manifest
from determinism_validation import (
    DeterminismValidator,
    InsufficientPilotError,
    compare_orderings,
    compute_window,
    required_root_changes,
)
from parquet_storage import ROUTING_SCHEMA
from routing_test_helpers import QWEN_TOP_K, routing_row

# 8 toy experts, top 4. The gap between the 4th and 5th is what the check measures.
CLEAR = [0.40, 0.25, 0.15, 0.10, 0.04, 0.03, 0.02, 0.01]          # gap 0.06
NEAR_TIE = [0.40, 0.25, 0.15, 0.0905, 0.0895, 0.01, 0.005, 0.005]  # gap 0.001
TIE = [0.40, 0.25, 0.15, 0.09, 0.09, 0.01, 0.005, 0.005]          # gap 0: an exact tie
TOP4 = [0, 1, 2, 3]
SWAPPED = [0, 1, 2, 4]  # the 4th expert lost the near-tie to the 5th


def write(path, rows):
    """rows: (sequence_id, position, layer, probs, selected_experts)."""
    records = [
        routing_row(sequence_id=s, token_position=p, layer_index=l, expert_probs=probs,
                    selected_experts=sel, selected_probs=[probs[e] for e in sel])
        for s, p, l, probs, sel in rows
    ]
    pq.write_table(pa.Table.from_pylist(records, schema=ROUTING_SCHEMA), path)


def grid(sequences=("s0",), positions=4, layers=4, probs=CLEAR):
    """Every (sequence, position, layer) with the same routing."""
    return {(s, p, l): (probs, TOP4) for s in sequences for p in range(positions) for l in range(layers)}


def write_grid(path, cells):
    write(path, [(s, p, l, probs, sel) for (s, p, l), (probs, sel) in sorted(cells.items())])


def changes(tmp_path, changed_cells, base=None):
    """Compare a reference run with one where changed_cells route differently."""
    base = base or grid()
    write_grid(tmp_path / "run_0_gsm8k.parquet", base)
    other = dict(base)
    other.update(changed_cells)
    write_grid(tmp_path / "run_1_gsm8k.parquet", other)
    return compare_orderings(tmp_path / "run_0_gsm8k.parquet", tmp_path / "run_1_gsm8k.parquet")


def roots(result):
    rows = result.changes[result.changes.is_root]
    return sorted(zip(rows.sequence_id, rows.token_position, rows.layer_index))


# ---------- which changes are root changes ----------

def test_a_change_with_no_earlier_change_is_a_root(tmp_path):
    result = changes(tmp_path, {("s0", 1, 2): (NEAR_TIE, SWAPPED)})
    assert len(result.changes) == 1 and roots(result) == [("s0", 1, 2)]


def test_a_later_change_on_the_same_token_is_a_knock_on(tmp_path):
    result = changes(tmp_path, {("s0", 1, 0): (NEAR_TIE, SWAPPED), ("s0", 1, 2): (CLEAR, SWAPPED)})
    assert roots(result) == [("s0", 1, 0)]


def test_a_change_after_an_earlier_tokens_lower_layer_change_is_a_knock_on(tmp_path):
    """Waleed's point (#128): attention carries a change to later tokens."""
    result = changes(tmp_path, {("s0", 0, 0): (NEAR_TIE, SWAPPED), ("s0", 3, 1): (CLEAR, SWAPPED)})
    assert roots(result) == [("s0", 0, 0)]


def test_an_earlier_tokens_change_at_a_higher_layer_is_not_a_cause(tmp_path):
    """A change at a higher layer can't reach a lower layer of a later token."""
    result = changes(tmp_path, {("s0", 0, 3): (NEAR_TIE, SWAPPED), ("s0", 2, 1): (NEAR_TIE, SWAPPED)})
    assert roots(result) == [("s0", 0, 3), ("s0", 2, 1)]


def test_a_later_tokens_change_is_not_a_cause(tmp_path):
    """Attention is causal: a later token can't affect an earlier one."""
    result = changes(tmp_path, {("s0", 3, 0): (NEAR_TIE, SWAPPED), ("s0", 1, 2): (NEAR_TIE, SWAPPED)})
    assert roots(result) == [("s0", 1, 2), ("s0", 3, 0)]


def test_sequences_are_independent(tmp_path):
    base = grid(sequences=("s0", "s1"))
    result = changes(tmp_path, {("s0", 0, 0): (NEAR_TIE, SWAPPED), ("s1", 2, 1): (NEAR_TIE, SWAPPED)}, base)
    assert roots(result) == [("s0", 0, 0), ("s1", 2, 1)]


# ---------- what counts as a change, and its gap ----------

def test_a_change_is_read_from_the_routers_choice(tmp_path):
    """Identical probabilities with an exact tie can still route differently; that's a change."""
    base = grid(probs=TIE)
    result = changes(tmp_path, {("s0", 0, 0): (TIE, SWAPPED)}, base)
    assert len(result.changes) == 1
    assert result.changes.gap.iloc[0] == 0.0


def test_the_same_experts_in_a_different_order_are_not_a_change(tmp_path):
    result = changes(tmp_path, {("s0", 0, 0): (CLEAR, list(reversed(TOP4)))})
    assert result.changes.empty and result.compared == 16


def test_the_gap_is_the_smaller_of_the_two_orderings_gaps(tmp_path):
    result = changes(tmp_path, {("s0", 0, 0): (NEAR_TIE, SWAPPED)})  # 0.06 in run 0, 0.001 in run 1
    assert result.changes.gap.iloc[0] == pytest.approx(0.001, abs=1e-6)


def test_missing_and_duplicate_rows_are_problems_not_passes(tmp_path):
    base = grid()
    write_grid(tmp_path / "run_0_gsm8k.parquet", base)
    fewer = dict(base)
    del fewer[("s0", 0, 0)]
    write_grid(tmp_path / "run_1_gsm8k.parquet", fewer)
    result = compare_orderings(tmp_path / "run_0_gsm8k.parquet", tmp_path / "run_1_gsm8k.parquet")
    assert any("not in both" in p for p in result.problems)

    rows = [(s, p, l, probs, sel) for (s, p, l), (probs, sel) in sorted(base.items())]
    write(tmp_path / "run_2_gsm8k.parquet", rows + rows[:1])
    result = compare_orderings(tmp_path / "run_0_gsm8k.parquet", tmp_path / "run_2_gsm8k.parquet")
    assert any("duplicate" in p for p in result.problems)


# ---------- computing the window from a pilot ----------

def test_the_required_count_comes_from_the_target_rate():
    assert required_root_changes(0.01) == 99
    assert required_root_changes(0.05) == 19


def pilot_with_roots(folder, gaps):
    """A pilot whose ordering 1 has one root change per gap, each in its own sequence."""
    folder.mkdir(parents=True, exist_ok=True)
    ref, other = [], []
    for i, g in enumerate(gaps):
        probs = [0.40, 0.25, 0.15, 0.09 + g / 2, 0.09 - g / 2, 0.01, 0.005, 0.005]
        ref.append((f"s{i}", 0, 0, probs, TOP4))
        other.append((f"s{i}", 0, 0, probs, SWAPPED))
    write(folder / "run_0_gsm8k.parquet", ref)
    write(folder / "run_1_gsm8k.parquet", other)


def test_the_window_is_the_margin_times_the_largest_root_gap(tmp_path):
    gaps = [0.0001 * (i % 7) for i in range(99)] + [0.0004]
    pilot_with_roots(tmp_path / "pilot", gaps)

    record = compute_window(tmp_path / "pilot", target_rate=0.01, margin=3.0, model="toy-moe")

    assert record["root_changes"] == 100
    assert record["largest_root_gap"] == pytest.approx(0.0006, abs=1e-6)
    assert record["window"] == pytest.approx(3 * record["largest_root_gap"])
    assert record["required_root_changes"] == 99 and record["margin"] == 3.0
    assert record["comparison"] == "ordering 0 against each other ordering"
    assert record["pilot_files"]  # each pilot file with its fingerprint


def test_a_pilot_with_too_few_root_changes_is_refused(tmp_path):
    pilot_with_roots(tmp_path / "pilot", [0.0001] * 98)
    with pytest.raises(InsufficientPilotError, match="98 root changes; at least 99"):
        compute_window(tmp_path / "pilot", target_rate=0.01, margin=3.0, model="toy-moe")


def test_orderings_are_compared_with_ordering_0_only(tmp_path):
    """Ordering 2 differs from 0 and from 1; only the 0-vs-2 comparison counts."""
    base = grid()
    write_grid(tmp_path / "run_0_gsm8k.parquet", base)
    write_grid(tmp_path / "run_1_gsm8k.parquet", base)
    changed = dict(base)
    changed[("s0", 0, 0)] = (NEAR_TIE, SWAPPED)
    write_grid(tmp_path / "run_2_gsm8k.parquet", changed)

    result = DeterminismValidator(tmp_path, window={"window": 0.01}).check_near_tie()

    assert result.root_changes == 1  # all pairs would count it twice


# ---------- the near-tie check on a collection ----------

def collection(tmp_path, changed_cells):
    base = grid()
    write_grid(tmp_path / "run_0_gsm8k.parquet", base)
    other = dict(base)
    other.update(changed_cells)
    write_grid(tmp_path / "run_1_gsm8k.parquet", other)
    return DeterminismValidator(tmp_path, window={"window": 0.003})


def test_a_root_change_outside_the_window_is_a_warning_not_a_failure(tmp_path):
    result = collection(tmp_path, {("s0", 1, 1): (CLEAR, SWAPPED)}).check_near_tie()  # gap 0.06
    assert result.status == "ran"
    assert result.root_changes == 1 and len(result.outside_window) == 1
    assert result.outside_window[0]["layer_index"] == 1


def test_knock_ons_are_reported_but_never_warned(tmp_path):
    result = collection(tmp_path, {("s0", 0, 0): (NEAR_TIE, SWAPPED),
                                   ("s0", 2, 3): (CLEAR, SWAPPED)}).check_near_tie()
    assert result.root_changes == 1 and result.knock_ons == 1
    assert result.outside_window == []


def test_without_a_window_record_the_check_does_not_run(tmp_path):
    write_grid(tmp_path / "run_0_gsm8k.parquet", grid())
    write_grid(tmp_path / "run_1_gsm8k.parquet", grid())
    result = DeterminismValidator(tmp_path).check_near_tie()
    assert result.status == "not_run" and "window" in result.reason


def test_the_window_can_be_read_from_a_record_file(tmp_path):
    record = tmp_path / "window.json"
    record.write_text(json.dumps({"window": 0.003, "pilot": "trial"}))
    validator = collection(tmp_path, {})
    validator = DeterminismValidator(tmp_path, window=record)
    assert validator.check_near_tie().window == 0.003


# ---------- the same-batch check ----------

def test_an_identical_repeat_passes_the_same_batch_check(tmp_path):
    write_grid(tmp_path / "run_0_gsm8k.parquet", grid())
    write_grid(tmp_path / "run_1_gsm8k.parquet", grid())
    write_grid(tmp_path / "repeat_run_0_gsm8k.parquet", grid())
    assert DeterminismValidator(tmp_path).check_same_batch().status == "passed"


def test_any_difference_in_a_repeat_fails_the_same_batch_check(tmp_path):
    write_grid(tmp_path / "run_0_gsm8k.parquet", grid())
    nudged = grid()
    nudged[("s0", 2, 2)] = ([CLEAR[0] - 1e-6, CLEAR[1] + 1e-6] + CLEAR[2:], TOP4)
    write_grid(tmp_path / "repeat_run_0_gsm8k.parquet", nudged)
    result = DeterminismValidator(tmp_path).check_same_batch()
    assert result.status == "failed" and result.differing_rows == 1


def test_without_a_repeat_the_same_batch_check_does_not_run(tmp_path):
    write_grid(tmp_path / "run_0_gsm8k.parquet", grid())
    write_grid(tmp_path / "run_1_gsm8k.parquet", grid())
    result = DeterminismValidator(tmp_path).check_same_batch()
    assert result.status == "not_run" and "repeat" in result.reason


def test_a_check_that_did_not_run_keeps_the_report_from_passing(tmp_path):
    write_grid(tmp_path / "run_0_gsm8k.parquet", grid())
    write_grid(tmp_path / "run_1_gsm8k.parquet", grid())
    write_manifest(tmp_path, {"status": "finished", "model": "toy-moe"})  # validate_all needs one (#134)
    report = DeterminismValidator(tmp_path, window={"model": "toy-moe", "window": 0.003}).validate_all()
    assert report.status == "incomplete" and not report.overall_passed
    assert "same-batch" in report.summary.lower() and "not run" in report.summary.lower()
