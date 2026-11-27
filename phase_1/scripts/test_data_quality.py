#!/usr/bin/env python3
"""
The data-quality check covers every probability column (#145).

check_data_quality() decides whether a collection's data can be trusted, and the
official collection passes or fails on it (#135). These tests break one thing at
a time in a single routing row and check that the matching count catches it.
CPU only: rows in the Phase 1 parquet schema, written to a temporary folder.
"""

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from determinism_validation import DeterminismValidator
from parquet_storage import ROUTING_SCHEMA
from routing_test_helpers import TOY_EXPERTS, routing_row

SAMPLE = Path(__file__).resolve().parent / "fixtures" / "trial_sample"
UNIFORM = 1 / TOY_EXPERTS


def check(tmp_path, *rows):
    """check_data_quality() on one file holding a clean row plus the given rows."""
    table = pa.Table.from_pylist([routing_row()] + list(rows), schema=ROUTING_SCHEMA)
    pq.write_table(table, tmp_path / "run_0_gsm8k.parquet")
    return DeterminismValidator(tmp_path).check_data_quality()


def test_a_clean_row_passes_with_every_count_zero(tmp_path):
    result = check(tmp_path)
    assert result.passed, result
    assert result.total_rows == 1
    assert (result.nan_count, result.inf_count, result.invalid_prob_sums, result.invalid_indices,
            result.selected_nonfinite, result.out_of_range_values, result.selected_prob_mismatches,
            result.selected_length_mismatches) == (0, 0, 0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_selected_probs_fail(tmp_path, bad):
    """Forced failure: selected_probs used to be ignored entirely."""
    result = check(tmp_path, routing_row(selected_probs=[bad, UNIFORM, UNIFORM, UNIFORM]))
    assert not result.passed
    assert result.selected_nonfinite == 1


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_selected_probs_outside_zero_to_one_fail(tmp_path, bad):
    """Forced failure: a negative or above-one selected probability."""
    result = check(tmp_path, routing_row(selected_probs=[bad, UNIFORM, UNIFORM, UNIFORM]))
    assert not result.passed
    assert result.out_of_range_values == 1


def test_expert_probs_outside_zero_to_one_fail_even_when_the_row_sums_to_one(tmp_path):
    """Forced failure: 1.2 and -0.2 cancel out, so the sum check alone passes this row."""
    probs = [1.2, -0.2] + [0.0] * (TOY_EXPERTS - 2)
    result = check(tmp_path, routing_row(expert_probs=probs, selected_experts=[0, 2, 3, 4],
                                         selected_probs=[1.2, 0.0, 0.0, 0.0]))
    assert not result.passed
    assert result.invalid_prob_sums == 0  # the old check's only probability test misses it
    assert result.out_of_range_values == 3  # -0.2 and 1.2 in expert_probs, 1.2 in selected_probs


def test_selected_probs_must_equal_expert_probs_at_the_selected_experts(tmp_path):
    """Forced failure: selected_probs that don't come from the row's expert_probs."""
    result = check(tmp_path, routing_row(selected_probs=[0.5, UNIFORM, UNIFORM, UNIFORM]))
    assert not result.passed
    assert result.selected_prob_mismatches == 1


def test_float32_rounding_is_not_a_mismatch(tmp_path):
    """The same value stored twice as float32 matches; only real differences count."""
    probs = [0.1, 0.2, 0.3, 0.15, 0.05, 0.1, 0.05, 0.05]
    result = check(tmp_path, routing_row(expert_probs=probs, selected_experts=[2, 1, 3, 0],
                                         selected_probs=[0.3, 0.2, 0.15, 0.1]))
    assert result.passed, result
    assert result.selected_prob_mismatches == 0


def test_selected_probs_of_a_different_length_fail(tmp_path):
    """Forced failure: three selected probabilities for four selected experts."""
    result = check(tmp_path, routing_row(selected_probs=[UNIFORM] * 3))
    assert not result.passed
    assert result.selected_length_mismatches == 1


def test_the_expert_count_comes_from_the_data_not_a_constant(tmp_path):
    """Forced failure: a row with no expert_probs has no valid selection. The old check
    assumed 60 experts for such a row, so indices 0-3 passed."""
    result = check(tmp_path, routing_row(expert_probs=[], selected_probs=[]))
    assert not result.passed
    assert result.invalid_indices == 4


def test_a_wider_model_is_checked_against_its_own_width(tmp_path):
    """Indices beyond 60 are valid for a row with 64 experts, and beyond 64 are not."""
    wide = [1 / 64] * 64
    ok = routing_row(expert_probs=wide, selected_experts=[60, 61, 62, 63], selected_probs=[1 / 64] * 4)
    bad = routing_row(expert_probs=wide, selected_experts=[60, 61, 62, 64], selected_probs=[1 / 64] * 4)
    assert check(tmp_path, ok).passed
    result = check(tmp_path, bad)
    assert not result.passed and result.invalid_indices == 1


def test_the_trial_sample_still_passes():
    """Real routing data (a 10% sample of the trial) passes every new check."""
    result = DeterminismValidator(SAMPLE).check_data_quality()
    assert result.passed, result
    assert result.total_rows > 0
    assert (result.selected_nonfinite, result.out_of_range_values, result.selected_prob_mismatches,
            result.selected_length_mismatches) == (0, 0, 0, 0)
