#!/usr/bin/env python3
"""
selected_probs must hold the selected experts' probabilities only (#101).

Validation takes k from the data, not a Qwen constant: full_pipeline.py is in
the model-agnostic part of Phase 1 (PHASE_1_MODEL_DECISION.md).
"""

import pytest

from full_pipeline import FullPipeline
from routing_test_helpers import (
    QWEN_TOP_K,
    SEQ_IDS,
    TOY_EXPERTS,
    TOY_PROMPTS,
    capture_each_alone,
    make_capture,
    make_toy_qwen,
    routing_row,
)


def test_selected_probs_are_probabilities_of_selected_experts():
    """#101: selected_probs == expert_probs[selected_experts], one per selected expert."""
    capture = make_capture(make_toy_qwen())
    rows = capture_each_alone(capture, TOY_PROMPTS, SEQ_IDS)
    for key, layer in rows.items():
        experts, probs = layer["selected_experts"], layer["selected_probs"]
        assert len(probs) == len(experts), key
        expected = [layer["expert_probs"][e] for e in experts]
        assert probs == pytest.approx(expected, abs=1e-6), key


def test_validate_row_accepts_consistent_row():
    """#101 (control): a well-formed row is valid."""
    assert FullPipeline()._validate_row(routing_row())


def test_validate_row_rejects_selected_length_mismatch():
    """#101: selected_probs must have one entry per selected expert."""
    row = routing_row(selected_probs=[1 / TOY_EXPERTS] * TOY_EXPERTS)
    assert not FullPipeline()._validate_row(row)


def test_validate_row_rejects_widths_that_change_within_a_run():
    """#101: expert_probs and selected_experts keep the same lengths across a run."""
    pipeline = FullPipeline()
    assert pipeline._validate_row(routing_row())

    wider_probs = routing_row(expert_probs=[1 / (TOY_EXPERTS * 2)] * (TOY_EXPERTS * 2))
    assert not pipeline._validate_row(wider_probs)

    fewer_selected = routing_row(
        selected_experts=[0, 1], selected_probs=[1 / TOY_EXPERTS] * 2
    )
    assert not pipeline._validate_row(fewer_selected)
    assert QWEN_TOP_K == len(routing_row()["selected_experts"])  # control: fixture is top-4
