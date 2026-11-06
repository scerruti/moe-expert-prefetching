#!/usr/bin/env python3
"""
Extraction must read each prompt's own gate rows (#99, reference test #102).

The gate output is flattened to [batch * padded_len, num_experts] and prompts
are left-padded, so a prompt's real tokens sit at the end of its block. Rows
from a batched run must equal rows from running each prompt on its own.
"""

import pytest

from routing_test_helpers import (
    SEQ_IDS,
    TOY_PROMPTS,
    capture_each_alone,
    capture_routing,
    make_capture,
    make_toy_qwen,
    requires_gpu,
)


def test_batched_rows_match_single_prompt_runs():
    """#99: rows from a left-padded batch equal rows from running each prompt alone."""
    capture = make_capture(make_toy_qwen())
    batched = capture_routing(capture, TOY_PROMPTS, SEQ_IDS)
    reference = capture_each_alone(capture, TOY_PROMPTS, SEQ_IDS)

    assert batched.keys() == reference.keys()
    for key, expected in reference.items():
        got = batched[key]
        assert sorted(got["selected_experts"]) == sorted(expected["selected_experts"]), key
        assert got["expert_probs"] == pytest.approx(expected["expert_probs"], abs=1e-5), key


REAL_PROMPTS = [
    "What is 7 times 8?",
    "Write a Python function that reverses a list of integers.",
    "A train travels 60 miles in 1.5 hours. What is its average speed?",
]


@requires_gpu
def test_real_model_batched_matches_single_prompt_runs(qwen_model):
    """#102: on the real model, batched rows match single-prompt rows.

    fp16 kernels aren't bitwise stable across batch shapes, so near-tied
    experts may swap; require top-k agreement on at least 98% of (token, layer)
    pairs. Wrong-row extraction (#99) agrees far less often.
    """
    model, tokenizer, _ = qwen_model
    capture = make_capture(model, device="cuda")
    try:
        prompts = [tokenizer.encode(p) for p in REAL_PROMPTS]
        pad_id = tokenizer.pad_token_id
        batched = capture_routing(capture, prompts, SEQ_IDS, pad_id)
        reference = capture_each_alone(capture, prompts, SEQ_IDS, pad_id)
    finally:
        capture.cleanup()

    assert batched.keys() == reference.keys()
    agree = sum(
        sorted(batched[k]["selected_experts"]) == sorted(reference[k]["selected_experts"])
        for k in reference
    )
    assert agree / len(reference) >= 0.98, f"top-k agreement {agree}/{len(reference)}"
    for key, layer in reference.items():
        assert len(layer["selected_experts"]) == model.config.num_experts_per_tok, key


def test_position_ids_count_real_tokens_from_zero():
    """#114 experiment: real tokens are numbered from 0 whatever the left padding."""
    import torch

    from forward_pass import position_ids_from_mask

    mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
    positions = position_ids_from_mask(mask)

    for row, row_mask in zip(positions, mask):
        assert row[row_mask == 1].tolist() == list(range(int(row_mask.sum())))


def test_forward_pass_gives_the_model_mask_based_positions():
    """#114 experiment: capture_batch passes position_ids built from the attention mask."""
    import torch

    from forward_pass import position_ids_from_mask

    model = make_toy_qwen()
    capture = make_capture(model)
    seen = {}
    handle = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.update(kwargs), with_kwargs=True
    )
    try:
        capture_routing(capture, TOY_PROMPTS, SEQ_IDS)
    finally:
        handle.remove()

    mask = seen["attention_mask"]
    assert torch.equal(seen["position_ids"], position_ids_from_mask(mask))
