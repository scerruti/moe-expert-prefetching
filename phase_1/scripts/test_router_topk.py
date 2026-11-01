#!/usr/bin/env python3
"""
The router hook must record Qwen1.5-MoE-A2.7B's top-4 experts (#100).

Phase 1 is locked to Qwen (PHASE_1_MODEL_DECISION.md), so the hook hardcodes
k=4 and refuses a model that routes differently.
"""

import pytest
import torch

from pytorch_forward_hooks import RoutingDataBuffer, register_router_hooks
from routing_test_helpers import (
    QWEN_TOP_K,
    SEQ_IDS,
    TOY_PROMPTS,
    capture_each_alone,
    capture_routing,
    make_capture,
    make_toy_qwen,
)


def test_selected_experts_has_qwen_top_k():
    """#100: every row records num_experts_per_tok (4) selected experts, not 2."""
    capture = make_capture(make_toy_qwen())
    rows = capture_each_alone(capture, TOY_PROMPTS, SEQ_IDS)
    for key, layer in rows.items():
        assert len(layer["selected_experts"]) == QWEN_TOP_K, key


def test_selected_experts_match_model_router_logits():
    """#100: hook top-k equals top-k of the model's own router logits (float32 softmax)."""
    model = make_toy_qwen()
    capture = make_capture(model)
    for prompt, seq_id in zip(TOY_PROMPTS, SEQ_IDS):
        rows = capture_routing(capture, [prompt], [seq_id])

        with torch.no_grad():
            out = model(input_ids=torch.tensor([prompt]), output_router_logits=True)
        capture.routing_buffer.get_all()  # drop events from this reference pass

        for layer_idx, logits in enumerate(out.router_logits):
            probs = torch.softmax(logits.float(), dim=-1)
            expected = torch.topk(probs, QWEN_TOP_K, dim=-1).indices
            for pos in range(len(prompt)):
                got = rows[(seq_id, pos, layer_idx)]["selected_experts"]
                assert sorted(got) == sorted(expected[pos].tolist()), (seq_id, pos, layer_idx)


def test_hook_registration_rejects_non_qwen_top_k():
    """#100: registering on a model whose num_experts_per_tok isn't 4 fails loudly."""
    with pytest.raises((AssertionError, ValueError)):
        register_router_hooks(make_toy_qwen(top_k=2), RoutingDataBuffer())
