#!/usr/bin/env python3
"""
The router gate must not be quantized to 4-bit (#103).

Expert weights are 4-bit to fit the A100; the gate is tiny (24 x 2048 x 60)
and decides the routing being studied, so it stays nn.Linear.
"""

from routing_test_helpers import requires_gpu


@requires_gpu
def test_router_gate_not_quantized(qwen_model):
    """#103: the router gate stays nn.Linear while expert weights are 4-bit."""
    model, _, _ = qwen_model
    moe = model.model.layers[0].mlp
    assert type(moe.experts[0].gate_proj).__name__ == "Linear4bit"  # control
    assert type(moe.gate).__name__ != "Linear4bit"
