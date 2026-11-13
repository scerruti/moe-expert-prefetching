#!/usr/bin/env python3
"""
Phase 1 loads Qwen1.5-MoE-A2.7B unquantized, in fp16 (#103).

Under 4-bit loading, bitsandbytes also quantized the router gate, so routing
decisions came from 4-bit weights. fp16 fits the A100 (28.7 GB of 42.4 GB),
so routing comes from the model as released.
"""

import pytest
import torch


def requires_gpu(test):
    """Excluded from CI (`-m "not gpu"`) and skipped without CUDA."""
    return pytest.mark.gpu(pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")(test))


@requires_gpu
def test_model_is_not_quantized(qwen_model):
    """#103: no layer is 4-bit; the router gate is a plain fp16 nn.Linear."""
    model, _, _ = qwen_model
    quantized = [name for name, m in model.named_modules() if type(m).__name__ == "Linear4bit"]
    assert quantized == [], f"{len(quantized)} 4-bit layers, e.g. {quantized[:3]}"

    gate = model.model.layers[0].mlp.gate
    assert type(gate) is torch.nn.Linear
    assert gate.weight.dtype == torch.float16
