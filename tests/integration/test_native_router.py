import pytest
import torch
from pytorch_forward_hooks import RoutingDataBuffer, create_router_hook
from transformers import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeTopKRouter

pytestmark = pytest.mark.integration


def test_native_fp16_router_capture_matches_linear_reference():
    config = Qwen2MoeConfig(
        hidden_size=32, num_experts=8, num_experts_per_tok=4, norm_topk_prob=False
    )
    gate = Qwen2MoeTopKRouter(config).half()
    with torch.no_grad():
        gate.weight.copy_(torch.arange(256).reshape(8, 32) / 1024)
    hidden = torch.arange(224, dtype=torch.float16).reshape(7, 32) / 128
    buffer = RoutingDataBuffer()
    handle = gate.register_forward_hook(create_router_hook(0, buffer))
    try:
        with torch.no_grad():
            logits, scores, indices = gate(hidden)
    finally:
        handle.remove()
    reference = torch.nn.functional.linear(hidden, gate.weight)
    probabilities = torch.softmax(reference.float(), dim=-1)
    expected_scores, expected_indices = torch.topk(probabilities, 4, dim=-1)
    captured = buffer.get_all()
    assert len(captured) == 1
    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
    torch.testing.assert_close(captured[0]["logits"], reference, rtol=0, atol=0)
    torch.testing.assert_close(captured[0]["probabilities"], probabilities, rtol=0, atol=0)
    assert torch.equal(captured[0]["topk_indices"], expected_indices)
    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(scores, expected_scores.half(), rtol=0, atol=0)
