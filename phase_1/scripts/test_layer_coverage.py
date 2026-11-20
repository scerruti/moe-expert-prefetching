"""
CPU tests: a batch is only accepted if every layer's router reported once (#97).

Uses a tiny, randomly initialised Qwen2MoE model, built from a config in code:
no download, no GPU.
"""
import pytest
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from forward_pass import ForwardPassCapture
from pytorch_forward_hooks import QWEN_TOP_K


@pytest.fixture
def capture():
    torch.manual_seed(0)
    config = Qwen2MoeConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        shared_expert_intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, num_experts=8, num_experts_per_tok=QWEN_TOP_K,
        decoder_sparse_step=1, norm_topk_prob=False,
    )
    capture = ForwardPassCapture("qwen1.5-moe-a2.7b", device="cpu")
    capture.model = Qwen2MoeForCausalLM(config).eval()
    capture.setup_hooks()
    yield capture
    for handle in capture.hooks:
        handle.remove()


def batch():
    tokens = {"input_ids": torch.tensor([[0, 5, 6, 7], [1, 2, 3, 4]]),
              "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])}
    return tokens, [[5, 6, 7], [1, 2, 3, 4]]


def test_batch_with_every_layer_is_accepted(capture):
    routing = capture.capture_batch(*batch())
    assert routing.num_layers == 3
    assert len(routing.layer_index) == 7 * 3  # 7 real tokens, 3 layers each


def test_batch_with_a_missing_layer_is_refused(capture):
    capture.hooks[1].remove()  # layer 1's hook never fires
    with pytest.raises(RuntimeError, match=r"layers \[0, 2\]"):
        capture.capture_batch(*batch())


def test_batch_with_a_layer_reported_twice_is_refused(capture):
    capture.routing_buffer.add(0, torch.zeros(1, 8), torch.zeros(1, 8), torch.zeros(1, 4, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="4 events"):
        capture.capture_batch(*batch())
