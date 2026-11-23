"""
CPU tests for the shared-model cleanup (#118), on a tiny Qwen2MoE model built in code.
"""
import pytest
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from shared_model import reset_after_test


@pytest.fixture
def model():
    config = Qwen2MoeConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        shared_expert_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, num_experts=8, num_experts_per_tok=4, decoder_sparse_step=1,
    )
    return Qwen2MoeForCausalLM(config).eval()


def hook_count(model):
    return sum(len(getattr(m, a)) for m in model.modules()
               for a in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"))


def test_hooks_on_any_module_are_removed(model):
    layer = model.model.layers[0]
    layer.self_attn.register_forward_hook(lambda *args: None)
    layer.mlp.register_forward_pre_hook(lambda *args: None)
    layer.mlp.gate.register_forward_hook(lambda *args: None)
    model.model.norm.register_full_backward_hook(lambda *args: None)
    assert hook_count(model) == 4

    reset_after_test(model)

    assert hook_count(model) == 0
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]))  # still runs after the reset


def test_model_left_in_training_mode_fails_and_is_restored(model):
    model.train()
    with pytest.raises(AssertionError, match="training mode"):
        reset_after_test(model)
    assert not model.training
    assert not any(m.training for m in model.modules())


def test_clean_model_passes(model):
    reset_after_test(model)
    assert not model.training
