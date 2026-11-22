"""Exercise the device_map path with a local checkpoint instead of GPU weights."""

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM


def test_local_qwen_checkpoint_loads_with_device_map(tmp_path):
    config = Qwen2MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=4,
        decoder_sparse_step=1,
    )
    original = Qwen2MoeForCausalLM(config).eval()
    original.save_pretrained(tmp_path)
    loaded = Qwen2MoeForCausalLM.from_pretrained(
        tmp_path,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()

    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    assert {str(device) for device in loaded.hf_device_map.values()} == {"cpu"}
    inputs = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = original(inputs).logits
        actual = loaded(inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
