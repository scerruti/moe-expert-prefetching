"""
Shared helpers for routing data-quality tests (issues #99-#105).

The toy model is a tiny, randomly initialised Qwen2MoE built directly from
Qwen2MoeConfig: no download, runs on CPU. Its routing is meaningless, but it
produces gate outputs with the same shape and semantics as the real model
(flattened [batch * padded_len, num_experts], top-4, unnormalised), which is
all the extraction code depends on. It exercises the same gate code path as
Colab only with the transformers version pinned in requirements.txt.
"""

import pytest
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from forward_pass import ForwardPassCapture

QWEN_TOP_K = 4  # Qwen1.5-MoE-A2.7B num_experts_per_tok
TOY_EXPERTS = 8
TOY_PAD_ID = 0

# Distinct lengths, so each prompt gets a different amount of left padding
TOY_PROMPTS = [[5, 17, 42], [7, 3, 99, 23, 61, 8, 14], [11, 2, 76, 30, 9]]
SEQ_IDS = ["p0", "p1", "p2"]

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU and bitsandbytes for 4-bit quantization",
)


def make_toy_qwen(top_k=QWEN_TOP_K, seed=0):
    """Tiny Qwen2MoE with the real model's routing shape: top-k of N experts, unnormalised."""
    torch.manual_seed(seed)
    config = Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=TOY_EXPERTS,
        num_experts_per_tok=top_k,
        decoder_sparse_step=1,
        norm_topk_prob=False,
    )
    return Qwen2MoeForCausalLM(config).eval()


def make_capture(model, device="cpu"):
    capture = ForwardPassCapture("qwen1.5-moe-a2.7b", device=device)
    capture.model = model
    capture.setup_hooks()
    return capture


def left_padded_batch(prompts, pad_id):
    """Left-pad token id lists the way the Qwen tokenizer does (padding_side='left')."""
    width = max(len(p) for p in prompts)
    input_ids = [[pad_id] * (width - len(p)) + p for p in prompts]
    attention_mask = [[0] * (width - len(p)) + [1] * len(p) for p in prompts]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
    }


def capture_routing(capture, prompts, seq_ids, pad_id=TOY_PAD_ID):
    """Run prompts as one batch; return {(sequence_id, position, layer): layer_data}."""
    routing = capture.run_forward_pass(
        token_tensors=left_padded_batch(prompts, pad_id),
        sequence_ids=seq_ids,
        token_ids_list=prompts,
        token_strs_list=[[str(t) for t in p] for p in prompts],
    )
    rows = {}
    for prompt in routing:
        for token in prompt.token_routing:
            for layer in token.layer_data:
                rows[(prompt.sequence_id, token.token_position, layer["layer_index"])] = layer
    return rows


def capture_each_alone(capture, prompts, seq_ids, pad_id=TOY_PAD_ID):
    """Reference: run every prompt on its own, so there is no batching or padding."""
    rows = {}
    for prompt, seq_id in zip(prompts, seq_ids):
        rows.update(capture_routing(capture, [prompt], [seq_id], pad_id))
    return rows


def routing_row(**overrides):
    """A well-formed parquet routing row (toy sizes); override fields to break it."""
    row = {
        "sequence_id": "s0",
        "token_id": 1,
        "token_str": "a",
        "token_position": 0,
        "layer_index": 0,
        "expert_probs": [1 / TOY_EXPERTS] * TOY_EXPERTS,
        "selected_experts": [0, 1, 2, 3],
        "selected_probs": [1 / TOY_EXPERTS] * QWEN_TOP_K,
        "dataset_source": "gsm8k",
    }
    row.update(overrides)
    return row
