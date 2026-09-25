#!/usr/bin/env python3
"""Shared, importable helpers for loading Phase 1 MoE models and tokenizers.

Other Phase 1 scripts (forward hooks, data collection loop) should import
from this module rather than importing each other directly, so each script
can still be run, debugged, or replaced on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "model_config.json"


def load_model_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the expected MoE architecture parameters from the project config file."""
    with open(config_path, encoding="utf-8") as config_file:
        return json.load(config_file)


def load_mixtral_model_and_tokenizer(
    model_id: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    Load Mixtral 8x7B and its tokenizer, ready for deterministic prefill telemetry.

    Sets the model to eval() mode so dropout is disabled and routing is
    reproducible, matching the determinism requirement in SYSTEM_DESIGN.md
    section 1.2 (same input must give bitwise-identical router output).

    Returns (model, tokenizer, meta). meta is read back from the loaded
    model's own config rather than hardcoded, so the same function keeps
    working unmodified when the research model is swapped in later.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # device_map="auto" lets Hugging Face split the model across whatever
    # GPUs (and CPU, as a last resort) are actually available, instead of
    # forcing it all onto one device. torch_dtype=bfloat16 halves memory
    # versus fp32, but Mixtral is still ~93GB at bf16, bigger than a single
    # Colab GPU. Loading a 4-bit quantized version would need a different
    # code path (bitsandbytes), not just a dtype swap; not implemented yet.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    model.eval()

    # model.eval() only turns off dropout and similar training-only behavior;
    # it does NOT stop PyTorch from tracking gradients. We set requires_grad
    # on every weight so nothing here can accidentally be trained, since this
    # script only ever needs to read the model's routing decisions.
    for param in model.parameters():
        param.requires_grad = False

    # These three field names are Hugging Face's specific names for Mixtral's
    # config, not our own naming. num_local_experts = how many expert
    # sub-networks exist per layer; num_experts_per_tok = how many of them
    # actually get used for a given word (this project's "top_k").
    meta = {
        "model_id": model_id,
        "num_layers": model.config.num_hidden_layers,
        "num_experts": model.config.num_local_experts,
        "top_k": model.config.num_experts_per_tok,
    }
    return model, tokenizer, meta


def verify_moe_architecture(meta: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """
    Compare a loaded model's architecture against the expected values from
    model_config.json. Returns a list of human-readable mismatches; an empty
    list means the architecture matches expectations.
    """
    mismatches = []
    for field in ("num_layers", "num_experts", "top_k"):
        actual_value = meta.get(field)
        expected_value = expected.get(field)
        # A None expected value means we deliberately haven't confirmed that
        # number yet (see research_model.num_layers in model_config.json),
        # not "the field should be empty", so we skip it instead of flagging
        # a false mismatch.
        if expected_value is not None and actual_value != expected_value:
            mismatches.append(f"{field}: expected {expected_value}, got {actual_value}")
    return mismatches


def describe_architecture(meta: dict[str, Any]) -> str:
    """Render architecture facts as text, always sourced from meta (never hardcoded)."""
    return f"{meta['num_experts']} experts, top-{meta['top_k']} routing, {meta['num_layers']} layers"


def check_tokenizer_roundtrip(tokenizer, sample_text: str) -> bool:
    """Encode then decode sample_text; True if the round trip is lossless."""
    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    # .strip() because tokenizers commonly add/drop a leading or trailing
    # space during decode; that is not the kind of mismatch this check cares
    # about, we only want to catch the text itself changing.
    return decoded_text.strip() == sample_text.strip()


def run_toy_forward_pass(model, tokenizer, sample_text: str) -> torch.Tensor:
    """
    Run a single no-grad forward pass on sample_text and return the output logits.

    Confirms the model is wired up correctly (no gradient tracking, no crash)
    before routing hooks get registered on it in a later issue.
    """
    # device_map="auto" can spread the model across several devices, but
    # grabbing the first parameter's device is enough here: our one-sentence
    # input just needs to land wherever the model's embedding layer is.
    device = next(model.parameters()).device
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    # torch.no_grad() is a second guard on top of requires_grad=False above;
    # keeping both means this still fails safe even if a future caller reuses
    # this function with a model that wasn't loaded through this file.
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits
