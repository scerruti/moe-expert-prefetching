#!/usr/bin/env python3
"""
Load Mixtral 8x7B and verify it is ready for Phase 1 telemetry collection.

Implements GitHub issue #4: the model loads without errors, ends up in
eval() mode, the tokenizer encodes/decodes correctly, and a toy forward pass
runs without gradient tracking. Run this on the target GPU box before
starting the router-hook work in issue #5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Lets this script find model_utils.py sitting next to it, no matter what
# directory you run "python model_loading.py" from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import (
    DEFAULT_CONFIG_PATH,
    check_tokenizer_roundtrip,
    describe_architecture,
    load_mixtral_model_and_tokenizer,
    load_model_config,
    run_toy_forward_pass,
    verify_moe_architecture,
)


def report_gpu_memory_usage() -> None:
    """Print peak GPU memory allocated so far; no-op on CPU-only machines."""
    import torch

    if not torch.cuda.is_available():
        print("No CUDA device available; skipping GPU memory report.")
        return
    peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak GPU memory allocated: {peak_allocated_gb:.2f} GB")


def check_architecture(meta: dict, mvp_config: dict) -> bool:
    """Print and return whether the loaded model matches the expected architecture."""
    architecture_mismatches = verify_moe_architecture(meta, mvp_config)
    if architecture_mismatches:
        print("Architecture verification FAILED:")
        for mismatch in architecture_mismatches:
            print(f"  - {mismatch}")
        return False
    print(f"Architecture verification passed ({describe_architecture(meta)}).")
    return True


def check_eval_mode(model) -> bool:
    """Print and return whether the model is in eval() mode (dropout disabled)."""
    # .training is PyTorch's own flag on every model, not something we set
    # ourselves; model.eval() in model_utils.py is what flips it to False.
    if model.training:
        print("Model is not in eval() mode.")
        return False
    print("Model is in eval() mode.")
    return True


def check_tokenizer(tokenizer, sample_text: str) -> bool:
    """Print and return whether the tokenizer round trip is lossless."""
    if not check_tokenizer_roundtrip(tokenizer, sample_text):
        print(f"Tokenizer round trip FAILED for sample text: {sample_text!r}")
        return False
    print("Tokenizer encode/decode round trip passed.")
    return True


def check_forward_pass(model, tokenizer, sample_text: str) -> bool:
    """Print and return whether a toy forward pass runs without tracking gradients."""
    logits = run_toy_forward_pass(model, tokenizer, sample_text)
    if logits.requires_grad:
        print("Forward pass unexpectedly tracked gradients.")
        return False
    print(f"Toy forward pass passed. Output logits shape: {tuple(logits.shape)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and verify Mixtral 8x7B for Phase 1.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to model_config.json with expected architecture parameters.",
    )
    parser.add_argument(
        "--sample-text",
        default=None,
        help="Override the tokenizer/forward-pass smoke-test sentence from the config file.",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    mvp_config = config["mvp_model"]
    sample_text = args.sample_text or config["tokenizer_smoke_test_text"]

    print(f"Loading {mvp_config['model_id']} ...")
    model, tokenizer, meta = load_mixtral_model_and_tokenizer(mvp_config["model_id"])
    print(f"Loaded. layers={meta['num_layers']} experts={meta['num_experts']} top_k={meta['top_k']}")

    # "and" chains short-circuit: if check_architecture fails, the later
    # checks never run and never print. That is intentional, an architecture
    # mismatch means we probably loaded the wrong model entirely, so the
    # other checks would not tell us anything useful anyway.
    checks_passed = (
        check_architecture(meta, mvp_config)
        and check_eval_mode(model)
        and check_tokenizer(tokenizer, sample_text)
        and check_forward_pass(model, tokenizer, sample_text)
    )
    # Report memory regardless of pass/fail. Even a failed run is useful to
    # know the memory cost of, especially for the Colab-fit question.
    report_gpu_memory_usage()

    if not checks_passed:
        return 1
    print("\nIssue #4 acceptance criteria: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
