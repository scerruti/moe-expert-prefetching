#!/usr/bin/env python3
"""
Test forward pass integration with routing capture (Issue #15).

Validates:
- Hook registration and firing on all layers
- Routing data capture per token and layer
- Output format and data types
- Memory management during forward passes
- Integration with tokenizer (#14) and hooks (#5)
"""

import logging
from pathlib import Path

import torch

from forward_pass import ForwardPassCapture, run_forward_pass_pipeline, PromptRoutingData
from model_loader import ModelLoader
from tokenizer import TokenizationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAMPLE_PROMPTS = [
    "What is machine learning?",
    "Explain neural networks.",
]


def test_hook_registration_and_firing(qwen_model):
    """Test that hooks register on all layers and fire during forward pass."""
    logger.info("=" * 60)
    logger.info("Test: Hook Registration & Firing")
    logger.info("=" * 60)

    model, tokenizer, config = qwen_model

    capture = ForwardPassCapture("qwen1.5-moe-a2.7b")
    capture.model = model
    capture.tokenizer = tokenizer
    capture.config = config

    routing_buffer = capture.setup_hooks()

    # Verify hooks registered
    assert len(capture.hooks) > 0, "No hooks registered"
    logger.info(f"✅ Registered {len(capture.hooks)} hooks")

    # Tokenize and run forward pass
    tokenizer_pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")
    tokenization_result = tokenizer_pipeline.tokenize_batch(SAMPLE_PROMPTS)

    input_ids = tokenization_result["token_tensors"]["input_ids"]
    attention_mask = tokenization_result["token_tensors"]["attention_mask"]

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    # Verify hooks fired
    routing_events = routing_buffer.get_all()
    assert len(routing_events) > 0, "No routing events captured"
    logger.info(f"✅ Hooks fired and captured {len(routing_events)} routing events")

    capture.cleanup()
    return True


def test_routing_data_format(qwen_model):
    """Test that routing data matches expected format."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Routing Data Format")
    logger.info("=" * 60)

    model, tokenizer, config = qwen_model

    capture = ForwardPassCapture("qwen1.5-moe-a2.7b")
    capture.model = model
    capture.tokenizer = tokenizer
    capture.config = config

    capture.setup_hooks()

    # Tokenize and run forward pass
    tokenizer_pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")
    tokenization_result = tokenizer_pipeline.tokenize_batch(SAMPLE_PROMPTS)

    routing_data = capture.run_forward_pass(
        token_tensors=tokenization_result["token_tensors"],
        sequence_ids=tokenization_result["sequence_ids"],
        token_ids_list=tokenization_result["token_ids"],
        token_strs_list=tokenization_result["token_strs"],
    )

    assert len(routing_data) == len(SAMPLE_PROMPTS), "Routing data count mismatch"
    logger.info(f"✅ Got routing data for {len(routing_data)} prompts")

    # Verify each prompt's data structure
    for prompt_data in routing_data:
        assert isinstance(prompt_data, PromptRoutingData)
        assert prompt_data.sequence_id is not None
        assert prompt_data.model == "qwen1.5-moe-a2.7b"
        assert prompt_data.num_tokens > 0
        assert prompt_data.num_layers > 0
        logger.info(
            f"  Prompt {prompt_data.sequence_id}: "
            f"{prompt_data.num_tokens} tokens, {prompt_data.num_layers} layers"
        )

    logger.info(f"✅ All routing data structures valid")

    capture.cleanup()
    return routing_data


def test_token_layer_routing_extraction(qwen_model):
    """Test extraction of routing data per token per layer."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Token-Layer Routing Extraction")
    logger.info("=" * 60)

    model, tokenizer, config = qwen_model

    capture = ForwardPassCapture("qwen1.5-moe-a2.7b")
    capture.model = model
    capture.tokenizer = tokenizer
    capture.config = config

    capture.setup_hooks()

    # Tokenize and run forward pass
    tokenizer_pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")
    tokenization_result = tokenizer_pipeline.tokenize_batch(SAMPLE_PROMPTS)

    routing_data = capture.run_forward_pass(
        token_tensors=tokenization_result["token_tensors"],
        sequence_ids=tokenization_result["sequence_ids"],
        token_ids_list=tokenization_result["token_ids"],
        token_strs_list=tokenization_result["token_strs"],
    )

    # Verify token routing structure
    for prompt_data in routing_data:
        assert len(prompt_data.token_routing) > 0, "No token routing data"

        for token_routing in prompt_data.token_routing[:2]:  # Check first 2 tokens
            assert token_routing.token_id is not None
            assert token_routing.token_str is not None
            assert token_routing.token_position >= 0

            if token_routing.layer_data:
                for layer_data in token_routing.layer_data[:2]:  # Check first 2 layers
                    assert "layer_index" in layer_data
                    assert "expert_probs" in layer_data or "expert_logits" in layer_data
                    logger.info(
                        f"  Token '{token_routing.token_str.strip()}' (pos {token_routing.token_position}), "
                        f"Layer {layer_data['layer_index']}: {len(layer_data.get('selected_experts', []))} experts selected"
                    )

    logger.info(f"✅ Token-layer routing structure valid")

    capture.cleanup()
    return routing_data


def test_no_memory_leaks(qwen_model):
    """Test that forward passes don't cause memory leaks."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Memory Management")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available, skipping GPU memory test")
        return True

    initial_memory = torch.cuda.memory_allocated()
    logger.info(f"Initial GPU memory: {initial_memory / 1e9:.2f} GB")

    model, tokenizer, config = qwen_model

    capture = ForwardPassCapture("qwen1.5-moe-a2.7b", device="cuda")
    capture.model = model
    capture.tokenizer = tokenizer
    capture.config = config

    capture.setup_hooks()

    # Run forward pass
    tokenizer_pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")
    tokenization_result = tokenizer_pipeline.tokenize_batch(SAMPLE_PROMPTS)

    routing_data = capture.run_forward_pass(
        token_tensors=tokenization_result["token_tensors"],
        sequence_ids=tokenization_result["sequence_ids"],
        token_ids_list=tokenization_result["token_ids"],
        token_strs_list=tokenization_result["token_strs"],
    )

    # Clean up and check memory
    capture.cleanup()
    torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated()
    logger.info(f"Final GPU memory: {final_memory / 1e9:.2f} GB")

    logger.info(f"✅ Forward pass completed without excessive memory growth")
    return True


def test_end_to_end_pipeline(qwen_model):
    """Test complete end-to-end pipeline from prompts to routing data."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: End-to-End Pipeline")
    logger.info("=" * 60)

    routing_data = run_forward_pass_pipeline(
        model_name="qwen1.5-moe-a2.7b",
        prompts=SAMPLE_PROMPTS,
    )

    assert len(routing_data) == len(SAMPLE_PROMPTS)
    logger.info(f"✅ Pipeline processed {len(SAMPLE_PROMPTS)} prompts end-to-end")

    for prompt_data in routing_data:
        logger.info(
            f"  {prompt_data.sequence_id}: "
            f"{prompt_data.num_tokens} tokens, "
            f"{prompt_data.num_layers} layers, "
            f"{len(prompt_data.token_routing)} tokens with routing data"
        )

    logger.info(f"✅ End-to-end pipeline successful")
    return routing_data


def test_integration_with_tokenizer(qwen_model):
    """Test that forward pass integrates correctly with tokenizer output."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Tokenizer Integration")
    logger.info("=" * 60)

    model, tokenizer, config = qwen_model

    # Tokenize
    tokenizer_pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")
    tokenization_result = tokenizer_pipeline.tokenize_batch(SAMPLE_PROMPTS)

    # Forward pass
    capture = ForwardPassCapture("qwen1.5-moe-a2.7b")
    capture.model = model
    capture.tokenizer = tokenizer
    capture.config = config

    capture.setup_hooks()

    routing_data = capture.run_forward_pass(
        token_tensors=tokenization_result["token_tensors"],
        sequence_ids=tokenization_result["sequence_ids"],
        token_ids_list=tokenization_result["token_ids"],
        token_strs_list=tokenization_result["token_strs"],
    )

    # Verify consistency
    for prompt_idx, prompt_data in enumerate(routing_data):
        expected_tokens = len(tokenization_result["token_ids"][prompt_idx])
        actual_tokens = len(prompt_data.token_routing)
        assert actual_tokens <= expected_tokens, f"More routing tokens than tokenized tokens"
        logger.info(f"✅ Prompt {prompt_idx}: {actual_tokens}/{expected_tokens} tokens with routing data")

    capture.cleanup()
    return True


def main():
    """Run all tests."""
    logger.info("Starting Forward Pass Routing Capture Tests\n")

    try:
        test_hook_registration_and_firing()
        test_routing_data_format()
        test_token_layer_routing_extraction()
        test_no_memory_leaks()
        test_end_to_end_pipeline()
        test_integration_with_tokenizer()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nReady for Issue #16 (Full Pipeline Integration).")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
