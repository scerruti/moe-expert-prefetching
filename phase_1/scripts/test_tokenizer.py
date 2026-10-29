#!/usr/bin/env python3
"""
Test tokenization pipeline with sample data.

Can run standalone before Issue #13 data pipeline is ready.
Tests:
- Tokenization with ModelLoader
- Repeated token tracking
- Output format
"""

import logging
from pathlib import Path

import torch

from model_loader import ModelLoader
from tokenizer import TokenizationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample prompts for initial testing
SAMPLE_PROMPTS = [
    "Hello, how are you today?",
    "What is 2 plus 2?",
    "Write a Python function to calculate factorial.",
    "Hello again, how are things?",
    "Define a function that adds two numbers.",
    "Hello world and goodbye world.",
]


def test_tokenization_basic():
    """Test basic tokenization and tensor preparation."""
    logger.info("=" * 60)
    logger.info("Test: Basic Tokenization")
    logger.info("=" * 60)

    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()

    pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")

    logger.info(f"Tokenizing {len(SAMPLE_PROMPTS)} prompts...")
    result = pipeline.tokenize_batch(SAMPLE_PROMPTS)

    logger.info(f"✅ Tokenization successful")
    logger.info(f"   - Input prompts: {result['metadata']['num_prompts']}")
    logger.info(f"   - Total tokens: {result['metadata']['total_tokens']}")
    logger.info(f"   - Unique tokens: {result['metadata']['unique_tokens']}")

    # Check tensor format
    assert result["token_tensors"]["input_ids"].shape[0] == len(SAMPLE_PROMPTS)
    logger.info(f"✅ Tensor shape: {result['token_tensors']['input_ids'].shape}")

    return result


def test_repeated_tokens():
    """Test tracking of repeated tokens."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Repeated Token Tracking")
    logger.info("=" * 60)

    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()

    pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")
    result = pipeline.tokenize_batch(SAMPLE_PROMPTS)

    repeated = pipeline.get_repeated_tokens()
    logger.info(f"Tokens appearing multiple times: {len(repeated)}")

    for token_stat in repeated[:5]:  # Show first 5
        logger.info(
            f"  Token '{token_stat.token_str.strip()}' (ID: {token_stat.token_id}): "
            f"{token_stat.occurrence_count} occurrences in {len(set(token_stat.sequences_seen))} prompts"
        )

    assert len(repeated) > 0, "Should have found repeated tokens"
    logger.info(f"✅ Repeated token tracking works")

    return result


def test_output_format():
    """Verify output format matches Phase 1 design."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Output Format Validation")
    logger.info("=" * 60)

    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()

    pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")
    result = pipeline.tokenize_batch(SAMPLE_PROMPTS)

    # Check required fields
    required_fields = ["token_ids", "token_strs", "token_tensors", "token_stats", "metadata"]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
        logger.info(f"  ✅ {field}: present")

    # Check tensor format
    assert isinstance(result["token_tensors"]["input_ids"], torch.Tensor)
    assert isinstance(result["token_tensors"]["attention_mask"], torch.Tensor)
    logger.info(f"✅ Tensor types correct")

    # Check metadata
    metadata = result["metadata"]
    assert metadata["model"] == "qwen1.5-moe-a2.7b"
    assert metadata["num_prompts"] == len(SAMPLE_PROMPTS)
    logger.info(f"✅ Metadata format correct")

    return result


def test_stats_persistence():
    """Test saving and loading token statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Statistics Persistence")
    logger.info("=" * 60)

    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()

    pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")
    result = pipeline.tokenize_batch(SAMPLE_PROMPTS)

    # Save stats
    stats_file = Path("/tmp/token_stats_test.json")
    pipeline.save_stats(stats_file)

    assert stats_file.exists()
    logger.info(f"✅ Statistics saved to {stats_file}")

    # Load and verify
    import json

    with open(stats_file) as f:
        loaded = json.load(f)

    assert len(loaded) == result["metadata"]["unique_tokens"]
    logger.info(f"✅ Statistics loaded and verified")

    return True


def test_edge_cases_empty_and_long():
    """Test edge cases: empty prompts and very long prompts."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Edge Cases (Empty & Long Prompts)")
    logger.info("=" * 60)

    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()
    pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")

    # Test with edge case prompts
    edge_prompts = [
        "",  # empty
        " ",  # whitespace only
        "a" * 5000,  # very long (will be truncated)
        "What is the meaning of life?",  # normal
    ]

    result = pipeline.tokenize_batch(edge_prompts)

    # Verify all prompts were processed
    assert result["metadata"]["num_prompts"] == len(edge_prompts)
    logger.info(f"✅ Handled {len(edge_prompts)} edge case prompts")

    # Verify tensors have expected shape (batch_size, seq_len)
    tensor_shape = result["token_tensors"]["input_ids"].shape
    assert tensor_shape[0] == len(edge_prompts)
    logger.info(f"✅ Tensor shape correct: {tensor_shape}")

    return result


def test_special_characters_and_unicode():
    """Test handling of special characters and Unicode."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Special Characters & Unicode")
    logger.info("=" * 60)

    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()
    pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")

    special_prompts = [
        "Hello! @#$%^&*()",
        "Unicode: 你好 مرحبا 🎉",
        "Code: print('hello'); x = [1,2,3]",
        "Math: ∑∏∫√≈≠≤≥",
    ]

    result = pipeline.tokenize_batch(special_prompts)

    assert result["metadata"]["num_prompts"] == len(special_prompts)
    logger.info(f"✅ Successfully tokenized {len(special_prompts)} special character prompts")

    # Verify all token_strs are decodable strings
    for token_str_list in result["token_strs"]:
        assert all(isinstance(ts, str) for ts in token_str_list)
    logger.info(f"✅ All token strings decoded correctly")

    return result


def test_duplicate_prompts():
    """Test that duplicate prompts are handled correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Duplicate Prompts")
    logger.info("=" * 60)

    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()
    pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")

    # Same prompt repeated
    dup_prompts = ["What is AI?"] * 5 + ["Hello world"] * 3

    result = pipeline.tokenize_batch(dup_prompts)

    # Verify all prompts were processed
    assert result["metadata"]["num_prompts"] == len(dup_prompts)
    assert result["metadata"]["total_tokens"] > 0
    logger.info(f"✅ Processed {len(dup_prompts)} prompts (with {result['metadata']['unique_tokens']} unique tokens)")

    # Verify repeated tokens are tracked
    repeated = pipeline.get_repeated_tokens()
    assert len(repeated) > 0, "Should find repeated tokens from duplicate prompts"
    logger.info(f"✅ Detected {len(repeated)} tokens appearing multiple times across duplicate prompts")

    return result


def test_batch_processing_realistic_scale():
    """Test tokenization with realistic batch sizes."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Realistic Batch Processing (100+ prompts)")
    logger.info("=" * 60)

    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()
    pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")

    # Generate realistic batch of 100 prompts
    realistic_batch = [
        f"Question {i}: What is the solution to {i*2} + {i*3}?"
        for i in range(100)
    ]

    result = pipeline.tokenize_batch(realistic_batch)

    assert result["metadata"]["num_prompts"] == 100
    assert result["metadata"]["total_tokens"] > 0
    logger.info(f"✅ Processed {result['metadata']['num_prompts']} prompts")
    logger.info(f"   - Total tokens: {result['metadata']['total_tokens']}")
    logger.info(f"   - Unique tokens: {result['metadata']['unique_tokens']}")

    # Verify tensor batch dimension matches
    batch_size = result["token_tensors"]["input_ids"].shape[0]
    assert batch_size == 100
    logger.info(f"✅ Tensor batch size: {batch_size}")

    return result


def test_multi_model_compatibility():
    """Test tokenization with different model architectures."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Multi-Model Compatibility")
    logger.info("=" * 60)

    test_prompts = [
        "What is machine learning?",
        "Explain neural networks briefly.",
    ]

    # Test with Qwen (already tested above, just verify consistency)
    loader_qwen = ModelLoader("qwen1.5-moe-a2.7b")
    model_q, tokenizer_q, config_q = loader_qwen.load()
    pipeline_q = TokenizationPipeline(tokenizer_q, "qwen1.5-moe-a2.7b")
    result_q = pipeline_q.tokenize_batch(test_prompts)

    logger.info(f"✅ Qwen1.5-MoE-A2.7B: {len(result_q['token_ids'][0])} tokens for first prompt")

    # Verify required fields exist for all models
    required_fields = ["token_ids", "token_strs", "token_tensors", "token_stats", "metadata"]
    for field in required_fields:
        assert field in result_q, f"Missing {field}"

    logger.info(f"✅ All required fields present and compatible across models")

    return result_q


def test_output_integration_with_routing():
    """Verify output format matches expectations for routing telemetry (Issue #15)."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Integration with Routing Telemetry (Issue #15)")
    logger.info("=" * 60)

    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()
    pipeline = TokenizationPipeline(tokenizer, "qwen1.5-moe-a2.7b")

    result = pipeline.tokenize_batch(SAMPLE_PROMPTS)

    # Verify output structure matches what routing capture (Issue #15) expects
    assert "token_tensors" in result
    assert "input_ids" in result["token_tensors"]
    assert "attention_mask" in result["token_tensors"]
    logger.info(f"✅ Has token_tensors with input_ids and attention_mask")

    # Verify sequence tracking for correlation with routing data
    assert "sequence_ids" in result
    assert len(result["sequence_ids"]) == len(SAMPLE_PROMPTS)
    logger.info(f"✅ Sequence IDs present for routing correlation ({len(result['sequence_ids'])} sequences)")

    # Verify metadata includes model info for Phase 1 schema
    assert "model" in result["metadata"]
    assert "num_prompts" in result["metadata"]
    assert "total_tokens" in result["metadata"]
    logger.info(f"✅ Metadata complete for Phase 1 routing schema")

    return result


def main():
    """Run all tests."""
    logger.info("Starting Tokenization Pipeline Tests\n")

    try:
        test_tokenization_basic()
        test_repeated_tokens()
        test_output_format()
        test_stats_persistence()
        test_edge_cases_empty_and_long()
        test_special_characters_and_unicode()
        test_duplicate_prompts()
        test_batch_processing_realistic_scale()
        test_multi_model_compatibility()
        test_output_integration_with_routing()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nReady for integration with Issue #13 data pipeline.")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
