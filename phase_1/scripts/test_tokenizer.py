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


def main():
    """Run all tests."""
    logger.info("Starting Tokenization Pipeline Tests\n")

    try:
        test_tokenization_basic()
        test_repeated_tokens()
        test_output_format()
        test_stats_persistence()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nReady for integration with Issue #13 data pipeline.")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
