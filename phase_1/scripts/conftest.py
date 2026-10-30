#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Phase 1 acceptance tests.

Addresses GPU memory management to prevent CUDA OOM during test runs.
"""

import pytest
import torch
import logging

from model_loader import ModelLoader

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def qwen_model():
    """
    Fixture to load Qwen1.5-MoE-A2.7B once and clean up after each test.

    Prevents GPU memory leak by properly freeing model between tests.
    Yields model, tokenizer, and config for use in tests.

    Usage:
        def test_something(qwen_model):
            model, tokenizer, config = qwen_model
            # use model...
    """
    loader = ModelLoader("qwen1.5-moe-a2.7b", load_in_4bit=True)
    model, tokenizer, config = loader.load()

    logger.info(f"Loaded model on device: {model.device}")

    # Yield to test
    yield model, tokenizer, config

    # Cleanup after test
    logger.info("Cleaning up model from GPU memory...")
    del model
    torch.cuda.empty_cache()
    logger.info("Model cleanup complete")


@pytest.fixture(scope="function")
def qwen_model_no_4bit():
    """
    Fixture for Qwen1.5-MoE without 4-bit quantization (float16).

    Use this when 4-bit quantization would interfere with test requirements.
    """
    loader = ModelLoader("qwen1.5-moe-a2.7b", load_in_4bit=False)
    model, tokenizer, config = loader.load()

    logger.info(f"Loaded model (float16) on device: {model.device}")

    yield model, tokenizer, config

    # Cleanup
    logger.info("Cleaning up model from GPU memory...")
    del model
    torch.cuda.empty_cache()
    logger.info("Model cleanup complete")
