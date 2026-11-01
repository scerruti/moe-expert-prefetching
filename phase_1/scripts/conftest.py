#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Phase 1 acceptance tests.

Addresses GPU memory management to prevent CUDA OOM during test runs.

Tests that need a CUDA GPU carry the `gpu` marker. CI runs `pytest -m "not gpu"`;
GPU tests run on Colab (#113). Any test using the qwen_model fixture is marked
`gpu`, and skipped without CUDA, automatically: loading the real model on a CPU
machine needs about 29 GB and has crashed laptops (#87).
"""

import pytest
import torch
import logging

from model_loader import ModelLoader

logger = logging.getLogger(__name__)

GPU_FIXTURES = {"qwen_model"}


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU and the real model (run on Colab, not in CI)")


def pytest_collection_modifyitems(config, items):
    skip_without_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    for item in items:
        if GPU_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.gpu)
            item.add_marker(skip_without_gpu)


@pytest.fixture(scope="session")
def qwen_tokenizer():
    """Qwen1.5-MoE-A2.7B's tokenizer only: a few MB, no model weights, runs on CPU."""
    return ModelLoader("qwen1.5-moe-a2.7b", device="cpu").load_tokenizer()


@pytest.fixture(scope="function")
def qwen_model():
    """
    Fixture to load Qwen1.5-MoE-A2.7B in fp16 (Phase 1 precision, #103) and
    clean up after each test.

    Prevents GPU memory leak by properly freeing model between tests.
    Yields model, tokenizer, and config for use in tests.

    Usage:
        def test_something(qwen_model):
            model, tokenizer, config = qwen_model
            # use model...
    """
    loader = ModelLoader("qwen1.5-moe-a2.7b")
    model, tokenizer, config = loader.load()

    logger.info(f"Loaded model on device: {model.device}")

    # Yield to test
    yield model, tokenizer, config

    # Cleanup after test
    logger.info("Cleaning up model from GPU memory...")
    del model
    torch.cuda.empty_cache()
    logger.info("Model cleanup complete")
