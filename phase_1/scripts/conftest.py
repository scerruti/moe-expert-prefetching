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
from shared_model import reset_after_test

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


@pytest.fixture(scope="session")
def _qwen_model_session():
    """
    Load Qwen1.5-MoE-A2.7B in fp16 (Phase 1 precision, #103) once per test session.

    Loading once means a failing test can't leave a second copy behind and run
    the GPU out of memory for every test after it.
    """
    model, tokenizer, config = ModelLoader("qwen1.5-moe-a2.7b").load()
    logger.info(f"Loaded model on device: {model.device}")
    yield model, tokenizer, config

    logger.info("Cleaning up model from GPU memory...")
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def qwen_model(_qwen_model_session):
    """
    The shared model. After each test, every hook it left on any module is
    removed, and the test fails if it left the model in training mode (#118).

    Usage:
        def test_something(qwen_model):
            model, tokenizer, config = qwen_model
    """
    yield _qwen_model_session
    reset_after_test(_qwen_model_session[0])
