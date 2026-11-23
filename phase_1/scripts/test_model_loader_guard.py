"""
CPU test: ModelLoader.load() refuses to start without a GPU, so a test that
loads the model directly fails in seconds instead of downloading 29 GB in CI.
"""
from unittest import mock

import pytest

from model_loader import ModelLoader


def test_load_refuses_without_a_gpu(monkeypatch):
    monkeypatch.delenv("MOE_ALLOW_CPU_MODEL_LOAD", raising=False)
    loader = ModelLoader("qwen1.5-moe-a2.7b", device="cpu")
    with mock.patch("model_loader.AutoModelForCausalLM.from_pretrained") as from_pretrained, \
            pytest.raises(RuntimeError, match="qwen_model fixture"):
        loader.load()
    from_pretrained.assert_not_called()


def test_load_on_cpu_is_allowed_when_asked(monkeypatch):
    monkeypatch.setenv("MOE_ALLOW_CPU_MODEL_LOAD", "1")
    loader = ModelLoader("qwen1.5-moe-a2.7b", device="cpu")
    with mock.patch("model_loader.AutoModelForCausalLM.from_pretrained") as from_pretrained, \
            mock.patch.object(ModelLoader, "load_tokenizer"), \
            mock.patch.object(ModelLoader, "_count_parameters", return_value=0):
        loader.load()
    from_pretrained.assert_called_once()
