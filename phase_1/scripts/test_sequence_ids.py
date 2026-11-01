#!/usr/bin/env python3
"""
Sequence IDs must be stable across processes, and prompts must be text (#104).

Python's str hash is salted per process, so hash(prompt) gives the same prompt
a different ID in every Colab session.
"""

import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from tokenizer import TokenizationPipeline

SCRIPTS_DIR = Path(__file__).resolve().parent


class FakeTokenizer:
    """Character-level stand-in for a Hugging Face tokenizer (no download)."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def __call__(self, texts, **kwargs):
        width = max(len(t) for t in texts)
        ids = [[0] * (width - len(t)) + [ord(c) for c in t] for t in texts]
        mask = [[0] * (width - len(t)) + [1] * len(t) for t in texts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


SEQUENCE_ID_SCRIPT = """
import sys
import torch
sys.path.insert(0, sys.argv[1])
from tokenizer import TokenizationPipeline
{fake_tokenizer}
result = TokenizationPipeline(FakeTokenizer(), "fake").tokenize_batch(
    ["What is 2 + 2?", "def add(a, b):"]
)
print(result["sequence_ids"])
"""


def sequence_ids_with_hash_seed(seed):
    script = SEQUENCE_ID_SCRIPT.format(
        fake_tokenizer=textwrap.dedent(inspect.getsource(FakeTokenizer))
    )
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run(
        [sys.executable, "-c", script, str(SCRIPTS_DIR)],
        env=env, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_sequence_ids_stable_across_processes():
    """#104: the same prompt gets the same sequence_id in different Python processes."""
    assert sequence_ids_with_hash_seed(1) == sequence_ids_with_hash_seed(2)


def test_tokenize_batch_rejects_dict_prompts():
    """#104: a loader dict must not be JSON-encoded and fed to the model (it holds the answer)."""
    example = {"question": "What is 2 + 2?", "reference_answer": "4"}
    with pytest.raises((TypeError, ValueError)):
        TokenizationPipeline(FakeTokenizer(), "fake").tokenize_batch([example])
