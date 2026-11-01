#!/usr/bin/env python3
"""
What Phase 1 collects, and how each row is identified (#104, #109).

- Rows carry stable "<dataset>-<split>-<example>" sequence IDs (#104).

CPU only: the toy Qwen2MoE stands in for the model, and a character-level
tokenizer for the Qwen tokenizer.
"""

import pyarrow.parquet as pq
import torch

from full_pipeline import FullPipeline, Prompt
from routing_test_helpers import make_capture, make_toy_qwen


class CharTokenizer:
    """Character-level tokenizer within the toy model's 128-token vocabulary; pads left with 0."""

    def encode(self, text):
        return [ord(c) % 127 + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def __call__(self, texts, **kwargs):
        encoded = [self.encode(t) for t in texts]
        width = max(len(e) for e in encoded)
        ids = [[0] * (width - len(e)) + e for e in encoded]
        mask = [[0] * (width - len(e)) + [1] * len(e) for e in encoded]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def toy_pipeline(**kwargs):
    pipeline = FullPipeline(**kwargs)
    pipeline.tokenizer = CharTokenizer()
    pipeline.capture = make_capture(make_toy_qwen())
    return pipeline


def test_rows_carry_prompt_sequence_ids(tmp_path):
    """#104: every parquet row carries its prompt's dataset-split-example ID."""
    prompts = [
        Prompt("gsm8k-test-00007", "What is 2 + 2?"),
        Prompt("mbpp-validation-602", "Write a function to add two numbers."),
        Prompt("gsm8k-train-00000", "Natalia sold clips."),
    ]
    output = tmp_path / "run.parquet"

    processed, rows = toy_pipeline(batch_size=2).run_single_pass(0, 42, prompts, "test", output)

    assert processed == len(prompts) and rows > 0
    ids = pq.read_table(output, columns=["sequence_id"]).column(0).to_pylist()
    assert set(ids) == {p.sequence_id for p in prompts}
