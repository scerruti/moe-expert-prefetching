#!/usr/bin/env python3
"""
Dataset loaders for Phase 1 data collection.

Supports:
- MBPP (Mostly Basic Programming Problems)
- GSM8K (Grade School Math 8K)

Usage:
    mbpp_loader = MBPPLoader()
    train_data = mbpp_loader.load(split="train")

    gsm_loader = GSM8KLoader()
    test_data = gsm_loader.load(split="test")
"""

from datasets import load_dataset
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MBPPLoader:
    """Load MBPP (Mostly Basic Programming Problems) dataset."""

    def __init__(self):
        """Initialize MBPP loader."""
        self.dataset_name = "google-research-datasets/mbpp"
        self.data = None
        logger.info("Initialized MBPP loader")

    def load(self, split: str = "train", num_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load MBPP dataset.

        Args:
            split: "train", "validation", or "test"
            num_examples: Limit number of examples (None = load all)

        Returns:
            List of examples, each with: task_id, text, code, test_list
        """
        logger.info(f"Loading MBPP {split} split...")

        dataset = load_dataset(self.dataset_name, split=split)

        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        examples = []
        for example in dataset:
            examples.append({
                "task_id": example.get("task_id"),
                "prompt": example.get("text", ""),
                "reference_code": example.get("code", ""),
                "test_cases": example.get("test_list", []),
                "source": "mbpp",
            })

        logger.info(f"Loaded {len(examples)} examples from MBPP {split}")
        return examples

    def get_split_sizes(self) -> Dict[str, int]:
        """Get sizes of each split."""
        splits = {}
        for split in ["train", "validation", "test"]:
            dataset = load_dataset(self.dataset_name, split=split)
            splits[split] = len(dataset)
        return splits


class GSM8KLoader:
    """Load GSM8K (Grade School Math 8K) dataset."""

    def __init__(self):
        """Initialize GSM8K loader."""
        self.dataset_name = "openai/gsm8k"
        self.data = None
        logger.info("Initialized GSM8K loader")

    def load(self, split: str = "train", num_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load GSM8K dataset.

        Args:
            split: "train" or "test"
            num_examples: Limit number of examples (None = load all)

        Returns:
            List of examples, each with: question, answer, reasoning
        """
        logger.info(f"Loading GSM8K {split} split...")

        # GSM8K needs 'main' configuration
        dataset = load_dataset(self.dataset_name, "main", split=split)

        if num_examples:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        examples = []
        for idx, example in enumerate(dataset):
            # Parse the answer to extract reasoning steps
            answer_text = example.get("answer", "")
            reasoning = self._extract_reasoning(answer_text)

            examples.append({
                "example_id": idx,
                "question": example.get("question", ""),
                "reference_answer": answer_text,
                "reasoning_steps": reasoning,
                "source": "gsm8k",
            })

        logger.info(f"Loaded {len(examples)} examples from GSM8K {split}")
        return examples

    @staticmethod
    def _extract_reasoning(answer_text: str) -> List[str]:
        """Extract step-by-step reasoning from GSM8K answer."""
        # GSM8K format: reasoning steps followed by #### final_answer
        if "####" in answer_text:
            reasoning_part = answer_text.split("####")[0].strip()
            steps = [s.strip() for s in reasoning_part.split("\n") if s.strip()]
            return steps
        return [answer_text]

    def get_split_sizes(self) -> Dict[str, int]:
        """Get sizes of each split."""
        splits = {}
        for split in ["train", "test"]:
            try:
                dataset = load_dataset(self.dataset_name, "main", split=split)
                splits[split] = len(dataset)
            except Exception as e:
                logger.warning(f"Could not load {split}: {e}")
                splits[split] = 0
        return splits


def test_loaders():
    """Quick test of dataset loaders."""
    logging.basicConfig(level=logging.INFO)

    # Test MBPP
    print("\n" + "="*60)
    print("Testing MBPP Loader")
    print("="*60)
    mbpp = MBPPLoader()
    mbpp_sizes = mbpp.get_split_sizes()
    print(f"MBPP splits: {mbpp_sizes}")

    mbpp_train = mbpp.load(split="train", num_examples=5)
    print(f"\n✅ Loaded {len(mbpp_train)} MBPP examples")
    if mbpp_train:
        print(f"Example: {mbpp_train[0]['prompt'][:100]}...")

    # Test GSM8K
    print("\n" + "="*60)
    print("Testing GSM8K Loader")
    print("="*60)
    gsm8k = GSM8KLoader()
    gsm8k_sizes = gsm8k.get_split_sizes()
    print(f"GSM8K splits: {gsm8k_sizes}")

    gsm8k_train = gsm8k.load(split="train", num_examples=5)
    print(f"\n✅ Loaded {len(gsm8k_train)} GSM8K examples")
    if gsm8k_train:
        print(f"Example: {gsm8k_train[0]['question'][:100]}...")


if __name__ == "__main__":
    test_loaders()
