#!/usr/bin/env python3
"""
Data collection skeleton for Phase 1 (Issue #13).

Loop structure:
    for run in range(num_runs):              # outer: randomized runs
        for dataset in ["gsm8k", "mbpp"]:     # middle: datasets
            for example in shuffled(dataset): # inner: prompts, seeded shuffle
                process_fn(...)

The per-prompt work (tokenization, forward pass with router hooks, storage)
is plugged in later via ``process_fn``; by default it is a no-op.

Usage:
    python data_collection.py --num-examples 10
"""

import argparse
import logging
import random
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

DATASET_ORDER = ["gsm8k", "mbpp"]
DEFAULT_NUM_RUNS = 5
DEFAULT_BASE_SEED = 42

# Field holding the prompt text in each loader's output
PROMPT_FIELDS = {"gsm8k": "question", "mbpp": "prompt"}

ProcessFn = Callable[[int, str, int, Dict[str, Any]], None]


def load_datasets(
    num_examples: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load GSM8K and MBPP via the existing loaders.

    Args:
        num_examples: Limit per dataset (None = full train split for GSM8K,
            all splits for MBPP)

    Returns:
        Dict mapping dataset name -> list of examples
    """
    # Imported here so the loop logic can be used/tested without `datasets`
    from dataset_loaders import GSM8KLoader, MBPPLoader

    gsm8k = GSM8KLoader().load(split="train", num_examples=num_examples)

    mbpp_loader = MBPPLoader()
    if num_examples:
        mbpp = mbpp_loader.load(split="train", num_examples=num_examples)
    else:
        mbpp = []
        for split in ["train", "validation", "test"]:
            mbpp.extend(mbpp_loader.load(split=split))

    return {"gsm8k": gsm8k, "mbpp": mbpp}


def get_prompt_text(dataset_name: str, example: Dict[str, Any]) -> str:
    """Return the prompt text for an example from the given dataset."""
    return example[PROMPT_FIELDS[dataset_name]]


def run_seed(base_seed: int, run_idx: int) -> int:
    """Seed for a given run; logged so every ordering is reproducible."""
    return base_seed + run_idx


def shuffled(examples: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Return a shuffled copy of examples using a dedicated RNG."""
    order = list(examples)
    random.Random(seed).shuffle(order)
    return order


def run_collection(
    datasets: Dict[str, List[Dict[str, Any]]],
    num_runs: int = DEFAULT_NUM_RUNS,
    base_seed: int = DEFAULT_BASE_SEED,
    process_fn: Optional[ProcessFn] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Execute the outer/middle/inner collection loops.

    Args:
        datasets: Dict mapping dataset name -> list of examples
        num_runs: Number of randomized runs (outer loop)
        base_seed: Base seed; run i uses base_seed + i
        process_fn: Called as process_fn(run_idx, dataset_name, position, example)
        show_progress: Display tqdm progress bar

    Returns:
        Run metadata: seeds per run and number of prompts processed
    """
    names = [n for n in DATASET_ORDER if n in datasets]
    names += [n for n in datasets if n not in names]

    total = num_runs * sum(len(datasets[n]) for n in names)
    seeds = []
    processed = 0

    with tqdm(
        total=total, desc="Collecting", unit="prompt", disable=not show_progress
    ) as pbar:
        for run_idx in range(num_runs):
            seed = run_seed(base_seed, run_idx)
            seeds.append(seed)
            logger.info(f"Run {run_idx + 1}/{num_runs} (seed={seed})")

            for dataset_name in names:
                pbar.set_postfix(run=f"{run_idx + 1}/{num_runs}", dataset=dataset_name)
                # Offset seed per dataset so datasets don't share an ordering
                order = shuffled(
                    datasets[dataset_name], seed * 1000 + names.index(dataset_name)
                )

                for position, example in enumerate(order):
                    if process_fn is not None:
                        process_fn(run_idx, dataset_name, position, example)
                    processed += 1
                    pbar.update(1)

    return {
        "num_runs": num_runs,
        "base_seed": base_seed,
        "run_seeds": seeds,
        "datasets": {n: len(datasets[n]) for n in names},
        "prompts_processed": processed,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1 data collection skeleton")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Limit examples per dataset (e.g. 10 for a smoke test)",
    )
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    datasets = load_datasets(num_examples=args.num_examples)
    for name, examples in datasets.items():
        logger.info(f"{name}: {len(examples)} examples")

    summary = run_collection(datasets, num_runs=args.num_runs, base_seed=args.seed)
    logger.info(f"Done: {summary}")


if __name__ == "__main__":
    main()
