#!/usr/bin/env python3
"""
Does the prefetch buffer scale with k, or stay constant? (k + m versus 1.5k)

Each row stores all expert probabilities, so we can apply a hypothetical top-k
cut-off to routing the model actually did with top-4. For each pair of orderings
and each k, needed_buffer gives the smallest m such that each ordering's top-k
is inside the other's top-(k+m).

This measures how crowded the ranking is around rank k. It does not simulate a
model built with that k: hidden states and cascades still come from top-4 routing.

Usage: python phase_1/scripts/k_scaling.py <collection output>/determinism
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

KEY = ["sequence_id", "token_position", "layer_index"]
KS = [1, 2, 3, 4, 6, 8, 12, 16]
TARGETS = [0.90, 0.95, 0.99]


def ranks(probs):
    """Each expert's 0-based rank per row, highest probability first."""
    return np.argsort(np.argsort(-probs, axis=1, kind="stable"), axis=1, kind="stable")


def needed_buffer(rank_a, rank_b, k):
    """Smallest m per row such that both top-k sets fit in the other's top-(k+m)."""
    worst_ab = np.where(rank_a < k, rank_b, -1).max(axis=1)
    worst_ba = np.where(rank_b < k, rank_a, -1).max(axis=1)
    return np.maximum(np.maximum(worst_ab, worst_ba) - k + 1, 0)


def load(path):
    table = pq.read_table(path, columns=KEY + ["expert_probs"])
    order = table.select(KEY).to_pandas().sort_values(KEY).index.to_numpy()
    probs = table.column("expert_probs").combine_chunks().flatten().to_numpy()
    return table.select(KEY).to_pandas().iloc[order].reset_index(drop=True), probs.reshape(len(order), -1)[order]


def smallest_m(histogram, target):
    """Smallest m covering the target share of changed decisions (those needing m >= 1)."""
    changed = histogram[1:]
    return int(np.searchsorted(np.cumsum(changed) / changed.sum(), target)) + 1 if changed.sum() else 0


def main(folder):
    folder = Path(folder)
    groups = {"all layers": lambda l: np.ones(len(l), bool), "layers 16-23": lambda l: l >= 16}
    hist = {g: {k: np.zeros(61, np.int64) for k in KS} for g in groups}
    gaps = {k: [] for k in KS}
    for dataset in ("gsm8k", "mbpp"):
        runs = sorted(folder.glob(f"run_*_{dataset}.parquet"))
        if len(runs) < 2:
            continue
        keys, probs_a = load(runs[0])
        rank_a = ranks(probs_a)
        sorted_a = -np.sort(-probs_a, axis=1)
        for k in KS:
            gaps[k].append(sorted_a[:, k - 1] - sorted_a[:, k])
        layers = keys["layer_index"].to_numpy()
        for path in runs[1:]:
            other_keys, probs_b = load(path)
            assert keys[KEY].equals(other_keys[KEY]), f"{path.name}: rows differ from {runs[0].name}"
            rank_b = ranks(probs_b)
            for k in KS:
                need = needed_buffer(rank_a, rank_b, k)
                for g, select in groups.items():
                    hist[g][k] += np.bincount(need[select(layers)], minlength=61)

    for g in groups:
        print(f"{g}:")
        print("   k  changed  median gap  m for 90%  m for 95%  m for 99%  (k+m95)/k  missed at m=2")
        for k in KS:
            h = hist[g][k]
            gap = np.median(np.concatenate(gaps[k]))
            needed = [smallest_m(h, t) for t in TARGETS]
            print(f"{k:4d}  {h[1:].sum() / h.sum():6.2%}  {gap:10.4f}  {needed[0]:9d}  {needed[1]:9d}  {needed[2]:9d}"
                  f"  {(k + needed[1]) / k:9.2f}  {h[3:].sum() / h.sum():12.3%}")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
