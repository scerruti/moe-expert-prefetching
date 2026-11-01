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
KS = list(range(1, 41))  # stops well short of 60, where top-(k+m) would hit the ceiling
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


def fractional_m(histogram, target):
    """Buffer at which coverage of changed decisions crosses target, interpolated between whole m."""
    changed = histogram[1:]
    if not changed.sum():
        return 0.0
    coverage = np.concatenate([[0.0], np.cumsum(changed) / changed.sum()])
    m = int(np.searchsorted(coverage, target))
    return m - 1 + (target - coverage[m - 1]) / (coverage[m] - coverage[m - 1])


def fit_forms(k, m):
    """Least-squares fits of m against k: {name: (a, b, r2, leave-one-out RMSE)}."""
    forms = {"linear": k, "sqrt": np.sqrt(k), "log2": np.log2(k)}
    results = {}
    for name, x in forms.items():
        b, a = np.polyfit(x, m, 1)
        r2 = 1 - ((m - a - b * x) ** 2).sum() / ((m - m.mean()) ** 2).sum()
        errors = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            bi, ai = np.polyfit(x[keep], m[keep], 1)
            errors.append(m[i] - ai - bi * x[i])
        results[name] = (a, b, r2, float(np.sqrt(np.mean(np.square(errors)))))
    return results


def main(folder):
    folder = Path(folder)
    groups = {"all layers": lambda l: np.ones(len(l), bool), "layers 16-23": lambda l: l >= 16}
    hist = {g: {k: np.zeros(61, np.int64) for k in KS} for g in groups}
    gaps = {g: {k: [] for k in KS} for g in groups}
    for dataset in ("gsm8k", "mbpp"):
        runs = sorted(folder.glob(f"run_*_{dataset}.parquet"))
        if len(runs) < 2:
            continue
        keys, probs_a = load(runs[0])
        rank_a = ranks(probs_a)
        sorted_a = -np.sort(-probs_a, axis=1)
        layers = keys["layer_index"].to_numpy()
        for g, select in groups.items():
            for k in KS:
                gaps[g][k].append((sorted_a[:, k - 1] - sorted_a[:, k])[select(layers)])
        for path in runs[1:]:
            other_keys, probs_b = load(path)
            assert keys[KEY].equals(other_keys[KEY]), f"{path.name}: rows differ from {runs[0].name}"
            rank_b = ranks(probs_b)
            for k in KS:
                need = needed_buffer(rank_a, rank_b, k)
                for g, select in groups.items():
                    hist[g][k] += np.bincount(need[select(layers)], minlength=61)

    ks = np.array(KS, dtype=float)
    for g in groups:
        print(f"{g}:")
        print("   k  changed  median gap  m90.frac  m95.frac  m99.frac  (k+m95)/k  missed at m=2")
        for k in KS:
            h = hist[g][k]
            gap = np.median(np.concatenate(gaps[g][k]))
            f = [fractional_m(h, t) for t in TARGETS]
            print(f"{k:4d}  {h[1:].sum() / h.sum():6.2%}  {gap:10.5f}  {f[0]:8.2f}  {f[1]:8.2f}  {f[2]:8.2f}"
                  f"  {(k + f[1]) / k:9.2f}  {h[3:].sum() / h.sum():12.3%}")
        for target in TARGETS:
            m = np.array([fractional_m(hist[g][k], target) for k in KS])
            for label, mask in (("k=1-40", ks >= 1), ("k=1-16", ks <= 16)):
                text = "  ".join(f"{n}: {a:.2f}{b:+.3f}x R2={r:.3f} LOO={e:.3f}"
                                 for n, (a, b, r, e) in fit_forms(ks[mask], m[mask]).items())
                print(f"  fit m{int(target * 100)} {label}: {text}")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
