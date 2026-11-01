#!/usr/bin/env python3
"""
Why does the prefetch buffer grow like sqrt(k), and would that carry over to other MoE models?

Tests each step of the argument in phase_1/docs/experiments/k_scaling.md against
collected routing, then runs a synthetic router with no Qwen routing in it.

Usage: python phase_1/scripts/k_scaling_theory.py <collection output>/determinism
"""
import sys
from pathlib import Path

import numpy as np

from k_scaling import fractional_m, load, needed_buffer, ranks

KS = [1, 2, 4, 8, 16]
COLLAPSE_KS = [1, 2, 3, 4, 6, 8, 12, 16, 24]
LAYER_GROUPS = {"layers 0-7": (0, 8), "layers 8-15": (8, 16), "layers 16-23": (16, 24)}
SEED = 0


def slope(x, y):
    """Least-squares slope of log y against log x."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def tail_exponent(values, low=0.95, high=0.999):
    """beta in P(|x| > v) ~ v^-beta, measured between two quantiles."""
    qs = np.linspace(low, high, 12)
    return -slope(np.quantile(values, qs), 1 - qs)


def hill(values, top=0.01):
    """Hill estimate of the tail exponent over the largest values."""
    v = np.sort(values)
    tail = v[int(len(v) * (1 - top)):]
    return float(1 / np.mean(np.log(tail[1:] / tail[0])))


def spacings(scores):
    """Gap between each pair of neighbouring ranks, per row, highest scores first."""
    return -np.diff(-np.sort(-scores, axis=1), axis=1)


def measure(a, b, ks):
    """{k: (change rate, m95, m99)} for two sets of scores for the same rows."""
    rank_a, rank_b = ranks(a), ranks(b)
    results = {}
    for k in ks:
        h = np.bincount(needed_buffer(rank_a, rank_b, k), minlength=a.shape[1] + 1)
        results[k] = (h[1:].sum() / h.sum(), fractional_m(h, 0.95), fractional_m(h, 0.99))
    return results


def synthetic(n_rows, n_experts, sigma, row_noise, rng):
    """Gaussian router scores, plus noise whose size is drawn per row from row_noise."""
    a = rng.normal(0, sigma, (n_rows, n_experts)).astype(np.float32)
    size = rng.choice(row_noise, n_rows)[:, None]
    return a, a + (size * rng.normal(0, 1, (n_rows, n_experts))).astype(np.float32)


def simulate(n_experts, ks, sigma, row_noise, n_rows=200_000):
    """measure() on a synthetic router. Seeded per size, so tables and figures agree."""
    rng = np.random.default_rng(SEED + n_experts)
    return measure(*synthetic(n_rows, n_experts, sigma, row_noise, rng), ks)


def gaussian_spacing(n_experts=60, n_rows=20000):
    """Mean gap per rank for standard Gaussian scores."""
    return spacings(np.random.default_rng(SEED).normal(0, 1, (n_rows, n_experts))).mean(axis=0)


def lattice_share(values, step=2.0 ** -9, tolerance=0.06):
    """Share of non-zero values within tolerance (as a fraction of step) of a multiple of step."""
    q = values[values > step * tolerance] / step
    return float((np.abs(q - np.round(q)) < tolerance).mean())


def recall_at(rank_true, rank_predicted, k, m):
    """Share of the true top-k found in the predicted top-(k+m), treating the other ordering as the prediction."""
    return float(((rank_true < k) & (rank_predicted < k + m)).sum() / (k * len(rank_true)))


def fit_sqrt(x, m):
    """(a, b) for m = a + b * sqrt(x)."""
    b, a = np.polyfit(np.sqrt(x), m, 1)
    return a, b


def rmse(x, m, fit):
    return float(np.sqrt(np.mean((m - fit[0] - fit[1] * np.sqrt(x)) ** 2)))


def step_report(name, a, b):
    gaps = spacings(a)
    mean_gap = gaps.mean(axis=0)
    order = np.argsort(-a, axis=1)
    delta = np.take_along_axis(b - a, order, axis=1)
    results = measure(a, b, KS)
    print(f"{name}: {len(a):,} comparisons")
    print("   k  spacing  spacing*k  P(gap<0.1 mean)  noise E|d|  changed  c=rate*s/E|d|  beta  hill   m95   m99")
    for k in KS:
        noise = np.abs(delta[:, k - 1] - delta[:, k])
        tail = noise[noise > 0]
        rate, m95, m99 = results[k]
        s = mean_gap[k - 1]
        print(f"{k:4d}  {s:7.4f}  {s * k:9.3f}  {(gaps[:, k - 1] < 0.1 * s).mean():15.3f}  {noise.mean():10.4f}  {rate:7.2%}"
              f"  {rate * s / noise.mean():13.2f}  {tail_exponent(tail):4.2f}  {hill(tail):4.2f}  {m95:4.2f}  {m99:4.2f}")
    k = np.array(KS, float)
    excess = np.array([results[int(x)][1] - 1 for x in k])
    print(f"  slopes on k: spacing {slope(k, mean_gap[np.array(KS) - 1]):.2f}, changed {slope(k, [results[int(x)][0] for x in k]):.2f}, "
          f"m95 {slope(k, excess + 1):.2f}, m95-1 " + (f"{slope(k, excess):.2f}" if (excess > 0).all() else "undefined (m95 below 1)"))
    return mean_gap


def main(folder):
    folder = Path(folder)
    parts = {}
    for dataset in ("gsm8k", "mbpp"):
        runs = sorted(folder.glob(f"run_*_{dataset}.parquet"))
        keys, reference = load(runs[0])
        reference = np.log(np.maximum(reference, 1e-30))
        others = [np.log(np.maximum(load(path)[1], 1e-30)) for path in runs[1:]]
        parts[dataset] = (np.tile(keys["layer_index"].to_numpy(), len(others)),
                          np.concatenate([reference] * len(others)), np.concatenate(others))
    layers = np.concatenate([p[0] for p in parts.values()])
    a = np.concatenate([p[1] for p in parts.values()])
    b = np.concatenate([p[2] for p in parts.values()])

    print("1. Each step of the argument\n")
    measured_gap = step_report("all rows", a, b)
    subsets = {"gsm8k": parts["gsm8k"][1:], "mbpp": parts["mbpp"][1:]}
    for name, (low, high) in LAYER_GROUPS.items():
        mask = (layers >= low) & (layers < high)
        subsets[name] = (a[mask], b[mask])
    for name, (sa, sb) in subsets.items():
        print()
        step_report(name, sa, sb)

    print("\n2. Spacing around the k the router was trained with (top-4), all rows\n")
    print("   k  " + "  ".join(f"{k:6d}" for k in range(1, 11)))
    print("mean  " + "  ".join(f"{measured_gap[k - 1]:6.3f}" for k in range(1, 11)))
    for name, (low, high) in LAYER_GROUPS.items():
        g = spacings(a[(layers >= low) & (layers < high)]).mean(axis=0)
        print(f"{name}: " + "  ".join(f"{g[k - 1]:.3f}" for k in range(1, 11)))

    print("\n3. Does the buffer follow the change rate, across layers and datasets?\n")
    groups = {}
    for name in ("gsm8k", "mbpp"):
        for layer_name, (low, high) in LAYER_GROUPS.items():
            mask = (parts[name][0] >= low) & (parts[name][0] < high)
            groups[f"{name} {layer_name}"] = measure(parts[name][1][mask], parts[name][2][mask], COLLAPSE_KS)
    table = np.array([[i, k, r, m95, m99] for i, g in enumerate(groups.values()) for k, (r, m95, m99) in g.items()])
    group, k, rate, m95, m99 = table.T
    for i, (name, g) in enumerate(groups.items()):
        print(f"  {name:24s}" + "  ".join(f"k={kk} {r:5.2%} m95={m:.2f}" for kk, (r, m, _) in g.items() if kk in (1, 4, 16)))
    print("\n  Held-out group error (RMSE of m95), fitting m95 = a + b*sqrt(x) on the other five groups:")
    print(f"  {'held out':24s}  x=k     x=change rate")
    errors = {"k": [], "rate": []}
    for i, name in enumerate(groups):
        out = group == i
        e_k = rmse(k[out], m95[out], fit_sqrt(k[~out], m95[~out]))
        e_r = rmse(rate[out], m95[out], fit_sqrt(rate[~out], m95[~out]))
        errors["k"].append(e_k)
        errors["rate"].append(e_r)
        print(f"  {name:24s}  {e_k:5.3f}  {e_r:5.3f}")
    print(f"  {'mean':24s}  {np.mean(errors['k']):5.3f}  {np.mean(errors['rate']):5.3f}")
    curve = fit_sqrt(rate, m95)
    curve99 = fit_sqrt(rate, m99)
    print(f"  all groups: m95 = {curve[0]:.2f} + {curve[1]:.2f}*sqrt(rate), RMSE {rmse(rate, m95, curve):.3f}; "
          f"m99 = {curve99[0]:.2f} + {curve99[1]:.2f}*sqrt(rate), RMSE {rmse(rate, m99, curve99):.3f}")

    print("\n4. Are Qwen's scores unusual? Spacing against Gaussian scores, same number of experts\n")
    gaussian_gap = gaussian_spacing()
    sigma = float((measured_gap[:16] * gaussian_gap[:16]).sum() / (gaussian_gap[:16] ** 2).sum())
    print(f"  scale fitted on k=1-16: {sigma:.3f}; slope on k, k=1-16: measured {slope(np.arange(1, 17), measured_gap[:16]):.2f}, "
          f"gaussian {slope(np.arange(1, 17), gaussian_gap[:16]):.2f}")
    print("   k  measured  gaussian")
    for kk in (1, 2, 4, 8, 16, 24, 30, 40, 50):
        print(f"{kk:4d}  {measured_gap[kk - 1]:8.4f}  {gaussian_gap[kk - 1] * sigma:8.4f}")
    print(f"  smallest spacing: measured k={int(measured_gap.argmin()) + 1}, gaussian k={int(gaussian_gap.argmin()) + 1} (of 60)")

    print("\n5. Synthetic router: Gaussian scores, per-row noise sizes drawn from Qwen. No Qwen routing.\n")
    row_noise = (b - a).std(axis=1)
    print(f"  per-row noise: median {np.median(row_noise):.4f}, q90 {np.quantile(row_noise, 0.9):.4f}, q99 {np.quantile(row_noise, 0.99):.4f}")
    grid = {8: [1, 2, 3, 4], 60: [1, 2, 4, 8, 16, 24, 30, 40], 256: [1, 2, 4, 8, 16, 32, 64, 128]}
    for n_experts, ks in grid.items():
        print(f"  {n_experts} experts:")
        for kk, (r, m, m_99) in simulate(n_experts, ks, sigma, row_noise).items():
            print(f"    k={kk:3d}  changed {r:6.2%}  m95 {m:5.2f}  m99 {m_99:5.2f}  curve predicts m95 {curve[0] + curve[1] * np.sqrt(r):5.2f}")
    for label, size in (("median", np.median(row_noise)), ("mean", row_noise.mean())):
        same = simulate(60, grid[60], sigma, np.array([size]))
        print(f"  60 experts, every row the {label} noise size: m95 " + str([round(float(v[1]), 2) for v in same.values()]))
    print("  Qwen, same k: m95 " + str([round(float(v[1]), 2) for v in measure(a, b, grid[60]).values()]))

    print("\n6. Is the noise made of fp16 rounding steps?\n")
    order = np.argsort(-a, axis=1)[:, 3:5]
    rows = np.arange(len(a))
    noise = np.abs((b[rows, order[:, 0]] - b[rows, order[:, 1]]) - (a[rows, order[:, 0]] - a[rows, order[:, 1]])).astype(np.float64)
    step = 2.0 ** -9
    print(f"  ranks 4 and 5: {lattice_share(noise):.1%} of non-zero |delta| lie on multiples of 2^-9 (chance: 12%)")
    nonzero = noise[noise > step * 0.06]
    print("  share of non-zero |delta| at 1 to 6 steps: " + ", ".join(f"{(np.round(nonzero / step) == n).mean():.1%}" for n in range(1, 7)))
    for name, (low, high) in LAYER_GROUPS.items():
        mask = (layers >= low) & (layers < high)
        print(f"  {name}: zero {(noise[mask] <= step * 0.06).mean():.1%}, median non-zero {np.median(noise[mask & (noise > step * 0.06)]) / step:.1f} steps, "
              f"99th percentile {np.quantile(noise[mask], 0.99) / step:.0f} steps")

    print("\n7. Recall with batch noise as the only error (k = 4; each ordering predicting the other)\n")
    rank_a, rank_b = ranks(a), ranks(b)
    deep = layers >= 16
    for m in (0, 1, 2, 4):
        whole = (recall_at(rank_a, rank_b, 4, m) + recall_at(rank_b, rank_a, 4, m)) / 2
        late = (recall_at(rank_a[deep], rank_b[deep], 4, m) + recall_at(rank_b[deep], rank_a[deep], 4, m)) / 2
        print(f"  m={m}: Recall@{4 + m} all layers {whole:.2%}, layers 16-23 {late:.2%}")


if __name__ == "__main__":
    main(sys.argv[1])
