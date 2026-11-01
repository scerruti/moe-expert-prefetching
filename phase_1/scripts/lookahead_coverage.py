#!/usr/bin/env python3
"""
Question 3 (#114): can a prefetch predictor absorb batch-dependent routing changes?

For each ordering compared with ordering 0, per (prompt, token, layer):
- changed: the top-4 sets differ
- coverage(m): the other ordering's top-4 is inside ordering 0's top-(4+m)
  (and, symmetrically, ordering 0's inside the other's top-(4+m))
- first flip: the earliest layer at which this token's top-4 changed (this pair)
- for lookahead H, a change at layer l is "unforeseeable" if the token's first
  flip is inside (l-H, l]: a predictor reading the live state at layer l-H saw an
  unchanged state. If the first flip is at or before l-H, the predictor already
  sees the diverged state ("foreseeable").

Usage: python phase_1/scripts/lookahead_coverage.py <collection output>/determinism
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEY = ["sequence_id", "token_position", "layer_index"]
K = 4
MS = [1, 2, 4, 8]
HS = [1, 2, 4, 8]
GROUPS = {"layers 0-7": range(0, 8), "layers 8-15": range(8, 16), "layers 16-23": range(16, 24)}


def load(path):
    t = pq.read_table(path, columns=KEY + ["expert_probs", "selected_experts"])
    df = t.select(KEY).to_pandas()
    n = len(df)
    probs = t.column("expert_probs").combine_chunks().flatten().to_numpy().reshape(n, -1).astype(np.float32)
    sel = t.column("selected_experts").combine_chunks().flatten().to_numpy().reshape(n, -1).astype(np.int64)
    df["i"] = np.arange(n)
    return df, probs, sel


def worst_rank(probs, experts):
    """0-based rank (in probs' descending order) of the lowest-ranked of these experts."""
    lowest = np.take_along_axis(probs, experts, axis=1).min(axis=1)
    return (probs > lowest[:, None]).sum(axis=1)


def compare(ref, other):
    a, pa_, sa = ref
    b, pb_, sb = other
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    ia, ib = m["i_a"].to_numpy(), m["i_b"].to_numpy()
    out = m[KEY].copy()
    out["changed"] = ~(np.sort(sa[ia], 1) == np.sort(sb[ib], 1)).all(1)
    rank = np.maximum(worst_rank(pa_[ia], sb[ib]), worst_rank(pb_[ib], sa[ia]))
    for mm in MS:
        out[f"cov{mm}"] = rank <= K - 1 + mm
    return out


def comparisons(folder):
    """One row per (ordering pair, prompt, token, layer): changed, coverage(m), first_flip."""
    folder = Path(folder)
    parts = []
    for dataset in ("gsm8k", "mbpp"):
        runs = sorted(folder.glob(f"run_*_{dataset}.parquet"))
        if len(runs) < 2:
            continue  # nothing to compare for this dataset
        ref = load(runs[0])
        for path in runs[1:]:
            c = compare(ref, load(path))
            c["pair"] = f"{dataset}:{path.stem}"
            parts.append(c)
    df = pd.concat(parts, ignore_index=True)

    token = ["pair", "sequence_id", "token_position"]
    first = df[df.changed].groupby(token)["layer_index"].min().rename("first_flip")
    return df.join(first, on=token)


def main(folder):
    df = comparisons(folder)
    ch = df[df.changed]
    n_all, n_ch = len(df), len(ch)
    print(f"{n_all:,} compared decisions; top-4 changed: {n_ch:,} ({n_ch / n_all:.2%})\n")

    print("1. Over-prediction: share of CHANGED decisions covered by predicting top-(4+m)")
    rows = []
    for name, layers in {"all layers": range(24), **GROUPS}.items():
        sub = ch[ch.layer_index.isin(layers)]
        rows.append([name, f"{len(sub) / len(df[df.layer_index.isin(layers)]):.2%}"]
                    + [f"{sub[f'cov{mm}'].mean():.1%}" for mm in MS])
    print(pd.DataFrame(rows, columns=["", "changed"] + [f"m={mm}" for mm in MS]).to_string(index=False), "\n")

    print("2. Lookahead: share of ALL decisions that changed unforeseeably (first flip within the last H layers)")
    rows = []
    for name, layers in {"all layers": range(24), **GROUPS}.items():
        sub_all = df[df.layer_index.isin(layers)]
        sub = ch[ch.layer_index.isin(layers)]
        r = [name, f"{len(sub) / len(sub_all):.2%}"]
        for h in HS:
            unforeseeable = sub.first_flip > sub.layer_index - h
            r.append(f"{unforeseeable.sum() / len(sub_all):.2%}")
        rows.append(r)
    print(pd.DataFrame(rows, columns=["", "all changes"] + [f"H={h}" for h in HS]).to_string(index=False), "\n")

    print("3. Both: share of ALL decisions still missed with lookahead H and top-(4+m)")
    rows = []
    for h in HS:
        unforeseeable = ch[ch.first_flip > ch.layer_index - h]
        rows.append([f"H={h}", f"{len(unforeseeable) / n_all:.2%}"]
                    + [f"{(~unforeseeable[f'cov{mm}']).sum() / n_all:.3%}" for mm in MS])
    print(pd.DataFrame(rows, columns=["", "unforeseeable"] + [f"missed, m={mm}" for mm in MS]).to_string(index=False), "\n")

    print("   Same, deepest layers only (16-23):")
    rows = []
    deep_all = df[df.layer_index >= 16]
    deep = ch[ch.layer_index >= 16]
    for h in HS:
        unforeseeable = deep[deep.first_flip > deep.layer_index - h]
        rows.append([f"H={h}", f"{len(unforeseeable) / len(deep_all):.2%}"]
                    + [f"{(~unforeseeable[f'cov{mm}']).sum() / len(deep_all):.3%}" for mm in MS])
    print(pd.DataFrame(rows, columns=["", "unforeseeable"] + [f"missed, m={mm}" for mm in MS]).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
