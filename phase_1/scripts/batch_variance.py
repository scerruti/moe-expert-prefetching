#!/usr/bin/env python3
"""
Where does batch composition change Phase 1 routing?

Usage: python phase_1/scripts/batch_variance.py <collection output>/determinism

Compares each ordering with ordering 0 (per dataset), aligned on
(sequence_id, token_position, layer_index), and reports:
- top-k flips and probability deviations > 0.01, by layer and by token position
- the cascade test: do large deviations follow an earlier-layer flip on the same token?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEY = ["sequence_id", "token_position", "layer_index"]
ATOL = 1e-2


def load(path):
    table = pq.read_table(path, columns=KEY + ["expert_probs", "selected_experts"])
    df = table.select(KEY).to_pandas()
    n = len(df)
    probs = table.column("expert_probs").combine_chunks().flatten().to_numpy().reshape(n, -1)
    sel = table.column("selected_experts").combine_chunks().flatten().to_numpy().reshape(n, -1)
    return df, probs, np.sort(sel, axis=1)


def compare(ref_path, other_path):
    a, pa_, sa = load(ref_path)
    b, pb_, sb = load(other_path)
    a["ia"], b["ib"] = np.arange(len(a)), np.arange(len(b))
    m = a.merge(b, on=KEY)
    ia, ib = m["ia"].to_numpy(), m["ib"].to_numpy()
    m["dev"] = np.abs(pa_[ia].astype(np.float64) - pb_[ib]).max(axis=1)
    m["flip"] = ~(sa[ia] == sb[ib]).all(axis=1)
    m["bitwise"] = (pa_[ia] == pb_[ib]).all(axis=1)
    return m[KEY + ["dev", "flip", "bitwise"]]


def main(folder):
    folder = Path(folder)
    rows = []
    for dataset in ("gsm8k", "mbpp"):
        runs = sorted(folder.glob(f"run_*_{dataset}.parquet"))
        for other in runs[1:]:
            c = compare(runs[0], other)
            c["dataset"], c["pair"] = dataset, other.stem
            rows.append(c)
    df = pd.concat(rows, ignore_index=True)
    df["big"] = df["dev"] > ATOL
    print(f"{len(df):,} compared rows | flips {df.flip.mean():.2%} | dev>{ATOL} {df.big.mean():.2%} | "
          f"bitwise {df.bitwise.mean():.2%} | max dev {df.dev.max():.3f}\n")

    print("By dataset:")
    print(df.groupby("dataset")[["flip", "big", "bitwise"]].mean().map("{:.2%}".format), "\n")

    print("By layer (fraction of rows):")
    by_layer = df.groupby("layer_index").agg(flip=("flip", "mean"), big=("big", "mean"),
                                             bitwise=("bitwise", "mean"), max_dev=("dev", "max"))
    print(by_layer.to_string(formatters={c: "{:.2%}".format for c in ("flip", "big", "bitwise")}
                             | {"max_dev": "{:.3f}".format}), "\n")

    print("By token position:")
    df["pos_bucket"] = pd.cut(df.token_position, [-1, 0, 4, 16, 64, 10_000],
                              labels=["0", "1-4", "5-16", "17-64", "65+"])
    print(df.groupby("pos_bucket", observed=True)[["flip", "big"]].mean().map("{:.2%}".format), "\n")

    # Cascade: for each (pair, token), the first layer where top-k flipped
    token = ["dataset", "pair", "sequence_id", "token_position"]
    first_flip = df[df.flip].groupby(token)["layer_index"].min().rename("first_flip")
    df = df.join(first_flip, on=token)
    df["after_flip"] = df.first_flip.notna() & (df.first_flip < df.layer_index)
    big, small = df[df.big], df[~df.big]
    print("Cascade test:")
    print(f"  rows with dev > {ATOL} that follow an earlier-layer flip on the same token: {big.after_flip.mean():.1%}")
    print(f"  rows with dev <= {ATOL} that do:                                            {small.after_flip.mean():.1%}")
    tokens_with_flip = df.groupby(token).flip.any()
    print(f"  tokens with at least one flip in any layer: {tokens_with_flip.mean():.1%} of compared tokens")
    big_no_prior = big[~big.after_flip]
    print(f"  large deviations with no earlier flip: {len(big_no_prior):,} rows; "
          f"by layer: {big_no_prior.layer_index.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main(sys.argv[1])
