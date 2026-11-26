"""Experiment #140, step 2: compare two runs of triton_kernel_check.py.

Compares every routing parquet file the two runs wrote, row by row:
the router probabilities bit for bit, and the top-4 expert sets.

    python phase_1/scripts/triton_fusion_compare.py RUN_A RUN_B

Use it twice: A vs A' (same setting; must be identical, or the runs aren't
comparable) and A vs B (fusion on vs off).

Throwaway experiment code: not merged; only the findings carry forward to #133.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["sequence_id", "token_position", "layer_index"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in KEY + ["expert_probs", "selected_experts"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    return df.sort_values(KEY, kind="stable").reset_index(drop=True)


def compare_file(a: pd.DataFrame, b: pd.DataFrame, name: str) -> dict:
    if len(a) != len(b) or not a[KEY].equals(b[KEY]):
        raise SystemExit(f"{name}: the two runs have different rows; they did not process the same prompts")
    pa, pb = np.stack(a["expert_probs"].to_numpy()), np.stack(b["expert_probs"].to_numpy())
    if pa.shape != pb.shape or pa.dtype != pb.dtype:
        raise SystemExit(f"{name}: probability arrays differ in shape or type: {pa.shape} {pa.dtype} vs {pb.shape} {pb.dtype}")
    # Bitwise: compare the raw bytes, so -0.0 vs 0.0 or NaN payloads count as differences.
    bits_a = pa.view(np.uint8).reshape(len(pa), -1)
    bits_b = pb.view(np.uint8).reshape(len(pb), -1)
    rows_bits_differ = (bits_a != bits_b).any(axis=1)
    with np.errstate(invalid="ignore"):
        absdiff = np.abs(pa.astype(np.float64) - pb.astype(np.float64))
    sa = [frozenset(x) for x in a["selected_experts"]]
    sb = [frozenset(x) for x in b["selected_experts"]]
    changed = np.array([x != y for x, y in zip(sa, sb)])
    per_layer = (
        pd.DataFrame({"layer": a["layer_index"], "changed": changed})
        .groupby("layer")["changed"].agg(["sum", "count"])
    )
    return {
        "rows": int(len(a)),
        "rows_with_any_probability_bit_difference": int(rows_bits_differ.sum()),
        "max_abs_probability_difference": float(np.nanmax(absdiff)) if absdiff.size else 0.0,
        "top4_set_changed_rows": int(changed.sum()),
        "top4_set_changed_share": float(changed.mean()) if len(changed) else 0.0,
        "top4_changed_by_layer": {int(k): f"{int(v['sum'])}/{int(v['count'])}" for k, v in per_layer.iterrows() if v["sum"]},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    args = ap.parse_args()

    files = {}
    for run in (args.run_a, args.run_b):
        report = json.loads((run / "report.json").read_text())
        if report.get("pipeline", {}).get("validation_passed") is not True:
            raise SystemExit(f"{run}: its pipeline did not complete and validate; its data can't be compared")
        files[run] = {p.relative_to(run / "collection") for p in (run / "collection").rglob("*.parquet")}
    if files[args.run_a] != files[args.run_b]:
        raise SystemExit(f"The runs wrote different files: {sorted(map(str, files[args.run_a] ^ files[args.run_b]))}")
    if not files[args.run_a]:
        raise SystemExit("No parquet files found; nothing was compared")

    settings = {r: json.loads((r / "report.json").read_text())["settings"] for r in (args.run_a, args.run_b)}
    results = {}
    for rel in sorted(files[args.run_a], key=str):
        results[str(rel)] = compare_file(load(args.run_a / "collection" / rel), load(args.run_b / "collection" / rel), str(rel))

    rows = sum(r["rows"] for r in results.values())
    bit_rows = sum(r["rows_with_any_probability_bit_difference"] for r in results.values())
    changed = sum(r["top4_set_changed_rows"] for r in results.values())
    summary = {
        "run_a": {"path": str(args.run_a), **settings[args.run_a]},
        "run_b": {"path": str(args.run_b), **settings[args.run_b]},
        "rows_compared": rows,
        "rows_with_any_probability_bit_difference": bit_rows,
        "top4_set_changed_rows": changed,
        "top4_set_changed_share": changed / rows,
        "verdict": "BITWISE IDENTICAL" if bit_rows == 0 else
                   f"DIFFERENT: {bit_rows} rows differ in probability bits, {changed} top-4 sets changed",
        "files": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
