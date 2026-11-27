#!/usr/bin/env python3
"""Build the trial sample used by test_trial_window.py from the full trial data.

Rule: every 10th prompt (sequence_id in sorted order, starting with the first),
keeping only the rows where some ordering's selected experts differ from
ordering 0's. Every column of the routing schema is kept, so the sample passes the
same data-quality check as a real collection (#145). Rows where nothing changed can't affect which changes are root
changes, and keeping whole prompts keeps every root change's causes in the sample.

Usage: python make_trial_sample.py data/phase_1_trial_38ef1da/determinism
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

KEY = ["sequence_id", "token_position", "layer_index"]
OUT = Path(__file__).resolve().parent / "determinism"


def main(source: Path) -> None:
    OUT.mkdir(exist_ok=True)
    for dataset in ("gsm8k", "mbpp"):
        runs = sorted(source.glob(f"run_*_{dataset}.parquet"), key=lambda p: int(p.name.split("_")[1]))
        tables = [pq.read_table(r).sort_by([(k, "ascending") for k in KEY]) for r in runs]
        sequences = sorted(set(tables[0].column("sequence_id").to_pylist()))
        keep_sequences = pa.array(sequences[::10])
        n = tables[0].num_rows
        selected = [np.sort(t.column("selected_experts").combine_chunks().flatten().to_numpy()
                            .reshape(n, -1), axis=1) for t in tables]
        changed = np.zeros(n, dtype=bool)
        for other in selected[1:]:
            changed |= (other != selected[0]).any(axis=1)
        in_sample = pc.is_in(tables[0].column("sequence_id"), value_set=keep_sequences).to_numpy(
            zero_copy_only=False)
        rows = pa.array(np.nonzero(changed & in_sample)[0])
        for run, table in zip(runs, tables):
            pq.write_table(table.take(rows), OUT / run.name, compression="zstd", compression_level=19)
        print(f"{dataset}: {len(keep_sequences)} of {len(sequences)} prompts, {len(rows)} rows per ordering")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
