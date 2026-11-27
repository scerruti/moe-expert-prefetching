"""Independent reference for the root-change figures: ordering 0 against each other
ordering, changes read from selected_experts. Written separately from the validator."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

KEY = ["sequence_id", "token_position", "layer_index"]

def load(path):
    t = pq.read_table(path, columns=KEY + ["expert_probs", "selected_experts"])
    df = t.select(KEY).to_pandas(); n = len(df)
    probs = t.column("expert_probs").combine_chunks().flatten().to_numpy().reshape(n, -1).astype(np.float64)
    sel = t.column("selected_experts").combine_chunks().flatten().to_numpy().reshape(n, -1)
    order = df.sort_values(KEY).index.to_numpy()
    return df.iloc[order].reset_index(drop=True), probs[order], np.sort(sel[order], axis=1)

def gap(p, k):
    s = -np.sort(-p, axis=1); return s[:, k - 1] - s[:, k]

folder = Path(sys.argv[1]); roots = []
for ds in ("gsm8k", "mbpp"):
    runs = sorted(folder.glob(f"run_*_{ds}.parquet"), key=lambda p: int(p.name.split("_")[1]))
    keys, a, sa = load(runs[0]); k = sa.shape[1]
    for path in runs[1:]:
        k2, b, sb = load(path); assert keys.equals(k2)
        changed = (sa != sb).any(1)
        g = np.minimum(gap(a, k), gap(b, k))
        df = keys.assign(g=g)[changed]
        for _, grp in df.groupby("sequence_id"):
            pos = grp.token_position.to_numpy(); lay = grp.layer_index.to_numpy()
            for p_, l_, g_ in zip(pos, lay, grp.g.to_numpy()):
                if not np.any((lay < l_) & (pos <= p_)):
                    roots.append(g_)
r = np.array(roots)
print(f"reference: root changes {len(r)}; largest gap {r.max():.8f}; within 0.001 {np.mean(r <= 0.001):.2%}")
