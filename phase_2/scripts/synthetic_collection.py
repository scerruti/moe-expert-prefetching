#!/usr/bin/env python3
"""
Synthetic Phase 1 collections for Phase 2 tests (#28).

Builds a collection folder the way the Phase 1 pipeline writes one: a finished
manifest.json and full/<dataset>.parquet in ROUTING_SCHEMA. The contents are
small and exact, so tests can check values, not just shapes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase_1" / "scripts"))
from parquet_storage import ROUTING_SCHEMA  # noqa: E402

MODEL = "toy-moe"
EXPERTS = 8
TOP_K = 4
LAYERS = 2


def routing_rows(dataset: str, prompts: Dict[str, int], seed: int = 0,
                 experts: int = EXPERTS, top_k: int = TOP_K, layers: int = LAYERS) -> List[dict]:
    """One row per (prompt, token, layer) for prompts {sequence_id: tokens}.

    expert_probs is a softmax of seeded random scores; selected_experts is the
    row's top k and selected_probs their probabilities, as the router hook records them.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sequence_id, tokens in prompts.items():
        for position in range(tokens):
            token_id = 1 + (position % 50)
            for layer in range(layers):
                scores = rng.normal(size=experts)
                probs = np.exp(scores - scores.max())
                probs = (probs / probs.sum()).astype(np.float32)
                top = np.argsort(-probs, kind="stable")[:top_k]
                rows.append({
                    "sequence_id": sequence_id,
                    "token_id": token_id,
                    "token_str": f"t{token_id}",
                    "token_position": position,
                    "layer_index": layer,
                    "expert_probs": probs.tolist(),
                    "selected_experts": top.astype(np.int16).tolist(),
                    "selected_probs": probs[top].tolist(),
                    "dataset_source": dataset,
                })
    return rows


def write_collection(folder: Path, datasets: Optional[Dict[str, List[dict]]] = None,
                     manifest: Optional[dict] = None, row_group_size: int = 7,
                     schema: pa.Schema = ROUTING_SCHEMA) -> Path:
    """Write a collection folder. By default: two small datasets, a finished manifest,
    and small row groups so streaming crosses several of them."""
    folder = Path(folder)
    if datasets is None:
        datasets = {
            "gsm8k": routing_rows("gsm8k", {"gsm8k-train-00000": 3, "gsm8k-test-00001": 2}, seed=1),
            "mbpp": routing_rows("mbpp", {"mbpp-test-11": 4}, seed=2),
        }
    (folder / "full").mkdir(parents=True, exist_ok=True)
    for dataset, rows in datasets.items():
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, folder / "full" / f"{dataset}.parquet", row_group_size=row_group_size)
    if manifest is not False:
        manifest = manifest or {"status": "finished", "model": MODEL}
        (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder
