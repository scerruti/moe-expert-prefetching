#!/usr/bin/env python3
"""
Loading a Phase 1 collection for Phase 2 (#28.1).

The loader opens a collection folder, requires a finished manifest, checks every
file against Phase 1's ROUTING_SCHEMA, reuses the Phase 1 data-quality check
(#145), and streams full/ as numpy arrays. The acceptance criteria include
Waleed's review points on #53. CPU only, on synthetic collections.
"""

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from synthetic_collection import EXPERTS, MODEL, TOP_K, routing_rows, write_collection
from routing_data import Collection, CollectionError

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def test_opens_a_finished_collection_and_names_its_datasets(tmp_path):
    collection = Collection(write_collection(tmp_path))
    assert collection.model == MODEL
    assert sorted(collection.datasets) == ["gsm8k", "mbpp"]


def test_streamed_and_whole_reads_give_the_same_arrays(tmp_path):
    """Waleed's point 2: both loading modes work, and agree."""
    collection = Collection(write_collection(tmp_path, row_group_size=3))
    batches = list(collection.batches("gsm8k"))
    assert len(batches) > 1  # small row groups, so streaming really is in pieces
    whole = collection.read("gsm8k")
    for name in ("sequence_id", "token_id", "token_position", "layer_index",
                 "expert_probs", "selected_experts", "selected_probs"):
        np.testing.assert_array_equal(np.concatenate([getattr(b, name) for b in batches]),
                                      getattr(whole, name), err_msg=name)


def test_arrays_hold_the_stored_values(tmp_path):
    rows = routing_rows("gsm8k", {"gsm8k-train-00000": 2}, seed=5)
    data = Collection(write_collection(tmp_path, {"gsm8k": rows})).read("gsm8k")
    assert len(data) == len(rows) == 2 * 2  # 2 tokens x 2 layers
    assert data.expert_probs.shape == (4, EXPERTS) and data.expert_probs.dtype == np.float32
    assert data.selected_experts.shape == (4, TOP_K) and data.selected_experts.dtype == np.int16
    np.testing.assert_array_equal(data.expert_probs[3], np.float32(rows[3]["expert_probs"]))
    np.testing.assert_array_equal(data.selected_experts[1], rows[1]["selected_experts"])
    assert data.sequence_id.tolist() == ["gsm8k-train-00000"] * 4
    assert data.layer_index.tolist() == [0, 1, 0, 1]
    assert (data.num_experts, data.top_k) == (EXPERTS, TOP_K)


def test_a_clean_collection_validates(tmp_path):
    report = Collection(write_collection(tmp_path)).validate()
    assert report.quality.passed
    assert report.quality.total_rows == (3 + 2 + 4) * 2
    assert report.schema_problems == []


def test_wrong_column_types_fail_schema_validation_and_are_named(tmp_path):
    """Waleed's point 3: types are compared, not just logged."""
    wrong = pa.schema([
        ("sequence_id", pa.string()), ("token_id", pa.string()), ("token_str", pa.string()),
        ("token_position", pa.int32()), ("layer_index", pa.int64()),
        ("expert_probs", pa.list_(pa.float64())), ("selected_experts", pa.list_(pa.int16())),
        ("selected_probs", pa.list_(pa.float32())), ("dataset_source", pa.string()),
    ])
    rows = [dict(r, token_id=str(r["token_id"])) for r in routing_rows("gsm8k", {"gsm8k-train-00000": 1})]
    collection = Collection(write_collection(tmp_path, {"gsm8k": rows}, schema=wrong))
    with pytest.raises(CollectionError) as error:
        collection.validate()
    message = str(error.value)
    for column in ("token_id", "layer_index", "expert_probs"):
        assert column in message
    assert "selected_experts" not in message  # only the wrong ones are named


def test_a_missing_column_fails_schema_validation(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "selected_probs"}
            for r in routing_rows("gsm8k", {"gsm8k-train-00000": 1})]
    schema = pa.schema([f for f in write_schema() if f.name != "selected_probs"])
    with pytest.raises(CollectionError, match="selected_probs: missing"):
        Collection(write_collection(tmp_path, {"gsm8k": rows}, schema=schema)).validate()


def write_schema():
    from parquet_storage import ROUTING_SCHEMA
    return ROUTING_SCHEMA


@pytest.mark.parametrize("column, bad, count_name", [
    ("expert_probs", math.nan, "NaN values (expert_probs)"),
    ("expert_probs", math.inf, "Inf values (expert_probs)"),
    ("selected_probs", math.nan, "Non-finite selected_probs"),
    ("selected_probs", -0.25, "Values outside [0, 1]"),
    ("selected_probs", 1.5, "Values outside [0, 1]"),
])
def test_bad_values_inside_list_columns_fail_validation(tmp_path, column, bad, count_name):
    """Waleed's point 4: bad probabilities appear in the report, so a clean report means
    the checks ran and passed. The checks themselves are Phase 1's (#145)."""
    rows = routing_rows("gsm8k", {"gsm8k-train-00000": 2})
    rows[1] = dict(rows[1], **{column: [bad] + rows[1][column][1:]})
    with pytest.raises(CollectionError) as error:
        Collection(write_collection(tmp_path, {"gsm8k": rows})).validate()
    assert count_name in str(error.value)


def test_a_folder_without_a_manifest_is_refused(tmp_path):
    """Forced failure: nothing would say which model collected the data."""
    with pytest.raises(CollectionError, match="manifest.json"):
        Collection(write_collection(tmp_path, manifest=False))


def test_an_unfinished_run_is_refused(tmp_path):
    """Forced failure: a partial collection would look complete."""
    with pytest.raises(CollectionError, match="'running'"):
        Collection(write_collection(tmp_path, manifest={"status": "running", "model": MODEL}))


def test_a_collection_from_another_model_is_refused_when_one_is_asked_for(tmp_path):
    folder = write_collection(tmp_path)
    assert Collection(folder, model=MODEL).model == MODEL
    with pytest.raises(CollectionError, match=f"{MODEL}.*qwen|qwen.*{MODEL}"):
        Collection(folder, model="qwen1.5-moe-a2.7b")


def test_a_collection_with_no_full_pass_is_refused(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"status": "finished", "model": MODEL}))
    with pytest.raises(CollectionError, match="full/"):
        Collection(folder)


def test_an_unknown_dataset_is_refused(tmp_path):
    with pytest.raises(CollectionError, match="'humaneval'"):
        Collection(write_collection(tmp_path)).read("humaneval")


def test_every_imported_package_is_in_requirements():
    """Waleed's point 1: the loader works after installing the documented requirements."""
    pinned = {re.split(r"[=<>~!\[ ]", line, maxsplit=1)[0].strip().lower().replace("-", "_")
              for line in (REPO / "requirements.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")}
    local = {p.stem for p in (REPO / "phase_1" / "scripts").glob("*.py")} | \
            {p.stem for p in HERE.glob("*.py")}
    for module in ("routing_data.py", "synthetic_collection.py"):
        tree = ast.parse((HERE / module).read_text())
        imported = {(n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
                    for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in (n.names if isinstance(n, ast.Import) else [n])}
        import sys
        third_party = {m for m in imported if m not in sys.stdlib_module_names and m not in local}
        assert third_party <= pinned, f"{module} imports {third_party - pinned}, not in requirements.txt"
