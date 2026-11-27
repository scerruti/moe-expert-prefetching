#!/usr/bin/env python3
"""
Load a Phase 1 collection for Phase 2 analysis (#28.1).

A collection folder, as the Phase 1 pipeline writes it (#134):

    manifest.json           what the run used; must say status "finished"
    full/<dataset>.parquet  every prompt once, one row per (prompt, token, layer)
    determinism/            orderings for validation (not read here)

    collection = Collection("data/phase_1_collection", model="qwen1.5-moe-a2.7b")
    collection.validate()                  # schema and data quality, or CollectionError
    for batch in collection.batches("gsm8k"):
        batch.expert_probs                 # [rows, experts] float32

Everything is read from the data and the manifest: the model, the number of
experts and k are never assumed. The checks are Phase 1's: ROUTING_SCHEMA for
the columns, and the validator's data-quality check (#145) for the values.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Phase 1's modules: the schema, the manifest reader and the data-quality check
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase_1" / "scripts"))
from determinism_validation import DataQualityCheckResult, DeterminismValidator  # noqa: E402
from parquet_storage import ROUTING_SCHEMA  # noqa: E402
from run_manifest import ManifestError, require_finished_manifest  # noqa: E402

LIST_COLUMNS = ("expert_probs", "selected_experts", "selected_probs")


class CollectionError(ValueError):
    """The collection can't be analysed: no finished manifest, wrong model, or bad data."""


@dataclass
class RoutingBatch:
    """Routing rows as arrays: one row per (prompt, token, layer)."""
    sequence_id: np.ndarray  # [rows] str, "<dataset>-<split>-<example>"
    token_id: np.ndarray  # [rows] int32
    token_position: np.ndarray  # [rows] int32
    layer_index: np.ndarray  # [rows] int32
    expert_probs: np.ndarray  # [rows, experts] float32, the router's softmax
    selected_experts: np.ndarray  # [rows, k] int16
    selected_probs: np.ndarray  # [rows, k] float32

    def __len__(self) -> int:
        return len(self.layer_index)

    @property
    def num_experts(self) -> int:
        return self.expert_probs.shape[1]

    @property
    def top_k(self) -> int:
        return self.selected_experts.shape[1]


@dataclass
class ValidationResult:
    schema_problems: List[str] = field(default_factory=list)
    quality: Optional[DataQualityCheckResult] = None


QUALITY_COUNTS = {
    "nan_count": "NaN values (expert_probs)",
    "inf_count": "Inf values (expert_probs)",
    "invalid_prob_sums": "Invalid prob sums",
    "invalid_indices": "Invalid indices",
    "selected_nonfinite": "Non-finite selected_probs",
    "out_of_range_values": "Values outside [0, 1]",
    "selected_prob_mismatches": "selected_probs not matching expert_probs",
    "selected_length_mismatches": "selected_probs length mismatches",
}


class Collection:
    """A finished Phase 1 collection, opened for analysis."""

    def __init__(self, folder, model: Optional[str] = None):
        """Refuses a folder without a finished manifest, with no full/ pass, or (when
        model is given) collected with a different model."""
        self.folder = Path(folder)
        try:
            self.manifest = require_finished_manifest(self.folder)
        except ManifestError as error:
            raise CollectionError(str(error)) from error
        self.model = self.manifest["model"]
        if model is not None and model != self.model:
            raise CollectionError(f"{self.folder} was collected with {self.model!r}, not {model!r}")
        self.files: Dict[str, Path] = {p.stem: p for p in sorted((self.folder / "full").glob("*.parquet"))}
        if not self.files:
            raise CollectionError(f"{self.folder} has no full/<dataset>.parquet files")

    @property
    def datasets(self) -> List[str]:
        return list(self.files)

    def _file(self, dataset: str) -> Path:
        if dataset not in self.files:
            raise CollectionError(f"No dataset {dataset!r} in {self.folder}; it has {self.datasets}")
        return self.files[dataset]

    def schema_problems(self) -> List[str]:
        """Each column missing from, or typed differently to, Phase 1's ROUTING_SCHEMA."""
        problems = []
        for dataset, path in self.files.items():
            schema = pq.read_schema(path)
            for expected in ROUTING_SCHEMA:
                if expected.name not in schema.names:
                    problems.append(f"{dataset}: {expected.name}: missing")
                elif not schema.field(expected.name).type.equals(expected.type):
                    problems.append(f"{dataset}: {expected.name}: {schema.field(expected.name).type}, "
                                    f"expected {expected.type}")
        return problems

    def validate(self) -> ValidationResult:
        """Check the schema, then the values with Phase 1's data-quality check.

        Raises CollectionError naming every problem found; returns the result when
        the collection is clean."""
        result = ValidationResult(schema_problems=self.schema_problems())
        if result.schema_problems:
            raise CollectionError("Schema doesn't match ROUTING_SCHEMA:\n  " + "\n  ".join(result.schema_problems))
        result.quality = DeterminismValidator(self.folder / "full").check_data_quality()
        if not result.quality.passed:
            found = [f"{label}: {getattr(result.quality, name)}"
                     for name, label in QUALITY_COUNTS.items() if getattr(result.quality, name)]
            raise CollectionError("Data quality check failed:\n  " + "\n  ".join(found + result.quality.notes))
        return result

    def batches(self, dataset: str) -> Iterator[RoutingBatch]:
        """Stream a dataset's full pass one row group at a time."""
        parquet = pq.ParquetFile(self._file(dataset))
        for group in range(parquet.num_row_groups):
            yield _to_batch(parquet.read_row_group(group))

    def read(self, dataset: str) -> RoutingBatch:
        """A dataset's whole full pass in memory."""
        return _to_batch(pq.read_table(self._file(dataset)))


def _matrix(column: pa.ChunkedArray, name: str) -> np.ndarray:
    """A list column as a [rows, width] array; every row must have the same width."""
    lists = column.combine_chunks()
    lengths = lists.value_lengths().to_numpy(zero_copy_only=False)
    rows = len(lists)
    if rows and (lengths != lengths[0]).any():
        raise CollectionError(f"{name}: rows have different lengths ({sorted(set(lengths.tolist()))})")
    width = int(lengths[0]) if rows else 0
    return lists.flatten().to_numpy(zero_copy_only=False).reshape(rows, width)


def _to_batch(table: pa.Table) -> RoutingBatch:
    return RoutingBatch(
        sequence_id=table.column("sequence_id").to_numpy(zero_copy_only=False),
        token_id=table.column("token_id").to_numpy(),
        token_position=table.column("token_position").to_numpy(),
        layer_index=table.column("layer_index").to_numpy(),
        **{name: _matrix(table.column(name), name) for name in LIST_COLUMNS},
    )
