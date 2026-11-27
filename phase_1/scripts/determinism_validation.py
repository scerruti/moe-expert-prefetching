#!/usr/bin/env python3
"""
Determinism and consistency validation (Issue #8).

Implements the check decided in phase_1/docs/determinism_check.md (#128):

- Same-batch check: an identical batch run twice must give bitwise-identical
  routing. Needs a repeated ordering (repeat_run_0_<dataset>.parquet); without
  one the check reports "not run", never "passed".
- Near-tie check across batch orders: ordering 0 is compared with each other
  ordering. A decision has changed when the router selected a different set of
  experts. A change is a root change when nothing changed at a lower layer on
  the same or an earlier token of the same sequence; every other change is a
  knock-on. Each root change's gap (the k-th minus the (k+1)-th expert's
  probability, the smaller of the two orderings') is compared with the model's
  near-tie window. Root changes outside it are warnings to investigate, not
  failures; knock-ons are reported, never judged.
- The window is computed from a pilot run (compute_window), never chosen.
- The model comes from the collection's manifest.json (#134). validate_all()
  refuses a collection without one, a run that didn't finish, and a window
  record for a different model.
- Data quality: no NaN/inf, probabilities sum to 1.0, valid indices.
- Expert activation: frequency per layer (from full/ when present).

Runs are compared on (sequence_id, token_position, layer_index) with a
vectorised join; missing or duplicated rows are problems that fail the check.
"""

import json
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from run_manifest import ManifestError, read_manifest, require_finished_manifest

logger = logging.getLogger(__name__)

KEY_COLUMNS = ["sequence_id", "token_position", "layer_index"]
ORDERING_FILE = re.compile(r"^run_(\d+)_(\w+)\.parquet$")
REPEAT_FILE = re.compile(r"^repeat_run_0_(\w+)\.parquet$")
COMPARISON = "ordering 0 against each other ordering"

# Project-wide constants for computing a model's near-tie window
# (phase_1/docs/determinism_check.md): a 1% target rate and a 3x margin.
TARGET_RATE = 0.01
MARGIN = 3.0


class InsufficientPilotError(ValueError):
    """The pilot has too few root changes to compute a window."""


def required_root_changes(target_rate: float) -> int:
    """Root changes a pilot needs so that a new one exceeds the pilot's largest
    with probability about 1/(n+1) <= target_rate: n >= 1/target_rate - 1."""
    return math.ceil(1 / target_rate) - 1


def _json_builtin(value: Any) -> Any:
    """Convert numpy values the json module can't serialise (#63)."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_keys(value: Any) -> Any:
    """Recursively make dict keys JSON-safe (numpy integer keys become ints)."""
    if isinstance(value, dict):
        return {(k.item() if isinstance(k, np.generic) else k): _json_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_keys(v) for v in value]
    return value


# ---------- results ----------

@dataclass
class SameBatchCheckResult:
    """An identical batch run twice must give bitwise-identical routing."""
    status: str = "not_run"  # "passed", "failed" or "not_run"
    reason: str = ""
    compared_rows: int = 0
    differing_rows: int = 0
    problems: List[str] = field(default_factory=list)


@dataclass
class NearTieCheckResult:
    """Cross-batch changes, with root changes checked against the near-tie window."""
    status: str = "not_run"  # "ran", "failed" (a data problem) or "not_run"
    reason: str = ""
    window: Optional[float] = None
    comparison: str = COMPARISON
    compared_decisions: int = 0
    changed_decisions: int = 0
    root_changes: int = 0
    knock_ons: int = 0
    largest_root_gap: Optional[float] = None
    outside_window: List[Dict[str, Any]] = field(default_factory=list)  # warnings, not failures
    problems: List[str] = field(default_factory=list)


@dataclass
class DataQualityCheckResult:
    """Result of data quality validation."""
    passed: bool = False
    total_rows: int = 0
    nan_count: int = 0
    inf_count: int = 0
    invalid_prob_sums: int = 0
    invalid_indices: int = 0
    selected_nonfinite: int = 0  # NaN or infinity in selected_probs (#145)
    out_of_range_values: int = 0  # finite expert_probs or selected_probs outside [0, 1]
    selected_prob_mismatches: int = 0  # rows whose selected_probs aren't expert_probs at the selected experts
    selected_length_mismatches: int = 0  # rows with a different number of selected_probs and selected_experts
    notes: List[str] = field(default_factory=list)


@dataclass
class ExpertActivationStats:
    """Statistics for expert activation frequency."""
    layer_index: int
    total_activations: int = 0
    per_expert_count: Dict[int, int] = field(default_factory=dict)
    activation_frequency: Dict[int, float] = field(default_factory=dict)
    mean_activation: float = 0.0
    std_activation: float = 0.0


@dataclass
class ValidationReport:
    """Complete validation report."""
    model: str
    dataset: str
    same_batch: SameBatchCheckResult = field(default_factory=SameBatchCheckResult)
    near_tie: NearTieCheckResult = field(default_factory=NearTieCheckResult)
    data_quality: DataQualityCheckResult = field(default_factory=DataQualityCheckResult)
    expert_stats: List[ExpertActivationStats] = field(default_factory=list)
    status: str = "failed"  # "passed", "incomplete" (a check couldn't run) or "failed"
    overall_passed: bool = False
    summary: str = ""


# ---------- reading runs ----------

def _list_lengths(column: pa.ChunkedArray) -> np.ndarray:
    lengths = pc.fill_null(pc.list_value_length(column), 0)
    return lengths.to_numpy().astype(np.int64)


def _list_values(column: pa.ChunkedArray) -> np.ndarray:
    return column.combine_chunks().flatten().to_numpy(zero_copy_only=False)


def _list_matrix(table: pa.Table, column: str) -> np.ndarray:
    """[rows, width] array from a list column whose rows all have one length."""
    lengths = _list_lengths(table.column(column))
    if len(lengths) and (lengths != lengths[0]).any():
        raise ValueError(f"{column} rows have different lengths")
    width = int(lengths[0]) if len(lengths) else 0
    return _list_values(table.column(column)).reshape(len(lengths), width)


def _topk_boundary_gap(probs: np.ndarray, k: int) -> np.ndarray:
    """Per row: probability of the k-th expert minus the (k+1)-th."""
    if probs.shape[1] <= k:
        return np.full(len(probs), np.inf)
    ranked = -np.sort(-probs, axis=1)
    return ranked[:, k - 1] - ranked[:, k]


def _load_run(path: Path):
    """Keys (with a row number), expert_probs and selected_experts of one run."""
    table = pq.read_table(path, columns=KEY_COLUMNS + ["expert_probs", "selected_experts"])
    keys = table.select(KEY_COLUMNS).to_pandas()
    keys["row"] = np.arange(len(keys))
    return keys, _list_matrix(table, "expert_probs"), _list_matrix(table, "selected_experts")


def _align(path_a: Path, path_b: Path, problems: List[str]):
    """Rows of two runs matched on their keys. Missing or duplicate rows are problems."""
    keys_a, probs_a, sel_a = _load_run(path_a)
    keys_b, probs_b, sel_b = _load_run(path_b)
    for keys, path in ((keys_a, path_a), (keys_b, path_b)):
        if keys.duplicated(KEY_COLUMNS).any():
            problems.append(f"{path.name}: duplicate (sequence_id, position, layer) rows")
    if probs_a.shape[1] != probs_b.shape[1] or sel_a.shape[1] != sel_b.shape[1]:
        problems.append(f"{path_b.name}: row widths differ from {path_a.name}")
        return None
    merged = keys_a.merge(keys_b, on=KEY_COLUMNS, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = merged["_merge"] == "both"
    missing = int((~both).sum())
    if missing:
        problems.append(f"{path_b.name}: {missing} rows not in both {path_a.name} and {path_b.name}")
    merged = merged[both]
    ia = merged["row_a"].to_numpy(dtype=np.int64)
    ib = merged["row_b"].to_numpy(dtype=np.int64)
    keys = merged[KEY_COLUMNS].reset_index(drop=True)
    return keys, probs_a[ia], probs_b[ib], sel_a[ia], sel_b[ib]


def _mark_roots(changes: pd.DataFrame) -> pd.Series:
    """True for root changes: no change at a lower layer on the same or an earlier
    token of the same sequence. Otherwise the change is a knock-on."""
    if changes.empty:
        return pd.Series([], dtype=bool, index=changes.index)
    per_layer = (changes.groupby(["sequence_id", "layer_index"], sort=True)["token_position"]
                 .min().rename("first_position").reset_index())
    # Earliest changed position at any strictly lower layer of the same sequence.
    per_layer["earliest_below"] = (per_layer.groupby("sequence_id")["first_position"]
                                   .transform(lambda s: s.cummin().shift(1)))
    below = changes.merge(per_layer[["sequence_id", "layer_index", "earliest_below"]],
                          on=["sequence_id", "layer_index"], how="left")["earliest_below"].to_numpy()
    positions = changes["token_position"].to_numpy()
    return pd.Series(np.isnan(below) | (positions < below), index=changes.index)


@dataclass
class OrderingComparison:
    """The changed decisions between two orderings, each with its gap and root flag."""
    changes: pd.DataFrame
    compared: int
    k: int
    problems: List[str]


def compare_orderings(path_a: Path, path_b: Path) -> OrderingComparison:
    """Compare two orderings of the same prompts.

    A decision has changed when the router selected a different set of experts
    (from selected_experts, so an exact tie shows which way it went). Its gap is
    the k-th minus the (k+1)-th expert's probability, the smaller of the two
    orderings' gaps, where k is the number of experts selected per token."""
    problems: List[str] = []
    aligned = _align(Path(path_a), Path(path_b), problems)
    if aligned is None:
        empty = pd.DataFrame(columns=KEY_COLUMNS + ["gap", "is_root"])
        return OrderingComparison(empty, 0, 0, problems)
    keys, probs_a, probs_b, sel_a, sel_b = aligned
    k = sel_a.shape[1]
    changed = (np.sort(sel_a, axis=1) != np.sort(sel_b, axis=1)).any(axis=1)
    gap = np.minimum(_topk_boundary_gap(probs_a.astype(np.float64), k),
                     _topk_boundary_gap(probs_b.astype(np.float64), k))
    changes = keys[changed].assign(gap=gap[changed]).reset_index(drop=True)
    changes["is_root"] = _mark_roots(changes)
    return OrderingComparison(changes, len(keys), k, problems)


def _orderings(folder: Path) -> Dict[str, List[Path]]:
    """Each dataset's ordering files, ordering 0 first. Uses determinism/ if present."""
    folder = Path(folder)
    if (folder / "determinism").is_dir():
        folder = folder / "determinism"
    found: Dict[str, List[tuple]] = {}
    for path in folder.glob("*.parquet"):
        m = ORDERING_FILE.match(path.name)
        if m:
            found.setdefault(m.group(2), []).append((int(m.group(1)), path))
    return {ds: [p for _, p in sorted(runs)] for ds, runs in sorted(found.items())}


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_window(pilot_dir: Path, target_rate: float = TARGET_RATE, margin: float = MARGIN,
                   model: Optional[str] = None) -> Dict[str, Any]:
    """The near-tie window for a model, from a pilot run with the real collection's settings.

    Window = margin x the largest root-change gap, comparing ordering 0 with each
    other ordering. Refuses a pilot with fewer than required_root_changes(target_rate)
    root changes. Returns the record to save with the model's settings.

    The model comes from the pilot's manifest.json (#134). A pilot from before
    manifests has none, and then the model must be named; it is never assumed."""
    pilot_dir = Path(pilot_dir)
    model = _pilot_model(pilot_dir, model)
    runs = _orderings(pilot_dir)
    if not runs:
        raise InsufficientPilotError(f"{pilot_dir}: no run_<n>_<dataset>.parquet orderings")
    gaps, k, files = [], None, []
    for dataset, paths in runs.items():
        if len(paths) < 2:
            raise InsufficientPilotError(f"{dataset}: {len(paths)} ordering; at least 2 are needed")
        for other in paths[1:]:
            result = compare_orderings(paths[0], other)
            if result.problems:
                raise ValueError(f"pilot data problems: {result.problems}")
            k = result.k
            gaps.extend(result.changes.loc[result.changes.is_root, "gap"].tolist())
        files.extend(paths)
    required = required_root_changes(target_rate)
    if len(gaps) < required:
        raise InsufficientPilotError(
            f"{pilot_dir}: {len(gaps)} root changes; at least {required} are needed for a "
            f"{target_rate:.0%} target rate. Add pilot prompts or orderings.")
    largest = float(max(gaps))
    return {
        "model": model,
        "experts_per_token": k,
        "window": margin * largest,
        "largest_root_gap": largest,
        "root_changes": len(gaps),
        "required_root_changes": required,
        "target_rate": target_rate,
        "margin": margin,
        "comparison": COMPARISON,
        "pilot": pilot_dir.name,
        "pilot_files": [{"file": p.name, "sha256": _fingerprint(p)} for p in files],
    }


def _pilot_model(pilot_dir: Path, model: Optional[str]) -> str:
    """The pilot's model: from its manifest, which a named model must agree with."""
    if read_manifest(pilot_dir) is None:
        if not model:
            raise ManifestError(
                f"{pilot_dir} has no manifest.json, so name the model the pilot ran: "
                "compute_window(pilot_dir, model=...)")
        return model
    recorded = require_finished_manifest(pilot_dir)["model"]
    if model and model != recorded:
        raise ManifestError(f"The pilot's manifest says {recorded!r}, not {model!r}")
    return recorded


# ---------- the validator ----------

class DeterminismValidator:
    """Validate determinism and consistency of routing data."""

    def __init__(self, parquet_dir: Path, window: Union[None, Dict[str, Any], Path, str] = None):
        """
        Args:
            parquet_dir: A collection output directory (with full/ and determinism/
                subdirectories) or a flat directory of run files.
            window: The model's near-tie window record (a dict, or a JSON file
                written from compute_window). Without one, the near-tie check
                reports "not run".
        """
        self.parquet_dir = Path(parquet_dir)
        self.parquet_files = sorted(self.parquet_dir.rglob("*.parquet"))
        if isinstance(window, (str, Path)):
            window = json.loads(Path(window).read_text())
        self.window = window
        self.manifest: Optional[Dict[str, Any]] = None
        logger.info(f"Found {len(self.parquet_files)} parquet files")

    def _files_in(self, subdir: str) -> List[Path]:
        """Files in the named subdirectory if there are any, else all files."""
        return [f for f in self.parquet_files if f.parent.name == subdir] or self.parquet_files

    def validate_all(self) -> ValidationReport:
        """Run all validation checks on a finished collection.

        Refuses (ManifestError) a collection with no manifest.json, one whose run
        didn't finish, or a window record for a different model than the
        manifest's: the wrong window would make the near-tie check meaningless (#134).
        """
        logger.info("=" * 60)
        logger.info("DETERMINISM & CONSISTENCY VALIDATION")
        logger.info("=" * 60)

        self.manifest = require_finished_manifest(self.parquet_dir)
        model = self.manifest["model"]
        if self.window:
            window_model = self.window.get("model")
            if not window_model:
                raise ManifestError("The window record names no model, so it can't be checked against "
                                    f"the collection's ({model!r})")
            if window_model != model:
                raise ManifestError(f"The window record is for {window_model!r}, but this collection "
                                    f"ran {model!r} (from its manifest.json)")

        report = ValidationReport(model=model, dataset=self._infer_dataset())
        report.data_quality = self.check_data_quality()
        if report.data_quality.passed:
            report.same_batch = self.check_same_batch()
            report.near_tie = self.check_near_tie()
            report.expert_stats = self.analyze_expert_activation()

        failed = (not report.data_quality.passed or report.same_batch.status == "failed"
                  or report.near_tie.status == "failed")
        not_run = report.same_batch.status == "not_run" or report.near_tie.status == "not_run"
        report.status = "failed" if failed else ("incomplete" if not_run else "passed")
        report.overall_passed = report.status == "passed"
        report.summary = self._generate_summary(report)
        return report

    def check_data_quality(self) -> DataQualityCheckResult:
        """Validate every probability column and index (#145).

        expert_probs: no NaN or infinity, every value in [0, 1], each row summing to 1.
        selected_experts: indices into that row's expert_probs; the width comes from
        the data, so a row with no expert_probs has no valid selection.
        selected_probs: finite, in [0, 1], one per selected expert, and equal to
        expert_probs at the selected experts.
        """
        logger.info("Checking data quality...")

        result = DataQualityCheckResult()
        columns = ["expert_probs", "selected_experts", "selected_probs"]

        for parquet_file in self.parquet_files:
            try:
                table = pq.read_table(parquet_file, columns=columns)
                prob_lengths = _list_lengths(table.column("expert_probs"))
                probs = _list_values(table.column("expert_probs")).astype(np.float64)
                rows = len(prob_lengths)
                result.total_rows += rows

                result.nan_count += int(np.isnan(probs).sum())
                result.inf_count += int(np.isinf(probs).sum())
                with np.errstate(invalid="ignore"):
                    result.out_of_range_values += int((np.isfinite(probs) & ((probs < 0) | (probs > 1))).sum())

                # Probability sums should be ~1.0 (rows summing to 0 or NaN are skipped)
                nonempty = prob_lengths > 0
                prob_offsets = np.concatenate([[0], np.cumsum(prob_lengths)[:-1]]).astype(np.int64)
                sums = np.zeros(rows)
                if probs.size:
                    sums[nonempty] = np.add.reduceat(probs, prob_offsets[nonempty])
                with np.errstate(invalid="ignore"):
                    bad_sum = (sums > 0) & ~((sums >= 0.99) & (sums <= 1.01))
                result.invalid_prob_sums += int(bad_sum.sum())

                # Selected experts must index into that row's own expert_probs
                selected_lengths = _list_lengths(table.column("selected_experts"))
                selected = _list_values(table.column("selected_experts")).astype(np.int64)
                row_of_selected = np.repeat(np.arange(rows), selected_lengths)
                width = prob_lengths[row_of_selected]
                in_range = (selected >= 0) & (selected < width)
                result.invalid_indices += int((~in_range).sum())

                # Selected probabilities: finite, in [0, 1], one per selected expert
                chosen_lengths = _list_lengths(table.column("selected_probs"))
                chosen = _list_values(table.column("selected_probs")).astype(np.float64)
                result.selected_nonfinite += int((~np.isfinite(chosen)).sum())
                with np.errstate(invalid="ignore"):
                    result.out_of_range_values += int((np.isfinite(chosen) & ((chosen < 0) | (chosen > 1))).sum())
                same_length = chosen_lengths == selected_lengths
                result.selected_length_mismatches += int((~same_length).sum())

                # ...and equal to expert_probs at the selected experts, where both can be compared
                comparable = in_range & same_length[row_of_selected]
                if comparable.any():
                    expected = probs[prob_offsets[row_of_selected[comparable]] + selected[comparable]]
                    actual = chosen[comparable] if same_length.all() else \
                        chosen[np.repeat(same_length, chosen_lengths)][comparable[np.repeat(same_length, selected_lengths)]]
                    with np.errstate(invalid="ignore"):
                        differs = np.abs(actual - expected) > 1e-6
                    result.selected_prob_mismatches += len(np.unique(row_of_selected[comparable][differs]))

            except Exception as e:
                logger.error(f"Error reading {parquet_file}: {e}")
                result.notes.append(f"Error reading {parquet_file}: {e}")

        problems = {
            "NaN values (expert_probs)": result.nan_count,
            "Inf values (expert_probs)": result.inf_count,
            "Invalid prob sums": result.invalid_prob_sums,
            "Invalid indices": result.invalid_indices,
            "Non-finite selected_probs": result.selected_nonfinite,
            "Values outside [0, 1]": result.out_of_range_values,
            "selected_probs not matching expert_probs": result.selected_prob_mismatches,
            "selected_probs length mismatches": result.selected_length_mismatches,
        }
        result.passed = not any(problems.values()) and not result.notes

        if result.passed:
            logger.info(f"✅ Data quality check PASSED ({result.total_rows} rows)")
        else:
            logger.error("❌ Data quality check FAILED:")
            for name, count in problems.items():
                if count:
                    logger.error(f"   {name}: {count}")

        return result

    def check_same_batch(self) -> SameBatchCheckResult:
        """Ordering 0 run twice with an identical batch must match bitwise."""
        result = SameBatchCheckResult()
        runs = _orderings(self.parquet_dir)
        folder = self.parquet_dir / "determinism" if (self.parquet_dir / "determinism").is_dir() else self.parquet_dir
        repeats = {REPEAT_FILE.match(p.name).group(1): p for p in folder.glob("*.parquet")
                   if REPEAT_FILE.match(p.name)}
        if not repeats:
            result.reason = ("no repeated ordering (repeat_run_0_<dataset>.parquet) in the collection, "
                             "so identical batches couldn't be compared")
            logger.warning(f"Same-batch check NOT RUN: {result.reason}")
            return result
        for dataset, repeat in sorted(repeats.items()):
            if dataset not in runs:
                result.problems.append(f"{repeat.name}: no run_0_{dataset}.parquet to compare with")
                continue
            aligned = _align(runs[dataset][0], repeat, result.problems)
            if aligned is None:
                continue
            _, probs_a, probs_b, sel_a, sel_b = aligned
            same = (probs_a == probs_b).all(axis=1) & (sel_a == sel_b).all(axis=1)
            result.compared_rows += len(same)
            result.differing_rows += int((~same).sum())
        ok = not result.problems and result.differing_rows == 0 and result.compared_rows > 0
        result.status = "passed" if ok else "failed"
        log = logger.info if ok else logger.error
        log(f"Same-batch check {result.status.upper()}: {result.differing_rows} of "
            f"{result.compared_rows} rows differ; {result.problems}")
        return result

    def check_near_tie(self) -> NearTieCheckResult:
        """Root changes across batch orders, checked against the model's near-tie window.

        Root changes outside the window are warnings to investigate, not failures.
        Missing or duplicate rows, or too few orderings, fail the check."""
        result = NearTieCheckResult()
        if not self.window or "window" not in self.window:
            result.reason = "no near-tie window record for this model (see compute_window)"
            logger.warning(f"Near-tie check NOT RUN: {result.reason}")
            return result
        result.window = float(self.window["window"])
        runs = _orderings(self.parquet_dir)
        if not runs:
            result.problems.append("no run_<n>_<dataset>.parquet orderings to compare")
        root_gaps = []
        for dataset, paths in runs.items():
            if len(paths) < 2:
                result.problems.append(f"{dataset}: {len(paths)} ordering; at least 2 are needed")
                continue
            for other in paths[1:]:
                comparison = compare_orderings(paths[0], other)
                result.problems.extend(comparison.problems)
                changes = comparison.changes
                result.compared_decisions += comparison.compared
                result.changed_decisions += len(changes)
                roots = changes[changes.is_root]
                result.root_changes += len(roots)
                result.knock_ons += len(changes) - len(roots)
                root_gaps.extend(roots.gap.tolist())
                for row in roots[roots.gap > result.window].itertuples(index=False):
                    result.outside_window.append({
                        "dataset": dataset, "ordering": other.name, "sequence_id": row.sequence_id,
                        "token_position": int(row.token_position), "layer_index": int(row.layer_index),
                        "gap": float(row.gap)})
        result.largest_root_gap = float(max(root_gaps)) if root_gaps else None
        result.status = "failed" if result.problems else "ran"
        if result.problems:
            logger.error(f"Near-tie check FAILED: {result.problems}")
        else:
            logger.info(f"Near-tie check: {result.changed_decisions} of {result.compared_decisions} "
                        f"decisions changed; {result.root_changes} root changes, {result.knock_ons} "
                        f"knock-ons; {len(result.outside_window)} outside the window {result.window:.6f}")
        return result

    def analyze_expert_activation(self) -> List[ExpertActivationStats]:
        """Expert activation frequency per layer, from full/ when present."""
        logger.info("Analyzing expert activation...")

        pair_counts: Dict[Tuple[int, int], int] = {}
        for parquet_file in self._files_in("full"):
            table = pq.read_table(parquet_file, columns=["layer_index", "selected_experts"])
            lengths = _list_lengths(table.column("selected_experts"))
            layers = np.repeat(table.column("layer_index").to_numpy(), lengths).astype(np.int64)
            experts = _list_values(table.column("selected_experts")).astype(np.int64)
            pairs, counts = np.unique(np.stack([layers, experts], axis=1), axis=0, return_counts=True) \
                if len(experts) else (np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64))
            for (layer, expert), count in zip(pairs.tolist(), counts.tolist()):
                pair_counts[(layer, expert)] = pair_counts.get((layer, expert), 0) + count

        layer_stats: Dict[int, Dict[int, int]] = {}
        for (layer, expert), count in pair_counts.items():
            layer_stats.setdefault(layer, {})[expert] = count

        stats_list = []
        for layer_idx in sorted(layer_stats):
            expert_counts = layer_stats[layer_idx]
            total = sum(expert_counts.values())
            stats = ExpertActivationStats(layer_index=layer_idx, total_activations=total)
            stats.per_expert_count = expert_counts
            stats.activation_frequency = {e: c / total for e, c in expert_counts.items()}
            frequencies = list(stats.activation_frequency.values())
            stats.mean_activation = float(np.mean(frequencies))
            stats.std_activation = float(np.std(frequencies))
            stats_list.append(stats)

        logger.info(f"✅ Analyzed {len(stats_list)} layers")
        return stats_list

    def _infer_dataset(self) -> str:
        """Infer dataset from parquet filenames."""
        if any("gsm8k" in f.name for f in self.parquet_files):
            return "gsm8k"
        if any("mbpp" in f.name for f in self.parquet_files):
            return "mbpp"
        return "mixed"

    def _generate_summary(self, report: ValidationReport) -> str:
        """Generate human-readable summary."""
        lines = [f"Model: {report.model}", f"Dataset: {report.dataset}", ""]
        dq = report.data_quality
        lines.append(f"{'✅' if dq.passed else '❌'} Data quality: {'PASSED' if dq.passed else 'FAILED'} "
                     f"({dq.total_rows} rows)")

        sb = report.same_batch
        mark = {"passed": "✅", "failed": "❌", "not_run": "⚠️ "}[sb.status]
        detail = sb.reason if sb.status == "not_run" else f"{sb.differing_rows} of {sb.compared_rows} rows differ"
        lines.append(f"{mark} Same-batch check: {sb.status.replace('_', ' ').upper()} ({detail})")

        nt = report.near_tie
        if nt.status == "not_run":
            lines.append(f"⚠️  Near-tie check: NOT RUN ({nt.reason})")
        else:
            mark = "❌" if nt.status == "failed" else ("⚠️ " if nt.outside_window else "✅")
            lines.append(
                f"{mark} Near-tie check ({nt.comparison}; window {nt.window:.6f}): "
                f"{nt.changed_decisions} of {nt.compared_decisions} decisions changed; "
                f"{nt.root_changes} root changes, {nt.knock_ons} knock-ons; "
                f"{len(nt.outside_window)} root changes outside the window (warnings to investigate)")
        for problem in sb.problems + nt.problems:
            lines.append(f"   - {problem}")

        lines.append("")
        lines.append({"passed": "🎉 ALL VALIDATION CHECKS PASSED",
                      "incomplete": "⚠️  INCOMPLETE - a check could not run (see above)",
                      "failed": "⚠️  VALIDATION FAILED - Review above results"}[report.status])
        return "\n".join(lines)

    def save_report(self, report: ValidationReport, output_path: Path) -> None:
        """Save validation report to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_dict = {
            "model": report.model,
            "dataset": report.dataset,
            "status": report.status,
            "window_record": self.window,
            "manifest": self.manifest,
            "same_batch": asdict(report.same_batch),
            "near_tie": asdict(report.near_tie),
            "data_quality": asdict(report.data_quality),
            "expert_stats": [asdict(s) for s in report.expert_stats],
            "overall_passed": report.overall_passed,
            "summary": report.summary,
        }
        with open(output_path, "w") as f:
            json.dump(_json_keys(report_dict), f, indent=2, default=_json_builtin)
        logger.info(f"✅ Report saved to {output_path}")
