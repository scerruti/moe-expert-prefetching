#!/usr/bin/env python3
"""
Determinism and consistency validation (Issue #8).

Validates Phase 1 routing data quality and consistency (#105):
- Determinism: the same prompts in different randomized orderings (the
  determinism/ subset runs) give the same expert probabilities, within a
  tolerance. Orderings change batch composition and padding, so fp16
  results needn't be bitwise equal; the exact-equality count is reported
  separately.
- Randomization consistency: top-k selections agree across orderings;
  a change is accepted only at a near-tie between the k-th and
  (k+1)-th expert.
- Data quality: no NaN/inf, probabilities sum to 1.0, valid indices
- Expert activation: frequency per layer (from full/ when present)

Runs are compared on (sequence_id, token_position, layer_index) with a
vectorised join; missing or duplicated rows fail the check.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

KEY_COLUMNS = ["sequence_id", "token_position", "layer_index"]
# Max |probability difference| between orderings. fp16 noise across batch
# shapes is far smaller; a real routing change is far larger.
DEFAULT_ATOL = 1e-2


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


@dataclass
class DeterminismCheckResult:
    """Result of determinism validation across orderings."""
    passed: bool = False
    identical_probabilities: int = 0  # rows whose probabilities are exactly (bitwise) equal
    total_comparisons: int = 0
    max_deviation: float = 0.0
    tolerance: float = DEFAULT_ATOL
    exceeding_tolerance: int = 0  # rows with a probability difference above tolerance
    missing_rows: int = 0  # rows present in one ordering but not the other
    notes: List[str] = field(default_factory=list)


@dataclass
class RandomizationCheckResult:
    """Result of randomization consistency validation."""
    passed: bool = False
    matching_top_k: int = 0
    total_shared_tokens: int = 0
    match_percentage: float = 0.0
    near_tie_flips: int = 0  # top-k changed, at a near-tie (accepted)
    unexplained_changes: int = 0  # top-k changed without a near-tie (fails)
    notes: List[str] = field(default_factory=list)


@dataclass
class DataQualityCheckResult:
    """Result of data quality validation."""
    passed: bool = False
    total_rows: int = 0
    nan_count: int = 0
    inf_count: int = 0
    invalid_prob_sums: int = 0
    invalid_indices: int = 0
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
    determinism: DeterminismCheckResult = field(default_factory=DeterminismCheckResult)
    randomization: RandomizationCheckResult = field(default_factory=RandomizationCheckResult)
    data_quality: DataQualityCheckResult = field(default_factory=DataQualityCheckResult)
    expert_stats: List[ExpertActivationStats] = field(default_factory=list)
    overall_passed: bool = False
    summary: str = ""


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


class DeterminismValidator:
    """Validate determinism and consistency of routing data."""

    def __init__(self, parquet_dir: Path, atol: float = DEFAULT_ATOL):
        """
        Initialize validator with parquet files.

        Args:
            parquet_dir: A collection output directory (with full/ and
                determinism/ subdirectories) or a flat directory of run files
            atol: Max probability difference accepted between orderings
        """
        self.parquet_dir = Path(parquet_dir)
        self.parquet_files = sorted(self.parquet_dir.rglob("*.parquet"))
        self.atol = atol
        self._comparison: Optional[Tuple[DeterminismCheckResult, RandomizationCheckResult]] = None
        logger.info(f"Found {len(self.parquet_files)} parquet files")

    def _files_in(self, subdir: str) -> List[Path]:
        """Files in the named subdirectory if there are any, else all files."""
        return [f for f in self.parquet_files if f.parent.name == subdir] or self.parquet_files

    def _runs_by_dataset(self) -> Dict[str, List[Path]]:
        runs = {}
        for dataset in ("gsm8k", "mbpp"):
            files = [f for f in self._files_in("determinism") if dataset in f.name]
            if files:
                runs[dataset] = files
        return runs

    def validate_all(self) -> ValidationReport:
        """Run all validation checks."""
        logger.info("=" * 60)
        logger.info("DETERMINISM & CONSISTENCY VALIDATION")
        logger.info("=" * 60)

        # Determine dataset and model from files
        dataset = self._infer_dataset()
        model = self._infer_model()

        report = ValidationReport(model=model, dataset=dataset)

        # Run all checks
        logger.info("\n1. Data Quality Check...")
        report.data_quality = self.check_data_quality()

        if report.data_quality.passed:
            logger.info("\n2. Determinism Check...")
            report.determinism = self.check_determinism()

            logger.info("\n3. Randomization Consistency Check...")
            report.randomization = self.check_randomization_consistency()

            logger.info("\n4. Expert Activation Analysis...")
            report.expert_stats = self.analyze_expert_activation()

        # Overall pass/fail
        report.overall_passed = (
            report.data_quality.passed and
            report.determinism.passed and
            report.randomization.passed
        )

        report.summary = self._generate_summary(report)

        return report

    def check_data_quality(self) -> DataQualityCheckResult:
        """Validate data quality: no NaN/inf, valid probabilities and indices."""
        logger.info("Checking data quality...")

        result = DataQualityCheckResult()

        for parquet_file in self.parquet_files:
            try:
                table = pq.read_table(parquet_file, columns=["expert_probs", "selected_experts"])
                prob_lengths = _list_lengths(table.column("expert_probs"))
                probs = _list_values(table.column("expert_probs")).astype(np.float64)
                result.total_rows += len(prob_lengths)

                result.nan_count += int(np.isnan(probs).sum())
                result.inf_count += int(np.isinf(probs).sum())

                # Probability sums should be ~1.0 (rows summing to 0 or NaN are skipped)
                nonempty = prob_lengths > 0
                sums = np.zeros(len(prob_lengths))
                if probs.size:
                    offsets = np.concatenate([[0], np.cumsum(prob_lengths)[:-1]])
                    sums[nonempty] = np.add.reduceat(probs, offsets[nonempty])
                with np.errstate(invalid="ignore"):
                    bad_sum = (sums > 0) & ~((sums >= 0.99) & (sums <= 1.01))
                result.invalid_prob_sums += int(bad_sum.sum())

                # Selected experts must index into that row's expert_probs
                selected_lengths = _list_lengths(table.column("selected_experts"))
                selected = _list_values(table.column("selected_experts")).astype(np.int64)
                num_experts = np.repeat(np.where(nonempty, prob_lengths, 60), selected_lengths)
                result.invalid_indices += int(((selected < 0) | (selected >= num_experts)).sum())

            except Exception as e:
                logger.error(f"Error reading {parquet_file}: {e}")
                result.notes.append(f"Error reading {parquet_file}: {e}")

        # Determine pass/fail
        result.passed = (
            result.nan_count == 0 and
            result.inf_count == 0 and
            result.invalid_prob_sums == 0 and
            result.invalid_indices == 0 and
            len(result.notes) == 0
        )

        if result.passed:
            logger.info(f"✅ Data quality check PASSED ({result.total_rows} rows)")
        else:
            logger.error(f"❌ Data quality check FAILED:")
            logger.error(f"   NaN values: {result.nan_count}")
            logger.error(f"   Inf values: {result.inf_count}")
            logger.error(f"   Invalid prob sums: {result.invalid_prob_sums}")
            logger.error(f"   Invalid indices: {result.invalid_indices}")

        return result

    def _load_run(self, path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Keys (with a row number), expert_probs and selected_experts of one run."""
        table = pq.read_table(path, columns=KEY_COLUMNS + ["expert_probs", "selected_experts"])
        keys = table.select(KEY_COLUMNS).to_pandas()
        keys["row"] = np.arange(len(keys))
        return keys, _list_matrix(table, "expert_probs"), _list_matrix(table, "selected_experts")

    def _compare_runs(self) -> Tuple[DeterminismCheckResult, RandomizationCheckResult]:
        """Compare every ordering of each dataset with its first ordering (computed once)."""
        if self._comparison is not None:
            return self._comparison

        det = DeterminismCheckResult(tolerance=self.atol)
        rnd = RandomizationCheckResult()
        problems = []

        runs = self._runs_by_dataset()
        if not runs:
            problems.append("No GSM8K or MBPP run files to compare")
        for dataset, files in runs.items():
            if len(files) < 2:
                problems.append(f"{dataset}: {len(files)} run, at least 2 orderings are needed")
                continue
            try:
                ref_keys, ref_probs, ref_selected = self._load_run(files[0])
                if ref_keys.duplicated(KEY_COLUMNS).any():
                    problems.append(f"{files[0].name}: duplicate (sequence_id, position, layer) rows")
                for other in files[1:]:
                    keys, probs, selected = self._load_run(other)
                    if keys.duplicated(KEY_COLUMNS).any():
                        problems.append(f"{other.name}: duplicate (sequence_id, position, layer) rows")
                    if probs.shape[1] != ref_probs.shape[1] or selected.shape[1] != ref_selected.shape[1]:
                        problems.append(f"{other.name}: row widths differ from {files[0].name}")
                        continue

                    merged = ref_keys.merge(keys, on=KEY_COLUMNS, how="outer",
                                            suffixes=("_a", "_b"), indicator=True)
                    both = merged["_merge"] == "both"
                    missing = int((~both).sum())
                    if missing:
                        det.missing_rows += missing
                        problems.append(f"{other.name}: {missing} rows not in both {files[0].name} and {other.name}")

                    ia = merged.loc[both, "row_a"].to_numpy(dtype=np.int64)
                    ib = merged.loc[both, "row_b"].to_numpy(dtype=np.int64)
                    pa_, pb_ = ref_probs[ia].astype(np.float64), probs[ib].astype(np.float64)

                    deviation = np.abs(pa_ - pb_).max(axis=1) if pa_.size else np.zeros(len(ia))
                    det.total_comparisons += len(ia)
                    det.identical_probabilities += int((ref_probs[ia] == probs[ib]).all(axis=1).sum())
                    det.exceeding_tolerance += int((deviation > self.atol).sum())
                    if len(deviation):
                        det.max_deviation = max(det.max_deviation, float(deviation.max()))

                    same = (np.sort(ref_selected[ia], axis=1) == np.sort(selected[ib], axis=1)).all(axis=1)
                    k = ref_selected.shape[1]
                    near_tie = ((_topk_boundary_gap(pa_, k) <= 2 * self.atol) |
                                (_topk_boundary_gap(pb_, k) <= 2 * self.atol))
                    rnd.total_shared_tokens += len(ia)
                    rnd.matching_top_k += int(same.sum())
                    rnd.near_tie_flips += int((~same & near_tie).sum())
                    rnd.unexplained_changes += int((~same & ~near_tie).sum())
            except Exception as e:
                problems.append(f"{dataset}: {e}")

        det.notes.extend(problems)
        rnd.notes.extend(problems)
        compared = det.total_comparisons > 0
        det.passed = compared and not problems and det.exceeding_tolerance == 0
        rnd.passed = compared and not problems and rnd.unexplained_changes == 0
        if rnd.total_shared_tokens:
            rnd.match_percentage = 100.0 * rnd.matching_top_k / rnd.total_shared_tokens

        self._comparison = (det, rnd)
        return self._comparison

    def check_determinism(self) -> DeterminismCheckResult:
        """
        Same prompts in different orderings give the same expert probabilities,
        within self.atol. Fails on missing data, missing or duplicate rows,
        or any row differing by more than the tolerance.
        """
        logger.info("Checking determinism...")
        result = self._compare_runs()[0]
        if result.passed:
            logger.info(
                f"✅ Determinism check PASSED ({result.total_comparisons} rows, "
                f"max deviation {result.max_deviation:.2e} <= {result.tolerance:.0e}; "
                f"{result.identical_probabilities} bitwise identical)"
            )
        else:
            logger.error(
                f"❌ Determinism check FAILED: {result.exceeding_tolerance} rows above "
                f"{result.tolerance:.0e}, max deviation {result.max_deviation:.2e}; {result.notes}"
            )
        return result

    def check_randomization_consistency(self) -> RandomizationCheckResult:
        """
        Top-k selections agree across orderings. A change is accepted only at
        a near-tie (k-th and (k+1)-th probabilities within 2 * atol).
        """
        logger.info("Checking randomization consistency...")
        result = self._compare_runs()[1]
        if result.passed:
            logger.info(
                f"✅ Randomization check PASSED ({result.match_percentage:.2f}% identical top-k, "
                f"{result.near_tie_flips} near-tie flips)"
            )
        else:
            logger.error(
                f"❌ Randomization check FAILED: {result.unexplained_changes} top-k changes "
                f"without a near-tie; {result.notes}"
            )
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

    def _infer_model(self) -> str:
        """Infer model name from context."""
        return "qwen1.5-moe-a2.7b"

    def _generate_summary(self, report: ValidationReport) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append(f"Model: {report.model}")
        lines.append(f"Dataset: {report.dataset}")
        lines.append("")

        if report.data_quality.passed:
            lines.append(f"✅ Data Quality: PASSED ({report.data_quality.total_rows} rows)")
        else:
            lines.append(f"❌ Data Quality: FAILED")

        det = report.determinism
        verdict = "✅ Determinism: PASSED" if det.passed else "❌ Determinism: FAILED"
        lines.append(
            f"{verdict} ({det.total_comparisons} rows; max deviation {det.max_deviation:.2e}, "
            f"tolerance {det.tolerance:.0e}; {det.identical_probabilities} bitwise identical)"
        )

        rnd = report.randomization
        verdict = "✅ Randomization: PASSED" if rnd.passed else "❌ Randomization: FAILED"
        lines.append(
            f"{verdict} ({rnd.match_percentage:.2f}% identical top-k; {rnd.near_tie_flips} "
            f"near-tie flips; {rnd.unexplained_changes} unexplained changes)"
        )
        for note in det.notes:
            lines.append(f"   - {note}")

        if report.overall_passed:
            lines.append("")
            lines.append("🎉 ALL VALIDATION CHECKS PASSED")
        else:
            lines.append("")
            lines.append("⚠️  VALIDATION FAILED - Review above results")

        return "\n".join(lines)

    def save_report(self, report: ValidationReport, output_path: Path) -> None:
        """Save validation report to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        report_dict = {
            "model": report.model,
            "dataset": report.dataset,
            "determinism": asdict(report.determinism),
            "randomization": asdict(report.randomization),
            "data_quality": asdict(report.data_quality),
            "expert_stats": [asdict(s) for s in report.expert_stats],
            "overall_passed": report.overall_passed,
            "summary": report.summary,
        }

        with open(output_path, "w") as f:
            json.dump(_json_keys(report_dict), f, indent=2, default=_json_builtin)

        logger.info(f"✅ Report saved to {output_path}")
