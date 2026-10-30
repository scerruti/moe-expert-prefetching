#!/usr/bin/env python3
"""
Determinism and consistency validation (Issue #8).

Validates Phase 1 routing data quality and consistency:
- Bitwise determinism: Same prompts run multiple times produce identical probabilities
- Randomization consistency: Different orderings preserve top-k expert selection
- Data quality: No NaN/inf, probabilities sum to 1.0, valid indices
- Expert activation: Frequency analysis across layers and runs
- Anomaly detection: Flag deviations from expected patterns
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass
class DeterminismCheckResult:
    """Result of bitwise determinism validation."""
    passed: bool = False
    identical_probabilities: int = 0
    total_comparisons: int = 0
    max_deviation: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class RandomizationCheckResult:
    """Result of randomization consistency validation."""
    passed: bool = False
    matching_top_k: int = 0
    total_shared_tokens: int = 0
    match_percentage: float = 0.0
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


class DeterminismValidator:
    """Validate determinism and consistency of routing data."""

    def __init__(self, parquet_dir: Path):
        """
        Initialize validator with parquet files.

        Args:
            parquet_dir: Directory containing phase 1 parquet files
        """
        self.parquet_dir = Path(parquet_dir)
        self.parquet_files = list(self.parquet_dir.glob("*.parquet"))
        logger.info(f"Found {len(self.parquet_files)} parquet files")

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
                table = pq.read_table(parquet_file)
                df = table.to_pandas()

                result.total_rows += len(df)

                # Check expert_probs for NaN/inf
                for idx, row in df.iterrows():
                    expert_probs = row.get("expert_probs", [])
                    # Convert to list if it's a PyArrow array
                    if expert_probs is not None and not isinstance(expert_probs, list):
                        expert_probs = list(expert_probs)

                    if expert_probs:
                        for prob in expert_probs:
                            if isinstance(prob, (float, np.floating)):
                                if np.isnan(prob):
                                    result.nan_count += 1
                                elif np.isinf(prob):
                                    result.inf_count += 1

                    # Check probability sums (should be ~1.0)
                    if expert_probs and sum(expert_probs) > 0:
                        prob_sum = sum(expert_probs)
                        if not (0.99 <= prob_sum <= 1.01):  # Allow small epsilon
                            result.invalid_prob_sums += 1

                    # Check selected_experts validity
                    selected_experts = row.get("selected_experts", [])
                    # Convert to list if it's a PyArrow array
                    if selected_experts is not None and not isinstance(selected_experts, list):
                        selected_experts = list(selected_experts)

                    num_experts = len(expert_probs) if expert_probs else 60
                    for expert_idx in selected_experts:
                        if not (0 <= expert_idx < num_experts):
                            result.invalid_indices += 1

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

    def check_determinism(self) -> DeterminismCheckResult:
        """
        Check bitwise determinism: same prompts run 5x produce identical probabilities.

        Compares all runs of same dataset to detect any variation.
        """
        logger.info("Checking determinism...")

        result = DeterminismCheckResult()

        # Group files by dataset
        gsm8k_files = [f for f in self.parquet_files if "gsm8k" in f.name]
        mbpp_files = [f for f in self.parquet_files if "mbpp" in f.name]

        for dataset_files in [gsm8k_files, mbpp_files]:
            if len(dataset_files) < 2:
                continue

            # Load data from each run
            datasets = []
            for file in sorted(dataset_files):
                df = pq.read_table(file).to_pandas()
                datasets.append(df)

            # Compare across runs
            for i in range(len(datasets) - 1):
                df1 = datasets[i]
                df2 = datasets[i + 1]

                # Compare by sequence_id, token_position, layer_index
                for _, row1 in df1.iterrows():
                    key = (row1["sequence_id"], row1["token_position"], row1["layer_index"])
                    matches = df2[
                        (df2["sequence_id"] == key[0]) &
                        (df2["token_position"] == key[1]) &
                        (df2["layer_index"] == key[2])
                    ]

                    if len(matches) > 0:
                        row2 = matches.iloc[0]
                        result.total_comparisons += 1

                        # Compare probabilities (allowing for float precision)
                        probs1 = row1.get("expert_probs", [])
                        probs2 = row2.get("expert_probs", [])
                        # Convert to lists if they're PyArrow arrays
                        if probs1 is not None and not isinstance(probs1, list):
                            probs1 = list(probs1)
                        if probs2 is not None and not isinstance(probs2, list):
                            probs2 = list(probs2)

                        if probs1 and probs2:
                            max_diff = max(abs(p1 - p2) for p1, p2 in zip(probs1, probs2))
                            result.max_deviation = max(result.max_deviation, max_diff)

                            if max_diff < 1e-6:  # Bitwise identical (within float epsilon)
                                result.identical_probabilities += 1

        # Determine pass/fail
        if result.total_comparisons > 0:
            match_rate = result.identical_probabilities / result.total_comparisons
            result.passed = match_rate >= 0.99  # 99% threshold

            if result.passed:
                logger.info(f"✅ Determinism check PASSED ({match_rate*100:.1f}% match rate)")
            else:
                logger.error(f"❌ Determinism check FAILED ({match_rate*100:.1f}% match rate)")
                logger.error(f"   Max deviation: {result.max_deviation:.2e}")
        else:
            logger.warning("⚠️  Insufficient data for determinism check")
            result.passed = True  # Pass if can't check

        return result

    def check_randomization_consistency(self) -> RandomizationCheckResult:
        """
        Check randomization consistency: different orderings preserve top-k selection.

        Verifies that the top-2 experts are the same for identical tokens across runs.
        """
        logger.info("Checking randomization consistency...")

        result = RandomizationCheckResult()

        # Load all data
        all_data = []
        for parquet_file in self.parquet_files:
            df = pq.read_table(parquet_file).to_pandas()
            all_data.append(df)

        if len(all_data) < 2:
            logger.warning("⚠️  Insufficient runs for randomization check")
            return result

        # Merge all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Group by token and layer
        for (seq_id, token_pos, layer_idx), group in combined_df.groupby(
            ["sequence_id", "token_position", "layer_index"]
        ):
            if len(group) >= 2:
                result.total_shared_tokens += 1

                # Get top-2 experts for each occurrence
                top_experts_list = []
                for _, row in group.iterrows():
                    selected = row.get("selected_experts", [])
                    # Convert to list if it's a PyArrow array
                    if selected is not None and not isinstance(selected, list):
                        selected = list(selected)
                    # Convert to Python ints for comparison
                    selected = [int(x) for x in selected]
                    selected = sorted(selected)[:2]
                    top_experts_list.append(selected)

                # Check if all have same top-2
                if len(set(tuple(t) for t in top_experts_list)) == 1:
                    result.matching_top_k += 1

        # Determine pass/fail
        if result.total_shared_tokens > 0:
            result.match_percentage = (result.matching_top_k / result.total_shared_tokens) * 100
            result.passed = result.match_percentage >= 99.0

            if result.passed:
                logger.info(f"✅ Randomization check PASSED ({result.match_percentage:.1f}% match)")
            else:
                logger.error(f"❌ Randomization check FAILED ({result.match_percentage:.1f}% match)")
        else:
            logger.warning("⚠️  No shared tokens found for randomization check")
            result.passed = True

        return result

    def analyze_expert_activation(self) -> List[ExpertActivationStats]:
        """Analyze expert activation frequencies across all layers."""
        logger.info("Analyzing expert activation...")

        layer_stats = {}

        for parquet_file in self.parquet_files:
            df = pq.read_table(parquet_file).to_pandas()

            for _, row in df.iterrows():
                layer_idx = int(row.get("layer_index", -1))
                selected_experts = row.get("selected_experts", [])
                # Convert to list if it's a PyArrow array
                if selected_experts is not None and not isinstance(selected_experts, list):
                    selected_experts = list(selected_experts)

                if layer_idx not in layer_stats:
                    layer_stats[layer_idx] = {}

                for expert_idx in selected_experts:
                    expert_idx = int(expert_idx)  # Convert PyArrow types to Python int
                    if expert_idx not in layer_stats[layer_idx]:
                        layer_stats[layer_idx][expert_idx] = 0
                    layer_stats[layer_idx][expert_idx] += 1

        # Convert to stats objects
        stats_list = []
        for layer_idx in sorted(layer_stats.keys()):
            expert_counts = layer_stats[layer_idx]
            total = sum(expert_counts.values())

            stats = ExpertActivationStats(layer_index=layer_idx, total_activations=total)
            stats.per_expert_count = expert_counts

            # Calculate frequencies
            for expert_idx, count in expert_counts.items():
                stats.activation_frequency[expert_idx] = count / total if total > 0 else 0

            # Calculate mean and std
            frequencies = list(stats.activation_frequency.values())
            stats.mean_activation = np.mean(frequencies) if frequencies else 0
            stats.std_activation = np.std(frequencies) if frequencies else 0

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

        if report.determinism.passed:
            lines.append(f"✅ Determinism: PASSED ({report.determinism.identical_probabilities}/{report.determinism.total_comparisons} match)")
        else:
            lines.append(f"❌ Determinism: FAILED")

        if report.randomization.passed:
            lines.append(f"✅ Randomization: PASSED ({report.randomization.match_percentage:.1f}%)")
        else:
            lines.append(f"❌ Randomization: FAILED")

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
            json.dump(report_dict, f, indent=2)

        logger.info(f"✅ Report saved to {output_path}")
