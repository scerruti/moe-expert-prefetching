#!/usr/bin/env python3
"""
Test determinism and consistency validation (Issue #8).

Validates:
- Bitwise determinism: Same prompts produce identical probabilities
- Randomization consistency: Different orderings preserve top-k
- Data quality: No NaN/inf, valid probabilities and indices
- Expert activation analysis
- Report generation
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from determinism_validation import (
    DeterminismValidator,
    ValidationReport,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_parquet_file(
    output_path: Path,
    num_rows: int = 100,
    dataset_name: str = "gsm8k",
    include_nan: bool = False,
    include_inf: bool = False,
) -> None:
    """Create a test parquet file with synthetic routing data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(num_rows):
        # Create synthetic routing data
        expert_probs = [0.1 * (j % 8) for j in range(60)]
        if include_nan and i == 0:
            expert_probs[0] = float('nan')
        if include_inf and i == 1:
            expert_probs[1] = float('inf')

        # Normalize probabilities
        prob_sum = sum(expert_probs)
        if prob_sum > 0 and not include_nan and not include_inf:
            expert_probs = [p / prob_sum for p in expert_probs]

        rows.append({
            "sequence_id": f"seq_{i % 10}",
            "token_id": 100 + (i % 20),
            "token_str": f"tok_{i % 20}",
            "token_position": i % 10,
            "layer_index": i % 24,
            "expert_probs": expert_probs[:60],
            "selected_experts": [0, 3, 10, 15][:4],
            "selected_probs": [0.4, 0.3, 0.2, 0.1][:4],
            "dataset_source": dataset_name,
        })

    schema = pa.schema([
        ("sequence_id", pa.string()),
        ("token_id", pa.int32()),
        ("token_str", pa.string()),
        ("token_position", pa.int32()),
        ("layer_index", pa.int32()),
        ("expert_probs", pa.list_(pa.float32())),
        ("selected_experts", pa.list_(pa.int16())),
        ("selected_probs", pa.list_(pa.float32())),
        ("dataset_source", pa.string()),
    ])

    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output_path)


def test_data_quality_check_valid():
    """Test data quality check on valid data."""
    logger.info("=" * 60)
    logger.info("Test: Data Quality Check (Valid Data)")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        create_test_parquet_file(tmpdir / "test_gsm8k.parquet", num_rows=100)

        validator = DeterminismValidator(tmpdir)
        result = validator.check_data_quality()

        assert result.passed
        assert result.total_rows == 100
        assert result.nan_count == 0
        assert result.inf_count == 0
        logger.info(f"✅ Valid data check passed ({result.total_rows} rows)")


def test_data_quality_check_with_nan():
    """Test data quality check detects NaN values."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Data Quality Check (NaN Detection)")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        create_test_parquet_file(tmpdir / "test_gsm8k.parquet", num_rows=100, include_nan=True)

        validator = DeterminismValidator(tmpdir)
        result = validator.check_data_quality()

        assert not result.passed
        assert result.nan_count > 0
        logger.info(f"✅ NaN detection working ({result.nan_count} NaN values found)")


def test_data_quality_check_with_inf():
    """Test data quality check detects inf values."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Data Quality Check (Inf Detection)")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        create_test_parquet_file(tmpdir / "test_gsm8k.parquet", num_rows=100, include_inf=True)

        validator = DeterminismValidator(tmpdir)
        result = validator.check_data_quality()

        assert not result.passed
        assert result.inf_count > 0
        logger.info(f"✅ Inf detection working ({result.inf_count} inf values found)")


def test_determinism_check():
    """Test determinism check on identical runs."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Determinism Check")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create two identical runs
        create_test_parquet_file(tmpdir / "run_0_gsm8k.parquet", num_rows=50)
        create_test_parquet_file(tmpdir / "run_1_gsm8k.parquet", num_rows=50)

        validator = DeterminismValidator(tmpdir)
        result = validator.check_determinism()

        assert result.passed
        assert result.total_comparisons > 0
        logger.info(
            f"✅ Determinism check: {result.identical_probabilities}/{result.total_comparisons} match "
            f"(max deviation: {result.max_deviation:.2e})"
        )


def test_randomization_consistency_check():
    """Test randomization consistency check."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Randomization Consistency Check")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multiple runs with same data
        for run in range(3):
            create_test_parquet_file(
                tmpdir / f"run_{run}_gsm8k.parquet",
                num_rows=50,
            )

        validator = DeterminismValidator(tmpdir)
        result = validator.check_randomization_consistency()

        logger.info(
            f"✅ Randomization check: {result.matching_top_k}/{result.total_shared_tokens} match "
            f"({result.match_percentage:.1f}%)"
        )


def test_expert_activation_analysis():
    """Test expert activation frequency analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Expert Activation Analysis")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        create_test_parquet_file(tmpdir / "test_gsm8k.parquet", num_rows=100)

        validator = DeterminismValidator(tmpdir)
        stats = validator.analyze_expert_activation()

        assert len(stats) > 0
        logger.info(f"✅ Analyzed {len(stats)} layers")

        for stat in stats[:3]:  # Show first 3 layers
            logger.info(
                f"  Layer {stat.layer_index}: "
                f"{stat.total_activations} total, "
                f"{len(stat.per_expert_count)} unique experts, "
                f"mean freq: {stat.mean_activation:.3f}, "
                f"std: {stat.std_activation:.3f}"
            )


def test_validation_report_generation():
    """Test complete validation report generation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Validation Report Generation")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test data
        for run in range(2):
            create_test_parquet_file(
                tmpdir / f"run_{run}_gsm8k.parquet",
                num_rows=50,
            )

        validator = DeterminismValidator(tmpdir)
        report = validator.validate_all()

        assert isinstance(report, ValidationReport)
        assert report.model is not None
        assert report.dataset is not None
        logger.info("\n" + report.summary)


def test_report_persistence():
    """Test saving and loading validation report."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Report Persistence")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test data
        create_test_parquet_file(tmpdir / "test_gsm8k.parquet", num_rows=50)

        # Generate and save report
        validator = DeterminismValidator(tmpdir)
        report = validator.validate_all()

        report_path = tmpdir / "validation_report.json"
        validator.save_report(report, report_path)

        assert report_path.exists()
        logger.info(f"✅ Report saved to {report_path}")

        # Load and verify
        with open(report_path) as f:
            loaded = json.load(f)

        assert loaded["model"] == report.model
        assert loaded["dataset"] == report.dataset
        logger.info(f"✅ Report loaded and verified")


def test_invalid_expert_indices():
    """Test detection of invalid expert indices."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Invalid Expert Index Detection")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create data with invalid expert indices
        output_path = tmpdir / "test_invalid.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(10):
            rows.append({
                "sequence_id": f"seq_{i}",
                "token_id": 100 + i,
                "token_str": f"tok_{i}",
                "token_position": i,
                "layer_index": 0,
                "expert_probs": [0.1] * 60,
                "selected_experts": [0, 100],  # 100 is invalid (>60)
                "selected_probs": [0.5, 0.5],
                "dataset_source": "test",
            })

        schema = pa.schema([
            ("sequence_id", pa.string()),
            ("token_id", pa.int32()),
            ("token_str", pa.string()),
            ("token_position", pa.int32()),
            ("layer_index", pa.int32()),
            ("expert_probs", pa.list_(pa.float32())),
            ("selected_experts", pa.list_(pa.int16())),
            ("selected_probs", pa.list_(pa.float32())),
            ("dataset_source", pa.string()),
        ])

        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, output_path)

        validator = DeterminismValidator(tmpdir)
        result = validator.check_data_quality()

        assert not result.passed
        assert result.invalid_indices > 0
        logger.info(f"✅ Invalid indices detected ({result.invalid_indices} found)")


def main():
    """Run all tests."""
    logger.info("Starting Determinism & Consistency Validation Tests\n")

    try:
        test_data_quality_check_valid()
        test_data_quality_check_with_nan()
        test_data_quality_check_with_inf()
        test_determinism_check()
        test_randomization_consistency_check()
        test_expert_activation_analysis()
        test_validation_report_generation()
        test_report_persistence()
        test_invalid_expert_indices()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nDeterminism validation ready for Phase 1 deployment.")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
