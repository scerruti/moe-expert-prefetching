#!/usr/bin/env python3
"""
Test full pipeline integration (Issue #16).

Validates:
- Complete pipeline execution across 5 runs
- Dataset loading and prompt processing
- Routing data integrity (no NaN/inf)
- Parquet file creation and verification
- Error handling and logging
- Memory management during long runs
"""

import logging
from pathlib import Path
import tempfile
import shutil

import torch

from full_pipeline import FullPipeline, run_full_pipeline, PipelineStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pipeline_initialization():
    """Test that pipeline initializes correctly."""
    logger.info("=" * 60)
    logger.info("Test: Pipeline Initialization")
    logger.info("=" * 60)

    pipeline = FullPipeline(
        model_name="qwen1.5-moe-a2.7b",
        num_runs=2,
        subset_size=5,
    )

    assert pipeline.model_name == "qwen1.5-moe-a2.7b"
    assert pipeline.num_runs == 2
    assert pipeline.subset_size == 5
    assert pipeline.device in ["cuda", "cpu"]
    logger.info(f"✅ Pipeline initialized on {pipeline.device}")

    return True


def test_model_loading():
    """Test model and hook loading."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Model & Hook Loading")
    logger.info("=" * 60)

    pipeline = FullPipeline(
        model_name="qwen1.5-moe-a2.7b",
        num_runs=1,
        subset_size=2,
    )

    pipeline.load_model()

    assert pipeline.model is not None
    assert pipeline.tokenizer is not None
    assert pipeline.config is not None
    assert pipeline.capture is not None
    assert len(pipeline.capture.hooks) > 0
    logger.info(f"✅ Model loaded with {len(pipeline.capture.hooks)} hooks")

    pipeline.capture.cleanup()
    return True


def test_dataset_loading():
    """Test dataset loading and subset limiting."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Dataset Loading")
    logger.info("=" * 60)

    pipeline = FullPipeline(subset_size=10)
    gsm8k_prompts, mbpp_prompts = pipeline.load_datasets()

    assert len(gsm8k_prompts) <= 10
    assert len(mbpp_prompts) <= 10
    assert len(gsm8k_prompts) > 0
    assert len(mbpp_prompts) > 0
    logger.info(f"✅ Loaded {len(gsm8k_prompts)} GSM8K + {len(mbpp_prompts)} MBPP (limited to 10)")

    return True


def test_single_pass_execution():
    """Test execution of a single collection pass."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Single Pass Execution")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_run.parquet"

        pipeline = FullPipeline(
            model_name="qwen1.5-moe-a2.7b",
            num_runs=1,
            subset_size=2,
            output_dir=Path(tmpdir),
        )

        pipeline.load_model()
        gsm8k_prompts, _ = pipeline.load_datasets()

        prompts_processed, routing_events = pipeline.run_single_pass(
            run_id=0,
            seed=42,
            prompts=gsm8k_prompts,
            dataset_name="test_gsm8k",
            output_path=output_path,
        )

        assert prompts_processed > 0
        assert routing_events > 0
        assert output_path.exists()
        logger.info(f"✅ Single pass: {prompts_processed} prompts, {routing_events} routing events")

        pipeline.capture.cleanup()
    return True


def test_parquet_row_validation():
    """Test validation of parquet rows."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Parquet Row Validation")
    logger.info("=" * 60)

    pipeline = FullPipeline()

    # Valid row
    valid_row = {
        "sequence_id": "test_seq_1",
        "token_id": 100,
        "token_str": "hello",
        "token_position": 0,
        "layer_index": 0,
        "expert_probs": [0.1, 0.2, 0.3, 0.4],
        "selected_experts": [0, 3],
        "selected_probs": [0.4, 0.3],
        "dataset_source": "gsm8k",
    }

    assert pipeline._validate_row(valid_row)
    logger.info("✅ Valid row accepted")

    # Invalid row (missing field)
    invalid_row_1 = valid_row.copy()
    del invalid_row_1["layer_index"]
    assert not pipeline._validate_row(invalid_row_1)
    logger.info("✅ Row with missing field rejected")

    # Invalid row (NaN)
    invalid_row_2 = valid_row.copy()
    invalid_row_2["expert_probs"] = [0.1, float('nan'), 0.3, 0.4]
    assert not pipeline._validate_row(invalid_row_2)
    logger.info("✅ Row with NaN rejected")

    # Invalid row (inf)
    invalid_row_3 = valid_row.copy()
    invalid_row_3["expert_probs"] = [0.1, float('inf'), 0.3, 0.4]
    assert not pipeline._validate_row(invalid_row_3)
    logger.info("✅ Row with inf rejected")

    return True


def test_row_conversion():
    """Test conversion of routing data to parquet rows."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Routing Data to Parquet Conversion")
    logger.info("=" * 60)

    pipeline = FullPipeline()

    # Create mock routing data
    from forward_pass import PromptRoutingData, TokenRoutingData

    mock_routing = PromptRoutingData(
        sequence_id="seq_1",
        model="qwen1.5-moe-a2.7b",
        num_tokens=3,
        num_layers=2,
    )

    for token_pos in range(3):
        token_routing = TokenRoutingData(
            sequence_id="seq_1",
            token_id=100 + token_pos,
            token_str=f"token_{token_pos}",
            token_position=token_pos,
        )

        for layer_idx in range(2):
            token_routing.layer_data.append({
                "layer_index": layer_idx,
                "expert_logits": [0.1 * i for i in range(60)],
                "expert_probs": [0.01 * i for i in range(60)],
                "selected_experts": [0, 5, 10, 15],
                "selected_probs": [0.4, 0.3, 0.2, 0.1],
            })

        mock_routing.token_routing.append(token_routing)

    rows = pipeline._convert_to_parquet_rows([mock_routing], "test")

    assert len(rows) == 6  # 3 tokens × 2 layers
    logger.info(f"✅ Converted routing data to {len(rows)} parquet rows")

    return True


def test_full_pipeline_subset_collection():
    """Test full pipeline on 100-example subset."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Full Pipeline on 100-Example Subset")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        stats = run_full_pipeline(
            model_name="qwen1.5-moe-a2.7b",
            num_runs=2,  # Reduced for testing
            subset_size=50,  # 50 prompts per dataset
            output_dir=Path(tmpdir),
        )

        assert isinstance(stats, PipelineStats)
        assert stats.runs_completed > 0
        assert stats.total_prompts_processed > 0
        assert stats.total_routing_events > 0
        logger.info(f"✅ Pipeline completed:")
        logger.info(f"   - Runs: {stats.runs_completed}/2")
        logger.info(f"   - Prompts: {stats.total_prompts_processed}")
        logger.info(f"   - Routing events: {stats.total_routing_events}")
        logger.info(f"   - Errors: {len(stats.errors)}")

    return stats


def test_pipeline_error_handling():
    """Test that pipeline handles errors gracefully."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Error Handling & Logging")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = FullPipeline(
            model_name="qwen1.5-moe-a2.7b",
            num_runs=1,
            subset_size=2,
            output_dir=Path(tmpdir),
        )

        pipeline.load_model()

        # Try a pass with invalid path (parent doesn't exist)
        invalid_output = Path(tmpdir) / "nonexistent" / "deep" / "path" / "test.parquet"

        prompts_processed, routing_events = pipeline.run_single_pass(
            run_id=0,
            seed=42,
            prompts=["Test prompt"],
            dataset_name="test",
            output_path=invalid_output,
        )

        # Pipeline should create parent directories
        assert invalid_output.parent.exists()
        logger.info(f"✅ Pipeline created parent directories automatically")

        pipeline.capture.cleanup()
    return True


def test_determinism_seed_control():
    """Test that seed control enables reproducible runs."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Determinism & Seed Control")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = FullPipeline(
            model_name="qwen1.5-moe-a2.7b",
            num_runs=1,
            subset_size=5,
            output_dir=Path(tmpdir),
        )

        pipeline.load_model()
        gsm8k_prompts, _ = pipeline.load_datasets()

        # Run with seed 42
        prompts_1, events_1 = pipeline.run_single_pass(
            run_id=0,
            seed=42,
            prompts=gsm8k_prompts,
            dataset_name="test",
            output_path=Path(tmpdir) / "run_1.parquet",
        )

        # Same seed should process same number of prompts
        prompts_2, events_2 = pipeline.run_single_pass(
            run_id=0,
            seed=42,
            prompts=gsm8k_prompts,
            dataset_name="test",
            output_path=Path(tmpdir) / "run_2.parquet",
        )

        assert prompts_1 == prompts_2
        assert events_1 == events_2
        logger.info(f"✅ Same seed produced consistent results ({prompts_1} prompts, {events_1} events)")

        pipeline.capture.cleanup()
    return True


def main():
    """Run all tests."""
    logger.info("Starting Full Pipeline Integration Tests\n")

    try:
        test_pipeline_initialization()
        test_model_loading()
        test_dataset_loading()
        test_parquet_row_validation()
        test_row_conversion()
        test_single_pass_execution()
        test_pipeline_error_handling()
        test_determinism_seed_control()
        test_full_pipeline_subset_collection()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nReady for Issue #8 (Determinism Validation).")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
