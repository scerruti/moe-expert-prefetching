#!/usr/bin/env python3
"""
Full data collection pipeline integration (Issue #16).

Orchestrates:
- Data collection skeleton (Issue #13) — 5 runs, outer/middle/inner loops
- Tokenization (Issue #14) — Model-specific token preparation
- Forward pass (Issue #15) — Hook-based routing telemetry capture
- Parquet storage (Issue #7) — Disk persistence with checksums

End-to-end Phase 1 data collection with comprehensive error handling and validation.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dataset_loaders import GSM8KLoader, MBPPLoader
from forward_pass import ForwardPassCapture
from model_loader import ModelLoader
from parquet_storage import RoutingParquetWriter, verify_parquet
from tokenizer import TokenizationPipeline

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for one randomized run."""
    run_id: int
    seed: int
    dataset: str
    model: str
    output_path: Path


@dataclass
class PipelineStats:
    """Aggregate statistics for full pipeline execution."""
    total_prompts_processed: int = 0
    total_tokens_processed: int = 0
    total_routing_events: int = 0
    runs_completed: int = 0
    errors: List[str] = None
    validation_passed: bool = False

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class FullPipeline:
    """Complete Phase 1 data collection pipeline."""

    def __init__(
        self,
        model_name: str = "qwen1.5-moe-a2.7b",
        num_runs: int = 5,
        subset_size: Optional[int] = None,
        output_dir: Path = Path("data/phase_1_collection"),
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        """
        Initialize full pipeline.

        Args:
            model_name: Model to use for collection
            num_runs: Number of randomized runs (default 5)
            subset_size: Limit to N examples per dataset (None = use all)
            output_dir: Directory for parquet files
            device: Device for model (default: cuda if available)
            batch_size: Prompts per forward pass (lower if GPU runs out of memory)
        """
        self.model_name = model_name
        self.num_runs = num_runs
        self.subset_size = subset_size
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.config = None
        self.capture = None
        self.stats = PipelineStats()

    def load_model(self) -> None:
        """Load model and setup forward pass capture."""
        logger.info(f"Loading {self.model_name}...")
        loader = ModelLoader(self.model_name, load_in_4bit=True)
        self.model, self.tokenizer, self.config = loader.load()
        self.model.to(self.device)
        self.model.eval()

        self.capture = ForwardPassCapture(self.model_name, device=self.device)
        self.capture.model = self.model
        self.capture.tokenizer = self.tokenizer
        self.capture.config = self.config
        self.capture.setup_hooks()

        logger.info(f"✅ Model loaded, hooks registered")

    def load_datasets(self) -> Tuple[List[str], List[str]]:
        """Load GSM8K and MBPP datasets."""
        logger.info("Loading datasets...")

        gsm8k_loader = GSM8KLoader()
        mbpp_loader = MBPPLoader()

        # Loaders return example dicts; the pipeline needs the prompt text
        gsm8k_prompts = [ex["question"] for ex in gsm8k_loader.load()]
        mbpp_prompts = [ex["prompt"] for ex in mbpp_loader.load()]

        if self.subset_size:
            gsm8k_prompts = gsm8k_prompts[:self.subset_size]
            mbpp_prompts = mbpp_prompts[:self.subset_size]

        logger.info(f"✅ Loaded {len(gsm8k_prompts)} GSM8K + {len(mbpp_prompts)} MBPP prompts")
        return gsm8k_prompts, mbpp_prompts

    def run_single_pass(
        self,
        run_id: int,
        seed: int,
        prompts: List[str],
        dataset_name: str,
        output_path: Path,
    ) -> Tuple[int, int]:
        """
        Execute one randomized collection pass.

        Args:
            run_id: Run number (0-4)
            seed: Random seed for this run
            prompts: List of prompts to process
            dataset_name: Dataset name for metadata
            output_path: Path to write parquet file

        Returns:
            Tuple of (prompts_processed, total_routing_events)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {run_id + 1}/{self.num_runs} (seed={seed})")
        logger.info(f"{'='*60}")

        random.seed(seed)
        torch.manual_seed(seed)

        # Shuffle prompts for this run
        shuffled_prompts = prompts.copy()
        random.shuffle(shuffled_prompts)

        prompts_processed = 0
        routing_events_count = 0

        try:
            # Process in mini-batches: a single forward pass over thousands of
            # padded prompts would exhaust GPU memory, and holding every
            # routing row in RAM before writing would exhaust host memory.
            tokenizer_pipeline = TokenizationPipeline(self.tokenizer, self.model_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            total = len(shuffled_prompts)
            logger.info(f"Processing {total} prompts in batches of {self.batch_size}...")

            with RoutingParquetWriter(output_path, flush_every=500) as writer:
                for start in range(0, total, self.batch_size):
                    batch_prompts = shuffled_prompts[start:start + self.batch_size]
                    tokenization_result = tokenizer_pipeline.tokenize_batch(batch_prompts)

                    routing_data_list = self.capture.run_forward_pass(
                        token_tensors=tokenization_result["token_tensors"],
                        sequence_ids=tokenization_result["sequence_ids"],
                        token_ids_list=tokenization_result["token_ids"],
                        token_strs_list=tokenization_result["token_strs"],
                    )

                    # One write_prompt call per prompt so flush_every counts prompts
                    for prompt_data in routing_data_list:
                        rows = self._convert_to_parquet_rows(
                            routing_data_list=[prompt_data],
                            dataset_name=dataset_name,
                        )
                        writer.write_prompt(rows)
                        routing_events_count += len(rows)

                    prompts_processed += len(batch_prompts)
                    if prompts_processed % (self.batch_size * 50) == 0 or prompts_processed == total:
                        logger.info(f"  {prompts_processed}/{total} prompts, {routing_events_count} routing records")

            # Verify parquet file integrity
            logger.info(f"Verifying parquet file integrity...")
            summary = verify_parquet(output_path, expected_rows=routing_events_count)
            logger.info(
                f"✅ Run {run_id + 1} complete: "
                f"{summary['rows']} rows, {summary['row_groups']} row groups, "
                f"compression {summary['compression_ratio']:.2f}:1"
            )

        except Exception as e:
            logger.error(f"❌ Run {run_id + 1} failed: {e}", exc_info=True)
            self.stats.errors.append(f"Run {run_id + 1}: {str(e)}")

        return prompts_processed, routing_events_count

    def _convert_to_parquet_rows(
        self,
        routing_data_list: List,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        """Convert PromptRoutingData to parquet schema rows."""
        rows = []

        for prompt_data in routing_data_list:
            for token_routing in prompt_data.token_routing:
                for layer_data in token_routing.layer_data:
                    row = {
                        "sequence_id": prompt_data.sequence_id,
                        "token_id": token_routing.token_id,
                        "token_str": token_routing.token_str,
                        "token_position": token_routing.token_position,
                        "layer_index": layer_data.get("layer_index", -1),
                        "expert_probs": layer_data.get("expert_probs", []),
                        "selected_experts": layer_data.get("selected_experts", []),
                        "selected_probs": layer_data.get("selected_probs", []),
                        "dataset_source": dataset_name,
                    }

                    # Validate row
                    if self._validate_row(row):
                        rows.append(row)
                    else:
                        logger.warning(f"Skipping invalid row: {row}")

        return rows

    def _validate_row(self, row: Dict[str, Any]) -> bool:
        """Validate that a parquet row has no NaN/inf and required fields."""
        required_fields = [
            "sequence_id", "token_id", "token_str", "token_position",
            "layer_index", "expert_probs", "selected_experts", "selected_probs",
            "dataset_source"
        ]

        # Check required fields
        if not all(field in row for field in required_fields):
            return False

        # Check for NaN/inf in probabilities
        if row["expert_probs"]:
            if not all(isinstance(p, (int, float)) and not (p != p or p == float('inf') or p == float('-inf')) for p in row["expert_probs"]):
                return False

        if row["selected_probs"]:
            if not all(isinstance(p, (int, float)) and not (p != p or p == float('inf') or p == float('-inf')) for p in row["selected_probs"]):
                return False

        return True

    def run_collection(self) -> PipelineStats:
        """Execute full data collection (5 runs, both datasets)."""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1 FULL PIPELINE EXECUTION")
        logger.info("=" * 60)

        self.load_model()
        gsm8k_prompts, mbpp_prompts = self.load_datasets()

        run_seeds = [42, 123, 456, 789, 999]  # Fixed seeds for reproducibility

        for run_id in range(self.num_runs):
            seed = run_seeds[run_id]

            # GSM8K pass
            gsm8k_output = self.output_dir / f"run_{run_id}_gsm8k.parquet"
            prompts, events = self.run_single_pass(
                run_id=run_id,
                seed=seed,
                prompts=gsm8k_prompts,
                dataset_name="gsm8k",
                output_path=gsm8k_output,
            )
            self.stats.total_prompts_processed += prompts
            self.stats.total_routing_events += events

            # MBPP pass
            mbpp_output = self.output_dir / f"run_{run_id}_mbpp.parquet"
            prompts, events = self.run_single_pass(
                run_id=run_id,
                seed=seed,
                prompts=mbpp_prompts,
                dataset_name="mbpp",
                output_path=mbpp_output,
            )
            self.stats.total_prompts_processed += prompts
            self.stats.total_routing_events += events

            self.stats.runs_completed += 1

            # Clear GPU cache between runs to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"Cleared GPU cache after run {run_id + 1}")

        # Final validation
        self._validate_pipeline_completion()

        self.capture.cleanup()

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Runs completed: {self.stats.runs_completed}/{self.num_runs}")
        logger.info(f"Prompts processed: {self.stats.total_prompts_processed}")
        logger.info(f"Routing events captured: {self.stats.total_routing_events}")
        logger.info(f"Errors: {len(self.stats.errors)}")

        if self.stats.errors:
            logger.error("Errors encountered:")
            for error in self.stats.errors:
                logger.error(f"  - {error}")
        else:
            logger.info("✅ All runs completed successfully")

        return self.stats

    def _validate_pipeline_completion(self) -> bool:
        """Validate that pipeline completed all required tasks."""
        if self.stats.runs_completed < self.num_runs:
            logger.error(f"Not all runs completed: {self.stats.runs_completed}/{self.num_runs}")
            self.stats.validation_passed = False
            return False

        if self.stats.total_routing_events == 0:
            logger.error("No routing events captured")
            self.stats.validation_passed = False
            return False

        if len(self.stats.errors) > 0:
            logger.error(f"{len(self.stats.errors)} errors occurred")
            self.stats.validation_passed = False
            return False

        logger.info("✅ Pipeline validation passed")
        self.stats.validation_passed = True
        return True


def run_full_pipeline(
    model_name: str = "qwen1.5-moe-a2.7b",
    num_runs: int = 5,
    subset_size: int = 100,
    output_dir: Path = Path("data/phase_1_collection"),
) -> PipelineStats:
    """
    Execute full Phase 1 data collection pipeline.

    Args:
        model_name: Model for collection
        num_runs: Number of randomized runs
        subset_size: Limit prompts per dataset (for testing)
        output_dir: Output directory for parquet files

    Returns:
        PipelineStats with execution results
    """
    pipeline = FullPipeline(
        model_name=model_name,
        num_runs=num_runs,
        subset_size=subset_size,
        output_dir=output_dir,
    )
    return pipeline.run_collection()
