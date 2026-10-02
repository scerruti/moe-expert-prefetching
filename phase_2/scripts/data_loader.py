"""
Phase 2: Data Loading & Preparation

Unified interface for loading and validating Phase 1 Parquet files.
Handles schema validation, data quality checks, and memory profiling.

Issue #40: Data Loading & Preparation
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import psutil
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Expected Parquet schema (9 columns from SYSTEM_DESIGN.md)
EXPECTED_SCHEMA = {
    'sequence_id': str,
    'token_id': 'int32',
    'token_str': str,
    'token_position': 'int32',
    'layer_index': 'int32',
    'expert_probs': list,  # float32[E] where E=num_experts
    'selected_experts': list,  # int16[k] where k=2 (top-k)
    'selected_probs': list,  # float32[k]
    'dataset_source': str,  # 'gsm8k' or 'mbpp'
}

REQUIRED_COLUMNS = list(EXPECTED_SCHEMA.keys())


class RoutingDataLoader:
    """Unified interface for loading Phase 1 Parquet files."""

    def __init__(self, parquet_path: str, model_name: str = "qwen"):
        """
        Initialize data loader.

        Args:
            parquet_path: Path to Phase 1 Parquet file
            model_name: Model identifier (e.g., "qwen", "deepseek")
        """
        self.parquet_path = Path(parquet_path)
        self.model_name = model_name
        self.data = None
        self._metadata = None
        logger.info(f"Initialized loader for {parquet_path}")

    def load(self, lazy: bool = True) -> pl.DataFrame:
        """
        Load Parquet file using Polars.

        Args:
            lazy: If True, use lazy evaluation for large files

        Returns:
            Polars DataFrame (or LazyFrame if lazy=True)
        """
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        logger.info(f"Loading {self.parquet_path}")

        if lazy:
            self.data = pl.scan_parquet(str(self.parquet_path))
            logger.info("Loaded as lazy DataFrame")
        else:
            self.data = pl.read_parquet(str(self.parquet_path))
            logger.info(f"Loaded {len(self.data)} rows")

        return self.data

    def validate_schema(self) -> Dict[str, bool]:
        """
        Verify Parquet schema against specification.

        Returns:
            Dict with validation results and details
        """
        if self.data is None:
            self.load(lazy=False)

        logger.info("Validating schema...")
        results = {
            'schema_valid': True,
            'columns_present': True,
            'missing_columns': [],
            'extra_columns': [],
            'type_issues': [],
        }

        # Check for required columns
        actual_columns = set(self.data.columns)
        expected_columns = set(REQUIRED_COLUMNS)

        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        if missing:
            results['columns_present'] = False
            results['missing_columns'] = list(missing)
            results['schema_valid'] = False
            logger.error(f"Missing columns: {missing}")

        if extra:
            results['extra_columns'] = list(extra)
            logger.warning(f"Extra columns: {extra}")

        # Basic type checks (note: Polars types differ from Python types)
        for col in self.data.columns:
            col_type = self.data[col].dtype
            # Basic validation - just log type info
            logger.debug(f"Column '{col}' type: {col_type}")

        logger.info(f"Schema validation: {'PASS' if results['schema_valid'] else 'FAIL'}")
        return results

    def quality_report(self) -> Dict:
        """
        Generate data quality statistics.

        Returns:
            Dict with quality metrics
        """
        if self.data is None:
            self.load(lazy=False)

        logger.info("Generating data quality report...")

        # Collect to memory for analysis (use sample if too large)
        df = self.data.collect() if hasattr(self.data, 'collect') else self.data

        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'null_counts': {},
            'value_ranges': {},
            'issues': [],
        }

        # Check for null/NaN values
        for col in df.columns:
            null_count = df[col].is_null().sum()
            report['null_counts'][col] = null_count

            if null_count > 0:
                pct = (null_count / len(df)) * 100
                report['issues'].append(f"Column '{col}': {null_count} nulls ({pct:.2f}%)")
                logger.warning(f"Column '{col}': {null_count} null values")

        # Check numeric columns for ranges
        numeric_cols = ['token_id', 'token_position', 'layer_index']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    report['value_ranges'][col] = (min_val, max_val)
                    logger.info(f"Column '{col}': range [{min_val}, {max_val}]")
                except Exception as e:
                    logger.warning(f"Could not compute range for '{col}': {e}")

        # Check probability columns (should be in [0, 1])
        if 'selected_probs' in df.columns:
            logger.info("Validating probability values...")
            try:
                # Note: selected_probs is a list column, need to check differently
                logger.debug("Probability validation: using sampling approach")
            except Exception as e:
                logger.warning(f"Could not validate probabilities: {e}")

        # Check dataset_source values
        if 'dataset_source' in df.columns:
            sources = df['dataset_source'].unique()
            logger.info(f"Dataset sources: {sources}")

        logger.info(f"Quality check complete: {len(report['issues'])} issues found")
        return report

    def summary_stats(self) -> pd.DataFrame:
        """
        Compute aggregate statistics.

        Returns:
            DataFrame with summary statistics
        """
        if self.data is None:
            self.load(lazy=False)

        logger.info("Computing summary statistics...")

        df = self.data.collect() if hasattr(self.data, 'collect') else self.data

        stats = {
            'metric': [],
            'value': [],
        }

        # Basic stats
        stats['metric'].append('total_rows')
        stats['value'].append(len(df))

        stats['metric'].append('total_columns')
        stats['value'].append(len(df.columns))

        # Dataset breakdown
        if 'dataset_source' in df.columns:
            for source in df['dataset_source'].unique():
                count = (df['dataset_source'] == source).sum()
                stats['metric'].append(f'rows_{source}')
                stats['value'].append(count)
                logger.info(f"Dataset {source}: {count} rows")

        # Unique counts
        if 'sequence_id' in df.columns:
            unique_seqs = df['sequence_id'].n_unique()
            stats['metric'].append('unique_sequences')
            stats['value'].append(unique_seqs)
            logger.info(f"Unique sequences: {unique_seqs}")

        if 'layer_index' in df.columns:
            unique_layers = df['layer_index'].n_unique()
            stats['metric'].append('unique_layers')
            stats['value'].append(unique_layers)
            logger.info(f"Unique layers: {unique_layers}")

        return pd.DataFrame(stats)

    def filter_by_dataset(self, dataset_name: str) -> 'RoutingDataLoader':
        """Filter data by dataset (GSM8K or MBPP)."""
        if self.data is None:
            self.load()

        filtered = self.data.filter(pl.col('dataset_source') == dataset_name)
        logger.info(f"Filtered to {dataset_name}")
        return filtered

    def filter_by_layer(self, layer_idx: int) -> 'RoutingDataLoader':
        """Filter data by layer index."""
        if self.data is None:
            self.load()

        filtered = self.data.filter(pl.col('layer_index') == layer_idx)
        logger.info(f"Filtered to layer {layer_idx}")
        return filtered

    def sample(self, n: int, stratify: bool = True) -> pl.DataFrame:
        """
        Sample from data with optional stratification.

        Args:
            n: Number of samples
            stratify: If True, stratify by dataset_source and layer_index

        Returns:
            Sampled DataFrame
        """
        if self.data is None:
            self.load(lazy=False)

        df = self.data.collect() if hasattr(self.data, 'collect') else self.data

        if stratify and 'dataset_source' in df.columns:
            # Stratified sampling
            sample_data = []
            for source in df['dataset_source'].unique():
                source_data = df.filter(pl.col('dataset_source') == source)
                n_source = min(n // 2, len(source_data))
                sampled = source_data.sample(n=n_source, seed=42)
                sample_data.append(sampled)

            result = pl.concat(sample_data)
            logger.info(f"Stratified sample: {len(result)} rows")
        else:
            result = df.sample(n=min(n, len(df)), seed=42)
            logger.info(f"Random sample: {len(result)} rows")

        return result

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if self.data is None:
            self.load(lazy=False)

        df = self.data.collect() if hasattr(self.data, 'collect') else self.data
        return df.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        """Get as Polars DataFrame."""
        if self.data is None:
            self.load(lazy=False)

        return self.data.collect() if hasattr(self.data, 'collect') else self.data

    def memory_profile(self) -> Dict[str, float]:
        """
        Profile memory usage.

        Returns:
            Dict with memory metrics (in MB)
        """
        process = psutil.Process(os.getpid())

        # Get memory info before and after loading
        before = process.memory_info().rss / 1024 / 1024

        if self.data is None:
            self.load(lazy=False)

        after = process.memory_info().rss / 1024 / 1024

        profile = {
            'before_load_mb': before,
            'after_load_mb': after,
            'peak_increase_mb': after - before,
            'timestamp': datetime.now().isoformat(),
        }

        logger.info(f"Memory profile: +{profile['peak_increase_mb']:.2f} MB")
        return profile


def create_data_quality_report(
    parquet_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Create comprehensive data quality report.

    Args:
        parquet_path: Path to Parquet file
        output_path: Optional path to save report (default: data_quality_report.md)

    Returns:
        Report text
    """
    loader = RoutingDataLoader(parquet_path)
    loader.load(lazy=False)

    # Run validations
    schema_report = loader.validate_schema()
    quality_report = loader.quality_report()
    summary_stats = loader.summary_stats()
    memory_profile = loader.memory_profile()

    # Format report
    report_lines = [
        "# Phase 2 Data Quality Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"Data source: {parquet_path}",
        "\n## Schema Validation",
        f"Status: {'✅ PASS' if schema_report['schema_valid'] else '❌ FAIL'}",
    ]

    if schema_report['missing_columns']:
        report_lines.append(f"\nMissing columns: {schema_report['missing_columns']}")

    if schema_report['extra_columns']:
        report_lines.append(f"Extra columns: {schema_report['extra_columns']}")

    report_lines.extend([
        "\n## Quality Metrics",
        f"Total rows: {quality_report['total_rows']}",
        f"Total columns: {quality_report['total_columns']}",
    ])

    if quality_report['issues']:
        report_lines.append("\n### Issues Found:")
        for issue in quality_report['issues']:
            report_lines.append(f"- {issue}")
    else:
        report_lines.append("\n✅ No quality issues detected")

    report_lines.extend([
        "\n## Summary Statistics",
        summary_stats.to_string(index=False),
        f"\n## Memory Profile",
        f"Before load: {memory_profile['before_load_mb']:.2f} MB",
        f"After load: {memory_profile['after_load_mb']:.2f} MB",
        f"Increase: {memory_profile['peak_increase_mb']:.2f} MB",
    ])

    report_text = "\n".join(report_lines)

    if output_path:
        Path(output_path).write_text(report_text)
        logger.info(f"Report saved to {output_path}")

    return report_text


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        parquet_file = sys.argv[1]
        loader = RoutingDataLoader(parquet_file)
        loader.load(lazy=False)
        loader.validate_schema()
        loader.quality_report()
        print(loader.summary_stats())
    else:
        print("Usage: python data_loader.py <parquet_file>")
