#!/usr/bin/env python3
"""
Parquet storage for Phase 1 routing traces (Issue #7).

One row per (prompt, token, layer). Rows are buffered per prompt and written
to disk as a new row group every `flush_every` prompts, so memory stays flat
during a multi-hour collection run.

Usage:
    with RoutingParquetWriter("data/gsm8k_routing_traces.parquet") as writer:
        for prompt_rows in ...:      # one list of row dicts per prompt
            writer.write_prompt(prompt_rows)

    summary = verify_parquet("data/gsm8k_routing_traces.parquet", expected_rows=n)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Column contract shared with Phase 2's loader (phase_2/scripts/data_loader.py).
# Lists are variable-length so one schema covers models with different E and k.
ROUTING_SCHEMA = pa.schema(
    [
        ("sequence_id", pa.string()),
        ("token_id", pa.int32()),
        ("token_str", pa.string()),
        ("token_position", pa.int32()),
        ("layer_index", pa.int32()),
        ("expert_probs", pa.list_(pa.float32())),
        ("selected_experts", pa.list_(pa.int16())),
        ("selected_probs", pa.list_(pa.float32())),
        ("dataset_source", pa.string()),
    ]
)

DEFAULT_FLUSH_EVERY = 500
DEFAULT_COMPRESSION = "zstd"


class RoutingParquetWriter:
    """Incrementally write routing rows to one Parquet file."""

    def __init__(
        self,
        path: Union[str, Path],
        flush_every: int = DEFAULT_FLUSH_EVERY,
        compression: str = DEFAULT_COMPRESSION,
    ):
        """
        Args:
            path: Output Parquet file (parent directories are created)
            flush_every: Prompts to buffer before writing a row group
            compression: Parquet codec ("zstd" or "snappy")
        """
        if flush_every < 1:
            raise ValueError(f"flush_every must be >= 1, got {flush_every}")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._writer = pq.ParquetWriter(
            self.path,
            ROUTING_SCHEMA,
            compression=compression,
            write_page_checksum=True,
        )
        self._buffer: List[Dict[str, Any]] = []
        self._buffered_prompts = 0
        self.rows_written = 0
        self.prompts_written = 0

    def write_prompt(self, rows: List[Dict[str, Any]]) -> None:
        """Buffer all rows for one prompt; flush when the buffer is full."""
        self._buffer.extend(rows)
        self._buffered_prompts += 1
        if self._buffered_prompts >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows as one row group."""
        if not self._buffer:
            return

        table = pa.Table.from_pylist(self._buffer, schema=ROUTING_SCHEMA)
        # from_pylist fills missing keys with nulls; a missing field is a bug upstream
        nulls = [name for name in table.column_names if table[name].null_count]
        if nulls:
            raise ValueError(f"Rows have missing values in columns: {nulls}")

        self._writer.write_table(table)
        self.rows_written += table.num_rows
        self.prompts_written += self._buffered_prompts
        logger.info(
            f"Wrote {table.num_rows} rows ({self._buffered_prompts} prompts) "
            f"to {self.path}"
        )
        self._buffer = []
        self._buffered_prompts = 0

    def close(self) -> None:
        """Flush remaining rows and finalize the file footer."""
        self.flush()
        self._writer.close()

    def __enter__(self) -> "RoutingParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            # Still close so the rows already flushed stay readable
            self._writer.close()


def verify_parquet(path: Union[str, Path], expected_rows: int) -> Dict[str, Any]:
    """
    Read a routing file back and check it is complete and intact.

    Args:
        path: Parquet file written by RoutingParquetWriter
        expected_rows: Number of rows that were written

    Returns:
        Dict with row count, row groups, file size and compression ratio

    Raises:
        ValueError: If the schema or row count does not match
    """
    path = Path(path)
    parquet_file = pq.ParquetFile(path, page_checksum_verification=True)

    schema = parquet_file.schema_arrow
    if not schema.equals(ROUTING_SCHEMA):
        raise ValueError(f"Schema mismatch in {path}:\n{schema}")

    # Full read decodes every page and verifies its CRC checksum
    table = parquet_file.read()
    if table.num_rows != expected_rows:
        raise ValueError(f"{path} has {table.num_rows} rows, expected {expected_rows}")

    file_bytes = path.stat().st_size
    return {
        "rows": table.num_rows,
        "row_groups": parquet_file.metadata.num_row_groups,
        "file_bytes": file_bytes,
        "in_memory_bytes": table.nbytes,
        "compression_ratio": table.nbytes / file_bytes,
    }
