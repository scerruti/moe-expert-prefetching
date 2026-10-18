import random
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parquet_storage import ROUTING_SCHEMA, RoutingParquetWriter, verify_parquet


def make_prompt_rows(prompt_idx, num_tokens=3, num_layers=2, num_experts=8, k=2):
    """Rows for one prompt, shaped like real router output (softmax + top-k)."""
    rng = random.Random(prompt_idx)
    rows = []
    for pos in range(num_tokens):
        for layer in range(num_layers):
            weights = [rng.random() for _ in range(num_experts)]
            probs = [w / sum(weights) for w in weights]
            top = sorted(range(num_experts), key=lambda e: -probs[e])[:k]
            rows.append(
                {
                    "sequence_id": f"gsm8k-{prompt_idx}",
                    "token_id": 100 + pos,
                    "token_str": f"tok{pos}",
                    "token_position": pos,
                    "layer_index": layer,
                    "expert_probs": probs,
                    "selected_experts": top,
                    "selected_probs": [probs[e] for e in top],
                    "dataset_source": "gsm8k",
                }
            )
    return rows


class ParquetStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "out" / "traces.parquet"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, prompts, **kwargs):
        with RoutingParquetWriter(self.path, **kwargs) as writer:
            for rows in prompts:
                writer.write_prompt(rows)
        return writer

    def test_round_trip_preserves_values(self):
        prompts = [make_prompt_rows(i) for i in range(3)]
        self.write(prompts)

        read = pq.read_table(self.path).to_pylist()
        expected = [row for rows in prompts for row in rows]
        self.assertEqual(len(read), len(expected))
        for got, want in zip(read, expected):
            for col in ["sequence_id", "token_id", "token_str", "token_position"]:
                self.assertEqual(got[col], want[col])
            self.assertEqual(got["selected_experts"], want["selected_experts"])
            # float32 storage: equal to ~7 significant digits
            for g, w in zip(got["expert_probs"], want["expert_probs"]):
                self.assertAlmostEqual(g, w, places=6)

    def test_schema_matches_phase_2_contract(self):
        self.write([make_prompt_rows(0)])
        self.assertEqual(
            pq.read_schema(self.path).names,
            [
                "sequence_id",
                "token_id",
                "token_str",
                "token_position",
                "layer_index",
                "expert_probs",
                "selected_experts",
                "selected_probs",
                "dataset_source",
            ],
        )
        self.assertTrue(pq.read_schema(self.path).equals(ROUTING_SCHEMA))

    def test_flushes_every_n_prompts(self):
        writer = RoutingParquetWriter(self.path, flush_every=2)
        writer.write_prompt(make_prompt_rows(0))
        self.assertEqual(writer.rows_written, 0)  # still buffered
        writer.write_prompt(make_prompt_rows(1))
        self.assertEqual(writer.rows_written, 12)  # flushed, buffer empty
        self.assertEqual(writer._buffer, [])
        for i in range(2, 5):
            writer.write_prompt(make_prompt_rows(i))
        writer.close()

        self.assertEqual(writer.prompts_written, 5)
        # 5 prompts at flush_every=2 -> groups of 2, 2, then 1 on close
        self.assertEqual(pq.ParquetFile(self.path).metadata.num_row_groups, 3)

    def test_variable_expert_counts(self):
        mixtral_like = make_prompt_rows(0, num_experts=8, k=2)
        qwen_like = make_prompt_rows(1, num_experts=60, k=4)
        self.write([mixtral_like, qwen_like])

        lengths = {len(r["expert_probs"]) for r in pq.read_table(self.path).to_pylist()}
        self.assertEqual(lengths, {8, 60})

    def test_missing_field_is_rejected(self):
        rows = make_prompt_rows(0)
        del rows[0]["layer_index"]
        writer = RoutingParquetWriter(self.path)
        writer.write_prompt(rows)
        with self.assertRaisesRegex(ValueError, "layer_index"):
            writer.flush()

    def test_rows_flushed_before_an_error_stay_readable(self):
        with (
            self.assertRaises(RuntimeError),
            RoutingParquetWriter(self.path, flush_every=1) as writer,
        ):
            writer.write_prompt(make_prompt_rows(0))
            raise RuntimeError("forward pass crashed")

        self.assertEqual(pq.read_table(self.path).num_rows, 6)

    def test_verify_reports_summary(self):
        self.write([make_prompt_rows(i) for i in range(4)], flush_every=2)
        summary = verify_parquet(self.path, expected_rows=24)
        self.assertEqual(summary["rows"], 24)
        self.assertEqual(summary["row_groups"], 2)
        self.assertGreater(summary["compression_ratio"], 0)

    def test_verify_catches_wrong_row_count(self):
        self.write([make_prompt_rows(0)])
        with self.assertRaisesRegex(ValueError, "expected 7"):
            verify_parquet(self.path, expected_rows=7)

    def test_verify_catches_corrupted_bytes(self):
        self.write([make_prompt_rows(i) for i in range(20)])
        data = bytearray(self.path.read_bytes())
        # Flip bytes inside the data pages, away from the header and footer
        middle = len(data) // 3
        for i in range(middle, middle + 16):
            data[i] ^= 0xFF
        self.path.write_bytes(bytes(data))

        with self.assertRaisesRegex(OSError, "CRC checksum"):
            verify_parquet(self.path, expected_rows=120)

    def test_invalid_flush_every(self):
        with self.assertRaises(ValueError):
            RoutingParquetWriter(self.path, flush_every=0)


if __name__ == "__main__":
    unittest.main()
