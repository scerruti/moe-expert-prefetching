import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import (
    DEFAULT_CONFIG_PATH,
    check_tokenizer_roundtrip,
    describe_architecture,
    load_model_config,
    verify_moe_architecture,
)


class DescribeArchitectureTests(unittest.TestCase):
    def test_description_is_built_from_meta_not_hardcoded(self):
        # Regression test: an earlier version of model_loading.py printed a
        # literal "8 experts, top-2 routing, 32 layers" string regardless of
        # which model was actually loaded. This must reflect meta instead.
        meta = {"num_experts": 64, "top_k": 6, "num_layers": 27}
        self.assertEqual(describe_architecture(meta), "64 experts, top-6 routing, 27 layers")


class VerifyMoeArchitectureTests(unittest.TestCase):
    def test_matching_architecture_returns_no_mismatches(self):
        meta = {"num_layers": 32, "num_experts": 8, "top_k": 2}
        expected = {"num_layers": 32, "num_experts": 8, "top_k": 2}
        self.assertEqual(verify_moe_architecture(meta, expected), [])

    def test_mismatched_expert_count_is_reported(self):
        meta = {"num_layers": 32, "num_experts": 16, "top_k": 2}
        expected = {"num_layers": 32, "num_experts": 8, "top_k": 2}
        mismatches = verify_moe_architecture(meta, expected)
        self.assertEqual(len(mismatches), 1)
        self.assertIn("num_experts", mismatches[0])

    def test_unset_expected_field_is_skipped_not_flagged(self):
        # research_model.num_layers is null in model_config.json until verified;
        # a null expected value must not produce a false-positive mismatch.
        meta = {"num_layers": 27, "num_experts": 64, "top_k": 6}
        expected = {"num_layers": None, "num_experts": 64, "top_k": 6}
        self.assertEqual(verify_moe_architecture(meta, expected), [])


class TokenizerRoundtripTests(unittest.TestCase):
    def test_lossless_roundtrip_returns_true(self):
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = [1, 2, 3]
        fake_tokenizer.decode.return_value = "hello world"
        self.assertTrue(check_tokenizer_roundtrip(fake_tokenizer, "hello world"))

    def test_lossy_roundtrip_returns_false(self):
        fake_tokenizer = MagicMock()
        fake_tokenizer.encode.return_value = [1, 2, 3]
        fake_tokenizer.decode.return_value = "hello wrld"
        self.assertFalse(check_tokenizer_roundtrip(fake_tokenizer, "hello world"))


class LoadModelConfigTests(unittest.TestCase):
    def test_config_file_has_required_mvp_fields(self):
        config = load_model_config(DEFAULT_CONFIG_PATH)
        mvp_config = config["mvp_model"]
        for field in ("model_id", "num_layers", "num_experts", "top_k", "router_module_path"):
            self.assertIn(field, mvp_config)

    def test_mvp_config_matches_locked_mixtral_architecture(self):
        config = load_model_config(DEFAULT_CONFIG_PATH)
        mvp_config = config["mvp_model"]
        self.assertEqual(mvp_config["num_layers"], 32)
        self.assertEqual(mvp_config["num_experts"], 8)
        self.assertEqual(mvp_config["top_k"], 2)


if __name__ == "__main__":
    unittest.main()
