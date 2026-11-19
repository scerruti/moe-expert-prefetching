"""Classify the legacy suite without dropping tests during CI migration."""

from pathlib import Path

import pytest

LEGACY_UNIT = {
    "test_data_collection.py",
    "test_model_loader_guard.py",
    "test_verify_environment.py",
}
LEGACY_INTEGRATION = {
    "test_batch_variance.py",
    "test_collection_scope.py",
    "test_extraction_rows.py",
    "test_forward_pass.py",
    "test_full_pipeline.py",
    "test_layer_coverage.py",
    "test_parquet_storage.py",
    "test_pinned_versions.py",
    "test_pytorch_forward_hooks.py",
    "test_router_gate_quantization.py",
    "test_router_topk.py",
    "test_selected_probs.py",
    "test_sequence_ids.py",
    "test_tokenizer.py",
}
LEVELS = {"unit", "integration", "e2e"}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        real_data = getattr(item.cls, "__name__", None) == "RealDataTests"
        if real_data:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.network)
        levels = {mark.name for mark in item.iter_markers()} & LEVELS
        if not levels:
            name = Path(str(item.path)).name
            if name in LEGACY_UNIT:
                item.add_marker(pytest.mark.unit)
            elif name in LEGACY_INTEGRATION:
                item.add_marker(pytest.mark.integration)
        levels = {mark.name for mark in item.iter_markers()} & LEVELS
        if len(levels) != 1:
            raise pytest.UsageError(f"{item.nodeid}: declare exactly one test-level marker")
        if "qwen_tokenizer" in item.fixturenames or item.name in {
            "test_dataset_loading",
            "test_load_datasets_covers_every_split",
        }:
            item.add_marker(pytest.mark.network)
        if (
            Path(str(item.path)).name == "test_pytorch_forward_hooks.py"
            and "toy_model" in item.fixturenames
        ):
            item.add_marker(pytest.mark.network)
