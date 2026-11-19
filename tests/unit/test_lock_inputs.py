import json
from pathlib import Path

import pytest

from scripts.update_locks import INPUTS, check, fingerprints

pytestmark = pytest.mark.unit


def test_lock_check_detects_changed_inputs_or_lock_content(tmp_path):
    for name in INPUTS:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture")
    metadata = tmp_path / "requirements/lock-inputs.json"
    metadata.write_text(json.dumps({"sha256": fingerprints(tmp_path)}))
    check(tmp_path)
    (tmp_path / "requirements.txt").write_text("changed dependency")
    with pytest.raises(ValueError, match="regenerate"):
        check(tmp_path)


def test_cpu_lock_preserves_workflow_index_selection():
    root = Path(__file__).resolve().parents[2]
    lock = (root / "requirements/cpu.lock").read_text()
    assert "\n--index-url " not in lock, "A lock index resets pip's CLI extra indexes"
