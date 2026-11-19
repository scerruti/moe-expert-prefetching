import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e
CLI = Path(__file__).resolve().parents[2] / "phase_1/scripts/verify_environment.py"


def test_environment_cli_accepts_installed_pins_and_creates_directories(tmp_path):
    result = subprocess.run(
        [sys.executable, str(CLI), "--create-dirs"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Environment is ready" in result.stdout
    for directory in ("data", "phase_1/scripts"):
        assert (tmp_path / directory / "README.md").is_file()


def test_environment_cli_rejects_missing_dependencies(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", str(CLI)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Missing or mismatched dependencies" in result.stdout
    assert "torch" in result.stdout and "transformers" in result.stdout
