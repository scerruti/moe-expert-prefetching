import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_gpu_runner_refuses_untracked_executable_input(tmp_path):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Test fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, env=env, check=True)
    (tmp_path / "tracked.txt").write_text("fixture\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, env=env, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, env=env, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    (tmp_path / "sitecustomize.py").write_text("# Untracked executable input\n")
    script = Path(__file__).resolve().parents[2] / "scripts/run_gpu_tests.sh"
    result = subprocess.run(
        ["bash", str(script), sha], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0
    assert "modified checkout" in result.stderr
