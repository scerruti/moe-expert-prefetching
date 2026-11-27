"""Run real pytest subprocesses so selecting one file cannot bypass the guard."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_probe(tmp_path, requirement, installed=None, *, nested=False):
    for name in ("conftest.py", "dependency_versions.py", "run_settings.py", "pytest.ini"):
        source = REPO_ROOT / name
        if source.exists():
            shutil.copyfile(source, tmp_path / name)
    (tmp_path / "requirements.txt").write_text(requirement)
    if installed is not None:
        distribution = tmp_path / "moe_pin_probe-1.0.dist-info"
        distribution.mkdir()
        (distribution / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: moe-pin-probe\nVersion: {installed}\n"
        )
    directory = tmp_path / "phase_1" / "scripts" if nested else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    if nested:
        shutil.copyfile(
            REPO_ROOT / "phase_1/scripts/conftest.py", directory / "conftest.py"
        )
        # If fixture setup imports scientific packages early, it hides the pin diagnostic.
        (directory / "torch.py").write_text(
            "raise RuntimeError('torch imported before pin guard')\n"
        )
        (directory / "model_loader.py").write_text(
            "raise RuntimeError('loader imported before pin guard')\n"
        )
    marker = tmp_path / "collected.txt"
    test = directory / "test_probe.py"
    test.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('collected')\n"
        "def test_selected_file():\n    assert True\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test), "-q"],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
    )
    return result, marker


@pytest.mark.parametrize("installed", [None, "2.0"])
def test_selected_file_stops_before_collection_for_missing_or_wrong_pin(
    tmp_path, installed
):
    result, marker = run_probe(tmp_path, "moe-pin-probe==1.0\n", installed)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, (
        result.stdout + result.stderr
    )
    assert "moe-pin-probe==1.0" in result.stdout + result.stderr
    expected = "not installed" if installed is None else "found 2.0"
    assert expected in result.stdout + result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("installed", ["1.0", "1.0+cpu", "1.0+cu126"])
def test_matching_public_version_allows_selected_file(tmp_path, installed):
    result, marker = run_probe(tmp_path, "moe-pin-probe==1.0\n", installed)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_text() == "collected"


def test_nested_target_checks_versions_before_scientific_imports(tmp_path):
    result, marker = run_probe(tmp_path, "moe-pin-probe==1.0\n", "2.0", nested=True)
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "moe-pin-probe==1.0" in output
    assert "imported before pin guard" not in output
    assert not marker.exists()


@pytest.mark.parametrize(
    "requirements", ["# no exact pins\n", "moe-pin-probe==1.0\nmoe-pin-probe==2.0\n"]
)
def test_invalid_requirements_fail_before_collection(tmp_path, requirements):
    result, marker = run_probe(tmp_path, requirements, "1.0")
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, (
        result.stdout + result.stderr
    )
    assert not marker.exists()
