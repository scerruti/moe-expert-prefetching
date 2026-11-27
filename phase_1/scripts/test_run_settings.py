"""
Run settings that must be in place before torch is imported (#133).

TRITON_DEFAULT_FP_FUSION is read when Triton starts, and pytest has already
imported torch by the time a test runs, so most of these tests run the entry
point in a subprocess, as test_dependency_startup.py does (#129).
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FUSION = "TRITON_DEFAULT_FP_FUSION"
APPLIED = "MOE_RUN_SETTINGS_APPLIED"


def clean_env(**changes):
    """This environment without the run settings, plus the given changes."""
    env = {k: v for k, v in os.environ.items() if k not in (FUSION, APPLIED)}
    env.update(changes)
    return env


def run_python(code, **env_changes):
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=clean_env(PYTHONPATH=str(REPO_ROOT), **env_changes),
    )


APPLY_AND_REPORT = (
    "import json, os, run_settings\n"
    "returned = run_settings.apply_run_settings()\n"
    f"print(json.dumps({{'returned': returned, 'fusion': os.environ.get({FUSION!r}),"
    f" 'applied': os.environ.get({APPLIED!r})}}))\n"
)


def test_apply_sets_fusion_to_exactly_one():
    result = run_python(APPLY_AND_REPORT)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == {"returned": {FUSION: "1"}, "fusion": "1", "applied": "1"}


def test_apply_accepts_the_required_value_already_set():
    result = run_python(APPLY_AND_REPORT, **{FUSION: "1"})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["fusion"] == "1"


@pytest.mark.parametrize("preset", ["0", "", "true"])
def test_apply_refuses_a_conflicting_value(preset):
    """Forced failure: an environment that already sets fusion differently stops the run."""
    result = run_python(APPLY_AND_REPORT + "print('continued')\n", **{FUSION: preset})
    assert result.returncode != 0
    assert "continued" not in result.stdout
    assert "RunSettingsError" in result.stderr
    assert f"{FUSION} is set to {preset!r}" in result.stderr
    assert "requires '1'" in result.stderr


def test_apply_refuses_after_torch_is_imported():
    """Forced failure: Triton may already have read its defaults, so it's too late."""
    result = run_python("import torch\n" + APPLY_AND_REPORT + "print('continued')\n")
    assert result.returncode != 0
    assert "continued" not in result.stdout
    assert "RunSettingsError" in result.stderr
    assert "torch was imported before the run settings were applied" in result.stderr


def test_apply_again_after_torch_is_harmless():
    """A second call (a re-run notebook cell, an inherited environment) doesn't refuse."""
    code = (
        "import os, run_settings\n"
        "run_settings.apply_run_settings()\n"
        "import torch\n"
        "run_settings.apply_run_settings()\n"
        f"print(os.environ[{FUSION!r}])\n"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


def test_require_refuses_when_settings_were_not_applied(monkeypatch):
    """Forced failure: code that runs the model refuses unless the entry point applied the settings."""
    import run_settings

    monkeypatch.delenv(APPLIED, raising=False)
    with pytest.raises(run_settings.RunSettingsError, match="apply_run_settings"):
        run_settings.require_run_settings()


def test_require_refuses_a_value_changed_after_applying(monkeypatch):
    import run_settings

    monkeypatch.setenv(FUSION, "0")
    with pytest.raises(run_settings.RunSettingsError, match=f"{FUSION} is set to '0'"):
        run_settings.require_run_settings()


def test_this_pytest_session_applied_the_settings_before_torch():
    """The root conftest.py applied them; apply refuses after torch, so the marker proves the order."""
    assert os.environ.get(FUSION) == "1"
    assert os.environ.get(APPLIED) == "1"


def run_pytest_probe(tmp_path, **env_changes):
    """Run pytest on one test file in a copy of the root setup, as test_dependency_startup.py does."""
    for name in ("conftest.py", "dependency_versions.py", "run_settings.py", "pytest.ini"):
        shutil.copyfile(REPO_ROOT / name, tmp_path / name)
    (tmp_path / "requirements.txt").write_text("moe-pin-probe==1.0\n")
    distribution = tmp_path / "moe_pin_probe-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Metadata-Version: 2.1\nName: moe-pin-probe\nVersion: 1.0\n")
    marker = tmp_path / "collected.txt"
    test = tmp_path / "test_probe.py"
    test.write_text(
        "import os\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('collected')\n"
        f"def test_fusion_is_set():\n    assert os.environ[{FUSION!r}] == '1'\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test), "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env=clean_env(PYTHONPATH=str(tmp_path), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", **env_changes),
    )
    return result, marker


def test_pytest_sets_fusion_for_the_tests_it_runs(tmp_path):
    result, marker = run_pytest_probe(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_text() == "collected"


def test_pytest_stops_before_collection_on_a_conflicting_value(tmp_path):
    """Forced failure: pytest launched with fusion set to 0 stops before running any test."""
    result, marker = run_pytest_probe(tmp_path, **{FUSION: "0"})
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert f"{FUSION} is set to '0'" in output
    assert not marker.exists()


def test_collection_refuses_before_loading_or_writing(tmp_path, monkeypatch):
    """Forced failure: FullPipeline won't collect unless the entry point applied the settings."""
    import run_settings
    from full_pipeline import FullPipeline

    monkeypatch.delenv(APPLIED, raising=False)

    def must_not_load(self):
        raise AssertionError("the model was loaded before the run settings were checked")

    monkeypatch.setattr(FullPipeline, "load_model", must_not_load)
    output_dir = tmp_path / "collection"
    with pytest.raises(run_settings.RunSettingsError):
        FullPipeline(output_dir=output_dir).run_collection()
    assert not output_dir.exists()
