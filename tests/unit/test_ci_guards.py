import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_tests
from scripts.audit_dependencies import audit_input

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "xml",
    [
        "<testsuites/>",
        "<testsuites><testcase><skipped/></testcase></testsuites>",
        '<!DOCTYPE testsuites [<!ENTITY injected "expanded">]><testsuites><testcase name="&injected;"/></testsuites>',
    ],
)
def test_test_runner_rejects_empty_or_skipped_suites(tmp_path, monkeypatch, xml):
    monkeypatch.setattr(run_tests, "__file__", str(tmp_path / "scripts/run_tests.py"))
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "unit"])

    def pytest_result(*args, **kwargs):
        (tmp_path / "reports/unit.xml").write_text(xml)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_tests.subprocess, "run", pytest_result)
    assert run_tests.main() == 1


def test_test_runner_preserves_pytest_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tests, "__file__", str(tmp_path / "scripts/run_tests.py"))
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "unit"])
    monkeypatch.setattr(
        run_tests.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=5)
    )
    assert run_tests.main() == 5


def test_test_runner_rejects_missing_coverage_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tests, "__file__", str(tmp_path / "scripts/run_tests.py"))
    monkeypatch.setattr(sys, "argv", ["run_tests.py", "unit"])

    def pytest_result(*args, **kwargs):
        (tmp_path / "reports/unit.xml").write_text("<testsuites><testcase/></testsuites>")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_tests.subprocess, "run", pytest_result)
    assert run_tests.main() == 1


def test_cpu_wheel_advisory_mapping_is_explicit(tmp_path):
    lock = tmp_path / "cpu.lock"
    lock.write_text("torch==2.15.0+cpu --hash=sha256:" + "0" * 64 + "\n")
    packages, aliases = audit_input(lock)
    assert packages == {"torch": "2.15.0"}
    assert aliases == [{"package": "torch", "installed": "2.15.0+cpu", "advisory": "2.15.0"}]


@pytest.mark.parametrize(
    "requirement", ["torch>=2", "torch==2.15.0", "example==1.0+private --hash=sha256:" + "0" * 64]
)
def test_audit_rejects_unfrozen_or_unreviewed_dependencies(tmp_path, requirement):
    lock = tmp_path / "invalid.lock"
    lock.write_text(requirement + "\n")
    with pytest.raises(ValueError):
        audit_input(lock)


def test_ordered_ci_cannot_succeed_after_a_failed_predecessor():
    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/cpu-tests.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request_target" not in workflow["on"]
    order = ["formatting", "lint", "security", "unit", "integration", "e2e"]
    for previous, stage in zip(order, order[1:]):
        assert workflow["jobs"][stage]["needs"] == previous
    gate = workflow["jobs"]["cpu-tests"]
    assert gate["needs"] == order
    assert gate["if"] == "always()"
    # The gate rejects skipped/cancelled results as well as explicit failures.
    assert "!= 'success'" in gate["steps"][0]["run"]
