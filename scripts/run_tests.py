"""Keep pytest's exit status and reject empty or entirely skipped CI suites."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=["unit", "integration", "e2e", "gpu"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    report = reports / f"{args.level}.xml"
    selector = "gpu" if args.level == "gpu" else f"{args.level} and not gpu"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            selector,
            "--junitxml",
            str(report),
            "-v",
            "--tb=short",
            "--cov=phase_1/scripts",
            "--cov-branch",
            f"--cov-report=xml:{reports / (args.level + '-coverage.xml')}",
            "--cov-report=term",
        ],
        cwd=root,
        check=False,
        env={**os.environ, "COVERAGE_FILE": str(reports / f".coverage-{args.level}")},
    )
    if result.returncode:
        return result.returncode
    try:
        cases = list(ET.parse(report).iter("testcase"))
    except (OSError, ET.ParseError, DefusedXmlException) as exc:
        print(f"Invalid test report: {exc}", file=sys.stderr)
        return 1
    if not cases or any(case.find("skipped") is not None for case in cases):
        print("Required suite is empty or contains skipped tests", file=sys.stderr)
        return 1
    try:
        coverage = ET.parse(reports / f"{args.level}-coverage.xml")
        if coverage.getroot().tag != "coverage":
            raise ValueError("Unexpected coverage report format")
    except (OSError, ValueError, ET.ParseError, DefusedXmlException) as exc:
        print(f"Missing or invalid coverage evidence: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
