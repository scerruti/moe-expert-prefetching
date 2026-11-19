"""Audit complete locks without resolving or installing application packages."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version
from pip_requirements_parser import RequirementsFile


def audit_input(lock):
    packages, aliases = {}, []
    for entry in RequirementsFile.from_file(str(lock)).requirements:
        req = entry.req
        pins = list(req.specifier)
        if req.url or len(pins) != 1 or pins[0].operator != "==" or not entry.hash_options:
            raise ValueError(f"{lock}: each package must have an exact version and hashes")
        name = canonicalize_name(req.name)
        if name in packages:
            raise ValueError(f"Duplicate locked package: {name}")
        version = Version(pins[0].version)
        if version.local:
            if name not in {"torch", "torchvision"} or version.local != "cpu":
                raise ValueError(f"No reviewed advisory identity mapping for {name}=={version}")
            aliases.append({"package": name, "installed": str(version), "advisory": version.public})
        packages[name] = version.public
    if not packages:
        raise ValueError("Cannot audit an empty lock")
    return packages, aliases


def audit(lock, output):
    packages, aliases = audit_input(lock)
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / f"{lock.stem}.audit-requirements.txt"
    frozen.write_text(
        "\n".join(f"{name}=={version}" for name, version in sorted(packages.items())) + "\n"
    )
    (output / f"{lock.stem}.identity-map.json").write_text(json.dumps(aliases, indent=2))
    report = output / f"{lock.stem}.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--strict",
            "--no-deps",
            "--disable-pip",
            "--progress-spinner",
            "off",
            "--format",
            "json",
            "--output",
            str(report),
            "-r",
            str(frozen),
        ],
        check=False,
        timeout=300,
    )
    if not report.exists():
        return 1
    dependencies = json.loads(report.read_text())["dependencies"]
    skipped = [entry for entry in dependencies if entry.get("skip_reason")]
    reported = {canonicalize_name(entry["name"]) for entry in dependencies}
    if skipped or reported != set(packages):
        print(f"{lock}: incomplete advisory coverage", file=sys.stderr)
        return 1
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("locks", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/audit"))
    args = parser.parse_args()
    failed = False
    for lock in args.locks:
        try:
            failed |= audit(lock, args.output) != 0
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            print(f"Audit failed for {lock}: {exc}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
