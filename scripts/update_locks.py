"""Generate Linux/Python 3.13 locks, or reject stale inputs without resolving."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROFILES = ("ci-tools", "cpu", "gpu")
INPUTS = ("requirements.txt", "scripts/update_locks.py") + tuple(
    f"requirements/{profile}.{suffix}" for profile in PROFILES for suffix in ("in", "lock")
)


def fingerprints(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in INPUTS}


def check(root):
    recorded = json.loads((root / "requirements/lock-inputs.json").read_text())
    if recorded["sha256"] != fingerprints(root):
        raise ValueError(
            "Dependency inputs or locks changed; regenerate with scripts/update_locks.py"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.check:
            check(root)
            return 0
        for profile in PROFILES:
            command = [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "compile",
                f"requirements/{profile}.in",
                "--python-version",
                "3.13",
                "--python-platform",
                "x86_64-unknown-linux-gnu",
                "--generate-hashes",
                "--no-build",
                "--quiet",
                "--output-file",
                f"requirements/{profile}.lock",
            ]
            if profile == "cpu":
                command.extend(["--torch-backend", "cpu"])
            subprocess.run(command, cwd=root, check=True, timeout=300)
        data = {
            "python": "3.13",
            "platform": "x86_64-unknown-linux-gnu",
            "sha256": fingerprints(root),
        }
        (root / "requirements/lock-inputs.json").write_text(json.dumps(data, indent=2) + "\n")
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Lock validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
