#!/usr/bin/env python3
"""Gather every flagged turn from a folder of review reports into one checklist.

Usage: checklist.py EXPORT_FOLDER [--out CHECKLIST.md]
"""

import argparse
from pathlib import Path


def build(folder):
    out, total = [], 0
    for f in sorted(Path(folder).glob("*.REVIEW.md")):
        rows = f.read_text().splitlines()
        flagged = [r for r in rows if "check redaction" in r]
        redacted = [r for r in rows if "REDACTED" in r]
        if not flagged:
            continue
        out += [f"## {f.name.replace('.REVIEW.md', '')}" + (f"  (already redacted: {len(redacted)})" if redacted else ""), ""]
        for r in flagged:
            c = [x.strip() for x in r.strip().strip("|").split("|")]
            out.append(f"- [ ] Turn {c[0]} ({c[1]}), {c[3]}: {c[4]}")
            total += 1
        out.append("")
    head = ["# Review checklist", "",
            f"**{total} turns to check.** Each is kept unless you ask to redact it. "
            "Search the record for `## Turn N`.", ""]
    return "\n".join(head + out), total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    text, total = build(a.folder)
    out = Path(a.out) if a.out else Path(a.folder) / "CHECKLIST.md"
    out.write_text(text)
    print(f"{total} flagged turn(s) -> {out}")


if __name__ == "__main__":
    main()
