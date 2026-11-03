#!/usr/bin/env python3
"""Refuse a tree whose tracked files GitHub would reject on push.

    python3 serverbench/scripts/check_git_blob_sizes.py

The `repository` job in `.github/workflows/cpu.yml` runs this.

## What GitHub actually enforces

A single blob over **100 MiB** is refused outright, and one over **50 MiB**
draws a warning. Neither is negotiable from our side and neither is visible
until a push fails -- by which time the blob is in the history and taking it
out means a rewrite, not a revert.

This checks the WORKING TREE's tracked files, which is what a push would carry
and what an author can still fix with one `git rm --cached`. It does not walk
history: a blob already committed is a different and larger problem, and a CI
job that failed for it every run afterwards would be noise rather than a gate.

Large artefacts belong outside the index: model weights, traces and run
evidence are regenerated or fetched per host.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: GitHub refuses a push carrying a blob larger than this.
HARD_LIMIT = 100 * 1024 * 1024
#: GitHub warns above this. Reported, not failed: a file this size is almost
#: always something that should have been generated or fetched instead.
WARN_LIMIT = 50 * 1024 * 1024


def tracked_files(root: Path) -> list[Path]:
    got = subprocess.run(["git", "ls-files", "-z"], cwd=str(root),
                         capture_output=True, text=True, check=False)
    if got.returncode != 0:
        print(f"not a git repository at {root}: {got.stderr.strip()}",
              file=sys.stderr)
        raise SystemExit(2)
    return [root / name for name in got.stdout.split("\0") if name]


def main(argv: list[str] | None = None) -> int:
    root = Path(subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True,
        text=True, check=False).stdout.strip() or ".")

    oversize: list[tuple[Path, int]] = []
    warn: list[tuple[Path, int]] = []
    checked = 0
    for path in tracked_files(root):
        try:
            size = path.stat().st_size
        except OSError:
            continue                     # deleted but still in the index
        checked += 1
        if size > HARD_LIMIT:
            oversize.append((path, size))
        elif size > WARN_LIMIT:
            warn.append((path, size))

    def mb(n: int) -> str:
        return f"{n / 1024 / 1024:.1f} MiB"

    for path, size in sorted(warn, key=lambda kv: -kv[1]):
        print(f"large: {path.relative_to(root)} is {mb(size)} "
              f"(GitHub warns above {mb(WARN_LIMIT)})")

    if oversize:
        print()
        for path, size in sorted(oversize, key=lambda kv: -kv[1]):
            print(f"TOO LARGE: {path.relative_to(root)} is {mb(size)}",
                  file=sys.stderr)
        print(
            f"\n{len(oversize)} tracked file(s) exceed GitHub's "
            f"{mb(HARD_LIMIT)} hard limit. A push carrying them is refused, "
            f"and once committed the fix is a history rewrite rather than a "
            f"revert.\n\nUntrack them and regenerate or fetch them instead:\n"
            f"    git rm --cached <path>\n"
            f"    echo '<path>' >> .gitignore\n",
            file=sys.stderr)
        return 1

    print(f"{checked} tracked file(s); none over GitHub's {mb(HARD_LIMIT)} "
          f"limit"
          + (f", {len(warn)} above the {mb(WARN_LIMIT)} warning line" if warn
             else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
