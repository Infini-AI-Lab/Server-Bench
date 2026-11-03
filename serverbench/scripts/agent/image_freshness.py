#!/usr/bin/env python3
"""Is the agent image built from the Dockerfiles in this checkout?

    image_freshness.py digest              the digest a build must stamp
    image_freshness.py check --image TAG   refuse if TAG was built from others

## Why this exists

A Dockerfile fix does nothing until the image is rebuilt. A pin test compares
the Dockerfile with the requirements file, but no test reads the image, so a
stale image (an old transformers that cannot load the task's model, a missing
tokenizer dependency) would break the starter and the brief's own gates.

## The rule

The build stamps `LABEL serverbench.image_inputs_sha256=<digest>`. The docker
launch path recomputes the digest from this checkout and refuses an image whose
label differs or is absent. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

LABEL = "serverbench.image_inputs_sha256"

#: Everything the agent image is built FROM, in build order. Adding a file to
#: the build means adding it here; the digest is only as honest as this list.
IMAGE_INPUTS = (
    "serverbench/environments/serving-agent/Dockerfile",
    "serverbench/environments/evolve-agent/Dockerfile",
)

REBUILD = "bash serverbench/scripts/deploy/03-build-images.sh"


def digest(repo: Path = REPO) -> str:
    """sha256 over each input's path and bytes, in `IMAGE_INPUTS` order."""
    h = hashlib.sha256()
    for relative in IMAGE_INPUTS:
        h.update(relative.encode("utf-8") + b"\0")
        h.update((repo / relative).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def image_label(image: str) -> str | None:
    """The label on a local image, "" if unlabelled, None if no such image."""
    got = subprocess.run(
        ["docker", "image", "inspect", "-f",
         "{{index .Config.Labels \"" + LABEL + "\"}}", image],
        capture_output=True, text=True, timeout=60, check=False)
    if got.returncode != 0:
        return None
    value = (got.stdout or "").strip()
    return "" if value in ("", "<no value>") else value


def problem(image: str, repo: Path = REPO) -> str:
    """Why `image` must not be used, or "" when it is current."""
    want = digest(repo)
    have = image_label(image)
    if have is None:
        return f"{image} does not exist locally. Build it: {REBUILD}"
    if have == "":
        return (f"{image} carries no {LABEL} label, so nothing says which "
                f"Dockerfiles built it, so it may be stale. Rebuild: {REBUILD}")
    if have != want:
        return (f"{image} was built from different Dockerfiles than this "
                f"checkout's (label {have[:12]}, checkout {want[:12]}). A "
                f"Dockerfile change reaches a run only through a rebuild. "
                f"Rebuild: {REBUILD}")
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("digest")
    check = sub.add_parser("check")
    check.add_argument("--image", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "digest":
        print(digest())
        return 0
    why = problem(args.image)
    if why:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    print(f"{args.image} is current ({digest()[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
