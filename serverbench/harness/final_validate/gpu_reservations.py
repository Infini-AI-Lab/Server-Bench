#!/usr/bin/env python3
"""Find live Docker containers that reserve a GPU, even while CUDA is idle.

`nvidia-smi` reports compute processes and resident memory.  A live container
can own a device while reporting neither: an agent may stop its server between
experiments and start it again minutes later.  Claims are the primary ownership
record, but jobs from another checkout can be missing one.  Docker's live
`DeviceRequests` is therefore conservatively treated as a host-wide reservation
signal, not a Docker-enforced exclusive lock. An operator may explicitly exempt
exact full container IDs through `SERVERBENCH_IGNORE_GPU_CONTAINER_IDS`; actual
GPU usage and lease checks are independent and must still pass.

The scan fails closed.  A card is not free when Docker cannot say whether a
running container owns it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _matches(request: dict[str, Any], index: int, uuid: str) -> bool:
    capabilities = request.get("Capabilities") or []
    is_gpu = (request.get("Driver") == "nvidia"
              or any("gpu" in group for group in capabilities))
    if not is_gpu:
        return False
    identifiers = {str(value) for value in request.get("DeviceIDs") or []}
    if str(index) in identifiers or (uuid and uuid in identifiers):
        return True
    # `--gpus all`, or a count without resolved IDs, gives the container a
    # daemon-selected card.  The host config does not say which one, so no card
    # can be proved free from that reservation.
    return not identifiers and int(request.get("Count") or 0) != 0


def reservations_for(index: int, uuid: str = "") -> tuple[str, ...]:
    """Names/IDs of running containers reserving ``index``.

    A diagnostic beginning with ``scan unavailable`` is returned on an
    unreadable Docker state.  Callers treat any returned value as held.
    Containers are inspected one at a time with a bound: a disappearing or
    wedged transient container cannot hang GPU scheduling indefinitely.
    """
    listed = _run(["docker", "ps", "-q"], 5.0)
    if listed is None or listed.returncode != 0:
        return ("scan unavailable: docker ps failed",)
    # Operator-authorized, process-local exception for a specific container.
    # Full IDs only: names/prefixes could accidentally exempt a replacement.
    # This bypasses Docker visibility, never compute/quiet/claim/lease checks.
    ignored_ids = {
        value.strip() for value in os.environ.get(
            "SERVERBENCH_IGNORE_GPU_CONTAINER_IDS", "").split(",")
        if len(value.strip()) == 64
        and all(c in "0123456789abcdef" for c in value.strip())
    }
    held = []
    for container_id in listed.stdout.split():
        inspected = _run(["docker", "inspect", container_id], 2.0)
        if inspected is None:
            return (f"scan unavailable: docker inspect {container_id} timed out",)
        if inspected.returncode != 0:
            # The container can disappear between `ps` and `inspect`.
            continue
        try:
            records = json.loads(inspected.stdout)
            record = records[0]
        except (IndexError, TypeError, ValueError):
            return (f"scan unavailable: malformed inspect for {container_id}",)
        if record.get("Id") in ignored_ids:
            continue
        requests = (record.get("HostConfig") or {}).get("DeviceRequests") or []
        if any(_matches(request, index, uuid) for request in requests):
            held.append(str(record.get("Name") or container_id).lstrip("/"))
    return tuple(held)


def _uuid(index: int) -> str:
    got = _run(["nvidia-smi", f"--id={index}", "--query-gpu=uuid",
                "--format=csv,noheader"], 5.0)
    return got.stdout.strip() if got and got.returncode == 0 else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, type=int)
    args = parser.parse_args(argv)
    held = reservations_for(args.device, _uuid(args.device))
    if held:
        print("; ".join(held))
        return 0                 # matches shell `gpu_is_held`: zero means held
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
