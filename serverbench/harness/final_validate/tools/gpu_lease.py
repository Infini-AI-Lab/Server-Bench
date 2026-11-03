#!/usr/bin/env python3
"""One device-ownership mechanism, holding BOTH locks, from the command line.

A host has two GPU-lock mechanisms that cannot see each other: the card claim
(`/tmp/serverbench-card-claims/gpu<N>/owner`) and the `with-gpu` flock. This
tool goes through `final_validate.lease`, which takes both, so a lease is
visible to callers of either:

* a claim and a flock are taken together, so a job checking only one still
  sees the card as held;
* removing a claim directory does not release the flock, so the kernel keeps
  defending the card;
* no claim is reaped while its flock is held: the kernel's answer beats the
  owner file's.

Bounded, always: an indefinite waiter for a multi-device window the host
cannot supply holds nothing while every single-device task behind it waits.

    gpu_lease.py --devices 1 --seconds 0        # what is in the way, take nothing
    gpu_lease.py --devices 2 --seconds 600      # take two, or give up in 10 min
    gpu_lease.py --status                       # who holds what, by both mechanisms
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
# `serverbench/harness/` on the path: the package root `final_validate` lives in.
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

from final_validate import lease as lease_mod  # noqa: E402


def status() -> dict:
    """Who holds what, by BOTH mechanisms, side by side."""
    out: dict = {"devices": {}, "note": (
        "a card is available only when nothing holds it by EITHER mechanism and "
        "no foreign compute process is on it. Zero utilisation is not "
        "availability")}
    for device in lease_mod._candidates():
        claim = lease_mod.CLAIMS / f"gpu{device}"
        owner = ""
        if (claim / "owner").is_file():
            owner = (claim / "owner").read_text("utf-8").strip()
        alive = None
        if owner.isdigit():
            try:
                os.kill(int(owner), 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
        out["devices"][f"gpu{device}"] = {
            "campaign_claim_owner": owner or None,
            "claim_owner_alive": alive,
            "claim_is_a_named_hold": bool(owner) and not owner.isdigit(),
            "with_gpu_flock_held": lease_mod._lock_is_held(device),
            "memory_used_mib": lease_mod._busy_mib(device),
            "foreign_processes": lease_mod._foreign_processes(device),
        }
    out["fenced_off"] = sorted(lease_mod._denied())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="acquisition deadline. 0 reports what is in the way "
                         "and takes nothing")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="seconds to hold the lease before releasing it")
    ap.add_argument("--deny", default="")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    if args.status:
        print(json.dumps(status(), indent=1))
        return 0

    deny = tuple(int(x) for x in args.deny.replace(" ", "").split(",")
                 if x.isdigit())
    try:
        got = lease_mod.acquire(args.devices, timeout_s=args.seconds, deny=deny)
    except lease_mod.NoDeviceAvailable as exc:
        print(str(exc))
        return 75            # EX_TEMPFAIL: not a failure of the work
    with got:
        print(json.dumps({"devices": list(got.devices), "uuids": got.uuids,
                          "device_spec": got.device_spec,
                          "owner_pid": got.owner_pid}, indent=1))
        if args.hold > 0:
            time.sleep(args.hold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
