#!/usr/bin/env python3
"""Hold a visible reservation on an IDLE GPU, and be provably absent otherwise.

    gpu_holder.py --device 4 --mib 256

WHY THIS EXISTS
---------------
A card a run owns reads 0 MiB for much of that run: the agent writes code, and
between every teardown and restart of its server the card is idle again. The
mechanisms that say the card is taken are invisible to another user:

  * docker `device_ids` controls what OUR CONTAINER sees and does not stop a
    host process from using the same physical card;
  * `/tmp/serverbench-card-claims/gpu4` is read by ServerBench tooling only;
  * `/tmp/serverbench-gpu-locks/gpu4.lock` likewise.

So to everyone else on a shared host an owned, idle card is indistinguishable
from a free one, and a neighbour can pick it in the window where it looks
empty. A small allocation makes the reservation legible in the one place
everybody looks, `nvidia-smi`.

## THE COST IT MUST NEVER HAVE

A holder resident on a card under measurement does two kinds of harm:

  * **a second CUDA context** (hundreds of MiB), which yielding the block does
    NOT free -- a context lives as long as the process that made it;
  * **a moving free-memory figure**, if it re-takes its block whenever the card
    dips: the agent sizes its KV cache against free memory that moves under
    it, and the final verdict runs with no holder at all, so the agent tunes
    against one machine and is judged on another.

The rule that removes both, and the reason it is a COUNT and not a threshold:

    anything else at all on this card  ->  release, and EXIT

Not "anything big". A memory threshold asks *is somebody using a lot of this
card*, and a profiling loop answers *sometimes* -- a 900 MiB microbenchmark
slips under a 1024 MiB bar and the holder stays resident right through the
measurement. The count has no window to slip through.

## BUT THE RESERVATION STILL HAS TO COME BACK

Exiting for good would reopen the window above: the agent's server goes down
between rounds, the card reads empty, a neighbour takes it.

So the holder is a two-phase loop, and **phase A has not imported torch**:

    A. WATCH   no CUDA context, no allocation, ~0 MiB on the card.
               Poll until the card has been CONTINUOUSLY EMPTY for
               --empty-for seconds.
    B. HOLD    import torch, allocate --mib, and stay only while the card is
               otherwise empty. The instant anything appears: free the block,
               then `os.execv` this process back to phase A.

`execv` rather than a `return` to the top of the loop, because that is what
frees the CUDA context -- the process image is replaced, the context dies with
it, and **the PID does not change**. The PID matters: the driver records this
holder with `took_process(pid)`, `--parent-pid` is checked against it, and the
journal's `card_held` names it. A holder that changed PID to shed its context would leak
out of every one of those.

`--empty-for` is what distinguishes the two kinds of idle. The gaps inside a
profiling loop are seconds; an agent writing code leaves the card empty for
much longer. At the default 120 s the holder never returns mid-benchmark and
always returns during the long stretches the reservation is for.

So, stated as a guarantee: **whenever anything else is on this card, nothing of
this holder is -- not the block, not the context.** And whenever the card is
genuinely, durably empty, `nvidia-smi` shows it taken.

IT MUST NOT OUTLIVE ITS RUN
---------------------------
Stale GPU memory with no owner is litter, and a holder that survives its run is
litter by construction. Two independent guarantees:

  * whatever started it (the evolve driver, which records it with
    `took_process`, or a GPU lease) releases it, before the final verdict for
    a run, so the card is clear when the verdict asks for it;
  * `--parent-pid` makes it exit on its own when its parent dies, so a crashed
    or SIGKILLed run cannot leave the holder behind. It is checked in BOTH
    phases, so a holder waiting to re-arm dies with its parent too.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

#: Set by the process on its way through `execv`, so a re-armed holder can say
#: in its log that it is a continuation rather than looking like a fresh start.
REARM_ENV = "SERVERBENCH_HOLDER_REARMS"


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, required=True,
                    help="HOST device index")
    ap.add_argument("--mib", type=int, default=256,
                    help="how much to hold, over and above the CUDA context")
    ap.add_argument("--parent-pid", type=int, default=0,
                    help="exit when this process is gone, so a crashed run "
                         "cannot leave the holder behind")
    ap.add_argument("--empty-for", type=float, default=120.0,
                    help="seconds the card must be CONTINUOUSLY empty before "
                         "the reservation is (re-)taken. Longer than any gap "
                         "inside a benchmark, far shorter than the stretches "
                         "the reservation exists for")
    ap.add_argument("--poll", type=float, default=10.0)
    return ap.parse_args(argv)


def others_on(device: int, mine: int) -> tuple[int, int]:
    """`(process count, MiB)` on this card that is NOT this process's.

    The COUNT is the field that decides anything. See the module docstring on
    why a memory threshold has a window a profiling loop slips through.
    """
    got = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits", "-i", str(device)],
        capture_output=True, text=True, timeout=60, check=False)
    count = total = 0
    for line in (got.stdout or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        if parts[0] == str(mine):
            continue
        count += 1
        total += int(parts[1])
    return count, total


def parent_gone(parent_pid: int) -> bool:
    if not parent_pid:
        return False
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False                    # alive, just not ours to signal
    return False


def rearm(args: argparse.Namespace) -> None:
    """Replace this process with a fresh copy of itself, same PID.

    The ONLY way to give back a CUDA context is to stop being the process that
    holds it. `execv` does that without changing the PID, which every other
    mechanism that tracks this holder depends on -- `took_process`,
    `--parent-pid`, and the run journal's `card_held`.
    """
    rearms = int(os.environ.get(REARM_ENV, "0")) + 1
    env_argv = [sys.executable, os.path.abspath(__file__),
                "--device", str(args.device), "--mib", str(args.mib),
                "--parent-pid", str(args.parent_pid),
                "--empty-for", str(args.empty_for),
                "--poll", str(args.poll)]
    os.environ[REARM_ENV] = str(rearms)
    sys.stdout.flush()
    os.execv(sys.executable, env_argv)          # noqa: S606 - our own file


def wait_until_durably_empty(args: argparse.Namespace, mine: int) -> bool:
    """Phase A. No torch, no context, nothing of ours on the card.

    Returns True when the card has been continuously empty for `--empty-for`,
    False if the run went away while waiting.
    """
    empty_since = None
    while True:
        if parent_gone(args.parent_pid):
            print(f"parent {args.parent_pid} is gone; not taking gpu"
                  f"{args.device}", flush=True)
            return False
        count, used = others_on(args.device, mine)
        now = time.monotonic()
        if count:
            if empty_since is not None:
                print(f"gpu{args.device} busy again ({count} process(es), "
                      f"{used} MiB); reservation stays off", flush=True)
            empty_since = None
        else:
            if empty_since is None:
                empty_since = now
                print(f"gpu{args.device} looks empty; waiting "
                      f"{args.empty_for:.0f}s to be sure it is not a gap "
                      f"between benchmark iterations", flush=True)
            elif now - empty_since >= args.empty_for:
                return True
        time.sleep(args.poll)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    mine = os.getpid()
    rearms = int(os.environ.get(REARM_ENV, "0"))
    print(f"holder up on host gpu{args.device} (pid {mine}"
          f"{f', re-arm #{rearms}' if rearms else ''}): holds {args.mib} MiB "
          f"only while the card is otherwise EMPTY, and releases its block and "
          f"its CUDA context the moment anything else appears", flush=True)

    running = True

    def stop(signum, frame):                                   # noqa: ARG001
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    # PHASE A -- no torch has been imported, so this process has no CUDA
    # context and contributes nothing to the card it is watching.
    if not wait_until_durably_empty(args, mine):
        return 0
    if not running:
        return 0

    # PHASE B -- take the reservation.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        import torch
    except ImportError:
        print("torch is not importable here, so no reservation was taken. "
              "The card is still claimed in the claims directory; it simply "
              "will not look busy to anyone reading nvidia-smi.",
              file=sys.stderr)
        return 3
    if not torch.cuda.is_available():
        print(f"device {args.device} is not visible to torch; nothing held",
              file=sys.stderr)
        return 4

    block = None

    def take() -> None:
        nonlocal block
        if block is None:
            block = torch.empty(args.mib * 1024 * 1024 // 2,
                                dtype=torch.float16, device="cuda:0")
            print(f"holding {args.mib} MiB on host gpu{args.device} "
                  f"(card was empty)", flush=True)

    def release(why: str) -> None:
        nonlocal block
        if block is not None:
            block = None
            torch.cuda.empty_cache()
            print(f"released gpu{args.device}: {why}", flush=True)

    take()
    while running:
        count, used = others_on(args.device, mine)
        if count:
            # ANYTHING ELSE ON THIS CARD ENDS THE RESERVATION, and ending it
            # means giving back the context too -- which is what `rearm` is
            # for. Whatever just appeared is a more visible reservation than
            # 256 MiB ever was, and it may be measuring.
            release(f"{count} other process(es), {used} MiB")
            print(f"gpu{args.device} is no longer empty: dropping the CUDA "
                  f"context as well and re-arming, so nothing of ours is "
                  f"resident while somebody works on this card", flush=True)
            rearm(args)                       # does not return
        if parent_gone(args.parent_pid):
            print(f"parent {args.parent_pid} is gone; releasing", flush=True)
            break
        time.sleep(args.poll)

    release("holder is shutting down")
    print(f"holder down on gpu{args.device}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
