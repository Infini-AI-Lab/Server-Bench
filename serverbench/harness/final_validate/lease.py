"""One device-ownership mechanism, holding BOTH existing locks. Bounded, and it cleans up.

THE PROBLEM THIS CLOSES
------------------------
Two independent lock mechanisms exist on a host and **neither can see the
other**:

* **card claims** -- a directory `/tmp/serverbench-card-claims/gpu<N>`
  holding an `owner` file, created atomically with `mkdir` by `take_card.sh`;
* **the `with-gpu` flock** -- `/tmp/serverbench-gpu-locks/gpu<N>.lock`, taken
  with `flock` by `scripts/deploy/with-gpu`.

A job holding one is invisible to a job checking the other, so two
measurements could share a card. The symptom is a plausible-looking number
roughly half as fast as the truth, which is far worse than a crash because it
gets published.

`acquire()` is the one ownership mechanism: it takes **both** locks for every
device, in ascending index order, and it is the only way the final validator
gets a card.

**Zero utilisation is not availability.** A card reads low for a moment while a
neighbour's engine is starting, and that is the window that steals a device
someone else is claiming. `--query-compute-apps` is consulted, not just
`memory.used`.

**A non-numeric claim owner is never reaped.** That is the `take_card.sh`
convention for a deliberate operator reservation (an `evolve-reserved` hold,
say, keeping this suite's schedulers off a card another user is running on).
A cleanup that "tidies" it would open exactly the window it closes.

**Acquisition is BOUNDED and failure is recorded, not waited out.** An
indefinite waiter for a multi-device request the host cannot satisfy holds
nothing, does no work, and starves every single-device task behind it.
`acquire()` raises `NoDeviceAvailable` at the deadline and the caller records
a blockage.

WHAT THIS DOES NOT DO
----------------------
It never signals a process it did not start, and it never removes a claim whose
owner is not this process. A foreign card is declined and named; it is not
waited out, and it is certainly not cleared.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from final_validate.gpu_reservations import reservations_for

CLAIMS = Path(os.environ.get("SERVERBENCH_CARD_CLAIMS",
                             "/tmp/serverbench-card-claims"))
LOCKS = Path(os.environ.get("GPU_LOCK_DIR", "/tmp/serverbench-gpu-locks"))

def _denied() -> frozenset[int]:
    """Cards the operator has fenced off for this host, from the environment.

    This exists because a claim is not always enough. A measurement stream that
    claims and releases a card PER ARM leaves the card unclaimed and idle
    between arms, and a claim-checking acquirer would take it and contend with
    work that is about to resume. A claim protects a card while
    it is held; only the operator knows which cards belong to a run that is
    still draining.

    `SERVERBENCH_LEASE_DENY=3,4,6` fences those off for every lease on this
    host. Deny is checked BEFORE availability, so a fenced card is never taken
    however free it looks.
    """
    raw = ",".join(
        filter(None, (
            os.environ.get("SERVERBENCH_EXCLUDED_GPUS", ""),
            os.environ.get("SERVERBENCH_LEASE_DENY", ""),
        ))
    )
    out = set()
    for piece in raw.replace(" ", "").split(","):
        if piece.isdigit():
            out.add(int(piece))
    return frozenset(out)

#: Above this, a card is treated as in use by work this lease does not cover.
BUSY_MIB = 500


def docker_device_spec(devices) -> str:
    """What `docker --gpus` wants for these cards, QUOTED.

    `--gpus` parses its value as CSV: a bare `device=0,1` is read as the fields
    `device=0` and `1`, and a bare number sets docker's Count -- `--gpus 7,8`
    hands the container the host's first two GPUs, with no visible error.
    Accepts a sequence of indices or a "7,8" string.
    """
    if isinstance(devices, str):
        devices = [d.strip() for d in devices.split(",") if d.strip()]
    names = [str(int(d)) for d in devices]
    if not names:
        raise ValueError("no devices named")
    return '"device=' + ",".join(names) + '"'


class NoDeviceAvailable(RuntimeError):
    """The requested devices could not be acquired inside the deadline.

    Carries what was in the way, so the caller records a blockage a person can
    act on rather than "no GPU".
    """


@dataclasses.dataclass
class Lease:
    """Devices this process owns, and the two locks proving it.

    Released exactly once, on `close()`, on context exit, and at interpreter
    exit. Releasing twice is safe; releasing something we did not take is
    impossible, because only descriptors and directories recorded here are
    touched.
    """

    devices: tuple[int, ...]
    uuids: dict[int, str]
    owner_pid: int
    _fds: dict[int, int] = dataclasses.field(default_factory=dict, repr=False)
    _claims: tuple[Path, ...] = ()
    _holders: tuple[subprocess.Popen, ...] = dataclasses.field(
        default_factory=tuple, repr=False)
    _closed: bool = False

    def __enter__(self) -> Lease:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def device_spec(self) -> str:
        return docker_device_spec(self.devices)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for process in self._holders:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(process.pid, signal.SIGTERM)
        for process in self._holders:
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                process.wait(timeout=10)
        for path in self._claims:
            # Only ever OUR claim, checked by the recorded owner. Removing
            # somebody else's is how a card gets handed to two jobs.
            try:
                if (path / "owner").read_text("utf-8").strip() == str(self.owner_pid):
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass
        for fd in self._fds.values():
            with contextlib.suppress(OSError):
                os.close(fd)          # closing the descriptor releases the flock
        self._fds.clear()


def _holder_interpreter() -> str:
    """Find a host interpreter that can make a lease visible in nvidia-smi."""
    named = os.environ.get("SERVERBENCH_HOLDER_PY", "").strip()
    candidates = [named] if named else []
    candidates += [str(Path.home() / "anaconda3" / "bin" / "python"),
                   "/usr/bin/python3", sys.executable]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not Path(candidate).is_file():
            continue
        seen.add(candidate)
        try:
            probe = subprocess.run(
                [candidate, "-c", "import torch,sys; "
                 "sys.exit(0 if torch.cuda.is_available() else 9)"],
                capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate
    return ""


def _visible_pids(device: int) -> set[int]:
    out = set()
    for line in _query("pid", "used_memory", "process_name", device=device):
        first = line.split(",", 1)[0].strip()
        if first.isdigit():
            out.add(int(first))
    return out


def _make_visible(lease: Lease, timeout_s: float = 30.0) -> None:
    """Bridge the lock-to-CUDA-start gap with a tiny balanced allocation.

    The claim and flock protect against ServerBench jobs. Shared-host schedulers
    commonly select cards from nvidia-smi and cannot see either one. A model
    load leaves minutes between acquiring the lease and allocating substantial
    memory, and another job entering that gap can make a tensor-parallel SGLang
    launch fail its memory-balance check. The same short-lived holder used by
    evolve closes the gap. It drops its CUDA context as soon as the launched
    server appears and exits with the leasing process.
    """
    interpreter = _holder_interpreter()
    script = Path(__file__).resolve().parents[2] / "scripts" / "gpu_holder.py"
    if not interpreter or not script.is_file():
        return
    holders = []
    try:
        for device in lease.devices:
            argv = [interpreter, str(script), "--device", str(device),
                    "--mib", "256", "--parent-pid", str(os.getpid()),
                    "--empty-for", "0", "--poll", "0.25"]
            holders.append(subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True))
        lease._holders = tuple(holders)
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in holders):
                break
            if all(process.pid in _visible_pids(device)
                   for process, device in zip(holders, lease.devices)):  # noqa: B905
                return
            time.sleep(0.25)
        raise NoDeviceAvailable(
            "leased GPU holder processes did not become visible within "
            f"{timeout_s:.0f}s; releasing the lease rather than leaving a "
            "model-load race on the shared host")
    except Exception:
        lease.close()
        raise


def _query(*fields: str, device: int | None = None) -> list[str]:
    argv = ["nvidia-smi", f"--query-{'compute-apps' if 'pid' in fields[0] else 'gpu'}"
            "=" + ",".join(fields), "--format=csv,noheader,nounits"]
    if device is not None:
        argv += ["-i", str(device)]
    try:
        got = subprocess.run(argv, capture_output=True, text=True, timeout=30,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in (got.stdout or "").splitlines() if line.strip()]


def _foreign_processes(device: int) -> list[str]:
    """Compute processes on this card that this process did not start.

    Checked in addition to `memory.used`, because a card reads low for a moment
    while a neighbour's engine is starting.
    """
    mine = {os.getpid()}
    out = []
    for line in _query("pid", "used_memory", "process_name", device=device):
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        if int(parts[0]) in mine:
            continue
        out.append(line)
    return out


def _busy_mib(device: int) -> int:
    rows = _query("memory.used", device=device)
    try:
        return int(float(rows[0]))
    except (IndexError, ValueError):
        return 1 << 30                    # unreadable is treated as busy


def _uuid(device: int) -> str:
    rows = _query("uuid", device=device)
    return rows[0] if rows else ""


def _lock_is_held(device: int) -> bool:
    """Whether ANYONE holds the `with-gpu` flock for this card.

    This is the load-bearing check. A claim's `owner` pid can be wrong -- a
    short-lived wrapper's `$$` rather than the long-lived driver -- and a
    reaper would then judge a live measurement dead and delete its claim out
    from under it. A flock cannot be wrong: the kernel holds it as
    long as the file descriptor is open, and releases it the moment the process
    dies however it dies.

    So a claim is never reaped while the flock is held, whatever its owner file
    says. Trying and immediately releasing is safe: `flock` is advisory and this
    only asks whether it COULD be taken.
    """
    path = LOCKS / f"gpu{device}.lock"
    if not path.exists():
        return False
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return True                       # unreadable is treated as held
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def _claim(device: int, owner_pid: int) -> Path | None:
    """Atomically claim one card, or return None.

    Three failure modes of a claim alone, and what stops each here:

    1. **A claim does not stop a flock-checking scheduler.** A claim and the
       `with-gpu` flock cannot see each other, so a job holding one is
       invisible to a job checking the other. `acquire()` takes BOTH, so a
       lease is visible to callers of either mechanism.
    2. **A claim can be destroyed by an unrelated `rm -rf`.** Removing the
       directory does not release the flock, so the lease survives and the
       card stays defended.
    3. **A claim can self-reap when its recorded owner exits before the work
       does.** `_lock_is_held` is consulted BEFORE any reap, and the kernel's
       answer beats the owner file's.
    """
    CLAIMS.mkdir(parents=True, exist_ok=True)
    path = CLAIMS / f"gpu{device}"
    try:
        path.mkdir()
    except FileExistsError:
        owner = ""
        with contextlib.suppress(OSError):
            owner = (path / "owner").read_text("utf-8").strip()
        # A NON-NUMERIC owner is a deliberate operator reservation and is never
        # reaped -- the `take_card.sh` convention.
        if not owner or not owner.isdigit():
            return None
        if _lock_is_held(device):
            # Somebody is measuring on this card RIGHT NOW, whatever the owner
            # file says about a pid.
            return None
        try:
            os.kill(int(owner), 0)
            return None                   # a live owner: not ours to take
        except ProcessLookupError:
            shutil.rmtree(path, ignore_errors=True)
        except PermissionError:
            return None                   # alive and someone else's
        try:
            path.mkdir()
        except OSError:
            return None
    except OSError:
        return None
    # The owner is THIS process, which holds the lease for its whole life. Never
    # a wrapper's pid, which would make the claim look dead to the next reaper.
    (path / "owner").write_text(str(owner_pid), "utf-8")
    (path / "what").write_text("final-validation", "utf-8")
    # A reaper that reads only this directory can still be taught the truth.
    (path / "flock").write_text(str(LOCKS / f"gpu{device}.lock"), "utf-8")
    (path / "since").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "utf-8")
    return path


def _flock(device: int) -> int | None:
    """Take the `with-gpu` lock for one card without blocking, or None."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    path = LOCKS / f"gpu{device}.lock"
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    with contextlib.suppress(OSError):
        os.write(fd, f"pid {os.getpid()}  final-validation  "
                     f"{time.strftime('%H:%M:%SZ', time.gmtime())}\n".encode())
    return fd


def _candidates() -> list[int]:
    """Every device this host reports, minus the deployment-local fence."""
    denied = _denied()
    out = []
    for row in _query("index"):
        with contextlib.suppress(ValueError):
            index = int(row)
            if index not in denied:
                out.append(index)
    return sorted(out)


def acquire(count: int, *, timeout_s: float = 900.0, poll_s: float = 15.0,
            allow: tuple[int, ...] = (), deny: tuple[int, ...] = ()) -> Lease:
    """Take `count` devices under BOTH locks, or raise inside `timeout_s`.

    Ascending index order, always, so two acquirers asking for overlapping sets
    cannot deadlock against each other. Partial acquisitions are released before
    the next attempt: holding two cards while waiting for a third is the
    starvation this bound exists to prevent.
    """
    if count < 1:
        raise ValueError("a lease needs at least one device")
    owner_pid = os.getpid()
    deadline = time.time() + max(0.0, timeout_s)
    blocked: dict[int, str] = {}
    attempt = 0
    while True:
        attempt += 1
        blocked = {}
        taken: list[int] = []
        fds: dict[int, int] = {}
        claims: list[Path] = []
        pool = [d for d in _candidates()
                if (not allow or d in allow) and d not in set(deny)]
        for device in pool:
            if len(taken) == count:
                break
            reserved = reservations_for(device, _uuid(device))
            if reserved:
                blocked[device] = (
                    f"reserved by running container(s): {list(reserved)[:3]}")
                continue
            foreign = _foreign_processes(device)
            if foreign:
                blocked[device] = f"compute processes not ours: {foreign[:2]}"
                continue
            busy = _busy_mib(device)
            if busy > BUSY_MIB:
                blocked[device] = f"{busy} MiB in use by work this lease does not cover"
                continue
            claim = _claim(device, owner_pid)
            if claim is None:
                blocked[device] = "claimed by another job"
                continue
            fd = _flock(device)
            if fd is None:
                shutil.rmtree(claim, ignore_errors=True)
                blocked[device] = "held under the with-gpu flock by another job"
                continue
            # Re-check AFTER both locks are held. Between the check above and
            # the lock there is always a window; a lock has no window, but only
            # for what it covers, and foreign work is not covered.
            if (reservations_for(device, _uuid(device))
                    or _foreign_processes(device)
                    or _busy_mib(device) > BUSY_MIB):
                os.close(fd)
                shutil.rmtree(claim, ignore_errors=True)
                blocked[device] = "became busy between the check and the lock"
                continue
            taken.append(device)
            fds[device] = fd
            claims.append(claim)
        if len(taken) == count:
            lease = Lease(devices=tuple(sorted(taken)),
                          uuids={d: _uuid(d) for d in taken},
                          owner_pid=owner_pid, _fds=fds,
                          _claims=tuple(claims))
            _make_visible(lease)
            atexit.register(lease.close)
            return lease
        # Release everything before waiting. Holding a partial set while waiting
        # for the rest is the multi-device starvation deadlock.
        for fd in fds.values():
            with contextlib.suppress(OSError):
                os.close(fd)
        for claim in claims:
            shutil.rmtree(claim, ignore_errors=True)
        if time.time() >= deadline:
            detail = "; ".join(f"gpu{d}: {why}" for d, why in sorted(blocked.items()))
            raise NoDeviceAvailable(
                f"could not acquire {count} device(s) within {timeout_s:.0f}s "
                f"after {attempt} attempt(s). What was in "
                f"the way: {detail or 'no candidate devices were reported'}. "
                f"Fenced off by SERVERBENCH_LEASE_DENY: "
                f"{sorted(_denied() | set(deny)) or 'none'}. "
                f"This is RECORDED as a blockage rather than waited out: an "
                f"indefinite waiter for a configuration the fleet cannot supply "
                f"idles the whole queue behind it")
        time.sleep(min(poll_s, max(0.0, deadline - time.time())))


# ------------------------------------------------------------------ external

class ExternalLease:
    """Cards another process already claimed for us, handed in by `--device`.

    Takes no lock and releases nothing: `close()` is a no-op, because tearing
    down somebody else's claim could hand their card to a third job. It still
    refuses a card carrying a compute process this program did not start,
    re-derived at the moment of use -- after a drain window, because
    `docker rm -f` returns before the driver has reclaimed the device memory of
    the previous arm.
    """

    DRAIN_SECONDS = 180.0

    def __init__(self, devices: tuple[int, ...],
                 *, reclaimable: tuple[str, ...] = ()) -> None:
        self.devices = tuple(devices)
        self.uuids = {d: _uuid(d) for d in self.devices}
        self.external = True
        reclaim_our_own(self.devices, mine=tuple(reclaimable))
        deadline = time.monotonic() + self.DRAIN_SECONDS
        foreign = self._foreign()
        while foreign and time.monotonic() < deadline:
            time.sleep(5.0)
            foreign = self._foreign()
        if foreign:
            raise SystemExit(
                f"device(s) {sorted(foreign)} carry a compute process this "
                f"program did not start, and still did after waiting "
                f"{self.DRAIN_SECONDS:.0f}s{_name_the_holders()}. A measurement "
                f"beside another job is not comparable, and taking the card "
                f"would disrupt that run. Nothing was launched")

    def _foreign(self) -> dict[int, str]:
        busy = {d: _foreign_processes(d) for d in self.devices}
        return {d: "; ".join(rows) for d, rows in busy.items() if rows}

    @property
    def device_spec(self) -> str:
        return docker_device_spec(self.devices)

    def close(self) -> None:
        """Deliberately nothing: we did not take this claim, so we may not drop it."""


#: Ours, and never reclaimed anyway: an oracle rebuild (`refs-<task>`) is
#: GPU-hours that cannot be resumed part-way.
NEVER_RECLAIM = ("refs-",)


def reclaim_our_own(devices: tuple[int, ...],
                    *, mine: tuple[str, ...] = ()) -> list[str]:
    """Remove this caller's OWN leftover containers still holding these cards.

    Ownership is the caller's explicit list of exact container names; naming
    nothing reclaims nothing. A name prefix cannot decide it on a shared host:
    `sb26...` matches every ServerBench container from any checkout. Bare host
    processes and unrecognised containers are never touched.
    """
    removed: list[str] = []
    # Nothing is authorised for reclamation. In particular, do not inspect
    # unrelated containers during the candidate -> reference transition.
    if not mine:
        return removed
    for device in devices:
        for line in _query("pid", device=device):
            pid = line.strip()
            name = _container_of(pid)
            if (not name or name not in mine
                    or any(name.startswith(p) for p in NEVER_RECLAIM)):
                continue
            subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                           text=True, timeout=180, check=False)
            removed.append(f"{name} (gpu{device}, pid {pid})")
    return removed


def _container_of(pid: str) -> str:
    """The container a host pid belongs to, from its cgroup, or ""."""
    import re
    try:
        cgroup = Path(f"/proc/{int(pid)}/cgroup").read_text("utf-8")
    except (OSError, ValueError):
        return ""
    for cid in re.findall(
            r"(?:docker[-/]|containerd.*?scope.*?)([0-9a-f]{12,64})", cgroup):
        try:
            got = subprocess.run(["docker", "inspect", "-f", "{{.Name}}", cid[:12]],
                                 capture_output=True, text=True, timeout=60,
                                 check=False)
        except (OSError, subprocess.TimeoutExpired):
            # Unknown ownership never authorises removal. The independent
            # foreign-process check still refuses an occupied device.
            return ""
        if got.returncode == 0:
            return (got.stdout or "").strip().lstrip("/")
    return ""


def _name_the_holders() -> str:
    """This suite's own running containers, so a refusal says whose card it is."""
    try:
        got = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=10,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    ours = sorted(n for n in (got.stdout or "").split()
                  if n.startswith(("finalval-", "sb2", "cal-")))
    if not ours:
        return ""
    return (f" -- this suite is also running {', '.join(ours[:3])}, so a card "
            f"held by one of those is not a neighbour's")


def wait_for_quiet(devices, *, timeout_s: float = 900.0,
                   quiet_mib: int = 256) -> str:
    """Block until no compute process runs on these cards and their memory is
    below `quiet_mib`, or return why not.

    Checked before a launch, so no process on a target card can be ours. A pid
    going away is not its memory going away, and a leaked CUDA context is a few
    hundred MiB -- hence a low floor AND a process check.
    """
    if isinstance(devices, str):
        devices = [int(d) for d in devices.split(",") if d.strip()]
    deadline = time.monotonic() + timeout_s
    last = ""
    while True:
        busy = []
        for device in devices:
            rows = _foreign_processes(device)
            used = _busy_mib(device)
            if rows:
                busy.append(f"card {device} runs {'; '.join(rows)}")
            elif used > quiet_mib:
                busy.append(f"card {device} holds {used} MiB")
        if not busy:
            return ""
        last = "; ".join(busy)
        if time.monotonic() >= deadline:
            return (f"cards {list(devices)} did not go quiet within "
                    f"{timeout_s:.0f}s (floor {quiet_mib} MiB, no compute "
                    f"process allowed): {last}")
        time.sleep(10)
