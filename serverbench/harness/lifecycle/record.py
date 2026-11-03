"""The run record: who this run is, what happened, and what state it is in.

Three things that are one thing. The identity is allocated once and never
inferred again; the journal is the only place a state change is written; the
state machines are the only thing allowed to move a run or a candidate, and
they move it by journalling. Reading any one of them without the other two
tells you nothing, and every phase of the loop touches all three.

## Identity
Run identity: allocated once, atomically, and never inferred afterwards.

Everything downstream keys off the identity minted here. The rules that matter:

* A run id is claimed by creating its directory with ``exist_ok=False``. Two
  launchers racing therefore cannot receive the same id -- the loser sees
  ``FileExistsError`` and draws again.
* A dry run and a test allocate in a *different namespace*, so neither consumes
  a production id.
* Resume takes the whole tuple. If any element differs it is a different run,
  and the answer is an error rather than a guess: resuming "the newest
  resumable workspace" can publish one run's codebase into another's lineage.

## Journal
One append-only journal per run. Every state change goes through it.

Deliberately plain: an ordered JSONL file, appended under an exclusive lock,
flushed and fsynced. There is no hash chain and no anchor here: tamper-evidence
against an agent running as the same uid is not achievable. What this must do
is survive a crash without a torn record, and let a reader reconstruct the run.

Nothing in the harness infers state from prose, a directory name, a container
name, or which file was modified last. If it is not an event here, it did not
happen.

## States
The two state machines, and the one function that moves either of them.

There are exactly two. Anything that happens to a candidate is *candidate*
state; a third machine for it would let a run report COMPLETED while the thing
it had built was still unsettled.

Every transition is checked against the table and written to the journal before
it is visible anywhere else, so a reader reconstructs the run from the journal
alone and there is no second state file to disagree with it.
"""
from __future__ import annotations

import dataclasses
import datetime
import enum
import fcntl
import json
import os
import re
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .utils import write_json_atomically

# --- identity --------------------------------------------------------------------

#: Production runs live here; nothing else may.
PRODUCTION = "evolve"
#: Dry runs, audits and tests. Same code, different tree.
SCRATCH = "evolve-scratch"

_RUN_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-\d{8}T\d{6}Z-[0-9a-f]{6}$")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")


class IdentityError(RuntimeError):
    """A provenance question was answered with a guess. Refuse instead."""


@dataclasses.dataclass(frozen=True)
class RunIdentity:
    """The immutable half of a run. Written once, at creation, and read back
    on every resume; a mismatch on any field is fatal."""

    run_id: str
    task_id: str
    lineage_id: str
    control_epoch: str
    workspace_id: str
    namespace: str
    created_at: str
    deadline_utc: str
    root: Path

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["root"] = str(self.root)
        d["kind"] = "run_identity"
        d["schema_version"] = 1
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RunIdentity:
        return cls(run_id=d["run_id"], task_id=d["task_id"],
                   lineage_id=d["lineage_id"],
                   control_epoch=d["control_epoch"],
                   workspace_id=d["workspace_id"],
                   namespace=d["namespace"], created_at=d["created_at"],
                   deadline_utc=d["deadline_utc"], root=Path(d["root"]))

    def matches(self, **expected: str) -> None:
        """Raise unless every supplied field is exactly what this run holds."""
        for field, want in expected.items():
            if want is None:
                continue
            got = getattr(self, field, None)
            if got != want:
                raise IdentityError(
                    f"this run's {field} is {got!r}, not {want!r}. Resume takes "
                    f"the exact tuple (run_id, task_id, lineage_id, "
                    f"workspace_id, control_epoch); a harness that repairs a "
                    f"mismatch is a harness that publishes one run's work into "
                    f"another run's lineage")


def evolve_space_root(repo: Path, namespace: str = PRODUCTION) -> Path:
    """Where a run's workspace, state and evidence live.

    Runs have their own tree, apart from models, images and other evidence
    under `serverbench/artifacts/`. The namespace is recorded in each run's
    identity.json.
    """
    return Path(repo) / "serverbench" / "evolve_space" / namespace


def allocate(*, repo: Path, task_id: str, lineage_id: str,
             control_epoch: str, deadline_utc: str,
             namespace: str = PRODUCTION, attempts: int = 32) -> RunIdentity:
    """Claim a fresh run id, atomically.

    The directory *is* the claim. There is no counter to read, no "latest" to
    scan and no lock file whose staleness would have to be judged.
    """
    base = evolve_space_root(repo, namespace) / task_id.lower()
    base.mkdir(parents=True, exist_ok=True)
    # The task's hash only. A run id ends up inside a unix socket path, and
    # AF_UNIX stops at 108 bytes; the task's full name would spend them on
    # words the directory above already says.
    from final_validate.spine import task_hash

    slug = task_hash(task_id)
    for _ in range(attempts):
        run_id = f"{slug}-{_stamp()}-{secrets.token_hex(3)}"
        root = base / run_id
        try:
            root.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        identity = RunIdentity(
            run_id=run_id, task_id=task_id, lineage_id=lineage_id,
            control_epoch=control_epoch, workspace_id=f"{run_id}/workspace",
            namespace=namespace, created_at=utc_now(),
            deadline_utc=deadline_utc, root=root)
        write_identity(identity)
        return identity
    raise IdentityError(
        f"could not claim a run id under {base} in {attempts} attempts")


def write_identity(identity: RunIdentity) -> None:
    write_json_atomically(identity.root / "identity.json", identity.to_dict())


def load(root: Path) -> RunIdentity:
    path = Path(root) / "identity.json"
    if not path.exists():
        raise IdentityError(
            f"{root} has no identity.json, so it is not a run this harness "
            f"created. Nothing is inferred from the directory name")
    return RunIdentity.from_dict(json.loads(path.read_text()))


def open_exact(*, repo: Path, run_id: str, task_id: str, lineage_id: str,
               workspace_id: str, control_epoch: str,
               namespace: str = PRODUCTION) -> RunIdentity:
    """Reopen one run, by its full tuple. The only way to resume."""
    if not _RUN_ID.match(run_id):
        raise IdentityError(f"{run_id!r} is not a run id this harness minted")
    root = evolve_space_root(repo, namespace) / task_id.lower() / run_id
    if not root.is_dir():
        raise IdentityError(
            f"no run {run_id} under {root.parent}. Resume never falls back to "
            f"another run, however recently it was written")
    identity = load(root)
    identity.matches(run_id=run_id, task_id=task_id, lineage_id=lineage_id,
                     workspace_id=workspace_id, control_epoch=control_epoch)
    return identity


# --- journal ---------------------------------------------------------------------

def _last_byte(path: Path) -> bytes:
    """The final byte of a file, or b"" when it is empty.

    Read separately from the append handle because that handle is opened in
    append mode: on Linux every write goes to the end regardless of the file
    position, so seeking backwards on it to look is legal but reads through a
    different buffer than the one being written.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                return b""
            handle.seek(-1, os.SEEK_END)
            return handle.read(1)
    except OSError:
        return b"\n"


class Journal:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "journal.jsonl"
        self.path.touch(exist_ok=True)

    def append(self, event_type: str, **payload: Any) -> dict:
        """Append one event and return it. Serialised across processes, so more
        than one writer can append to the same run."""
        event = {"at_utc": utc_now(), "event": event_type, **payload}
        line = json.dumps(event, sort_keys=True, default=str) + "\n"
        # O_APPEND already serialises the offset update, and a line this size
        # goes out in one write(). The lock is held anyway because the
        # guarantee is about the FILE, not about typical line lengths: a
        # payload longer than the buffer, a non-append writer, or a filesystem
        # without atomic append would all reintroduce the tear it prevents.
        with open(self.path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                # Close a torn final line before appending to it.
                #
                # `__iter__` skips a line it cannot decode, so a crash
                # mid-append costs that event, which is unavoidable. Without
                # this, the next append would land on the same line and the
                # two would become one undecodable string, losing a completed
                # record too (on a crash-and-resume, `run_resumed`).
                #
                # Under the same LOCK_EX as the write, so two writers cannot
                # both decide to repair. The repair is journalled rather than
                # done quietly: a healed tear is evidence about the run, and
                # silence would make a crashed run read like a clean one.
                fh.seek(0, os.SEEK_END)
                torn = fh.tell() > 0 and _last_byte(self.path) != b"\n"
                if torn:
                    fh.write("\n")
                    fh.write(json.dumps(
                        {"at_utc": utc_now(), "event": "journal_tear_repaired",
                         "kind": "journal_repair",
                         "why": ("the previous append did not finish; its line "
                                 "was closed so this one is not lost with it")},
                        sort_keys=True) + "\n")
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return event

    def __iter__(self) -> Iterator[dict]:
        if not self.path.exists():
            return iter(())
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A partial final line can only be a crash mid-append. It is
                # reported by being skipped here and kept on disk, not healed.
                continue
        return iter(out)

    def events(self, event_type: str | None = None) -> list[dict]:
        return [e for e in self
                if event_type is None or e.get("event") == event_type]

    def last(self, event_type: str) -> dict | None:
        found = self.events(event_type)
        return found[-1] if found else None


# --- states ----------------------------------------------------------------------

class RunState(str, enum.Enum):
    CREATED = "CREATED"
    WORKSPACE_READY = "WORKSPACE_READY"
    AGENT_RUNNING = "AGENT_RUNNING"
    ROUND_BOUNDARY = "ROUND_BOUNDARY"
    STOPPED_EARLY = "STOPPED_EARLY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class CandidateState(str, enum.Enum):
    CAPTURED = "CAPTURED"
    #: The deterministic boundary checks.
    CHECKING = "CHECKING"
    CHECKS_COMPLETE = "CHECKS_COMPLETE"
    MATERIALIZED = "MATERIALIZED"
    REGISTERED = "REGISTERED"
    #: Immutable and visible, but deliberately not selected as the final head.
    CHECKPOINTED = "CHECKPOINTED"
    ACCEPTED = "ACCEPTED"


RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.WORKSPACE_READY, RunState.FAILED,
                                 RunState.CANCELLED}),
    RunState.WORKSPACE_READY: frozenset({RunState.AGENT_RUNNING,
                                         RunState.FAILED, RunState.CANCELLED}),
    RunState.AGENT_RUNNING: frozenset({RunState.ROUND_BOUNDARY,
                                       RunState.STOPPED_EARLY,
                                       RunState.COMPLETED, RunState.FAILED,
                                       RunState.CANCELLED}),
    # A boundary returns to development: the agent never waits for the checks.
    RunState.ROUND_BOUNDARY: frozenset({RunState.AGENT_RUNNING,
                                        RunState.STOPPED_EARLY,
                                        RunState.COMPLETED, RunState.FAILED,
                                        RunState.CANCELLED}),
    # An early stop is continuable *into the same run*, and into nothing else.
    RunState.STOPPED_EARLY: frozenset({RunState.AGENT_RUNNING,
                                       RunState.COMPLETED, RunState.FAILED,
                                       RunState.CANCELLED}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
}

CANDIDATE_TRANSITIONS: dict[CandidateState, frozenset[CandidateState]] = {
    CandidateState.CAPTURED: frozenset({CandidateState.CHECKING}),
    CandidateState.CHECKING: frozenset({CandidateState.CHECKS_COMPLETE}),
    # A failed boundary check still reaches CHECKS_COMPLETE. The checks report
    # form and portability facts and do not gate publication: the candidate
    # becomes visible with the failure recorded against it, not hidden.
    CandidateState.CHECKS_COMPLETE: frozenset({CandidateState.MATERIALIZED}),
    CandidateState.MATERIALIZED: frozenset({CandidateState.REGISTERED}),
    # A registered version is published on ancestry alone, and what it is
    # worth is decided afterwards by the final verdict outside Harbor.
    CandidateState.REGISTERED: frozenset({CandidateState.CHECKPOINTED,
                                          CandidateState.ACCEPTED}),
    CandidateState.CHECKPOINTED: frozenset({CandidateState.ACCEPTED}),
    CandidateState.ACCEPTED: frozenset(),
}


class TransitionError(RuntimeError):
    """A move the machine does not have. Never repaired, never forced."""


def _move(kind: str, table: dict, subject: str, frm, to, journal: Journal,
          **detail) -> None:
    allowed = table.get(frm, frozenset())
    if to not in allowed:
        raise TransitionError(
            f"{kind} {subject}: {frm.value} -> {to.value} is not a transition "
            f"this machine has. From {frm.value} it may go to "
            f"{sorted(s.value for s in allowed) or 'nowhere'}")
    journal.append(f"{kind}_state", subject=subject, frm=frm.value,
                   to=to.value, **detail)


def move_run(journal: Journal, run_id: str, frm: RunState, to: RunState,
             **detail) -> RunState:
    _move("run", RUN_TRANSITIONS, run_id, frm, to, journal, **detail)
    return to


def move_candidate(journal: Journal, candidate_id: str, frm: CandidateState,
                   to: CandidateState, **detail) -> CandidateState:
    _move("candidate", CANDIDATE_TRANSITIONS, candidate_id, frm, to, journal,
          **detail)
    return to


def run_state(journal: Journal) -> RunState:
    """Reconstructed from the journal. There is no other source."""
    last = journal.last("run_state")
    return RunState(last["to"]) if last else RunState.CREATED


def candidate_state(journal: Journal, candidate_id: str) -> CandidateState | None:
    seen = [e for e in journal.events("candidate_state")
            if e.get("subject") == candidate_id]
    return CandidateState(seen[-1]["to"]) if seen else None


def candidates(journal: Journal) -> dict[str, CandidateState]:
    out: dict[str, CandidateState] = {}
    for e in journal.events("candidate_state"):
        out[e["subject"]] = CandidateState(e["to"])
    return out
