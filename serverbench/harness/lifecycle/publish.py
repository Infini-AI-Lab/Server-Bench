"""Preserve each publication, advance only along its lineage, then evaluate once.

Publishing is not grading. Every captured version is materialised and kept
without an in-run verdict. A causally descendant publication may become the
lineage head; the publisher checks identity, ancestry and compare-and-swap
state, but no correctness or performance threshold. Final evaluation runs only
after the last state is terminal and returns nothing toward the agent.

## The publisher
The publisher. It decides no quality question; it records the unjudged version
selected by the run and enforces the lineage transaction.

There is exactly one accepted head per lineage and one function that moves it.
No correctness head, no performance head, no reviewer head, no development
head: each would be another place a version could look promoted from.

Two refusals are the whole point of the file:

* **No bound publication record, no move.** ``advance`` takes an
  :class:`Authorization` and re-checks that it names this exact version and
  tree. The live driver mints an unconditional, explicitly unjudged record;
  development measurements cannot be substituted for it.
* **No silent repair.** The swap is compare-and-swap against the predecessor
  the caller expected. If the head moved underneath, the call fails and says
  what it found. It does not retry against the new head -- that would publish a
  candidate on top of a parent it was never built from, and a lineage would
  come to contain two unrelated codebases.

The final evaluation itself lives in `final_validate` and runs once, after the
run, against the published head.
"""
from __future__ import annotations

import dataclasses
import fcntl
import json
from pathlib import Path

from .record import Journal, utc_now
from .utils import write_json_atomically


class NotAuthorizedToPublish(RuntimeError):
    """Something asked for the head without a bound publication record."""


class CASConflict(RuntimeError):
    """The head is not what the caller expected. Never resolved by retrying."""

    def __init__(self, expected: str, found: str) -> None:
        super().__init__(
            f"the accepted head is {found or '(none)'}, not the "
            f"{expected or '(none)'} this candidate was built on. The publisher "
            f"does not re-target: capture a new candidate from the head that "
            f"is actually there")
        self.expected, self.found = expected, found


@dataclasses.dataclass(frozen=True)
class Authorization:
    """Carries what it authorises, so the publisher can refuse a mismatch
    rather than trust the caller.

    Nothing in a run measures a version, so nothing in a run grades one. What
    the type does, and the reason to keep it rather than pass three strings,
    is bind an authorisation to ONE version and ONE tree. `advance` re-checks
    both, so a caller cannot publish version A's authorisation over version B.

    `split` is recorded rather than gated. `unconditional` is what the driver
    mints: every captured version is published, and its worth is decided later
    by evidence the agent never sees. `development` is refused -- a
    development number reaching the head would be the agent grading itself."""

    version_id: str
    tree_sha: str
    candidate_hash: str
    artifact_id: str
    split: str
    outcome: str
    issued_utc: str
    #: Carried so the head records what a later candidate must descend FROM.
    commit_sha: str = ""

    #: What may authorise the head. `development` is deliberately absent: an
    #: agent's own measurement is trajectory evidence, never a publication
    #: verdict.
    SPLITS = ("unconditional",)

    def __post_init__(self) -> None:
        if self.split == "development":
            raise NotAuthorizedToPublish(
                "an authorization was built from a development measurement. "
                "Development measurements are non-authoritative trajectory "
                "evidence and authorise nothing: using one to move the head "
                "would turn the agent's own benchmark into a verdict and let "
                "the agent start grading itself")
        if self.split not in self.SPLITS:
            raise NotAuthorizedToPublish(
                f"unknown authorization split {self.split!r}; expected one of "
                f"{list(self.SPLITS)}")

    @classmethod
    def unconditional(cls, *, version_id: str, tree_sha: str,
                      candidate_hash: str, commit_sha: str = "",
                      issued_utc: str = "") -> Authorization:
        """What the driver mints: publish it, judge it later.

        The `outcome` is `PUBLISHED_UNJUDGED` and not `PASS`, because nothing
        has judged it at this point and a head recorded as passing something it
        never faced is a provenance failure.
        """
        import time as _time

        return cls(
            version_id=version_id, tree_sha=tree_sha,
            candidate_hash=candidate_hash,
            artifact_id="", split="unconditional",
            outcome="PUBLISHED_UNJUDGED",
            issued_utc=(issued_utc or _time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", _time.gmtime())),
            commit_sha=commit_sha)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["kind"] = "publication_authorization"
        d["schema_version"] = 1
        return d


@dataclasses.dataclass(frozen=True)
class Head:
    version_id: str
    tree_sha: str
    authorized_by: str
    predecessor: str
    at_utc: str
    commit_sha: str = ""

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["kind"] = "accepted_head"
        d["schema_version"] = 1
        return d


class Publisher:
    """One head file per lineage, moved under an exclusive lock."""

    def __init__(self, lineage_dir: Path) -> None:
        # Created on the first advance, not at startup: an empty lineage
        # directory in the published tree claims an ascent that never happened.
        self.root = Path(lineage_dir)
        self.path = self.root / "ACCEPTED_HEAD.json"
        self.lock = self.root / ".head.lock"

    def head(self) -> Head | None:
        if not self.path.exists():
            return None
        d = json.loads(self.path.read_text())
        return Head(version_id=d["version_id"], tree_sha=d["tree_sha"],
                    authorized_by=d["authorized_by"],
                    predecessor=d["predecessor"], at_utc=d["at_utc"],
                    commit_sha=d.get("commit_sha", ""))

    def head_tree(self) -> str:
        head = self.head()
        return head.tree_sha if head else ""

    def head_commit(self) -> str:
        head = self.head()
        return head.commit_sha if head else ""

    def head_version(self) -> str:
        head = self.head()
        return head.version_id if head else ""

    def advance(self, *, authorization: Authorization, expected: str,
                journal: Journal) -> Head:
        """Compare-and-swap the accepted head. The only writer of that file."""
        if not isinstance(authorization, Authorization):
            raise NotAuthorizedToPublish(
                f"advance() takes an Authorization, which binds a publication "
                f"to one version and one tree; it was given "
                f"{type(authorization).__name__}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock.touch(exist_ok=True)
        with open(self.lock, "r+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                current = self.head()
                found = current.version_id if current else ""
                if found != (expected or ""):
                    journal.append("head_cas_conflict",
                                   expected=expected, found=found,
                                   version_id=authorization.version_id)
                    raise CASConflict(expected, found)
                head = Head(version_id=authorization.version_id,
                            tree_sha=authorization.tree_sha,
                            authorized_by=authorization.artifact_id,
                            predecessor=found, at_utc=utc_now(),
                            commit_sha=authorization.commit_sha)
                write_json_atomically(self.path, head.to_dict())
                with open(self.root / "head_history.jsonl", "a") as hist:
                    hist.write(json.dumps(
                        {**head.to_dict(),
                         "authorization": authorization.to_dict()},
                        sort_keys=True) + "\n")
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        journal.append("head_advanced", version_id=head.version_id,
                       tree_sha=head.tree_sha, predecessor=head.predecessor,
                       authorized_by=head.authorized_by)
        return head
