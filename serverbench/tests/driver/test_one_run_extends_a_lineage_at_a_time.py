"""Two live runs in one lineage interleave, and each ends up with the other's head.

Version numbers in a lineage are allocated from its DIRECTORY, not from a
run's own journal. Two runs extending one lineage at once therefore interleave
their version numbers, and a run's accepted head can end up being a version the
other run captured, whose `version_registered` event lives in the other run's
journal. Final validation then correctly refuses -- there is no tree to measure
-- and the run's hours end with no verdict, because nothing refused at launch.

## What these tests hold

A lineage carries a lock naming the run extending it. A second run is refused,
and told to take its own lineage. A lock whose process is gone is not a lock:
a driver killed mid-run must not fence its lineage off forever, so liveness is
the pid rather than the file's existence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for path in (str(REPO / "serverbench" / "harness"),):
    if path not in sys.path:
        sys.path.insert(0, path)

from lifecycle import run as run_mod  # noqa: E402


def test_a_lineage_with_no_lock_has_no_writer(tmp_path) -> None:
    assert run_mod.lineage_live_writer(tmp_path) is None


def test_a_claimed_lineage_reports_this_process(tmp_path) -> None:
    run_mod.claim_lineage(tmp_path, "run-0001")
    live = run_mod.lineage_live_writer(tmp_path)
    assert live is not None
    assert live["run_id"] == "run-0001"
    assert live["pid"] == os.getpid()


def test_releasing_clears_it(tmp_path) -> None:
    run_mod.claim_lineage(tmp_path, "run-0001")
    run_mod.release_lineage(tmp_path)
    assert run_mod.lineage_live_writer(tmp_path) is None


def test_releasing_twice_is_not_an_error(tmp_path) -> None:
    """It is registered with atexit and may also be released explicitly."""
    run_mod.claim_lineage(tmp_path, "run-0001")
    run_mod.release_lineage(tmp_path)
    run_mod.release_lineage(tmp_path)


def test_a_lock_whose_process_is_gone_is_ignored(tmp_path) -> None:
    """Otherwise a killed driver fences its lineage off permanently.

    The pid used is one that can never belong to a live process.
    """
    # `pid_max` is the exclusive upper bound, so it is never a live pid. A
    # reaped child's pid, by contrast, can be recycled between the wait and
    # the check.
    dead = int(Path("/proc/sys/kernel/pid_max").read_text("utf-8").strip())
    (tmp_path / run_mod.LINEAGE_LOCK).write_text(json.dumps(
        {"run_id": "run-0002", "pid": dead,
         "since_utc": "2026-01-01T00:00:00Z"}), "utf-8")
    assert run_mod.lineage_live_writer(tmp_path) is None, (
        "a lock left by a dead driver still fences the lineage")


def test_a_corrupt_lock_does_not_fence_the_lineage(tmp_path) -> None:
    (tmp_path / run_mod.LINEAGE_LOCK).write_text("{not json", "utf-8")
    assert run_mod.lineage_live_writer(tmp_path) is None


def test_a_lock_without_a_pid_is_not_trusted(tmp_path) -> None:
    (tmp_path / run_mod.LINEAGE_LOCK).write_text(
        json.dumps({"run_id": "run-0003"}), "utf-8")
    assert run_mod.lineage_live_writer(tmp_path) is None


def test_the_lock_is_not_mistaken_for_a_published_version(tmp_path) -> None:
    """It lives in the lineage directory beside the versions."""
    assert run_mod.LINEAGE_LOCK.startswith("."), (
        "the lock is not hidden, so a lineage listing counts it as a version")


# --- the refusal itself ----------------------------------------------------

def test_the_driver_refuses_a_lineage_another_run_is_extending() -> None:
    """Checked as a property of the launch path, which needs no GPU to read.

    The message has to say what to do instead -- an operator who wanted two
    agents on one task at once has a legitimate goal and needs the other
    lineage, not a bare refusal.
    """
    body = (REPO / "serverbench" / "harness" / "lifecycle"
            / "run.py").read_text("utf-8")
    assert "lineage_live_writer(lineage_dir)" in body, (
        "the launch path does not check for a live writer on the lineage")
    start = body.index("live = lineage_live_writer(lineage_dir)")
    # Flattened, and checked against phrases that sit INSIDE one string
    # literal: the message is built from adjacent f-strings, so a phrase
    # spanning two of them is separated by `" f"` in the source and no amount
    # of whitespace flattening will join it.
    refusal = " ".join(body[start:start + 2000].split())
    assert "already being extended" in refusal
    assert "seed from the starter" in refusal, (
        "the refusal does not tell the operator how to run two agents on one "
        "task, which is the thing they were trying to do")
    assert "--lineage-id" in refusal, (
        "the refusal does not name the flag that gives this run its own "
        "lineage")
    assert "return 2" in refusal, (
        "the launch path warns about a live writer and proceeds anyway")


def test_the_claim_is_released_however_the_run_ends() -> None:
    body = (REPO / "serverbench" / "harness" / "lifecycle"
            / "run.py").read_text("utf-8")
    assert "atexit.register(release_lineage" in body, (
        "the lineage claim is taken but only released on a clean path, so a "
        "crashed driver would leave it behind")


def test_a_resume_is_not_fenced_out_of_its_own_lineage() -> None:
    """A resume is the same run continuing; its own lock must not block it."""
    body = (REPO / "serverbench" / "harness" / "lifecycle"
            / "run.py").read_text("utf-8")
    assert 'live.get("run_id") != resuming_run_id' in body, (
        "the guard compares against something other than the resuming run's "
        "own id, so resuming a run would be refused by its own lock")


@pytest.mark.parametrize("name", ["claim_lineage", "release_lineage",
                                  "lineage_live_writer"])
def test_the_helpers_exist_and_are_importable(name) -> None:
    assert hasattr(run_mod, name)
