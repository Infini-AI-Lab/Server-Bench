"""Isolation, proved from inside the agent's own container.

This is the acceptance criterion most likely to fail quietly, because a
host-side check passes every day and keeps passing after somebody adds a bind
mount. So the probe runs INSIDE, with the driver's own mount set.

The heavy test is marked `slow` and skipped without docker or the agent image;
the recorded evidence from the last real run is checked on a deployed host
(`host` marker), so a tree that has never run the probe cannot look like one
that passed it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness" / "final_validate" / "tools"))

EVIDENCE = REPO / "serverbench" / "artifacts" / "isolation" / "agent-container.json"
IMAGE = "serverbench/evolve-agent:cu128"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    got = subprocess.run(["docker", "image", "inspect", IMAGE],
                         capture_output=True, timeout=120, check=False)
    return got.returncode == 0


@pytest.mark.host
def test_the_recorded_probe_found_no_breach() -> None:
    assert EVIDENCE.is_file(), (
        f"{EVIDENCE} is missing. Isolation is an acceptance criterion and a "
        f"tree that has never run the probe must not look like one that "
        f"passed it: run final_validate/tools/prove_isolation.py")
    body = json.loads(EVIDENCE.read_text("utf-8"))
    assert body["ok"] is True, body.get("breaches")
    assert not body["breaches"]
    assert body["files_walked"] > 100, (
        "the probe walked almost nothing, so its silence means nothing")


@pytest.mark.host
def test_the_evidence_names_the_private_paths_it_checked() -> None:
    body = json.loads(EVIDENCE.read_text("utf-8"))
    checked = set(body["paths"])
    for needle in ("final_validation", "serving_tasks", "server_output"):
        assert any(needle in p for p in checked), (
            f"the probe never looked for {needle!r}, so its pass says nothing "
            f"about it")
    assert all(present is False for present in body["paths"].values())


@pytest.mark.host
def test_the_evidence_shows_no_route_out_of_the_container() -> None:
    """`No agent-callable endpoint may reveal private final results.`"""
    body = json.loads(EVIDENCE.read_text("utf-8"))
    for endpoint, outcome in body["network"].items():
        assert outcome != "open", (
            f"{endpoint} was reachable from inside the agent's container")


@pytest.mark.host
def test_the_evidence_states_what_it_does_not_prove() -> None:
    body = json.loads(EVIDENCE.read_text("utf-8"))
    assert "no GPU was attached" in body["caveat"], (
        "a proof that does not say what it skipped invites a reader to assume "
        "it covered everything")


PROBE = REPO / "serverbench" / "harness" / "final_validate" / "tools" / "prove_isolation.py"


def test_the_probe_uses_the_drivers_own_mount_set() -> None:
    """Reimplementing the mounts would prove a property of the probe, not the run."""
    source = PROBE.read_text("utf-8")
    assert "harbor_env.mounts_for(" in source, (
        "the probe computes its own mounts, so it can drift from the mount set "
        "the driver actually uses and keep passing")


@pytest.mark.host
def test_every_path_the_probe_forbids_is_a_path_that_exists() -> None:
    """A forbidden path that is not there is a check that passes for free.

    If the private package moves and the probe keeps its old path, the entry
    meant to prove the private program is unreachable proves only that a
    missing directory is unreachable, while the real one goes unchecked.

    `/opt/serverbench` is exempt: it is a conventional install location checked
    in case, and it is deliberately absent on a development host.
    """
    import prove_isolation

    absent = [path for path in prove_isolation.FORBIDDEN_PATHS
              if not Path(path).exists() and not path.startswith("/opt/")]
    assert not absent, (
        f"the probe forbids paths that are not on this host, so those entries "
        f"cannot fail: {absent}")
    private = str(REPO / "serverbench" / "harness" / "final_validate")
    assert private in prove_isolation.FORBIDDEN_PATHS, (
        f"{private} is the private final-validation program and the probe does "
        f"not check that it is unreachable")


@pytest.mark.slow
def test_the_private_program_is_unreachable_from_inside_the_container() -> None:
    if not _docker_available():
        pytest.skip(f"docker or {IMAGE} is not available on this host")
    import prove_isolation

    report = prove_isolation.run(IMAGE)
    assert report["ok"], report.get("breaches") or report.get("why")
    assert report["imports"] == dict.fromkeys(report["imports"], False)
