"""The boot check that says "no other run's directory is mounted" must mean it.

A mount is foreign when its source sits beside this run rather than inside it,
derived from the run's own layout rather than matched against path literals
(literals rot when run ids or the run space are renamed). The classification
is a helper so it can be exercised without starting a container; these cases
are what `docker inspect` would show.
"""

from __future__ import annotations

import sys

from paths import SERVERBENCH

sys.path.insert(0, str(SERVERBENCH / "harness"))

from lifecycle.workspace import foreign_mounts  # noqa: E402


def _layout(tmp_path):
    """`<namespace>/<task>/<run-id>/workspace`, as a real run is laid out."""
    task = tmp_path / "evolve_space" / "evolve" / "some_task"
    mine = task / "run-b-6a9aaf" / "workspace"
    theirs = task / "run-a-bf19ca" / "workspace"
    mine.mkdir(parents=True)
    theirs.mkdir(parents=True)
    return mine, theirs


def test_this_run_s_own_mounts_are_not_foreign(tmp_path):
    mine, _ = _layout(tmp_path)
    spec = f"{mine}->/workspace {mine.parent}/service->/run/serverbench"
    assert foreign_mounts(spec, mine) == []


def test_another_run_s_workspace_is_caught_wherever_it_is_mounted(tmp_path):
    """Another run's workspace is foreign even at an unrelated destination."""
    mine, theirs = _layout(tmp_path)
    spec = f"{mine}->/workspace {theirs}->/opt/other"
    found = foreign_mounts(spec, mine)
    assert found == [f"{theirs}->/opt/other"], (
        f"another run's workspace mounted at /opt/other was not reported: {found}"
    )


def test_a_sibling_run_s_evidence_directory_is_caught(tmp_path):
    """Not just its workspace -- its journal and qualification too."""
    mine, theirs = _layout(tmp_path)
    spec = f"{mine}->/workspace {theirs.parent}->/opt/evidence"
    assert foreign_mounts(spec, mine)


def test_the_model_and_the_wheelhouse_are_not_foreign(tmp_path):
    """The three legitimate mounts live outside the namespace entirely."""
    mine, _ = _layout(tmp_path)
    model = tmp_path / "artifacts" / "models" / "qwen3-4b"
    wheels = tmp_path / "artifacts" / "env-input"
    model.mkdir(parents=True)
    wheels.mkdir(parents=True)
    spec = (f"{mine}->/workspace {model}->/models/qwen3-4b "
            f"{wheels}->/opt/env-input")
    assert foreign_mounts(spec, mine) == []


def test_an_empty_mount_list_is_not_a_finding(tmp_path):
    mine, _ = _layout(tmp_path)
    assert foreign_mounts("", mine) == []


def test_a_mount_source_that_does_not_exist_does_not_crash(tmp_path):
    """`docker inspect` can name a source that has since been removed."""
    mine, _ = _layout(tmp_path)
    spec = f"{mine}->/workspace {tmp_path}/gone/away->/opt/x"
    assert foreign_mounts(spec, mine) == []
