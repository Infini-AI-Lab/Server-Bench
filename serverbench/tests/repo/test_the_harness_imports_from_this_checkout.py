"""The supported environment must resolve `serving_evolve` to THIS tree.

The harness venv may carry an editable install of `serverbench`, and an
editable install records one absolute path. If that path points at another
checkout or a stale snapshot, every entry point that imports the package
without first putting this repository on `sys.path` runs a different harness
while appearing to run this one -- silently, until a module that exists in one
tree and not the other happens to be imported. Freezing a task stamps its
digest, so doing it from stale code is not a cosmetic problem.

Scripts are safe when they do

    sys.path.insert(0, str(REPO / "serverbench" / "harness"))

before importing, which wins over the installed finder.

The environment may intentionally be shared by several checkouts. The
supported entry path is therefore `serverbench/scripts/env.sh`, which puts the
checkout containing that script ahead of any editable-install metadata. A
test that invokes the interpreter without sourcing that file tests the shared
venv's last editable install, not this repository's runtime contract.

No GPU, no network.
"""

from __future__ import annotations

import pathlib
import subprocess

from paths import SERVERBENCH

REPO = SERVERBENCH.parent


def _resolved(module: str) -> pathlib.Path:
    """Where a fresh interpreter in this checkout's supported env finds it."""
    env_script = SERVERBENCH / "scripts" / "env.sh"
    command = (
        f'. "{env_script}" && exec "$SERVERBENCH_PY" -c '
        f'\'import {module}, pathlib; '
        f'print(pathlib.Path({module}.__file__).parent)\'')
    got = subprocess.run(
        ["bash", "-c", command],
        capture_output=True, text=True, timeout=120, cwd=str(REPO))
    assert got.returncode == 0, (
        f"a bare `import {module}` fails in this environment:\n{got.stderr[-600:]}")
    return pathlib.Path(got.stdout.strip())


def test_serving_evolve_resolves_inside_this_repository():
    where = _resolved("serving_evolve")
    assert where.is_relative_to(REPO), (
        f"`import serving_evolve` resolves to {where}, which is outside "
        f"{REPO}. scripts/env.sh did not put this checkout first; do not repair "
        f"this by repointing a virtualenv shared with another checkout")
    assert where == SERVERBENCH / "harness" / "serving_evolve", where


def test_the_console_script_runs_this_tree():
    """`serverbench evolve doctor` is offered as the way to check the tree."""
    binary = SERVERBENCH / ".venv" / "bin" / "serverbench"
    if not binary.is_file():
        import pytest
        pytest.skip("no console script installed in this environment")
    got = subprocess.run([str(binary), "evolve", "doctor"],
                         capture_output=True, text=True, timeout=300,
                         cwd=str(REPO))
    assert "PASS" in got.stdout, (
        f"`serverbench evolve doctor` did not pass:\n"
        f"{got.stdout[-800:]}\n{got.stderr[-400:]}")


def test_every_script_puts_this_checkout_on_the_path_before_importing():
    """A script must not rely on the editable install being pointed correctly.

    The install is one absolute path recorded once, and nothing keeps it
    pointed at this checkout. Scripts that do

        sys.path.insert(0, str(REPO / "serverbench" / "harness"))

    are immune; a script that does not imports whatever tree the install
    names.
    """
    offenders = []
    base = SERVERBENCH / "scripts"
    assert base.is_dir(), base
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text("utf-8")
        imports_harness = any(
            line.startswith(("from serving_evolve", "import serving_evolve",
                             "from kernel_evolve", "import kernel_evolve",
                             "from lifecycle", "import lifecycle"))
            for line in text.splitlines())
        if imports_harness and "sys.path.insert" not in text:
            offenders.append(str(path.relative_to(SERVERBENCH)))
    assert not offenders, (
        "these scripts import the harness at module scope without putting this "
        "checkout on sys.path first, so they resolve through the editable "
        "install:\n" + "\n".join(f"    {o}" for o in offenders)
        + "\n  Add: sys.path.insert(0, str(Path(__file__).resolve().parents[N] "
          "/ 'harness'))")
