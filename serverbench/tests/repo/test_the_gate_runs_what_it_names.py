"""Every check `correctness-gate.sh` names has to exist.

The gate is the thing run before a change ships, so a step in it that cannot
run is worse than no step: it reports a failure that has nothing to do with the
change, and after the second time nobody reads it. The usual cause is a check
script retired while the line invoking it stays, so every script path and
pytest package the gate names must be on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "serverbench" / "scripts" / "correctness-gate.sh"

#: `"$PY_BIN" "$HERE/checks/foo.py"` and friends -- a script path the gate
#: hands to the interpreter, rather than a module or a pytest package.
INVOKED = re.compile(r'"\$(?:HERE|SERVERBENCH|REPO)(/[^"]+\.py)"')


def _gate_text() -> str:
    assert GATE.is_file(), GATE
    return GATE.read_text("utf-8")


def test_every_script_the_gate_invokes_is_on_disk() -> None:
    missing = []
    for match in INVOKED.finditer(_gate_text()):
        relative = match.group(1).lstrip("/")
        for root in (REPO / "serverbench" / "scripts", REPO / "serverbench",
                     REPO):
            if (root / relative).is_file():
                break
        else:
            missing.append(relative)
    assert not missing, (
        "correctness-gate.sh invokes script(s) that are not on disk: "
        + ", ".join(sorted(missing))
        + ". A retired check has to be removed from the gate or replaced by "
          "what covers the same guarantee -- left in place it fails the "
          "gate for a reason unrelated to whatever is being shipped.")


def test_every_pytest_package_the_gate_runs_exists() -> None:
    """A renamed or removed test package breaks the gate too."""
    text = _gate_text()
    tests = REPO / "serverbench" / "tests"
    named = set(re.findall(r"-m pytest ((?:[a-z_]+/ )+)-q", text))
    packages = {part for group in named for part in group.split()}
    assert packages, "the gate names no pytest package; has its shape changed?"
    missing = sorted(p for p in packages if not (tests / p.rstrip("/")).is_dir())
    assert not missing, (
        f"correctness-gate.sh runs test package(s) that do not exist: "
        f"{', '.join(missing)}")


def test_a_failing_step_actually_stops_the_gate() -> None:
    """`fail` must exit non-zero, or the gate is a printed opinion."""
    text = _gate_text()
    assert re.search(r"^fail\(\)\s*\{[^}]*exit 1", text, re.MULTILINE), (
        "fail() has to exit non-zero. Note that a gate run through a pipe "
        "still looks successful unless the caller sets `set -o pipefail`.")
