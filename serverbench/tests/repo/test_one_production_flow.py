"""One production flow: `scripts/evolve` -> `harness/lifecycle`.

There is no second driver, no smoke-only variant of the entrypoint, and no
supervisor above it; the entrypoint itself holds no logic.
"""

from __future__ import annotations

import subprocess
import sys

from paths import REPO, SERVERBENCH

sys.path.insert(0, str(SERVERBENCH / "harness"))

ENTRYPOINT = SERVERBENCH / "scripts" / "evolve"


def test_the_entrypoint_is_the_only_thing_that_starts_a_run():
    """Exactly one file imports the lifecycle driver's `main`."""
    out = subprocess.run(
        ["git", "grep", "-l", "-E", r"from lifecycle\.run import|lifecycle\.run\.main",
         "--", "serverbench/scripts", "serverbench/harness"],
        cwd=REPO, capture_output=True, text=True)
    importers = sorted(p for p in out.stdout.split() if p)
    assert importers == ["serverbench/scripts/evolve"], (
        f"{importers} import the driver's entry function. There is one "
        f"production entrypoint; a second one is a second way for a run to "
        f"begin, and 'which launcher did this run come from' becomes a "
        f"question the evidence cannot answer."
    )


def test_the_entrypoint_holds_no_logic():
    """It exists to put the harness on the path and call one function."""
    body = [line for line in ENTRYPOINT.read_text("utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]
    assert len(body) < 40, (
        f"{ENTRYPOINT.name} has grown to {len(body)} lines of code. Every "
        f"decision belongs in harness/lifecycle/run.py, so that the entrypoint "
        f"cannot acquire a policy of its own."
    )
