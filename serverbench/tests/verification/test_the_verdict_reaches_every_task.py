"""The published-version verdict has one entry point, and every task reaches it.

`measure_published.py` runs `final_validate/scenario.py` for every arm. Each
live task module must therefore define the `invocation()` that scenario offers,
and no second verdict path may come back beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

TOOLS = REPO / "serverbench" / "harness" / "final_validate" / "tools"
TASKS = REPO / "serving_tasks"


def _task_ids() -> list[str]:
    return sorted(p.name for p in TASKS.glob("*")
                  if (p / "contract.yaml").is_file())


def test_every_task_defines_the_invocation_the_verdict_offers() -> None:
    from final_validate import tasks

    missing = [t for t in _task_ids()
               if not hasattr(tasks.python_module(t), "invocation")]
    assert not missing, f"no invocation() for the scenario verdict: {missing}"


def test_the_published_version_verdict_has_one_entry_point() -> None:
    body = (TOOLS / "measure_published.py").read_text("utf-8")
    assert 'SCENARIO = PACKAGE / "scenario.py"' in body
    assert "validate.py" not in body
