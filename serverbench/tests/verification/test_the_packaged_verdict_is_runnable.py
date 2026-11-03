"""The verdict copy shipped to the agent must actually run.

It is shipped so the agent can check its own server against the real gates
before the clock runs out -- first-hour checks such as "does replay fidelity
hold" or "does this tree start at all". `scenario.py` does

    from final_validate import spine, tasks

and the copy ships as the package `final_verdict`, so the import has to work
under both package names.

The copy must stay BYTE-IDENTICAL to the source -- that identity is what makes
it the same logic -- so the import has to work under both names rather than be
rewritten on the way in.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "serverbench" / "harness"))

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
TASKS = REPO / "serving_tasks"
SOURCE = REPO / "serverbench" / "harness" / "final_validate"


def test_the_scenario_entry_point_imports_in_the_copy() -> None:
    task = "qwen3_5_9b_standard_2xL40"
    got = subprocess.run(
        [PY, "-m", "final_verdict.scenario", "--help"],
        cwd=str(TASKS / task / "tests"), capture_output=True, text=True,
        timeout=120)
    assert got.returncode == 0, (
        f"the packaged verdict cannot be run by the agent it is shipped "
        f"for:\n{(got.stderr or got.stdout)[-600:]}")
    assert "--endpoint" in got.stdout, (
        "the agent needs --endpoint to point it at its own running server")


def test_the_packaged_performance_path_imports_its_shared_policy() -> None:
    """The cheap correctness path never imports the performance policy."""
    task = "qwen3_5_9b_standard_2xL40"
    got = subprocess.run(
        [PY, "-c", (
            "from final_verdict import scenario; "
            "m=scenario._shared_agentx_module(); "
            "print(m.SCENARIO_DURATION_FLOOR_S)")],
        cwd=str(TASKS / task / "tests"), capture_output=True, text=True,
        timeout=120)
    assert got.returncode == 0, (
        "the packaged performance path cannot import its shared scenario "
        f"policy:\n{(got.stderr or got.stdout)[-600:]}")
    assert got.stdout.strip() == "900"


def test_packaged_performance_uses_agentx_wrapper_outside_candidate_venv(
        tmp_path: Path) -> None:
    """Harbor keeps AgentX separate from candidate serving dependencies."""
    task = "qwen3_5_9b_standard_2xL40"
    wrapper = tmp_path / "aiperf"
    wrapper.write_text("#!/bin/sh\nexit 0\n", "utf-8")
    wrapper.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    got = subprocess.run(
        [PY, "-S", "-c", (
            "from final_verdict import scenario; "
            "print(scenario._pinned_aiperf_argv(['aiperf','profile'])[0])")],
        cwd=str(TASKS / task / "tests"), capture_output=True, text=True,
        timeout=120, env=env)
    assert got.returncode == 0, (got.stderr or got.stdout)[-600:]
    assert Path(got.stdout.strip()).resolve() == wrapper.resolve()


def test_task_invocation_pins_the_tokenizer_revision() -> None:
    task = "qwen3_5_9b_standard_2xL40"
    probe = (
        "from final_verdict.tasks import "
        "qwen3_5_9b_standard_2xl40 as t; "
        "c=t.invocation('http://x',t.MODEL_ID,'/tmp/a',concurrency=2); "
        "print(c[c.index('--tokenizer')+1]); "
        "print(c[c.index('--tokenizer-revision')+1])")
    got = subprocess.run(
        [PY, "-c", probe], cwd=str(TASKS / task / "tests"),
        capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, (got.stderr or got.stdout)[-600:]
    assert got.stdout.splitlines() == [
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    ]


def test_the_source_still_runs_as_a_script() -> None:
    """Outside Harbor it is invoked as a path, not as a module."""
    got = subprocess.run([PY, str(SOURCE / "scenario.py"), "--help"],
                         cwd=str(REPO), capture_output=True, text=True,
                         timeout=120)
    assert got.returncode == 0, (got.stderr or got.stdout)[-600:]


def test_every_task_harbor_entry_refuses_rather_than_scoring() -> None:
    """The verdict runs on the host; Harbor's entry must never write a score."""
    for task in sorted(TASKS.glob("*")):
        if not (task / "contract.yaml").is_file():
            continue
        body = (task / "tests" / "test.sh").read_text("utf-8")
        assert "exit 1" in body and "reward" not in body.split("exit 1")[1]
        assert "final_validate/scenario.py" in body


def test_the_document_states_the_evaluators_interface() -> None:
    """Its two modes, not advice about when to use them.

    The document states the interface and what each mode measures; the agent
    decides when. An instruction about how to work ("Check CORRECTNESS early")
    does not belong in it.
    """
    import sys as _s
    from pathlib import Path as _PathType

    root = _PathType(__file__).resolve().parents[3]
    if str(root / "serverbench" / "harness") not in _s.path:
        _s.path.insert(0, str(root / "serverbench" / "harness"))
    from lifecycle import instructions

    body = instructions.build_brief(
        task_dir=root / "serving_tasks" / "qwen3_5_9b_standard_2xL40",
        model_id="", hours=8.0, carried=False)
    assert "python3 -m final_verdict.scenario" in body
    assert "--check-protocol-context" in body and "--measure-only" in body
    assert "Check CORRECTNESS early" not in body

