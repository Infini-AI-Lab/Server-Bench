"""Workspace publication and agent-facing brief contracts."""

from __future__ import annotations

import sys

from paths import SERVERBENCH

if str(SERVERBENCH / "harness") not in sys.path:
    sys.path.insert(0, str(SERVERBENCH / "harness"))

from lifecycle import instructions  # noqa: E402

TASK = (SERVERBENCH.parent / "serving_tasks"
        / "qwen3_5_2b_standard_1xL40")

# --------------------------------------------- what the workspace hands over
def test_a_modular_serving_system_is_captured_whole(tmp_path):
    """The brief tells the agent to split scheduler, KV cache and kernels into
    modules, so capture follows source at any depth: a file it misses, such as
    `engine/attention/flash.py`, is not published and is not there when the
    tree's own server is started.
    """
    from lifecycle import workspace as ws_mod

    ws = tmp_path / "ws"
    for rel in ("server.py", "serve.sh", "engine/scheduler.py",
                "engine/kv_cache.py", "engine/attention/flash.py",
                "kernels/triton/decode.py", "api/openai/chat.py"):
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text("mine\n")
    got = set(ws_mod.candidate_manifest(ws))
    assert "engine/attention/flash.py" in got
    assert "kernels/triton/decode.py" in got
    assert "api/openai/chat.py" in got
    assert len(got) == 7, sorted(got)


def test_a_vendored_environment_is_still_not_candidate_source(tmp_path):
    """Capture at any depth still excludes virtualenvs, node_modules and
    bytecode caches: they are installed, not written by the agent."""
    from lifecycle import workspace as ws_mod

    ws = tmp_path / "ws"
    (ws / "server.py").parent.mkdir(parents=True, exist_ok=True)
    (ws / "server.py").write_text("mine\n")
    for rel in (".venv/lib/python3.12/site-packages/torch/__init__.py",
                "node_modules/a/b/c.py", "__pycache__/x/y/z.py"):
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text("theirs\n")
    assert set(ws_mod.candidate_manifest(ws)) == {"server.py"}


def test_delivered_task_and_benchmark_are_not_published_as_candidate_source(
        tmp_path):
    """What the harness delivers (`microbench/`, `task_context/`) is mapped for
    the agent but never captured as the candidate's own source."""
    from lifecycle import workspace as ws_mod

    ws = tmp_path / "ws"
    for rel in ("server.py", "microbench/microbench-inputs.json",
                "task_context/qwen3_5_9b_standard_2xL40/task.yaml"):
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text("delivered\n")
    assert set(ws_mod.candidate_manifest(ws)) == {"server.py"}

    rendered = ws_mod._map(ws, ws_mod.candidate_manifest(ws))
    assert "`microbench/` --" in rendered
    assert "`task_context/` --" in rendered
    assert "`measure/`" not in rendered
    assert "`baselines/`" not in rendered


def test_the_agent_has_a_live_clock_that_is_not_candidate_source(tmp_path):
    """The budget is inspectable during the run, not frozen in the brief."""
    import subprocess

    from lifecycle import workspace as ws_mod

    ws = tmp_path / "ws"
    ws.mkdir()
    clock = ws_mod.deliver_clock(ws, "2099-01-02T03:04:05Z")
    got = subprocess.run([str(clock)], check=True, capture_output=True,
                         text=True).stdout
    assert "now_utc=" in got
    assert "deadline_utc=2099-01-02T03:04:05Z" in got
    assert "remaining_seconds=" in got
    assert "status=RUNNING" in got
    assert "clock" not in ws_mod.candidate_manifest(ws)


def test_the_brief_tells_the_agent_how_to_read_its_remaining_time():
    """The brief points at `./clock`, the live source of the remaining budget."""
    text = instructions.build_brief(
        task_dir=TASK, model_id="Qwen/Qwen3.5-9B", hours=2.0,
        carried=False)
    assert "`./clock`" in text
    assert "remaining budget" in text


def test_the_instructed_brief_routes_work_without_repeating_the_skill():
    """The compact brief routes evidence; detailed architecture stays in skill."""
    import re as _re

    from lifecycle import instructions as brief_mod
    task = (SERVERBENCH.parent / "serving_tasks" /
            "qwen3_5_9b_instructed_2xL40")
    assert task.is_dir(), task
    flat = _re.sub(r"\s+", " ", brief_mod.build_brief(
        task_dir=task, model_id="Qwen/Qwen3.5-9B", hours=2.0,
        carried=False))
    for shown in ("lightweight replay gate",
                  "--check-protocol-context", "./full-test --duration N --wait",
                  "delivered snapshot", "1800-second", "extreme-length"):
        assert shown in flat, f"the instructed evidence routing lost {shown}"
    assert "serving-system-evolution-instructed" in flat


def test_the_document_names_nothing_that_is_not_delivered():
    """The inventory is what the workspace HAS, in both directions.

    A document that names a directory the workspace does not have sends the
    agent looking for something that is not there; a delivered directory the
    document does not name goes unused.
    """
    body = _brief()
    for absent in ("knowledge/", "kernel_evolve/", "MISSION.md", "DEMO.md"):
        assert absent not in body, (
            f"the document names {absent!r}, which is not delivered into a "
            f"workspace")
    # The converse: the reference skills are delivered, so they are named.
    assert ".claude/skills/" in body, (
        "the workspace carries reference skills and the document does not "
        "say so")


def test_the_run_plan_defaults_no_reference_material(tmp_path):
    """Reference material reaches a workspace only when a run names it; none is
    defaulted from the repository."""
    from lifecycle.run import RunPlan

    plan = RunPlan(task_dir=tmp_path, repo=SERVERBENCH.parent, gpus=(0,),
                   hours=1.0, image="x", model_dir=tmp_path,
                   env_input=tmp_path, lineage_root=tmp_path)
    for field in ("skill_dir", "kernel_skill_dir", "kernelwiki_skill_dir",
                  "kernelwiki_overlay_dir", "ncu_skill_dir", "agents_dir"):
        assert getattr(plan, field) is None, (
            f"{field} has a default, so every workspace would receive it")


def test_no_correctness_pool_reaches_the_workspace(tmp_path):
    """No answer-bearing pool is delivered, public half or not.

    How output correctness is judged -- and with which benchmark -- is
    private. A run plan has no field for a pool, and delivery removes any pool
    directory a package carries.
    """
    from lifecycle import run as run_mod
    from lifecycle import workspace as ws_mod

    assert "development_samples" not in run_mod.RunPlan.__dataclass_fields__
    boot = (SERVERBENCH / "harness" / "lifecycle" / "workspace.py").read_text(
        "utf-8")
    assert "no correctness code or answer pool reached the workspace" in boot
    assert "pool" in boot[boot.index("no correctness code or answer pool"):][:400]
    assert ws_mod.boot_contract


# ------------------------------------------------ what a published version is
def test_a_published_version_can_be_measured_again(tmp_path):
    """A published version carries its reproduction kit.

    Source and numbers alone cannot be checked; with the kit, somebody holding
    the version directory can run the workload against it again.
    """
    from lifecycle.boundary import REPRODUCTION_KIT, _deliver_reproduction_kit

    ws, out = tmp_path / "ws", tmp_path / "v1"
    out.mkdir(parents=True)
    for name in REPRODUCTION_KIT:
        (ws / name).mkdir(parents=True, exist_ok=True)
        (ws / name / "thing.txt").write_text("delivered\n")
    (ws / "server.py").write_text("mine\n")

    _deliver_reproduction_kit(out, ws)
    for name in REPRODUCTION_KIT:
        assert (out / name / "thing.txt").is_file(), f"{name} was not published"
    assert not (out / "toolkit").exists(), (
        "the measurement implementation must stay on the harness side; the "
        "version gets the client, which is the point of handing one out")


def test_a_missing_kit_directory_does_not_fail_the_publication(tmp_path):
    """A version missing part of its kit is still a version."""
    from lifecycle.boundary import _deliver_reproduction_kit

    ws, out = tmp_path / "ws", tmp_path / "v1"
    ws.mkdir()
    out.mkdir()
    (ws / "measure").mkdir()
    (ws / "measure" / "bench").write_text("client\n")
    _deliver_reproduction_kit(out, ws)          # no tests/, baselines/, wait/
    assert (out / "measure" / "bench").is_file()


# ---------------------------------------------------- what the brief advises
#
# A brief is rendered from the harness and the task, and a prompt defect is a
# claim about the harness that is not true. These assert the claims that are
# cheapest to get wrong.
class TheBriefAdvisesWhatTheImageCanActuallyDo:
    """Grouping only; pytest collects the functions below."""


def _brief(name="qwen3_5_9b_standard_2xL40"):
    """The brief of a live task, `qwen3_5_9b_standard_2xL40` unless a test
    names another."""
    import sys

    from paths import REPO, SERVERBENCH
    if str(SERVERBENCH / "harness") not in sys.path:
        sys.path.insert(0, str(SERVERBENCH / "harness"))
    task_dir = REPO / "serving_tasks" / name
    # `build_brief` does not raise on a missing directory -- it renders a brief
    # with every task-derived field blank -- so a renamed task must fail here,
    # where the subject is known.
    assert task_dir.is_dir(), f"{name} is not a live task"
    return instructions.build_brief(task_dir=task_dir,
                                    model_id="Qwen/Qwen3-4B", hours=2.0,
                                    carried=False)


def test_the_document_names_this_task_s_own_metric_and_operating_point():
    """Read from the contract, so neither can be a retyped stale value."""
    text = _brief()
    head = text.index("## The goal")
    goal = text[head:text.index("## The p90 bars")]
    assert "`dollars_per_dollar`" in goal
    assert "`gross_dollars_per_dollar`" in goal
    # The load is the candidate's; the SGLang point is stated, not imposed.
    assert "declared by the candidate in `concurrency.json`" in text
    from final_validate import task_spec
    spec = task_spec.read("qwen3_5_9b_standard_2xL40")
    p90 = next(level for level in spec.service_levels if level.percentile == 90)
    assert f"offered session concurrency **{spec.reference_concurrency}**" in text
    assert f"<= **{float(p90.tpot_ms):g} ms**" in text
    assert "None" not in goal


def test_the_document_states_that_the_three_hard_gates_bind():
    """The replay, EOS and correctness gates are stated as three separate hard
    gates, never folded into one "correctness", and latency is not a gate."""
    flat = _flat(_brief())
    assert ("The replay gate, the EOS gate and the correctness gate are three separate "
            "hard gates") in flat
    assert "Latency is not a hard gate" in flat
    assert "## The three hard gates" in flat
    for heading in ("**1. The replay gate.**", "**2. The EOS (end-of-sequence) gate.**",
                    "**3. The correctness gate.**"):
        assert heading in flat, heading
    assert "report the result on its own `EOS GATE` line" in flat
    assert "Output correctness is verified privately" in flat
    assert "A complete AgentX replay measures the score" in flat
    assert "The verdict's window is 1800 s" in flat
    assert ("a candidate that fails any one of them is excluded and scores nothing, "
            "whatever its speed") in flat


def _flat(text: str) -> str:
    """Prose with its line breaks collapsed.

    These guards match sentences, and a brief is wrapped for reading, so a
    phrase would be "absent" whenever a wrap lands in the middle of it, after
    an edit that changes no meaning.
    """
    return " ".join(text.split())


def test_the_document_states_that_correctness_is_evaluated_first() -> None:
    """The FACT about how the verdict reads the evidence.

    Not an order of work: the document does not tell the agent what to do
    first. It says which gate the verdict reads first, which is a property of
    the measurement.
    """
    flat = _flat(_brief())
    assert "They are evaluated before the score." in flat
    assert ("reads the correctness and replay-fidelity hard gates, then scores "
            "every profiling request against the p90 bars") in flat


def test_the_document_describes_the_publication_contract() -> None:
    """What publication does and does not do, and what gets judged."""
    flat = " ".join(_brief().split())
    assert "freezes the working tree as an immutable version" in flat
    assert "runs no evaluation" in flat
    assert "returns no verdict or number" in flat
    assert "measures the **final published head** and nothing else" in flat
    assert "captures a changed working tree once" in flat
    assert "BOUNDARY_CHECKS.md" in flat


def test_an_unknown_stop_condition_is_rejected() -> None:
    """`--stop` is validated against the allowed values and nothing else.

    STOP_CONDITIONS is the set of allowed values, not prose for the brief; the
    chosen one is recorded in the run's evidence.
    """
    assert "not-a-real-condition" not in instructions.STOP_CONDITIONS
    assert set(instructions.STOP_CONDITIONS) == {
        "beat-all-baselines", "beat-one-baseline", "meet-slo", "timebox"}
    driver = (SERVERBENCH / "harness" / "lifecycle" / "run.py").read_text("utf-8")
    assert "choices=sorted(instructions_mod.STOP_CONDITIONS)" in driver
