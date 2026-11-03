"""Both task templates render a complete task that every reader accepts.

A task assembled by hand across many files drifts: statements disagree with
the contract they restate, and a new task can pass a table-driven test by not
being in its table.

`serving_tasks/_manage/new_task.py` renders the shared task data from
`_template/shared/` and the mode-specific files from `_template/<mode>/`, and
generates everything derived from them. This runs it for
both modes into a temporary tree and reads the result back through the
production readers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

TEMPLATES = REPO / "serving_tasks" / "_template"
NEW_TASK = REPO / "serving_tasks" / "_manage" / "new_task.py"

#: What a rendered task must carry, whichever mode it is for.
REQUIRED = ("contract.yaml", "reference.yaml", "task.toml", "MODE.md",
            "environment/Dockerfile")
SHARED = ("contract.yaml", "reference.yaml")
PER_MODE = ("task.toml", "MODE.md", "Dockerfile.in")


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(                                     # noqa: S603
        [sys.executable, str(NEW_TASK), *argv],
        capture_output=True, text=True, timeout=300, check=False,
        cwd=str(REPO))


def test_both_modes_have_a_template() -> None:
    for name in SHARED:
        assert (TEMPLATES / "shared" / name).is_file(), f"no shared {name}"
    for mode in ("docker", "modal"):
        for name in PER_MODE:
            assert (TEMPLATES / mode / name).is_file(), (
                f"the {mode} template has no {name}")
        for name in SHARED:
            assert not (TEMPLATES / mode / name).exists(), (
                f"{mode}/{name} duplicates the shared template")


@pytest.mark.parametrize("mode,suffix", [("docker", "2xL40"),
                                         ("modal", "2xL40S")])
@pytest.mark.parametrize("profile", ["standard", "instructed"])
def test_a_template_renders_with_no_placeholder_left(mode, suffix, profile) -> None:
    """A task created with a `{{...}}` still in it is not a task."""
    got = _run(
        "--mode", mode, "--profile", profile, "--task", f"probe_{profile}_{suffix}",
        "--model-id", "Qwen/Qwen3.5-9B", "--revision", "c" * 40,
        "--title", "Probe", "--question", "A probe.",
        "--usd-per-gpu-hour", "0.82", "--price-in", "0.1", "--price-out", "0.2",
        "--p90-ttft", "10000", "--p90-tpot", "60")
    assert got.returncode == 0, (got.stderr or got.stdout)[-600:]
    assert "GUIDANCE.md" not in got.stdout
    for name in REQUIRED:
        assert f"would write {name}" in got.stdout, (
            f"{mode}: the plan does not include {name}\n{got.stdout}")


def test_profile_cannot_silently_disagree_with_the_task_id():
    sys.path.insert(0, str(REPO / "serving_tasks/_manage"))
    from profiles import profile_of
    assert profile_of("probe_instructed_2xL40S") == "instructed"
    assert profile_of("probe_standard_modal_2xL40S") == "standard"
    assert profile_of("probe_instructed_modal_2xL40S") == "instructed"
    with pytest.raises(ValueError, match="disagrees"):
        profile_of("probe_standard_2xL40", "instructed")


def test_every_new_task_is_request_scored_p90_only() -> None:
    """Every task is born request_p90_billable_tokens_v4, p90 alone."""
    got = _run(
        "--mode", "docker", "--task", "probe_instructed_2xL40",
        "--model-id", "Qwen/Qwen3.5-9B", "--revision", "c" * 40,
        "--title", "Probe", "--question", "A probe.",
        "--derived-from", "probe_source_instructed_2xL40",
        "--usd-per-gpu-hour", "0.82", "--price-in", "0.1", "--price-out", "0.2",
        "--p90-ttft", "10000", "--p90-tpot", "60")
    assert got.returncode == 0, (got.stderr or got.stdout)[-600:]
    source = NEW_TASK.read_text("utf-8")
    assert "kind: request_p90_billable_tokens_v4" in source
    assert "gross_objective: gross_dollars_per_dollar" in source
    assert "p95_both" not in source.split("def substitutions", 1)[1].split(
        "def render", 1)[0]
    assert 'CONCURRENCY_POLICY = "candidate"' in source
    assert "RANDOM_SEED = None" in source
    body = (TEMPLATES / "shared" / "contract.yaml").read_text("utf-8")
    assert "objective: dollars_per_dollar" in body
    assert "{{REQUEST_SCORING}}" in body
    assert "percentile: 95" not in body and "P95" not in body
    reference = (TEMPLATES / "shared" / "reference.yaml").read_text("utf-8")
    assert "eos_audit: true" in reference
    for mode in ("docker", "modal"):
        toml = (TEMPLATES / mode / "task.toml").read_text("utf-8")
        assert "gate_p95" not in toml and "request_p90_billable_tokens_v4" in toml


def test_a_new_task_refuses_a_p95_row() -> None:
    got = _run(
        "--mode", "docker", "--task", "probe_instructed_2xL40",
        "--model-id", "Qwen/Qwen3.5-9B", "--revision", "c" * 40,
        "--title", "Probe", "--question", "A probe.",
        "--usd-per-gpu-hour", "0.82", "--price-in", "0.1", "--price-out", "0.2",
        "--p90-ttft", "10000", "--p90-tpot", "60",
        "--p95-ttft", "30000", "--p95-tpot", "70")
    assert got.returncode != 0
    assert "unrecognized arguments: --p95-ttft" in (got.stdout + got.stderr)


def test_built_modal_tasks_have_no_inherited_measurement():
    """Modal L40S tasks are uncalibrated: no load, and no reference load."""
    import yaml
    from final_validate import tasks
    for task in ("qwen3_5_9b_standard_modal_2xL40S",
                 "gemma4_e4b_it_standard_modal_1xL40S"):
        reference = yaml.safe_load((REPO / "serving_tasks" / task / "reference.yaml").read_text())
        assert "reference_accuracy" not in reference["correctness"]
        assert "concurrency_search" not in reference
        module = tasks.python_module(task)
        assert module.OFFERED_CONCURRENCY is None
        assert module.REFERENCE_CONCURRENCY is None
        assert not (REPO / "serving_tasks" / task / "GUIDANCE.md").exists()


def test_modal_refuses_a_card_modal_does_not_offer() -> None:
    """`_1xL40` cannot run on Modal, and the refusal says which mode can.

    `modalctl.gpu_spec` requests the card by the name in the task id and
    refuses anything it does not know, so a modal-mode task named after a
    local card would fail at setup rather than at authoring time.
    """
    got = _run(
        "--mode", "modal", "--task", "probe_standard_1xL40",
        "--model-id", "Qwen/Qwen3.5-9B", "--revision", "c" * 40,
        "--title", "Probe", "--question", "A probe.",
        "--usd-per-gpu-hour", "0.82", "--price-in", "0.1", "--price-out", "0.2",
        "--p90-ttft", "10000", "--p90-tpot", "60")
    assert got.returncode != 0
    assert "Modal does not offer" in (got.stdout + got.stderr)
    assert "--mode docker" in (got.stdout + got.stderr)


def test_a_task_id_without_hardware_is_refused() -> None:
    got = _run(
        "--mode", "docker", "--task", "probe_standard",
        "--model-id", "Qwen/Qwen3.5-9B", "--revision", "c" * 40,
        "--title", "Probe", "--question", "A probe.",
        "--usd-per-gpu-hour", "0.82", "--price-in", "0.1", "--price-out", "0.2",
        "--p90-ttft", "10000", "--p90-tpot", "60")
    assert got.returncode != 0
    assert "does not end in its hardware" in (got.stdout + got.stderr)


def test_a_task_id_with_an_ordinal_is_refused() -> None:
    """A task is named by what it is; its hash, not a position, is its handle."""
    got = _run(
        "--mode", "docker", "--task", "T1_probe_standard_2xL40",
        "--model-id", "Qwen/Qwen3.5-9B", "--revision", "c" * 40,
        "--title", "Probe", "--question", "A probe.",
        "--usd-per-gpu-hour", "0.82", "--price-in", "0.1", "--price-out", "0.2",
        "--p90-ttft", "10000", "--p90-tpot", "60")
    assert got.returncode != 0
    assert "drop the ordinal prefix" in (got.stdout + got.stderr)


@pytest.mark.parametrize("mode", ("docker", "modal"))
def test_the_mode_document_says_which_mode_it_is(mode) -> None:
    """A reader who opens only `MODE.md` learns where the container comes from."""
    body = (TEMPLATES / mode / "MODE.md").read_text("utf-8")
    assert body.startswith(f"# {mode.capitalize()} mode")
    if mode == "docker":
        assert "removed at teardown" in body
        assert "relays HTTPS to that one provider" in body
        assert "modal" not in body.lower().replace("modal mode", "")
    else:
        assert "Modal sandbox" in body
        assert "Only this run's sandboxes" in body
        assert "kept by\ndefault" in body
        assert "purge refuses" in body


def test_the_templates_carry_no_measured_evidence() -> None:
    """A new task has measured nothing, and its reference must not pretend to.

    `reference_accuracy` and `concurrency_search` are the two things that can
    only come from a run on the task's own hardware. A template that shipped
    either would hand a new task a number from somebody else's cards.
    """
    body = (TEMPLATES / "shared" / "reference.yaml").read_text("utf-8")
    for line in body.splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert "reference_accuracy" not in line
        assert "concurrency_search" not in line
        assert "adopted" not in line
