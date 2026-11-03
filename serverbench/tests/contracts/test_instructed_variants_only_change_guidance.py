"""The instructed cells change guidance, never the measured experiment."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "serving_tasks"

PAIRS = (
    (
        "qwen3_5_2b_standard_1xL40",
        "qwen3_5_2b_instructed_1xL40",
        "Qwen3.5 2B Standard",
        "Qwen3.5 2B Instructed",
    ),
    (
        "qwen3_5_9b_standard_2xL40",
        "qwen3_5_9b_instructed_2xL40",
        "Qwen3.5 9B Standard",
        "Qwen3.5 9B Instructed",
    ),
    (
        "gemma4_e4b_it_standard_1xL40",
        "gemma4_e4b_it_instructed_1xL40",
        "Gemma 4 E4B IT Standard",
        "Gemma 4 E4B IT Instructed",
    ),
)


def task_hash(task_id: str) -> str:
    from final_validate.spine import task_hash as of

    return of(task_id)


def _files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def _canonical_paths(root: Path, task_id: str) -> dict[str, Path]:
    """Map task-specific module names onto one comparable identity."""
    # `agents/` is guidance too: the instructed team's subagent definitions.
    return {
        str(relative).replace(task_id.lower(), "<task-id>"): relative
        for relative in _files(root) if relative.parts[0] != "agents"
    }


@pytest.mark.parametrize("standard,instructed,standard_title,instructed_title", PAIRS)
def test_task_payload_is_identical_after_identity_and_guidance_are_removed(
        standard: str, instructed: str,
        standard_title: str, instructed_title: str) -> None:
    """Model, scenario, gates, starter, environment and verdict stay equal."""
    left = TASKS / standard
    right = TASKS / instructed
    left_paths = _canonical_paths(left, standard)
    right_paths = _canonical_paths(right, instructed)
    assert left_paths.keys() == right_paths.keys()

    for canonical in sorted(left_paths):
        relative = left_paths[canonical]
        other = right_paths[canonical]
        if relative in {Path("instruction.md"), Path("README.md")}:
            continue
        if canonical.startswith("tests/final_verdict/tasks/<task-id>"):
            continue
        expected = (left / relative).read_text("utf-8")
        expected = expected.replace(standard, instructed)
        expected = expected.replace(standard_title, instructed_title)
        expected = expected.replace(task_hash(standard), task_hash(instructed))
        # These are task identity labels, not experiment parameters.
        if (canonical.startswith("demo/")
                or canonical == "tests/test_public.py"):
            expected = expected.replace(standard[:4], instructed[:4])
        actual = (right / other).read_text("utf-8")
        if canonical == "contract.yaml":
            # `derived_from` is provenance, which each task states for itself.
            expected, actual = (
                "".join(line for line in text.splitlines(keepends=True)
                        if not line.lstrip().startswith("derived_from:"))
                for text in (expected, actual))
        assert actual == expected, canonical


@pytest.mark.parametrize("standard,instructed,standard_title,instructed_title", PAIRS)
def test_final_validator_differs_only_in_task_identity(
        standard: str, instructed: str,
        standard_title: str, instructed_title: str) -> None:
    del standard_title, instructed_title
    from final_validate import tasks

    standard_module = tasks.python_module(standard)
    instructed_module = tasks.python_module(instructed)

    for name in (
            # PROFILE is the task title -- identity, like the task id, and
            # allowed to differ.
            "MODEL_ID", "MODEL_REVISION", "OFFERED_CONCURRENCY",
            "CONCURRENCY_POLICY", "REFERENCE_CONCURRENCY",
            "SCENARIO", "CORPUS", "CORPUS_SHA256", "ARRIVAL_PROCESS",
            "VERDICT_DURATION_S", "EVOLVE_DURATION_S", "RANDOM_SEED",
            "ENVIRONMENT"):
        assert getattr(standard_module, name) == getattr(instructed_module, name)
    for optional in ("MAX_CONTEXT_LENGTH", "REQUIRED_CONTEXT_LENGTH"):
        assert getattr(standard_module, optional, None) == getattr(
            instructed_module, optional, None)
    # Identical apart from the freshly drawn workload seed.
    def without_seed(cmd):
        at = cmd.index("--random-seed")
        return cmd[:at] + cmd[at + 2:]

    assert without_seed(standard_module.invocation("u", "m", "a", concurrency=3)) == (
        without_seed(instructed_module.invocation("u", "m", "a", concurrency=3)))


@pytest.mark.parametrize("standard,instructed,standard_title,instructed_title", PAIRS)
def test_instructed_prompt_is_a_real_distinct_treatment(
        standard: str, instructed: str,
        standard_title: str, instructed_title: str) -> None:
    del standard_title, instructed_title
    ordinary = (TASKS / standard / "instruction.md").read_text("utf-8")
    coached = (TASKS / instructed / "instruction.md").read_text("utf-8")
    assert ordinary != coached
    assert "## Rapid-feedback serving evolution" not in ordinary
    assert "## Rapid-feedback serving evolution" in coached
    assert "knowledge_instructed/" not in ordinary
    assert "knowledge_instructed/" not in coached
    assert ".claude/skills/serving-system-evolution-instructed/" in coached
    assert ".claude/skills/KernelWiki/" not in ordinary
    assert ".claude/skills/KernelWiki/" in coached
