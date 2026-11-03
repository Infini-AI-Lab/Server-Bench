"""How output correctness is checked never reaches an agent.

The agent is told correctness is verified privately and must build its own
verification. So no task package carries correctness code or answer data, and
no instruction or README names the method or the data it draws on.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from final_validate import package_grader

TASKS = Path(__file__).resolve().parents[3] / "serving_tasks"
LIVE = sorted(p for p in TASKS.glob("*") if (p / "contract.yaml").is_file())
LEAKS = re.compile(
    r"gsm8k|gsm 8k|math500|regret|teacher.forc|plain checkpoint|"
    r"plain-checkpoint|plain reference|reference-probability|agentx_reference",
    re.IGNORECASE)


@pytest.mark.parametrize("task", LIVE, ids=lambda p: p.name)
def test_no_correctness_module_or_pool_is_packaged(task):
    root = task / package_grader.GRADER_DIR
    assert root.is_dir()
    present = [n for n in package_grader.NEVER_PACKAGED if (root / n).exists()]
    assert present == []


@pytest.mark.parametrize("task", LIVE, ids=lambda p: p.name)
@pytest.mark.parametrize("name", ["instruction.md", "README.md"])
def test_documents_do_not_describe_the_method(task, name):
    text = (task / name).read_text("utf-8")
    assert not LEAKS.search(text), LEAKS.search(text).group(0)


def test_the_shared_list_never_names_a_correctness_module():
    assert not set(package_grader.SHARED_MODULES) & set(
        package_grader.NEVER_PACKAGED)


#: Words that describe HOW correctness is judged. The packaged evaluator is
#: read by the agent, so none may appear in any file of it -- names, comments
#: and docstrings included: a rule name, a limit or a sample size in a comment
#: tells an agent what to aim at.
PACKAGE_LEAKS = re.compile(
    r"regret|logprob|log-prob|teacher.forc|plain.checkpoint|plain.reference|"
    r"reference_fidelity|agentx_reference|gsm8k|math500|sample_size|"
    r"reference_arm|timed_agentx|eos_audit|padding.scan|stop.probe",
    re.IGNORECASE)


@pytest.mark.parametrize("task", LIVE, ids=lambda p: p.name)
def test_no_packaged_file_describes_the_method(task):
    root = task / package_grader.GRADER_DIR
    hits = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        for number, line in enumerate(
                path.read_text("utf-8", errors="ignore").splitlines(), 1):
            if PACKAGE_LEAKS.search(line):
                hits.append(f"{path.relative_to(task)}:{number}: {line.strip()[:90]}")
    assert not hits, "\n".join(hits[:20])


@pytest.mark.parametrize("task", LIVE, ids=lambda p: p.name)
def test_a_package_names_no_other_task(task):
    """A workspace carries its own task: no other task id, no other model."""
    import yaml
    from final_validate.spine import task_hash

    others = {name for p in LIVE if p != task
              for name in (p.name, task_hash(p.name))}
    others |= {f"T{n:03d}" for n in range(1, 60)}     # no ordinal task ids
    own_model = yaml.safe_load((task / "contract.yaml").read_text())[
        "token_prices"]["model"].split("/")[-1].lower()
    models = {yaml.safe_load((p / "contract.yaml").read_text())[
        "token_prices"]["model"].split("/")[-1].lower()
        for p in LIVE} - {own_model}
    hits = []
    for path in sorted((task / package_grader.GRADER_DIR).rglob("*.py")):
        text = path.read_text("utf-8", errors="ignore")
        for token in sorted(others):
            if re.search(rf"\b{token}\b", text):
                hits.append(f"{path.name}: {token}")
        for model in sorted(models):
            if model in text.lower():
                hits.append(f"{path.name}: {model}")
    assert not hits, hits[:20]
