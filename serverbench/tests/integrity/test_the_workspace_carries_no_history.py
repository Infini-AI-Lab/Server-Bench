"""Nothing an agent can read tells it the harness's history or its timing.

Every file below is delivered into an evolve workspace, where the coding agent
reads it: the brief, the task's two data files as delivered, the starter, the
packaged evaluator, the workspace tools the harness writes, the blocking
waiters, the GPU-clock helper and the repository's own reference skills.

A comment in any of them is part of what the agent is told. A date says when
the suite was built or last changed; an anecdote about an earlier run, another
agent, a model, an operator decision, a host or another task tells the agent
about evidence it must not have -- what other systems did on this workload and
when. The technical reason a rule exists belongs in those files, stated as a
timeless fact; the episode that taught it does not.

Out of scope: the third-party submodules (`skills/ncu-report-skill`,
`skills_instructed/KernelWiki`), whose dates are public facts of separate
repositories, and dates inside public URLs (a blog post's slug). Host-only
modules that never reach a workspace are not scanned.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from lifecycle import delivery, workspace
from paths import REPO, SERVERBENCH

TASKS = REPO / "serving_tasks"
LIVE = sorted(p for p in TASKS.glob("*") if (p / "contract.yaml").is_file())

#: A calendar date or year-month: `2026-09-17`, `2026-09`.
DATE = re.compile(r"\b20\d\d-\d\d(?:-\d\d)?\b")

#: A public URL may carry a date in its slug; that is the page's address.
URL = re.compile(r"https?://\S+")

#: Agent models, hosts and users, and the shapes of a history anecdote.
LEAK = re.compile(
    r"\bGLM\b|deepseek|minimax|\bkimi\b|\bopus\b|\bsonnet\b|\bcodex\b|gemini|"
    r"lovelace|zhuominc|xinrui|"
    r"operator (?:instruction|text)[,:]|owner decision|"
    r"an agent found|\b(?:two|three|both) (?:evolves|agents)\b|"
    r"\bevolves? of\b|consecutive (?:full-window )?runs|one run made|"
    r"(?:found|observed|decided|retired|reviewed|pulled|accessed) (?:on )?20\d\d",
    re.IGNORECASE)

#: Another task's id. A task's own id is how its files name it.
TASK_ID = re.compile(r"\bT0\d\d\b")

#: Directories under a delivered tree that are build noise, never delivered.
NOISE = ("__pycache__", ".pytest_cache", ".git")

#: The repository's own skills, as delivered. Submodules are excluded above.
SKILLS = (
    SERVERBENCH / "skills" / "kernel-evolution",
    SERVERBENCH / "skills" / "serving-system-evolution",
    SERVERBENCH / "skills_instructed" / "kernel-evolution",
    SERVERBENCH / "skills_instructed" / "serving-system-evolution-instructed",
    SERVERBENCH / "skills_instructed" / "ncu-report-skill",
    SERVERBENCH / "skills_instructed" / "_overlays",
    SERVERBENCH / "skills_instructed" / "README.md",
)


def _files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and not any(n in p.parts for n in NOISE))


def _hits(label: str, text: str, *, own: str = "") -> list[str]:
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        bare = URL.sub("", line)
        reasons = [m.group(0) for m in DATE.finditer(bare)]
        reasons += [m.group(0) for m in LEAK.finditer(bare)]
        reasons += [m.group(0) for m in TASK_ID.finditer(bare)
                    if m.group(0) != own]
        if reasons:
            found.append(f"{label}:{number}: {reasons} {line.strip()[:100]}")
    return found


def _read(path: Path) -> str:
    return path.read_text("utf-8", errors="ignore")


@pytest.mark.parametrize("task", LIVE, ids=lambda p: p.name)
def test_the_task_files_a_workspace_receives_carry_no_history(task, tmp_path):
    own = task.name[:4]
    hits: list[str] = []
    for name in ("instruction.md", "contract.yaml"):
        hits += _hits(name, _read(task / name), own=own)
    # reference.yaml as it is delivered: its private block stripped.
    delivered = tmp_path / "reference.yaml"
    workspace._write_without(task / "reference.yaml", delivered,
                             workspace.PRIVATE_KEYS["reference.yaml"])
    hits += _hits("reference.yaml (delivered)", _read(delivered), own=own)
    for tree in ("demo", "tests"):
        for path in _files(task / tree):
            hits += _hits(str(path.relative_to(task)), _read(path), own=own)
    assert not hits, "\n".join(hits[:30])


def test_the_workspace_tools_carry_no_history(tmp_path):
    templates = {
        "version": workspace.SCRIPT,
        "checkpoint": workspace.CHECKPOINT_SCRIPT,
        "promote": workspace.PROMOTE_SCRIPT,
        "clock": workspace.CLOCK_SCRIPT,
        "full-test": workspace.FULL_TEST_SCRIPT,
        "measurements": workspace.MEASUREMENTS_SCRIPT,
        "wait/README.md": delivery.WAIT_README,
        "MAP.md": workspace._map(tmp_path, ["server.py", "serve.sh"]),
    }
    hits: list[str] = []
    for label, text in templates.items():
        hits += _hits(label, text)
    for path in _files(delivery.WAITERS):
        hits += _hits(f"wait/{path.name}", _read(path))
    gpu_clock = SERVERBENCH / "scripts" / "agent" / "gpu_clock.py"
    hits += _hits("gpu-clock", _read(gpu_clock))
    assert not hits, "\n".join(hits[:30])


@pytest.mark.parametrize("root", SKILLS,
                         ids=lambda p: str(p.relative_to(SERVERBENCH)))
def test_the_reference_skills_carry_no_history(root):
    assert root.exists(), f"{root} is delivered but missing"
    hits: list[str] = []
    for path in _files(root):
        hits += _hits(str(path.relative_to(SERVERBENCH)), _read(path))
    assert not hits, "\n".join(hits[:30])


def test_the_scan_sees_what_it_claims_to():
    """A pattern that matches nothing proves nothing; check it bites."""
    assert _hits("x", "# found on 2026-09-17, nine runs")
    assert _hits("x", "(operator instruction, 2026-09-30)")
    assert _hits("x", "both agents of that night", own="T001")
    assert _hits("x", "see T029", own="T001")
    assert not _hits("x", "this task is T001", own="T001")
    assert not _hits("x", "[blog](https://vllm.ai/blog/2025-01-27-v1)")
    assert not _hits("x", "certifi==2026.7.22")
