"""Where a task's files and parameters live.

A task's data is `contract.yaml`, `reference.yaml` and its harness module;
everything else in the task directory is generated from them (see
`taskctl.py`). This module names those places:

* the task and module paths, and the list of live tasks;
* `SCALARS`, the one key each `taskctl set` parameter is written to;
* which files are generated and which evidence trees are never rewritten;
* `verdict_drift`, the check that each task's packaged evaluator is
  byte-identical to `serverbench/harness/final_validate/`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Where the tasks live. `SERVERBENCH_TASK_ROOT` points the tooling at a
#: disposable copy, so a parameter change can be tested without touching the
#: tree live runs read.
TASKS = Path(os.environ.get("SERVERBENCH_TASK_ROOT")
             or REPO / "serving_tasks").resolve()

MODULES = REPO / "serverbench" / "harness" / "final_validate" / "tasks"

#: The package the evaluator copy under `tests/final_verdict/` is built from.
VERDICT_SOURCE = REPO / "serverbench" / "harness" / "final_validate"

#: Generated inside a task and never edited there.
GENERATED = ("tests/final_verdict", "tests/replay-verify")

#: Suffixes of the text files a task id is rewritten in by `taskctl rename`.
PROSE_SUFFIXES = (".md", ".sh", ".py", ".yaml", ".toml", ".json")

#: Trees that record runs and measurements. A run under a task id stays under
#: that id, so a rename never rewrites them.
EVIDENCE = (
    "serverbench/artifacts",
    "serverbench/evolve_space",
    "server_output",
    "serving_tasks/_concurrency_search_archive",
    "evaluation_progress",
)

#: Where a task id is named rather than recorded, and must follow a rename.
REFERRERS = (
    "serverbench/tests",
    "serverbench/harness/final_validate",
    "serverbench/harness",
    "serverbench/scripts",
    "serverbench/docs",
    "serverbench/README.md",
    "serverbench/OPERATIONS.md",
    "serving_tasks/README.md",
    "CLAUDE.md",
)


def task_dir(task_id: str) -> Path:
    return TASKS / task_id


def module_path(task_id: str) -> Path:
    """The harness module for a task, named in lower case."""
    return MODULES / f"{task_id.lower()}.py"


def live_tasks() -> list[str]:
    # The rule of `final_validate.spine.is_task_dir`: a directory with a
    # contract is a task; `_*` holds none.
    return sorted(p.name for p in TASKS.iterdir()
                  if p.is_dir() and not p.name.startswith(("_", "."))
                  and (p / "contract.yaml").is_file())


def is_generated(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(rel == g or rel.startswith(g + "/") for g in GENERATED)


def authored_files(task_id: str) -> list[Path]:
    """Every hand-authored text file of a task, plus its module."""
    root = task_dir(task_id)
    out = [p for p in sorted(root.rglob("*"))
           if p.is_file() and p.suffix in PROSE_SUFFIXES
           and not is_generated(p, root)
           and ".pytest_cache" not in p.parts
           and "__pycache__" not in p.parts]
    if module_path(task_id).is_file():
        out.append(module_path(task_id))
    return out


def references(task_id: str) -> list[Path]:
    """Files outside the task that name it, excluding recorded evidence."""
    out: list[Path] = []
    lowered = task_id.lower()
    for where in REFERRERS:
        root = REPO / where
        paths = [root] if root.is_file() else (
            sorted(root.rglob("*")) if root.is_dir() else [])
        for path in paths:
            if not path.is_file() or path.suffix not in PROSE_SUFFIXES:
                continue
            rel = path.relative_to(REPO).as_posix()
            if any(rel.startswith(e + "/") for e in EVIDENCE):
                continue
            if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            try:
                body = path.read_text("utf-8", errors="ignore")
            except OSError:
                continue
            if task_id in body or lowered in body:
                out.append(path)
    return out


def verdict_drift(task_id: str) -> list[str]:
    """Where the packaged evaluator differs from `serverbench/harness/final_validate/`.

    A candidate judged in the container and on the host must be judged by the
    same code. The one allowed difference is the generated `__init__.py`.
    """
    inside = task_dir(task_id) / "tests" / "final_verdict"
    if not inside.is_dir():
        return [f"{task_id} carries no tests/final_verdict"]

    import sys
    for path in (str(REPO / "serverbench" / "harness"),):
        if path not in sys.path:
            sys.path.insert(0, path)
    from final_validate import package_grader

    bad: list[str] = []
    replay_entry = task_dir(task_id) / "tests" / "replay-verify"
    if not replay_entry.is_file():
        bad.append(f"{task_id}/tests/replay-verify is missing")
    elif replay_entry.read_text("utf-8") != package_grader.REPLAY_VERIFY_SH.format(
            task=task_id):
        bad.append(f"{task_id}/tests/replay-verify differs from the generated "
                   f"entry point")
    elif not replay_entry.stat().st_mode & 0o111:
        bad.append(f"{task_id}/tests/replay-verify is not executable")

    for copy in sorted(inside.rglob("*.py")):
        rel = copy.relative_to(inside)
        source = VERDICT_SOURCE / rel
        if not source.is_file():
            bad.append(f"{task_id}/tests/final_verdict/{rel} has no source in "
                       f"final_validate/")
            continue
        theirs = copy.read_bytes()
        if theirs == source.read_bytes():
            continue
        if rel.name == "__init__.py" and len(theirs) < 200 and b'"""' in theirs:
            continue
        bad.append(f"{task_id}/tests/final_verdict/{rel} differs from "
                   f"serverbench/harness/final_validate/{rel}; run `taskctl render`")
    return bad


# ------------------------------------------------------------------ scalars

@dataclass(frozen=True)
class Structured:
    """One key whose value is a parameter."""

    #: Relative to the task directory, or "MODULE" for the harness module.
    file: str
    key: str
    indent: int
    #: The enclosing key, where one key name appears under several parents.
    under: str | None = None
    #: "yaml" keys are written `key: value`; "python" keys `key = value`.
    syntax: str = "yaml"

    def path(self, task_id: str) -> Path:
        return (module_path(task_id) if self.file == "MODULE"
                else task_dir(task_id) / self.file)


def _contract(key: str, under: str | None = None) -> Structured:
    return Structured("contract.yaml", key, 2 if under else 0, under=under)


def _module(key: str) -> Structured:
    return Structured("MODULE", key, 0, syntax="python")


#: Every `taskctl set` parameter and the one place it is written. The p90 bar
#: itself is a list item and is written by `taskctl._write_gates`.
SCALARS: dict[str, tuple[Structured, ...]] = {
    "usd_per_gpu_hour": (_contract("usd_per_gpu_hour"),),
    "usd_per_million_input": (_contract("usd_per_million_input", "token_prices"),),
    "usd_per_million_output": (_contract("usd_per_million_output", "token_prices"),),
    "usd_per_million_cached_input": (
        _contract("usd_per_million_cached_input", "token_prices"),),
    "cache_rate_is": (_contract("cache_rate_is", "token_prices"),),
    "price_model": (_contract("model", "token_prices"),),
    "price_source": (_contract("source", "token_prices"),),
    "reference_concurrency": (_module("REFERENCE_CONCURRENCY"),),
    "model_id": (_module("MODEL_ID"),),
    "model_revision": (_module("MODEL_REVISION"),),
    "max_context_length": (_module("MAX_CONTEXT_LENGTH"),),
    "required_context_length": (_module("REQUIRED_CONTEXT_LENGTH"),),
    "answer_sample_size": (Structured("reference.yaml", "answer_sample_size", 2,
                                      under="correctness"),),
    "max_absolute_accuracy_drop": (Structured("reference.yaml", "max_absolute_accuracy_drop",
                                         2, under="correctness"),),
    "max_token_regret": (Structured("reference.yaml", "max_token_regret", 2,
                                    under="correctness"),),
    "reference_arm": (Structured("reference.yaml", "reference_arm", 2,
                                 under="correctness"),),
}


def formatted(value: float, like: str) -> str:
    """Render `value` the way `like` is written: commas, underscores, or bare.

    A fractional bar (a 7.7 ms TPOT) is written as it is; rounding it to a
    whole millisecond would move the bar.
    """
    if float(value) != round(value):
        return f"{float(value):g}"
    whole = round(value)
    if "," in like:
        return f"{whole:,}"
    if "_" in like:
        return f"{whole:_}"
    return str(whole)
