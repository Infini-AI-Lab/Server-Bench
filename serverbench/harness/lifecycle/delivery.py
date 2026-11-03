"""Copy the run's inputs into the workspace and record every delivered file.

What a workspace receives besides the task data and the brief:

    seed/                      the starting source (promoted to the root later)
    tests/                     the task's packaged evaluator
    .claude/agents/            subagent definitions, when the run supplies them
    .claude/settings.json      Claude Code project files (settings, hooks/), when
    .claude/hooks/             the run supplies them
    .claude/skills/<name>/     the serving, profiler-report and kernel skills,
                               with any file carrying upstream serving-framework
                               source held back
    .claude/skills/KernelWiki/ the pinned KernelWiki, unabridged, plus overlay
    wait/                      the blocking waiters

`deliver` returns `{workspace-relative path: sha256}` for everything it copied
and `{destination: why}` for everything it held back, so what a run received is
a record rather than an assumption.
"""

from __future__ import annotations

import contextlib
import dataclasses
import shutil
from pathlib import Path

from serving_evolve.hashing import digest_file

#: Import statements that mark a file as upstream serving-framework source.
#: Import lines rather than bare names: prose that discusses a framework stays.
UPSTREAM_MARKERS = ("from sglang", "import sglang", "from vllm", "import vllm",
                    "import tensorrt_llm", "from tensorrt_llm",
                    "import sgl_kernel", "from sgl_kernel")

WAITERS = Path(__file__).resolve().parent / "waiting"

# Keep benchmark-specific clock guidance in this repository, not in a dirty
# upstream submodule. Custom skill sources are left untouched.
NCU_SKILL = Path(__file__).resolve().parents[2] / "skills" / "ncu-report-skill"
NCU_OVERLAY = NCU_SKILL.parent / "_overlays" / "ncu-report-skill"

#: The waiters' own README, delivered beside them as `wait/README.md`.
WAIT_README = (WAITERS / "README.md").read_text("utf-8")


@dataclasses.dataclass(frozen=True)
class Sources:
    """Where each delivered input comes from on this host. None = absent."""

    seed: Path | None = None
    public_tests: Path | None = None
    agents: Path | None = None
    #: Claude Code project files layered into `.claude/` (instructed tasks:
    #: the settings and hook that re-state the team workflow).
    claude_project: Path | None = None
    serving_skill: Path | None = None
    ncu_report_skill: Path | None = None
    kernel_skill: Path | None = None
    kernelwiki_skill: Path | None = None
    #: Repository additions layered over the clean KernelWiki submodule.
    kernelwiki_overlay: Path | None = None
    #: The task's model id. A skill's per-model notes (`references/model-*.md`)
    #: for any OTHER model are not delivered: a workspace carries its own task.
    model_id: str = ""


def _other_model_notes(delivered: dict[str, str], model_id: str) -> list[str]:
    """Delivered model-note paths that are about a different model."""
    own = f"model-{model_id.split('/')[-1].lower()}.md"
    return [path for path in delivered
            if "/references/model-" in path and not path.endswith("/" + own)]


@dataclasses.dataclass(frozen=True)
class Delivery:
    mounted: dict[str, str]
    withheld: dict[str, str]


def _copy_tree(src: Path, dst: Path, workspace: Path) -> dict[str, str]:
    """Copy a file or tree; every copy is made writable, whatever its source.

    A published version is frozen read-only, and seeding a run from one is the
    supported way to continue a lineage, so the copy must not inherit 0444.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
        return {str(dst.relative_to(workspace)): digest_file(dst)}
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    delivered: dict[str, str] = {}
    for path in sorted(dst.rglob("*")):
        with contextlib.suppress(OSError):
            path.chmod(path.stat().st_mode | (0o200 if path.is_file() else 0o300))
        if path.is_file():
            delivered[str(path.relative_to(workspace))] = digest_file(path)
    return delivered


#: A skill bundle's own repository records, kept in the checkout and never
#: delivered. `provenance.yaml` names the upstream repositories a bundle was
#: distilled from and the harness's licence and exclusion bookkeeping; it is
#: what `skill_bundle.validate_bundle` checks, and nothing an agent needs.
SKILL_RECORDS_NOT_DELIVERED = frozenset({"provenance.yaml"})


def _copy_skill(src: Path, dst: Path, workspace: Path) -> tuple[dict[str, str], list[str]]:
    """Copy a skill bundle, leaving out files that carry upstream framework source.

    Filtered by content rather than by path: a vendored reference can store an
    upstream pull request's kernel files anywhere, and handing an agent SGLang's
    own implementation is the one thing an offline task must not do.
    """
    delivered: dict[str, str] = {}
    dropped: list[str] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name in SKILL_RECORDS_NOT_DELIVERED:
            continue
        relative = path.relative_to(src)
        try:
            text = path.read_text("utf-8", errors="ignore")
        except OSError:
            text = ""
        if any(marker in text for marker in UPSTREAM_MARKERS):
            dropped.append(str(relative))
            continue
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        delivered[str(target.relative_to(workspace))] = digest_file(target)
    return delivered, dropped


def deliver_waiters(workspace: Path) -> dict[str, str]:
    """Put the blocking waiters in `wait/`, always and over whatever the seed had."""
    target = Path(workspace) / "wait"
    target.mkdir(parents=True, exist_ok=True)
    delivered: dict[str, str] = {}
    for script in sorted(WAITERS.glob("*.sh")):
        copy = target / script.name
        shutil.copy2(script, copy)
        copy.chmod(0o755)
        delivered[str(copy.relative_to(workspace))] = digest_file(copy)
    (target / "README.md").write_text(WAIT_README, "utf-8")
    delivered["wait/README.md"] = digest_file(target / "README.md")
    return delivered


def deliver(workspace: Path, sources: Sources) -> Delivery:
    """Copy every present source into the workspace. Absent sources are skipped."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    mounted: dict[str, str] = {}
    withheld: dict[str, str] = {}

    for source, destination in ((sources.seed, "seed"),
                                (sources.public_tests, "tests"),
                                (sources.agents, ".claude/agents"),
                                (sources.claude_project, ".claude")):
        if source is not None and Path(source).exists():
            mounted.update(_copy_tree(Path(source), workspace / destination,
                                      workspace))
    mounted.update(deliver_waiters(workspace))

    for source in (sources.serving_skill, sources.ncu_report_skill,
                   sources.kernel_skill):
        if source is None or not Path(source).exists():
            continue
        destination = f".claude/skills/{Path(source).name}"
        delivered, dropped = _copy_skill(Path(source), workspace / destination,
                                         workspace)
        if (Path(source).resolve() == NCU_SKILL.resolve()
                and (Path(source) / "SKILL.md").is_file()):
            if not (NCU_OVERLAY / "reference" / "09-common-issues.md").is_file():
                raise FileNotFoundError(f"Missing NCU clock-guidance overlay: {NCU_OVERLAY}")
            patched, excluded = _copy_skill(NCU_OVERLAY, workspace / destination,
                                            workspace)
            delivered.update(patched)
            dropped.extend(excluded)
        for path in _other_model_notes(delivered, sources.model_id):
            (workspace / path).unlink()
            del delivered[path]
            withheld[path] = "a note about another task's model"
        mounted.update(delivered)
        if dropped:
            withheld[destination + "/"] = (
                f"{len(dropped)} file(s) held back: they carry upstream "
                f"serving-framework source. First few: {dropped[:3]}")

    if sources.kernelwiki_skill is not None and Path(sources.kernelwiki_skill).exists():
        target = workspace / ".claude/skills/KernelWiki"
        mounted.update(_copy_tree(Path(sources.kernelwiki_skill), target, workspace))
        if (sources.kernelwiki_overlay is not None
                and Path(sources.kernelwiki_overlay).is_dir()):
            mounted.update(_copy_tree(Path(sources.kernelwiki_overlay), target,
                                      workspace))
    return Delivery(mounted=mounted, withheld=withheld)
