"""Fast deterministic form and portability checks at publication.

The checks record facts about a frozen checkpoint; they do not run correctness
or performance evaluation and do not accept or reject the checkpoint.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

#: The candidate's start contract. The harness runs this file verbatim, so a
#: tree without it cannot be started at all -- a lifecycle fact, checkable
#: without running anything.
ENTRY_POINT = "serve.sh"

PASS, FAIL, SKIPPED = "PASS", "FAIL", "SKIPPED"


@dataclasses.dataclass(frozen=True)
class BoundaryChecks:
    """What was established about one captured tree, deterministically."""

    candidate_id: str
    tree_sha: str
    checks: tuple[dict[str, Any], ...]
    seconds: float

    @property
    def failed(self) -> tuple[dict[str, Any], ...]:
        return tuple(c for c in self.checks if c["outcome"] == FAIL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "deterministic_boundary_checks",
            # Explicitly null: no reviewer approves a candidate, and a
            # consumer looking for an approval sees its absence rather than a
            # value nobody produced.
            "reviewer": None,
            "verdict": None,
            "candidate_id": self.candidate_id,
            "tree_sha": self.tree_sha,
            "checks": [dict(c) for c in self.checks],
            "failed": [c["name"] for c in self.failed],
            "seconds": round(self.seconds, 3),
            "note": ("these checks are recorded, not gating. They do not run "
                     "correctness or performance evaluation. A failure is "
                     "written into the agent's workspace as "
                     "BOUNDARY_CHECKS.md and does not reject the checkpoint"),
        }


#: Where the agent develops, and where the verdict serves from. They differ,
#: and that difference is what `portable_paths` exists to catch.
WORKSPACE_ABS = "/workspace"
APP_DIR_ABS = "/app"

#: The files the verdict actually executes. `serve.sh` is run verbatim and
#: everything else follows from what it starts; these three are where a
#: hardcoded path stops a server from booting rather than merely being untidy.
ENTRY_CHAIN = frozenset({"serve.sh", "start_server.sh", "server.py"})

#: How the verdict hands a candidate its weights: mounted at this path, with
#: the path itself exported under this name. Both are `final_validate.deploy`'s
#: constants; they are restated here because the harness core is stdlib-only
#: so it can run inside a bare image, and `tests/contracts` asserts the two
#: against what `deploy.launch` actually passes.
MODEL_MOUNT = "/models/target"
MODEL_PATH_ENV = "MODEL_PATH"


def _check(name: str, outcome: str, detail: str) -> dict[str, Any]:
    return {"name": name, "outcome": outcome, "detail": detail}


def required_profile_check(tree: Path, evidence_root: Path) -> dict[str, Any]:
    """Audit exact source coverage without running a profiler on a serving GPU.

    Receipts are development evidence, not trusted final correctness results.
    Profile collection itself verifies native CUDA events. The boundary checks
    source hashes and the existence of the recorded native capture only; it
    never parses multi-GB traces while an AgentX performance window is running.
    """
    files = {}
    for path in sorted(tree.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(tree))] = hashlib.sha256(
                str(path.stat().st_mode & 0o777).encode() + b'\0' + path.read_bytes()).hexdigest()
    for path in evidence_root.rglob('profile.json'):
        try:
            report = json.loads(path.read_text())
            artifacts = report.get('artifacts') or {}
            # Container paths map into this run's own /workspace/agentx.
            exists = bool(artifacts) and all(
                (evidence_root / Path(name).relative_to('/workspace/agentx')).is_file()
                for name in artifacts)
            if (report.get('schema') == 'serving_evolution.profile/1'
                    and report.get('status') == 'passed'
                    and report.get('backend') in {'torch', 'nsys', 'ncu'}
                    and report.get('gpu_kernel_events', 0) > 0
                    and report.get('files') == files and exists):
                return _check('required_design_profile', PASS,
                              f'current deployed source has CUDA profiler evidence: {path}')
        except (OSError, ValueError, TypeError):
            continue
    return _check('required_design_profile', FAIL,
                  'MISSING_OR_STALE: every instructed design change must be checked with '
                  'torch.profiler, nsys or ncu on its real GPU execution. Run the delivered '
                  'scripts/profile_candidate.py and design_review.py. Keep diagnostic files '
                  'under /workspace/agentx; profile the exact deployable tree. This checkpoint '
                  'is saved as an unverified experiment, not a verified iteration. Repair or '
                  'revert before continuing from it. No profiler is run in the timed window.')


def run(tree_dir: Path, *, candidate_id: str, tree_sha: str,
        extra_markers: tuple[str, ...] = (),
        delivered_from: Path | None = None,
        evidence_root: Path | None = None,
        delivery_report: dict | None = None) -> BoundaryChecks:
    """Every deterministic boundary check, over one frozen tree."""
    started = time.perf_counter()
    tree = Path(tree_dir)
    checks: list[dict[str, Any]] = []
    if delivery_report is not None:
        errors = delivery_report.get('errors', [])
        checks.append(_check('runtime_resource_delivery', FAIL if errors else PASS,
                             '; '.join(errors) if errors else
                             'No statically detected missing runtime resources; this is not a boot test.'))
        checks.append({**_check('delivery_filter_report', PASS,
                               'Full filtering rules and exclusions are recorded; '
                               'excluded directories are pruned, not scanned.'),
                       'report': delivery_report})
    if evidence_root is not None:
        checks.append(required_profile_check(tree, evidence_root))
        completed = []
        for path in (evidence_root / 'full-tests').glob('*/full-test.json'):
            try:
                report = json.loads(path.read_text())
                if (report.get('schema') == 'serving_evolution.full_test/1'
                        and report.get('evaluation_scope') == 'development_full_test'
                        and report.get('complete') is True):
                    completed.append(path)
            except (OSError, ValueError):
                continue
        checks.append(_check('evolve_full_test_coverage', PASS if completed else FAIL,
                             f'{len(completed)} complete public full-verifier test(s) in this evolve; '
                             'this does not certify later source changes.' if completed else
                             'MISSING: complete ./full-test --wait during this evolve; reserve repair time. '
                             'protocol/performance-only checks do not replace it.'))

    # -- lifecycle: can this tree be started at all? -------------------------
    entry = tree / ENTRY_POINT
    checks.append(_check(
        "entry_point", PASS if entry.is_file() else FAIL,
        f"{ENTRY_POINT} is present" if entry.is_file() else
        f"{ENTRY_POINT} is missing, and the harness runs it verbatim -- this "
        f"tree cannot be started"))

    # -- workload: does the tree declare the load the verdict offers? -------
    #
    # The offered concurrency is the agent's to choose and the starter declares
    # none, so a tree that never wrote one cannot be measured. Told here, at
    # every publication, not by the verdict.
    load = tree / "concurrency.json"
    if load.is_file():
        try:
            body = json.loads(load.read_text("utf-8"))
            value = body.get("offered_concurrency") if isinstance(body, dict) else None
        except (OSError, ValueError):
            value = None
        good = type(value) is int and 1 <= value <= 256
        checks.append(_check(
            "offered_concurrency", PASS if good else FAIL,
            f"concurrency.json declares offered_concurrency {value}" if good else
            "concurrency.json must be {\"offered_concurrency\": N} with N an "
            f"integer in [1, 256]; got {value!r}. The verdict cannot measure "
            "this tree"))
    else:
        checks.append(_check(
            "offered_concurrency", FAIL,
            "concurrency.json is missing: the tree declares no offered load, "
            "so ./full-test and the verdict cannot measure it. Write "
            "{\"offered_concurrency\": N}"))

    # -- lifecycle: can it start from a tree with NO .venv? ------------------
    #
    # A defect invisible until the run is over. `.venv` is gigabytes of
    # packages rebuildable from the frozen wheelhouse, so it is deliberately NOT
    # published with a version -- but the final verdict re-serves the PUBLISHED
    # tree, in a fresh container, by running `bash serve.sh`. A serve.sh that
    # assumes `.venv` is already there dies at verdict time with
    #
    #     serve.sh: /app/.venv/bin/python missing - build .venv first
    #
    # and the run records NO_RESULT with NOTHING MEASURED.
    #
    # The task's brief states the requirement, but a brief is a hope. This
    # is the check, and it runs at every publication boundary so the agent is
    # told while it can still fix it -- hours before the verdict, not after.
    #
    # Static on purpose: actually starting the server here would load torch and
    # take the device, which does not belong at a boundary. What it looks for
    # is the shape that fails -- naming a venv interpreter with nothing in the
    # file that could create one.
    if entry.is_file():
        try:
            body = entry.read_text("utf-8", errors="ignore")
        except OSError:
            body = ""
        uses_venv = bool(re.search(r"\.venv/bin/(python|uvicorn)", body))
        builds_venv = bool(re.search(r"uv\s+venv|python3?\s+-m\s+venv|"
                                     r"virtualenv\s", body))
        if not uses_venv:
            outcome, detail = SKIPPED, (
                f"{ENTRY_POINT} does not run out of a .venv, so there is "
                f"nothing for it to rebuild")
        elif builds_venv:
            outcome, detail = PASS, (
                f"{ENTRY_POINT} runs out of a .venv AND can create one, so the "
                f"published tree can be served by the verdict")
        else:
            outcome, detail = FAIL, (
                f"{ENTRY_POINT} runs the server from a .venv and contains no "
                f"command that could build one. `.venv` is NOT published with "
                f"a version -- it is ~6.5 GB and rebuildable -- and the final "
                f"verdict re-serves the published tree in a fresh container by "
                f"running `bash {ENTRY_POINT}` verbatim. As written, that will "
                f"die with `.venv/bin/python missing` and the whole run will be "
                f"recorded NO_RESULT with nothing measured. Make the build "
                f"conditional inside {ENTRY_POINT}: if the interpreter is not "
                f"executable, `uv venv .venv --python 3.12` then `uv pip "
                f"install --offline --no-index --find-links "
                f"/opt/env-input/wheelhouse -r "
                f"/opt/env-input/resolved.requirements.txt`")
        checks.append(_check("entry_point_rebuilds_its_env", outcome, detail))

    # -- build: does every Python file in the tree parse? --------------------
    #
    # Not an import and not a run: importing a candidate's server would load
    # torch, take the device and possibly start listening, none of which belongs
    # at a publication boundary. A syntax error, though, is a certainty that
    # this tree will not start, and it is free to find.
    broken: list[str] = []
    counted = 0
    for path in sorted(tree.rglob("*.py")):
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        counted += 1
        try:
            ast.parse(path.read_text("utf-8", errors="ignore"),
                      filename=str(path))
        except SyntaxError as exc:
            broken.append(f"{path.relative_to(tree)}:{exc.lineno}: {exc.msg}")
        except OSError:
            continue
    checks.append(_check(
        "python_parses", PASS if not broken else FAIL,
        f"{counted} file(s) parse" if not broken else
        f"{len(broken)} file(s) will not parse: {broken[:3]}"))

    # -- lifecycle: PID 1 is a live container supervisor -------------------
    #
    # Development and final validation both put an init/supervisor at PID 1
    # and run the candidate below it. Treating `getppid() == 1` as proof that
    # a parent died is therefore not portable: it can kill a healthy process
    # that was launched directly by the supervisor. The launchers share the
    # supervisor invariant; this cheap check makes the fragile assumption
    # visible at each checkpoint without starting a server or touching a GPU.
    pid1_assumptions: list[str] = []
    for path in sorted(tree.rglob("*.py")):
        rel = path.relative_to(tree)
        if any(part in {"__pycache__", ".venv", "tests", "devtools"}
               for part in rel.parts):
            continue
        try:
            parsed = ast.parse(path.read_text("utf-8", errors="ignore"),
                               filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(parsed):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue
            sides = (node.left, node.comparators[0])
            call = next((side for side in sides if isinstance(side, ast.Call)
                         and isinstance(side.func, ast.Attribute)
                         and side.func.attr == "getppid"), None)
            one = next((side for side in sides if isinstance(side, ast.Constant)
                        and side.value == 1 and not isinstance(side.value, bool)), None)
            if call is not None and one is not None:
                pid1_assumptions.append(f"{rel}:{node.lineno}")
    checks.append(_check(
        "pid1_parent_assumption", PASS if not pid1_assumptions else FAIL,
        ("no runtime Python code equates parent PID 1 with an orphan")
        if not pid1_assumptions else
        (f"{len(pid1_assumptions)} runtime comparison(s) treat parent PID 1 "
         "as a special condition: " + "; ".join(pid1_assumptions[:5]) +
         ". Development and final validation both use a live container "
         "supervisor at PID 1. Do not use `os.getppid() == 1` alone as proof "
         "that a parent died; use PR_SET_PDEATHSIG with a before/after parent "
         "identity check or track the expected parent PID explicitly.")))

    # -- portability: will this tree start anywhere but where it was built? --
    #
    # THE VERDICT DOES NOT MOUNT YOUR WORKSPACE.
    #
    # The agent develops in `/workspace`. `final_validate/deploy.py` copies the
    # published snapshot and mounts it at `/app`, exports `APP_DIR=/app`, and
    # runs `serve.sh` VERBATIM. So an absolute path into `/workspace` exists
    # for the whole of development and does not exist at the only moment that
    # decides anything:
    #
    #     File "/app/server.py", line 38, in _spawn_worker
    #       with open("/workspace/tests/serve_pids.txt", "w") as f:
    #     FileNotFoundError: [Errno 2] No such file or directory
    #
    # Reported, never blocking -- same as every other check here. It is
    # delivered to the agent as feedback so it can fix it while it still has
    # a clock.
    #
    # WHICH FILES, THOUGH. A check that fires on every candidate stops being
    # read. A mention of `/workspace` in a dev tool the verdict never executes
    # does not stop the server starting, and failing it would be noise.
    #
    # So: the ENTRY-POINT CHAIN fails, everything else is reported. The chain
    # is what the verdict actually runs, and a path hardcoded there is the
    # difference between a measurement and a FileNotFoundError.
    on_serving_path: list[str] = []
    elsewhere: list[str] = []
    for path in sorted(tree.rglob("*")):
        if path.suffix not in {".py", ".sh"} or not path.is_file():
            continue
        rel = path.relative_to(tree)
        if any(part in {"__pycache__", ".venv", "task_context", "tests"}
               for part in rel.parts):
            continue
        try:
            body = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        for number, line in enumerate(body.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or WORKSPACE_ABS not in line:
                continue
            where = f"{rel}:{number}: {stripped[:90]}"
            (on_serving_path if rel.name in ENTRY_CHAIN
             else elsewhere).append(where)
            break
    note = ""
    if elsewhere:
        note = (f" ({len(elsewhere)} more outside the entry-point chain, which "
                f"the verdict never runs: {'; '.join(elsewhere[:2])})")
    checks.append(_check(
        "portable_paths", PASS if not on_serving_path else FAIL,
        ("nothing the verdict runs hardcodes the development workspace path"
         + note)
        if not on_serving_path else
        (f"{len(on_serving_path)} file(s) the verdict RUNS hardcode "
         f"{WORKSPACE_ABS!r}, which does not exist there -- it mounts this "
         f"tree at {APP_DIR_ABS!r} and exports APP_DIR. Use $APP_DIR, or a "
         f"path relative to the script's own directory: "
         + "; ".join(on_serving_path[:3]) + note)))

    # -- scope: does the candidate reach for the held-out surface? -----------
    #
    # The held-out oracle is withheld by content hash at delivery. That answers
    # "was it handed over"; this answers "is it being reached for" -- a
    # candidate that builds the path at runtime is invisible to a delivery-time
    # hash and plain in its source.
    try:
        from serving_evolve.evaluator import integrity
    except Exception as exc:                                   # noqa: BLE001
        checks.append(_check(
            "source_scan", SKIPPED,
            f"the integrity probes are not importable here ({exc!r}), so the "
            f"scan did not run. NOT the same as a clean scan"))
        checks.append(_check("evaluation_surface", SKIPPED,
                             "not run for the same reason"))
    else:
        scan = integrity.scan_sources(tree, extra_markers=extra_markers,
                                      delivered_from=delivered_from)
        checks.append(_check(
            "source_scan", PASS if scan["passed"] else FAIL, scan["reason"]))
        surface = integrity.evaluation_surface(tree)
        checks.append(_check(
            "evaluation_surface", PASS if surface.get("passed", True) else FAIL,
            str(surface.get("reason", ""))))

    # -- can the verdict's container tell this tree where the model is? -----
    #
    # The checkpoint is mounted at `/models/target` and its path is exported
    # as `MODEL_PATH`; that is how the task document says a candidate finds
    # its weights. A loader rewritten as
    #
    #     source = os.environ.get("MY_MODEL_DIR") or MODEL_ID
    #
    # drops the `MODEL_PATH` link. If `MY_MODEL_DIR` is exported only by the
    # agent's own `start_server.sh`, which the verdict does not run -- it runs
    # `serve.sh` -- then in the verdict's offline container the fallback
    # reaches for the hub id and the run ends
    #
    #     LOAD FAILED OSError: We couldn't connect to 'https://huggingface.co'
    #     status DID_NOT_START
    #
    # while every other check here passes.
    #
    # One fact, decided statically: does anything the verdict RUNS name the
    # mounted checkpoint at all. A tree that does cannot be certain to start,
    # but a tree that does not cannot find its weights except by luck.
    names_checkpoint: list[str] = []
    own_variables: set[str] = set()
    for path in sorted(tree.rglob("*")):
        if path.suffix not in {".py", ".sh"} or not path.is_file():
            continue
        rel = path.relative_to(tree)
        if any(part in {"__pycache__", ".venv", "task_context", "tests"}
               for part in rel.parts):
            continue
        try:
            body = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        # A tree may route the path through its own variable, in any file:
        # `serve.sh` exporting one for `server.py` is how this is meant to
        # work, and only the chain the verdict runs is examined below.
        own_variables.update(
            re.findall(r"(?:^|\n)\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=", body))
        if rel.name not in ENTRY_CHAIN:
            continue
        for number, line in enumerate(body.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if MODEL_PATH_ENV in line or MODEL_MOUNT in line:
                names_checkpoint.append(f"{rel}:{number}")
    checks.append(_check(
        "serves_the_mounted_checkpoint",
        PASS if names_checkpoint else FAIL,
        (f"the entry-point chain reads the mounted checkpoint "
         f"({names_checkpoint[0]})")
        if names_checkpoint else
        (f"nothing the verdict RUNS ({sorted(ENTRY_CHAIN)}) names "
         f"{MODEL_PATH_ENV!r} or {MODEL_MOUNT!r}. The verdict starts "
         f"`{ENTRY_POINT}` in an offline container with the checkpoint mounted "
         f"at {MODEL_MOUNT} and its path in {MODEL_PATH_ENV}; a loader that "
         f"reaches for a hub id, or for a variable only this workspace's "
         f"`start_server.sh` exports, has no way to find the weights there. "
         f"Variables this tree sets for itself: "
         f"{sorted(own_variables)[:6]}")))

    return BoundaryChecks(candidate_id=candidate_id, tree_sha=tree_sha,
                          checks=tuple(checks),
                          seconds=time.perf_counter() - started)
