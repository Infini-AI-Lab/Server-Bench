"""Materialise an immutable candidate in a clean evaluation environment.

**Immutable.** The thing measured is the registered snapshot, never the mutable
development workspace. `materialise()` refuses a tree whose `VERSION.json`
declares `mutable: true` and refuses one whose recorded tree digest does not
match what is on disk: the snapshot being graded is the registered immutable
version, not an uncommitted mutable workspace.

**Clean.** The snapshot is copied to a private staging directory that the coding
agent has no path to, and the container is built from that copy. Nothing from a
live workspace, no leftover artefacts, no `__pycache__` carrying a stale module.

**Bounded lifetime.** Cleanup is registered the moment the container
exists, not written at the end of the happy path. A server left running holds
the whole card and the next task to ask for that device is refused -- correctly,
because a contended device is not a measurement. A crash between an arm's last
block and its cleanup line would otherwise cost every later task on that card
its run.

This module starts no coding agent, modifies no candidate, and computes no
score; it only materialises and runs the selected snapshot.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

#: The candidate's own contract, from each task's `demo/` starter: the harness
#: runs `serve.sh` VERBATIM with no extra environment, and the server listens on
#: this port. Wiring anything else in would measure a server the agent was never
#: told to build.
SERVE_SCRIPT = "serve.sh"
CONTAINER_PORT = 8000
APP_DIR = "/app"

#: Where the model is mounted. The blob mounts below are computed against this
#: path, because the snapshot's links are RELATIVE and only resolve once the
#: snapshot is at a known location inside the container.
MODEL_MOUNT = "/models/target"

#: Where a SPECULATIVE DRAFT checkpoint is mounted, when an arm uses one.
#:
#: A draft model is a second HuggingFace snapshot in its own directory, so the
#: target's mount does not cover it. Handed a host path that does not exist
#: inside the container, transformers falls through to treating it as a Hub
#: repo id and fails with an `OSError` about repo-id syntax, which says nothing
#: about a missing mount.
DRAFT_MOUNT = "/models/draft"


def relocate_draft_model(serve_extra: str) -> tuple[str, Path | None]:
    """Resolve, mount, and rewrite a speculative draft checkpoint.

    Task contracts name a portable HuggingFace repo id and optional revision,
    not a developer's home directory. Offline execution resolves that id in
    the host cache, mounts the repository root so snapshot symlinks retain
    access to ``blobs/``, and rewrites the server flag to the container path.
    An explicit host snapshot remains supported. An already-container path is
    left unchanged.
    """
    flag = "--speculative-draft-model-path"
    parts = shlex.split(serve_extra or "")
    if flag not in parts:
        return serve_extra, None
    value = parts[parts.index(flag) + 1]
    host = Path(value)
    if not host.is_absolute():
        try:
            from final_validate import model_identity
        except ImportError:               # the packaged verifier copy
            from . import model_identity
        resolved = model_identity.local_snapshot_of(value)
        if resolved is None:
            return serve_extra, None
        revision_flag = "--speculative-draft-model-revision"
        if revision_flag in parts:
            revision = parts[parts.index(revision_flag) + 1]
            pinned = resolved.parent / revision
            if not pinned.is_dir():
                raise FileNotFoundError(
                    f"draft model {value}@{revision} is not in the local "
                    f"HuggingFace cache")
            resolved = pinned
        host = resolved
    elif not host.is_dir():
        return serve_extra, None          # already in-container
    # Hand over the REPO ROOT, not the snapshot. A HuggingFace snapshot's
    # files are relative symlinks into a sibling `blobs/`, and mounting the
    # snapshot alone leaves every one of them dangling -- or, if its blobs are
    # mounted the way the target's are, collides with the target on `/blobs`.
    root, inside = host, DRAFT_MOUNT
    if host.parent.name == "snapshots":
        root = host.parent.parent
        inside = f"{DRAFT_MOUNT}/snapshots/{host.name}"
    parts[parts.index(flag) + 1] = inside
    return shlex.join(parts), root

#: Where the frozen build inputs are bind-mounted, and what they are.
#:
#: `scripts/agent/harbor_env.py::mounts_for` binds the build-input bundle at
#: `/opt/env-input:ro` in the AGENT's container, and the brief tells the agent
#: in as many words that the pinned wheels live there:
#:
#:     --find-links /opt/env-input/wheelhouse -r /opt/env-input/resolved.requirements.txt
#:
#: External validation supplies the same frozen build-input bundle that was
#: available during development. It is mounted READ-ONLY: a candidate that could edit
#: its own pins would not be measured against the environment it was given.
ENV_INPUT_MOUNT = "/opt/env-input"

#: The immutable AgentX runtime mounted into the coding agent's container.
#: Candidates may legitimately use its frozen Python packages, so final
#: validation must reproduce this read-only part of the development runtime.
AGENTX_MOUNT = "/opt/agentx"

#: What `/opt/env-input` must contain to be the real thing rather than an empty
#: directory that mounts cleanly and fails at `uv pip install`.
ENV_INPUT_MARKERS = ("wheelhouse", "resolved.requirements.txt")


def env_input_dir() -> Path | None:
    """The frozen wheelhouse bundle on this host, or None if absent.

    Derived, never hardcoded to a particular machine: `$SERVERBENCH_ENV_INPUT`
    wins, else the path `scripts/server_evolve/evolve-common.sh` defaults to,
    resolved against THIS checkout rather than any path in a document.

    Returns None rather than raising: not every task's candidate needs the
    wheelhouse, and a host that has none should still be able to measure the
    ones that do not. `launch()` turns the absence into a legible failure at
    the point where it actually matters.
    """
    named = os.environ.get("SERVERBENCH_ENV_INPUT", "").strip()
    if named:
        candidate = Path(named).expanduser()
        return candidate if candidate.is_dir() else None
    repo = Path(__file__).resolve().parents[3]
    default = (repo / "serverbench" / "artifacts" / "env-input")
    # The directory can exist and be empty (a failed build, say). Merely
    # existing is not evidence that it is the frozen bundle, and returning it
    # turns an infrastructure omission into a candidate startup failure.
    complete = (default.is_dir()
                and all((default / marker).exists()
                        for marker in ENV_INPUT_MARKERS))
    return default if complete else None


def _build_input_flags(target: str, log: Path | None = None) -> list[str]:
    """`-v`/`-e` flags giving a candidate the environment it was built against.

    Baselines get nothing: they are measured in their own vendor images and
    have no business seeing our wheelhouse.
    """
    if target != "candidate":
        return []
    source = env_input_dir()
    if source is None:
        return []
    missing = [m for m in ENV_INPUT_MARKERS if not (source / m).exists()]
    if missing:
        # Loud, because a half-populated bundle produces exactly the failure
        # this constant exists to prevent, one layer deeper.
        raise ServerDidNotStart(
            f"the build-input bundle at {source} is missing {missing}. The "
            f"candidate is told by its brief to install from "
            f"{ENV_INPUT_MOUNT}/wheelhouse, so mounting an incomplete bundle "
            f"would fail inside the container and read as a broken candidate")
    if log is not None:
        with contextlib.suppress(OSError):
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"[verdict] mounting build inputs "
                         f"{source} -> {ENV_INPUT_MOUNT} (ro)\n")
    return [
        "-v", f"{source.resolve()}:{ENV_INPUT_MOUNT}:ro",
        # uv needs a WRITABLE cache even for `--offline --no-index`; the agent
        # container gets one at /opt/uv-cache. Without a writable HOME-derived
        # default this is the difference between a venv and a crash.
        "-e", "UV_CACHE_DIR=/var/tmp/uv-cache",
        "-e", "HF_HUB_OFFLINE=1",
    ]


def _agentx_runtime_flags(target: str, log: Path | None = None) -> list[str]:
    """Give a candidate the same frozen AgentX runtime it developed with."""
    if target != "candidate":
        return []
    named = os.environ.get("SERVERBENCH_AGENTX_RUNTIME", "").strip()
    source: Path | None
    if named:
        source = Path(named).expanduser()
    else:
        # `serverbench/scripts/agent/` is a directory of scripts, not a
        # package on the harness path: put it there, as its other callers do.
        agent_scripts = str(Path(__file__).resolve().parents[2] / "scripts" / "agent")
        if agent_scripts not in sys.path:
            sys.path.insert(0, agent_scripts)
        try:
            import harbor_env
        except (ImportError, ModuleNotFoundError):
            source = None
        else:
            source = harbor_env.agentx_runtime()
    if source is None:
        return []
    missing = [marker for marker in ("venv", "bin/aiperf")
               if not (source / marker).exists()]
    if missing:
        raise ServerDidNotStart(
            f"the AgentX runtime at {source} is missing {missing}. The agent "
            f"developed with that runtime at {AGENTX_MOUNT}; mounting a partial "
            "tree would turn an evaluator deployment defect into a candidate "
            "startup failure")
    if log is not None:
        with contextlib.suppress(OSError):
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"[verdict] mounting AgentX runtime "
                         f"{source} -> {AGENTX_MOUNT} (ro)\n")
    return [
        "-v", f"{source.resolve()}:{AGENTX_MOUNT}:ro",
        "-e", f"AGENTX_ROOT={AGENTX_MOUNT}",
        "-e", (f"PATH={AGENTX_MOUNT}/bin:/usr/local/cuda/bin:"
               "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
    ]


#: Containers this process started and has not yet removed.
_LIVE: set[str] = set()

#: Where each live container's own output belongs, so RELEASING a container
#: saves it -- not only the `CandidateServer.stop()` path. The scenario verdict
#: tears its container down through `_release_all`, and without this the
#: traceback of an engine that died mid-replay would go with the container.
_LOGS: dict[str, Path] = {}


class MaterialisationRefused(RuntimeError):
    """The candidate is not an immutable snapshot, or does not bind to its record."""


class ServerDidNotStart(RuntimeError):
    """The candidate's server never answered. An INFRASTRUCTURE-or-candidate fact.

    Which of the two it is cannot be decided here: a server that fails to boot
    may be a broken candidate (a valid failure) or a missing image (an invalid
    attempt). The caller decides with the rest of the evidence; raising a typed
    error rather than returning a zero is what keeps that decision possible.
    """


class DockerControlPlaneError(RuntimeError):
    """Docker itself did not answer; no claim about candidate behavior exists."""


def inspect_image(image: str) -> bool:
    """Whether an image exists, with Docker failure kept distinct from absent."""
    try:
        got = subprocess.run(["docker", "image", "inspect", image],
                             capture_output=True, text=True, timeout=120,
                             check=False)
    except subprocess.TimeoutExpired as exc:
        raise DockerControlPlaneError(
            f"docker image inspect did not answer within {exc.timeout}s for "
            f"{image!r}") from exc
    return got.returncode == 0


def _inspect_running(name: str, *, attempts: int = 3,
                     timeout_s: float = 20.0) -> bool:
    """Read container liveness, retrying only control-plane timeouts.

    A false result is a candidate/container fact. Repeated TimeoutExpired is a
    Docker daemon fact. Collapsing both into ``ServerDidNotStart`` would make
    a hung `docker inspect` look like a broken candidate.
    """
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            got = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", name],
                capture_output=True, text=True, timeout=timeout_s, check=False)
            return got.returncode == 0 and (got.stdout or "").strip() == "true"
        except subprocess.TimeoutExpired as exc:
            errors.append(f"attempt {attempt}: {exc.timeout}s")
    raise DockerControlPlaneError(
        f"docker inspect did not answer for container {name!r} after "
        f"{attempts} attempts ({', '.join(errors)}). The container's state "
        "is unknown; this is invalid infrastructure evidence, not a candidate "
        "startup failure")


def _blob_mounts(model_dir: Path) -> list[str]:
    return _blob_mounts_at(model_dir, MODEL_MOUNT)


def _blob_mounts_at(model_dir: Path, mount: str) -> list[str]:
    """Also mount whatever the model directory's symlinks point at.

    A HuggingFace snapshot is a farm of RELATIVE symlinks
    (`config.json -> ../../blobs/<sha>`), and the bytes live in a sibling
    `blobs/`. Bind-mounting the snapshot alone mounts the links and not their
    targets, so every file in the model directory dangles inside the container.

    The failure does not name its cause: a server reports `Unrecognized model
    in /models/target. Should have a model_type key in its config.json`
    because config.json is a dangling link.

    Same rule as `scripts/agent/run_agent.py::blob_mounts`; reproduced rather
    than imported because that module is an agent-container script and this
    is the final validator.
    """
    if not model_dir.is_dir():
        return []
    targets: dict[str, Path] = {}
    for entry in sorted(model_dir.iterdir()):
        if not entry.is_symlink():
            continue
        raw = os.readlink(entry)
        if os.path.isabs(raw) or os.path.dirname(raw) in ("", "."):
            continue                      # same-directory link: already mounted
        host = (model_dir / os.path.dirname(raw)).resolve()
        inside = os.path.normpath(os.path.join(mount, os.path.dirname(raw)))
        if host.is_dir():
            targets[inside] = host
    flags: list[str] = []
    for inside, host in sorted(targets.items()):
        flags += ["-v", f"{host}:{inside}:ro"]
    return flags


def capture_container_log(name: str) -> Path | None:
    """Write a live container's output beside the harness's own lines.

    Idempotent: `docker logs` returns the whole log every time, so the file is
    rewritten as the harness's `[verdict]` lines followed by the container's
    output, and calling this twice changes nothing. Never raises -- losing a
    log must not cost a verdict.
    """
    log = _LOGS.get(name)
    if log is None:
        return None
    with contextlib.suppress(Exception):
        got = subprocess.run(["docker", "logs", name], capture_output=True,
                             text=True, timeout=120, check=False)
        if got.returncode != 0:
            return log
        harness = ""
        if log.is_file():
            harness = "".join(
                line for line in log.read_text("utf-8", errors="replace")
                .splitlines(keepends=True) if line.startswith("[verdict]"))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(harness + "\n--- container output ---\n"
                       + (got.stdout or "") + (got.stderr or ""), "utf-8")
    return log


def capture_live_logs() -> None:
    """Save every live container's output now, without stopping anything.

    Called as soon as a measurement returns. A verdict killed later -- by a
    supervisor's SIGKILL, which no handler can intercept -- then still leaves
    the output that covers the measurement it was killed after.
    """
    for name in list(_LIVE):
        capture_container_log(name)


def _release(name: str, *, settle_s: float = 60.0) -> None:
    """Remove the container and WAIT for its processes to leave the cards.

    `docker rm -f` returns as soon as the container record is gone, which is
    not when its CUDA contexts are released, so the next launch's quiet-card
    check could run against a card that has not finished emptying. A
    straggler of a few hundred MiB slips under any megabyte threshold; the
    fix belongs here, at the point that knows which container it just killed,
    rather than in a wider tolerance downstream.

    Best effort by design: this never raises and never blocks past
    `settle_s`. A card that will not empty is the next caller's problem to
    report, and `wait_for_quiet_cards` says so by name.
    """
    # THE SERVER'S OWN OUTPUT FIRST. After `rm -f` it is gone for good.
    capture_container_log(name)
    with contextlib.suppress(Exception):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                       text=True, timeout=120, check=False)
    _LIVE.discard(name)
    _LOGS.pop(name, None)
    deadline = time.time() + settle_s
    while time.time() < deadline:
        try:
            got = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"name=^{name}$"],
                capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            return
        if not (got.stdout or "").strip():
            # The record is gone. Give the driver a moment to reap the
            # contexts: a pid going away is not its memory going away.
            time.sleep(3)
            return
        time.sleep(2)


@atexit.register
def _release_all() -> None:
    for name in list(_LIVE):
        _release(name)


def _release_and_die(signum, frame) -> None:                    # noqa: ARG001
    """Tear down on a signal, then die as if unhandled.

    `atexit` does NOT run on SIGTERM: Python's default handler terminates the
    process without unwinding. So a run stopped by `timeout`, by an operator's
    `kill`, or by a supervisor leaves its container running -- holding the whole
    card, with its claim already released because the flock died with the
    process. Card occupied, nothing recording who occupies it, which is the
    worst of both.

    Re-raising with the default handler after cleanup keeps the exit status
    honest -- a run killed by SIGTERM still reports as killed by SIGTERM.
    """
    _release_all()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    with contextlib.suppress(ValueError, OSError):
        signal.signal(_sig, _release_and_die)


def tree_digest(root: Path) -> str:
    """sha256 over the tree's file names and contents, in sorted order.

    Excludes `__pycache__` and `*.pyc`: they are build products of whichever
    interpreter last imported the tree, so including them would make the digest
    depend on who looked at the snapshot rather than on what it contains.
    """
    h = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class Materialised:
    """A private, verified copy of one immutable candidate."""

    source: Path
    path: Path
    candidate_id: str
    tree_sha256: str
    version_record: dict


def materialise(snapshot: Path, into: Path, *, expected_candidate_id: str = "",
                expected_tree_sha: str = "") -> Materialised:
    """Copy an immutable snapshot into a clean private directory, or refuse."""
    src = Path(snapshot).resolve()
    if not src.is_dir():
        raise MaterialisationRefused(f"{src} is not a directory")
    record: dict = {}
    version_json = src / "VERSION.json"
    if version_json.is_file():
        try:
            record = json.loads(version_json.read_text("utf-8"))
        except ValueError as exc:
            raise MaterialisationRefused(
                f"{version_json} would not parse ({exc}). A candidate whose "
                f"own record cannot be read does not bind to anything") from None
    if record.get("mutable") is True:
        raise MaterialisationRefused(
            f"{src} declares `mutable: true`. Final validation grades the "
            f"REGISTERED immutable version, never a development workspace: a "
            f"number measured against a tree that can still change is a number "
            f"about nothing in particular")
    candidate_id = str(record.get("version_id") or expected_candidate_id or src.name)
    if expected_candidate_id and candidate_id != expected_candidate_id:
        raise MaterialisationRefused(
            f"the snapshot at {src} identifies itself as {candidate_id!r} and "
            f"the controller asked for {expected_candidate_id!r}. Evidence "
            f"that does not bind to its candidate is evidence about nothing")

    target = Path(into).resolve()
    if target.exists() and any(target.iterdir()):
        raise MaterialisationRefused(
            f"{target} is not empty. A clean evaluation environment means "
            f"clean: a leftover artefact or a stale __pycache__ is exactly the "
            f"kind of thing that makes one run differ from another for reasons "
            f"nobody records")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    digest = tree_digest(target)
    recorded = str(record.get("tree_sha") or "")
    if expected_tree_sha and digest != expected_tree_sha:
        raise MaterialisationRefused(
            f"the materialised tree digests to {digest[:12]}… and the "
            f"controller expected {expected_tree_sha[:12]}…")
    return Materialised(source=src, path=target, candidate_id=candidate_id,
                        tree_sha256=digest,
                        version_record={**record, "recorded_tree_sha": recorded})


@dataclasses.dataclass
class CandidateServer:
    """One candidate serving system, running in its own container.

    The launch mirrors what measured the three baselines -- one device set, an
    offline model mount, no network route out, an explicit port -- because a
    candidate measured under different conditions than the baselines is not
    comparable with them, and comparability is the whole point of the datasheet.
    """

    name: str
    host: str
    port: int
    log: Path
    device_spec: str
    #: `"candidate"` or one of the three pinned arms. Recorded so every number
    #: says which target produced it -- a datasheet whose rows do not name their
    #: arm is a comparison nobody can check.
    target: str = "candidate"
    #: The body fields that make a BASELINE do the same work as the oracle.
    #: Empty for a candidate, on purpose.
    extra_body: dict = dataclasses.field(default_factory=dict)

    def __enter__(self) -> CandidateServer:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    #: The candidate tree this server was started from, when it was
    #: materialised into an evidence directory. Set by `launch`, and the only
    #: reason it is held here is so `stop` can reclaim what serving built.
    served_tree: Path | None = None

    def stop(self) -> None:
        self.capture_log()
        _release(self.name)
        if self.served_tree is not None:
            reclaim_rebuilt_environment(self.served_tree)

    def capture_log(self) -> Path:
        with contextlib.suppress(Exception):
            got = subprocess.run(["docker", "logs", self.name],
                                 capture_output=True, text=True, timeout=120,
                                 check=False)
            self.log.parent.mkdir(parents=True, exist_ok=True)
            self.log.write_text((got.stdout or "") + "\n" + (got.stderr or ""),
                                "utf-8")
        return self.log


#: What `serve.sh` builds at launch and what the verdict must not archive.
#:
#: A published version deliberately omits its `.venv` -- gigabytes of torch
#: rebuildable from the frozen wheelhouse, and the brief makes rebuilding it a
#: requirement of `serve.sh`. But the verdict materialises the tree INTO its own
#: evidence directory and serves it there, so `serve.sh` rebuilds the thing
#: right back, inside the artefact. Nothing COPIES the venv here -- it is
#: created in place -- so it is reclaimed at teardown rather than excluded.
REBUILDABLE = (".venv", "__pycache__", ".uv-cache", ".pytest_cache")


def reclaim_rebuilt_environment(tree: Path) -> int:
    """Remove what serving rebuilt, and return the bytes freed.

    Safe by construction and by timing. By construction, because everything in
    :data:`REBUILDABLE` is derived: the environment comes from the frozen
    wheelhouse and the caches from running the code. By timing, because the
    tree digest that binds this evidence to its candidate is taken in
    `materialise`, BEFORE the server ever starts -- so nothing removed here was
    ever part of what was hashed, and no verdict can be invalidated by it.

    The measurement itself lives beside the tree, not inside it: `MEASURED.json`,
    the aiperf exports, the server log. This touches none of them.

    Every failure is swallowed. This runs during teardown, after a measurement
    that may have cost an hour on two cards, and a disk-reclaim problem must
    never be the thing that loses it.
    """
    freed = 0
    tree = Path(tree)
    for name in REBUILDABLE:
        for found in tree.rglob(name):
            if not found.is_dir():
                continue
            with contextlib.suppress(OSError):
                freed += sum(f.stat().st_size for f in found.rglob("*")
                             if f.is_file())
            with contextlib.suppress(OSError):
                shutil.rmtree(found, ignore_errors=True)
    if freed:
        print(f"  reclaimed {freed / 1024 ** 3:.1f} GB of rebuilt environment "
              f"from {tree.name}; the measurement beside it is untouched")
    return freed


#: How long a CANDIDATE may take to answer readiness. Agents are told this limit
#: and plan against it (`lifecycle/full_test.py` mirrors it), so it never moves.
CANDIDATE_READY_TIMEOUT_S = 900.0
#: How long a pinned BASELINE engine may take: the harness's own server, which no
#: agent builds or sees. The private correctness stage stands the plain
#: checkpoint up on SGLang, whose prefill CUDA-graph capture for the largest
#: model can take well over the candidate's limit on a loaded host. The
#: concurrency search gives the same engine 1800 s
#: (`concurrency.Plan.ready_timeout`).
BASELINE_READY_TIMEOUT_S = 1800.0


def launch(target: str, *, image: str, model_dir: Path, device_spec: str,
           port: int, name: str, log: Path,
           candidate: Materialised | None = None,
           tp: int = 1, dp: int = 1, ready_timeout_s: float | None = None,
           env: dict[str, str] | None = None,
           draft_model_dir: Path | None = None,
           serve_extra: str = "") -> CandidateServer:
    """Stand up ONE measurement target and wait for it to answer.

    `target` is `"candidate"` or one of the three pinned baseline arms. Both
    go through the same code, and this is where that holds: everything after
    this function -- the load generator, the correctness judgement, the
    report -- does not know which it measured. **A candidate and a baseline
    are comparable because they are measured by the same instrument**, and
    that comparability is the point of the benchmark.

    Only two things differ, and both are properties of the target rather than of
    the measurement:

    * a candidate is its own tree, bind-mounted and started with its `serve.sh`;
      a baseline is a pinned image started with the framework's own command;
    * a baseline is handed the `extra_body` that makes it do the same work as
      the oracle. **The candidate is not**: getting semantic equivalence right is
      part of what the task asks of it, and handing it the flag would answer the
      question on its behalf.

    `serve_extra` is appended to a BASELINE's serve command verbatim, and is
    ignored for the candidate. It exists because a task can need a framework
    flag that is not a tuning choice but a precondition for serving the
    workload at all: a corpus whose turns exceed the model's native context
    under `forbid_input_truncation` needs RoPE scaling, without which a server
    rejects part of the workload and the scenario aborts at warmup -- an arm
    that could not be offered the work, not an arm that lost.

    It comes from the TASK's `reference.yaml`, frozen with the task, never from
    this function and never chosen at measurement time. Every result records
    the string it ran under, because a baseline number whose launch line is not
    recorded is a number nobody can reproduce.

    Raises `ServerDidNotStart` rather than returning something unusable, so the
    caller must decide what the failure MEANS instead of measuring an absent
    server and reporting zeros.
    """
    from .arms import spec_for

    if ready_timeout_s is None:
        ready_timeout_s = (CANDIDATE_READY_TIMEOUT_S if target == "candidate"
                           else BASELINE_READY_TIMEOUT_S)
    spec = spec_for(target)
    image = image or spec.get("image", "")
    # THE DRAFT CHECKPOINT IS RELOCATED FOR EVERY CALLER, HERE.
    #
    # A baseline launch line names its speculative draft by HOST path, because
    # that is where an operator pins a revision. The container mounts only the
    # target, so the path resolves to nothing and transformers reads it as a
    # Hub repo id. Doing this before `argv` is built means the bind mount and
    # the rewritten flag cannot disagree.
    #
    # Applied before the caller's own `draft_model_dir`, and idempotent, so a
    # caller that already relocated (concurrency.point does) is unaffected.
    if serve_extra and target != "candidate":
        serve_extra, _found_draft = relocate_draft_model(serve_extra)
        if _found_draft is not None and draft_model_dir is None:
            draft_model_dir = _found_draft

    if not image:
        raise ServerDidNotStart(
            f"no image for target {target!r}. A candidate is measured in the "
            f"task's own agent image, which the controller supplies")

    if not inspect_image(image):
        raise ServerDidNotStart(
            f"the image {image!r} is not on this host. Pulling it is a separate "
            f"decision rather than a step this takes: on a shared machine a "
            f"pull that runs out part-way leaves the filesystem full for "
            f"everyone. This is NOT a result about {target!r} -- an arm that "
            f"did not run is not an arm that lost")
    _release(name)                        # a leftover of OUR OWN, by exact name

    argv = [
        # Match the Harbor development container's process model. Without an
        # init process the command below becomes PID 1, while development
        # launches the candidate below the container supervisor. A multi-rank
        # candidate that checks its worker's parent identity would otherwise
        # observe different process trees in development and final validation.
        "docker", "run", "-d", "--init", "--name", name,
        "--gpus", device_spec,
        # Bound to loopback: the measurement client runs on this host and
        # nothing else should be able to reach a target under test.
        "-p", f"127.0.0.1:{port}:{CONTAINER_PORT if target == 'candidate' else port}",
        "--shm-size", spec.get("shm", "8g"),
        "-v", f"{Path(model_dir).resolve()}:{MODEL_MOUNT}:ro",
        *_blob_mounts(Path(model_dir).resolve()),
        # The draft checkpoint, with its blobs, when the arm speculates.
        # THE DRAFT'S REPO ROOT, mounted whole, with NO separate blob mounts.
        #
        # Both snapshots' symlinks read `../../blobs/<sha>`, which normalises
        # to `/blobs` from either mount -- so mounting the draft's blobs the
        # way the target's are mounted collides:
        #     docker: Error response from daemon: Duplicate mount point: /blobs
        # Mounting the repo root instead (the `models--org--name/` directory,
        # containing both `snapshots/` and `blobs/`) lets the relative links
        # resolve inside that one mount, with nothing to collide.
        *(["-v", f"{Path(draft_model_dir).resolve()}:{DRAFT_MOUNT}:ro"]
          if draft_model_dir else []),
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
    ]
    if target == "sglang":
        # Loaded by Python before `sglang.launch_server`. The pinned SGLang
        # build has a one-attribute Gemma 4 constructor omission that crashes
        # the first forward pass; the compatibility module restores exactly
        # that assignment and is a no-op for every other model.
        compat = Path(__file__).resolve().parent / "sglang_compat"
        if compat.is_dir():
            argv += ["-v", f"{compat}:/serverbench-sglang-compat:ro",
                     "-e", "PYTHONPATH=/serverbench-sglang-compat"]
    if target == "candidate":
        if candidate is None:
            raise ServerDidNotStart(
                "the candidate target needs a materialised snapshot")
        # EVERY VARIABLE THE TASK DOCUMENT PROMISES, because the document is
        # what the candidate was built against. The evolve container exports
        # `APP_DIR`, `MODEL_PATH`, `PORT` and `MODEL_ID` and the task
        # instruction names all four; a server that read one the verdict
        # omitted would work in development and then not start here. A
        # variable the instruction never promised is the candidate's defect.
        argv += ["-v", f"{Path(candidate.path).resolve()}:{APP_DIR}",
                 "-e", f"APP_DIR={APP_DIR}", "-e", "MODEL_PATH=/models/target",
                 "-e", f"PORT={CONTAINER_PORT}"]
        # `MODEL_ID` arrives through `env`, which the caller fills from the
        # task, and is exported by the loop below with everything else.
        # The frozen wheelhouse the brief told this agent to install from --
        # see `ENV_INPUT_MOUNT`.
        argv += _build_input_flags(target, log=log)
        argv += _agentx_runtime_flags(target, log=log)
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]

    parallel = ""
    if tp > 1 and spec.get("tp_flag"):
        parallel += spec["tp_flag"].format(tp=tp)
    if dp > 1 and spec.get("dp_flag"):
        parallel += spec["dp_flag"].format(dp=dp)
    command = shlex.split(
        spec["serve"].format(model="/models/target", port=port,
                             parallel=parallel))
    if serve_extra and target != "candidate":
        command += shlex.split(serve_extra)
    argv += [image, *command]

    got = subprocess.run(argv, capture_output=True, text=True, timeout=900,
                         check=False)
    if got.returncode != 0:
        raise ServerDidNotStart(f"docker run failed: {(got.stderr or '')[:400]}")
    _LIVE.add(name)
    if log is not None:
        _LOGS[name] = Path(log)
    server = CandidateServer(name=name, host="127.0.0.1", port=port, log=log,
                             device_spec=device_spec, target=target,
                             extra_body=dict(spec.get("extra_body") or {}),
                             # Only a materialised candidate has a tree in the
                             # evidence directory to reclaim. A pinned baseline
                             # arm serves from its own image and owns nothing
                             # here, so it passes None and `stop` does nothing.
                             served_tree=(candidate.path if candidate is not None
                                          else None))

    # Readiness, in a few lines of stdlib: the question is one GET.
    import urllib.error
    import urllib.request

    def answering(path: str, timeout_s: float) -> bool:
        try:
            with urllib.request.urlopen(              # noqa: S310 - loopback
                    f"http://127.0.0.1:{port}{path}", timeout=timeout_s) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    ready_path = spec.get("ready_path", "/health")
    deadline = time.time() + ready_timeout_s
    while time.time() < deadline:
        remaining = max(0.1, deadline - time.time())
        # Most arms answer within five seconds. SGLang's /health generates a
        # token, and Gemma 4 on the pinned build compiles Triton kernels on its
        # first request because CUDA-graph capture is unavailable. Its arm
        # opts into a longer request without changing the overall deadline.
        request_timeout = min(
            remaining, float(spec.get("ready_request_timeout_s", 5.0)))
        if answering(ready_path, request_timeout):
            return server
        if not _inspect_running(name):
            tail = server.capture_log().read_text("utf-8")[-1500:]
            server.stop()
            raise ServerDidNotStart(
                f"the {target} container exited before answering "
                f"{ready_path}. Last output:\n{tail}")
        time.sleep(2.0)
    tail = server.capture_log().read_text("utf-8")[-1500:]
    server.stop()
    raise ServerDidNotStart(
        f"the {target} server did not answer {ready_path} within "
        f"{ready_timeout_s:.0f}s. Last output:\n{tail}")
