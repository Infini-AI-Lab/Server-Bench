"""Modal transport for the shared scenario verifier; no backend-specific scoring.

Performance is measured over loopback inside a fresh sandbox. After it is
terminated, a separate pinned SGLang sandbox audits the committed replies.
Only evidence crosses the control plane, never timed request traffic.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import secrets
import shlex
import signal
import sys
import tarfile
import tempfile
from pathlib import Path

from . import candidate_bundle, scenario, spine

REPO = Path(__file__).resolve().parents[3]
REMOTE = "/opt/serverbench-verifier"
#: Root-only: the held-out questions for the candidate phase. The candidate
#: server runs as `agent` and cannot read it.
PRIVATE = "/opt/serverbench-private"


def _modalctl():
    sys.path.insert(0, str(REPO / "serving_tasks/_modal"))
    import modalctl
    return modalctl


def _exec(sandbox, command: list[str], *, log: Path, timeout: int = 14400):
    """Redirect remotely to avoid pipe backpressure; download after the phase."""
    remote_log = "/tmp/verifier-phase.log"
    process = sandbox.exec("bash", "-c",
                           "cd " + REMOTE + " && " + shlex.join(command)
                           + " >" + remote_log + " 2>&1",
                           timeout=timeout)
    code = process.wait()
    try:
        sandbox.filesystem.copy_to_local(remote_log, str(log))
    except Exception:
        if code == 0:
            raise
    if code:
        tail = log.read_text(errors="replace")[-4000:] if log.exists() else "no phase log"
        raise RuntimeError(f"Modal verifier phase exited {code}: {tail}")


def _upload(sandbox, archive: Path, destination: str):
    sandbox.filesystem.copy_from_local(str(archive), "/tmp/verifier-upload.tar")
    command = ("mkdir -p " + shlex.quote(destination)
                           + " && tar xf /tmp/verifier-upload.tar -C "
                           + shlex.quote(destination))
    if destination == REMOTE:
        command += f" && chown -R root:root {REMOTE} && chmod -R go-w {REMOTE}"
    elif destination == PRIVATE:
        command += f" && chown -R root:root {PRIVATE} && chmod -R go-rwx {PRIVATE}"
    process = sandbox.exec("bash", "-c", command, timeout=600)
    if process.wait() != 0:
        raise RuntimeError("INFRASTRUCTURE_FAILURE: Modal evidence upload failed")


def _download(sandbox, remote: str, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    process = sandbox.exec("tar", "cf", "/tmp/verifier-download.tar", "-C", remote, ".", timeout=600)
    if process.wait() != 0:
        raise RuntimeError("INFRASTRUCTURE_FAILURE: Modal evidence archive failed")
    with tempfile.TemporaryDirectory(prefix="modal-evidence-") as temporary:
        archive = Path(temporary) / "evidence.tar"
        sandbox.filesystem.copy_to_local("/tmp/verifier-download.tar", str(archive))
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            for member in members:
                if (not (member.isfile() or member.isdir()) or
                        not (destination / member.name).resolve().is_relative_to(destination.resolve())):
                    raise ValueError(f"unsafe Modal evidence member: {member.name}")
            handle.extractall(destination, members=members, filter="data")


def _harness_archive(task: str, destination: Path):
    """Only trusted evaluator source and this task's contract, never run history."""
    with tarfile.open(destination, "w") as archive:
        for path in sorted((REPO / "serverbench/harness/final_validate").rglob("*.py")):
            if "__pycache__" not in path.parts:
                archive.add(path, arcname=str(path.relative_to(REPO / "serverbench")))
        for name in ("contract.yaml", "reference.yaml"):
            path = REPO / "serving_tasks" / task / name
            archive.add(path, arcname=f"task_context/{task}/{name}")
        archive.add(REPO / "serverbench/artifacts/agentx/aiperf-pin.json",
                    arcname="artifacts/agentx/aiperf-pin.json")


def _reference_image():
    return _serving_image("sglang")


def _serving_image(arm: str):
    import docker_to_modal
    import modal

    from .arms import image_for
    runtime, _ = docker_to_modal.agentx_paths()
    interpreter_root, interpreter_mount = docker_to_modal.agentx_interpreter_paths(runtime)
    image = (modal.Image.from_registry(image_for(arm))
             .add_local_dir(str(runtime), "/opt/agentx", copy=True)
             .add_local_dir(str(interpreter_root), interpreter_mount, copy=True))
    return docker_to_modal.verifier_distribution_layers(image, runtime)


def preflight(task: str, version: Path | None = None, target: str = "candidate"):
    """Refuse an unmeasurable workload before requesting paid resources.

    The candidate is offered the load its tree declares in `concurrency.json`;
    a baseline arm the task's stated SGLang reference point.
    """
    contract = spine.load_contract(task)
    module = scenario._module_of(task)
    if (module.EVOLVE_DURATION_S, module.VERDICT_DURATION_S) != (1800, 1800):
        raise ValueError("both replay windows must be 1800 seconds")
    try:
        if getattr(module, "CONCURRENCY_POLICY", "fixed") != "candidate":
            offered = module.OFFERED_CONCURRENCY
        elif target == "candidate":
            try:
                offered = scenario._shared_agentx_module().candidate_concurrency(
                    version or Path("/nonexistent"))
            except ValueError as exc:
                raise ValueError(f"offered concurrency is undecided: {exc}") from None
        else:
            offered = getattr(module, "REFERENCE_CONCURRENCY", None)
            if offered is None:
                raise ValueError(
                    "offered concurrency is undecided: the task states no "
                    "SGLang reference point to measure a baseline arm at")
        module.invocation("http://127.0.0.1:8000", "/models/target",
                          "/evaluation/agentx/candidate", concurrency=offered)
    except ValueError as exc:
        raise ValueError(f"INFRASTRUCTURE_FAILURE: task configuration: {exc}") from exc
    return contract


def judge(task: str, version: Path, out: Path, *, keep: bool = False,
          development: bool = False, candidate_tree_sha: str = "",
          target: str = "candidate") -> dict:
    """Same two phases for final validation and public full-test.

    `keep` is accepted and ignored: owned GPU sandboxes always stop.
    Logs/evidence are retained locally, including when a phase fails.
    """
    del keep
    task = spine.resolve_task_dir(task).name
    if not Path(version).is_dir():
        raise ValueError(f"candidate directory does not exist: {version}")
    version_record = Path(version) / "VERSION.json"
    if version_record.is_file() and json.loads(version_record.read_text()).get("mutable") is True:
        raise ValueError("DELIVERY_FAILED: final validation requires an immutable published version")
    delivery = candidate_bundle.audit(version) if target == "candidate" else {"files": [], "errors": []}
    if delivery["errors"]:
        raise ValueError("DELIVERY_FAILED: " + "; ".join(delivery["errors"]))
    if target == "candidate" and "serve.sh" not in delivery["files"]:
        raise ValueError("DELIVERY_FAILED: candidate has no delivered serve.sh")
    preflight(task, Path(version), target)
    ctl = _modalctl()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "delivery-report.json").write_text(json.dumps(delivery, indent=2))
    active = None
    phase = "candidate"
    previous = {}

    def interrupted(signum, frame):
        if active is not None:
            active.terminate()
        raise InterruptedError(f"Modal verifier interrupted by signal {signum}")

    import threading
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
    try:
        with tempfile.TemporaryDirectory(prefix="modal-verdict-") as temporary:
            root = Path(temporary)
            harness = root / "harness.tar"
            _harness_archive(task, harness)
            source = root / "candidate.tar"
            fingerprint = candidate_bundle.fingerprint(version, delivery["files"])
            if candidate_tree_sha and fingerprint != candidate_tree_sha:
                raise ValueError("SOURCE_CHANGED: Modal upload does not match the requested source digest")
            candidate_tree_sha = fingerprint
            questions = None
            if not development:
                # Drawn on the host, where the held-out pool lives. The
                # candidate sandbox gets the questions only; the answers go to
                # the reference sandbox, which runs no candidate code.
                from . import private_correctness
                size, _ = private_correctness.answer_settings(spine.load_contract(task))
                questions = private_correctness.select_questions(size, secrets.randbits(64))
                (out / "reference-answers.json").write_text(json.dumps(
                    {q["id"]: q["answer"] for q in questions}))
                (root / "questions.json").write_text(json.dumps(
                    [{"id": q["id"], "question": q["question"]} for q in questions]))
                with tarfile.open(root / "questions.tar", "w") as archive:
                    archive.add(root / "questions.json", arcname="questions.json")
            with tarfile.open(source, "w") as archive:
                for name in delivery["files"]:
                    archive.add(Path(version) / name, arcname=name, recursive=False)
            for phase in ("candidate", "reference"):
                try:
                    image_args = {}
                    if phase == "reference":
                        image_args["image"] = _reference_image()
                    elif target != "candidate":
                        image_args["image"] = _serving_image(target)
                    active = ctl.create_sandbox(
                        task, purpose=f"{phase}-verdict", hours=4,
                        workdir="/app" if phase == "candidate" else REMOTE,
                        network="none", env={"APP_DIR": "/app", "PORT": "8000"},
                        **image_args)
                except Exception as exc:
                    raise RuntimeError(f"INFRASTRUCTURE_FAILURE: Modal allocation: {exc}") from exc
                (out / f"{phase}-sandbox.json").write_text(json.dumps({
                    "id": active.object_id, "backend": "modal", "task": task,
                    "gpu": ctl.gpu_spec(task), "network": "none"}))
                _upload(active, harness, REMOTE)
                if phase == "candidate" and target == "candidate":
                    _upload(active, source, "/app")
                if phase == "candidate" and questions is not None:
                    _upload(active, root / "questions.tar", PRIVATE)
                elif phase == "reference":
                    from . import agentx_reference
                    artifacts = out / "agentx" / target
                    with tarfile.open(root / "evidence.tar", "w") as archive:
                        for name in ("candidate-phase.json", "reference-answers.json"):
                            if (out / name).is_file():
                                archive.add(out / name, arcname=name)
                        # The EOS gate scans every timed reply in the sandbox;
                        # without the export that scan fails closed there.
                        try:
                            raw = agentx_reference._raw_export(artifacts)
                        except FileNotFoundError:
                            raw = None
                        if raw is not None:
                            archive.add(raw, arcname=str(raw.relative_to(out)))
                    _upload(active, root / "evidence.tar", "/evaluation")
                command = ["env", f"PYTHONPATH={REMOTE}/harness",
                           f"SERVERBENCH_TASK_CONTEXT={REMOTE}/task_context",
                           "/opt/agentx/venv/bin/python", "-m", "final_validate.modal_worker", phase,
                           "--task", task, "--target", target,
                           "--candidate-tree-sha", candidate_tree_sha]
                if development:
                    command.append("--development-workload")
                elif phase == "candidate":
                    command += ["--questions", f"{PRIVATE}/questions.json"]
                try:
                    _exec(active, command, log=out / f"{phase}.phase.log")
                finally:
                    try:
                        _download(active, "/evaluation", out)
                    except Exception as exc:
                        (out / f"{phase}-collection-error.txt").write_text(str(exc))
                active.terminate(wait=True)
                active = None
            result = json.loads((out / f"scenario-{target}.json").read_text())
            (out / f"scenario-{target}.json").write_text(json.dumps(result, indent=1) + "\n")
            return result
    except BaseException as exc:
        (out / "modal-failure.json").write_text(json.dumps({
            "phase": phase, "complete": False, "why": f"{type(exc).__name__}: {exc}"}, indent=2))
        if isinstance(exc, Exception) and phase == "reference":
            raise RuntimeError(f"INFRASTRUCTURE_FAILURE: reference audit did not complete: {exc}") from exc
        if isinstance(exc, Exception) and (out / "candidate-ready.json").is_file():
            raise RuntimeError(
                f"MEASUREMENT_FAILED: server started but verification did not finish: {exc}") from exc
        raise
    finally:
        if active is not None:
            with contextlib.suppress(Exception):
                active.terminate()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--version", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--development-workload", action="store_true")
    parser.add_argument("--candidate-tree-sha", default="")
    parser.add_argument("--target", default="candidate", choices=("candidate", *spine.ARMS))
    args = parser.parse_args(argv)
    try:
        judge(args.task, args.version, args.out, keep=args.keep,
              development=args.development_workload, candidate_tree_sha=args.candidate_tree_sha,
              target=args.target)
    except (RuntimeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
