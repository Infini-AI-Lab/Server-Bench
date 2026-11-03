#!/usr/bin/env python3
"""Execute one source-bound public check, or an optional compatibility cascade.

Run inside the task container with an already reserved GPU set. Supply a clean
deployable source directory (serve.sh and its dependencies), not /workspace.
This is a development convenience, NOT a security boundary or final verdict.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def digest_tree(root: Path) -> dict[str, str]:
    """Fingerprint source, including modes; reject unresolved external inputs."""
    result = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(p in {"__pycache__", ".venv", ".git"} for p in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"stage actual files, not a symlink: {path}")
        if path.is_file():
            result[str(rel)] = hashlib.sha256(
                str(path.stat().st_mode & 0o777).encode() + b"\0" + path.read_bytes()
            ).hexdigest()
    return result


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    body = json.loads(path.read_text())
    if not isinstance(body, dict):
        raise ValueError(f"expected a report object: {path}")
    return body


def stop(process: subprocess.Popen) -> None:
    # Kill only the session we created, including workers surviving their parent.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=3)
    process.wait()


def execute(command: list[str], *, cwd: Path, env: dict, log: Path,
            timeout: float) -> dict:
    began = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            stop(process)
    return {"command": command, "exit_code": process.returncode,
            "timed_out": timed_out, "wall_seconds": time.monotonic() - began,
            "log": str(log)}


def stage_holds(stage: str, report: dict, task: str,
                candidate_sha: str = "") -> bool:
    if report.get("task_id") != task:
        return False
    if stage == "replay":
        return (report.get("schema") == "final_validate.lightweight_replay_verification/1"
                and report.get("kind") == "lightweight_replay_verification"
                and (report.get("replay_fidelity") or {}).get("holds") is True)
    if (report.get("schema") != "final_validate.development_evidence/1"
            or report.get("mode") != stage or report.get("final_verdict") is not False):
        return False
    if candidate_sha and (report.get("measurement_status") != "COMPLETE"
                          or report.get("candidate_tree_sha") != candidate_sha):
        return False
    if stage == "correctness":
        return report.get("public_correctness_holds") is True
    value = report.get("objective_value")
    required = ["full_window_complete", "replay_fidelity_holds"]
    if "measured_gates_hold" in report:
        required.append("measured_gates_hold")
    return (all(report.get(key) is True for key in required)
            and type(value) in (int, float) and math.isfinite(value))


def wait_ready(server: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("owned server exited before readiness; see server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise TimeoutError("server readiness timed out; see server.log")


def selected_stages(stage: str | None, through: str | None) -> list[str]:
    """Full includes actual replay fidelity; never prepend its cheap proxy."""
    if stage:
        return [stage]
    if through == "full":
        return ["full", "correctness"]
    if through == "correctness":
        return ["replay", "correctness"]
    return ["replay"]


def snapshot_unchanged(frozen: Path, before: dict, tests: Path,
                       evaluator_before: dict) -> bool:
    """The original development tree is intentionally free to keep changing."""
    current = digest_tree(frozen)
    return (all(current.get(name) == value for name, value in before.items())
            and digest_tree(tests) == evaluator_before)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--tests-dir", type=Path, default=Path("/workspace/tests"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--design-evidence", type=Path,
                        help="passing source-bound iteration receipt; required for instructed full windows")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stage", choices=("replay", "correctness", "full"),
                           help="run exactly this benchmark method")
    selection.add_argument("--through", choices=("replay", "correctness", "full"),
                           help="compatibility cascade ending at this stage")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--correctness-timeout", type=float, default=3600)
    parser.add_argument("--full-timeout", type=float, default=7200)
    args = parser.parse_args(argv)
    source, run = args.candidate_dir.resolve(), args.run_dir.resolve()
    if "_instructed_" in args.task and "full" in selected_stages(args.stage, args.through):
        if not args.design_evidence:
            parser.error("instructed full windows require --design-evidence from design_review.py check")
        from design_review import validate_receipt
        try:
            validate_receipt(args.design_evidence, source, args.task)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    if ("_instructed_" in args.task
            and Path("/workspace/AGENT_TASK.md").is_file()):
        evidence_root = Path("/workspace/agentx").resolve()
        if run != evidence_root and evidence_root not in run.parents:
            parser.error(f"run-dir must be below {evidence_root}")
    if run == source or source in run.parents:
        parser.error("run-dir must be outside candidate-dir")
    if not (source / "serve.sh").is_file():
        parser.error("candidate-dir must contain a deployable serve.sh")
    if any((source / name).exists() for name in (".venv", ".git", "tests", "task_context")):
        parser.error("stage a clean deployment tree, without workspace, venv or evaluator")
    if any(not math.isfinite(t) or t <= 0 for t in (
            args.startup_timeout, args.correctness_timeout, args.full_timeout)):
        parser.error("timeouts must be finite and positive")
    # No attaching to a possibly stale endpoint. Never kill an existing listener.
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    run.mkdir(parents=True, exist_ok=False)
    frozen = run / "candidate"
    tests = args.tests_dir.resolve()
    sys.path.insert(0, str(tests))
    from final_verdict import candidate_bundle
    delivery = candidate_bundle.audit(source)
    write_json(run / 'delivery-report.json', delivery)
    if delivery['errors']:
        raise ValueError('delivery check failed: ' + '; '.join(delivery['errors']))
    frozen.mkdir()
    source_files = digest_tree(source)
    before = {name: source_files[name] for name in delivery['files']}
    for name in delivery['files']:
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    if digest_tree(frozen) != before:
        raise ValueError("source changed while staging the trial")
    evaluator_before = digest_tree(tests)
    record = {
        "schema": "serving_evolution.trial/1", "task_id": args.task,
        "source": str(source), "candidate": str(frozen), "files": before,
        "candidate_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        "evaluator_files": evaluator_before, "stages": [], "status": "running",
        "qualified_public": False, "final_verdict": False,
        "scope": "public development only; independent held-out verdict still required",
        "python": sys.version, "model_dir": str(args.model_dir.resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    candidate_sha = record["candidate_sha256"]
    version = ""
    version_file = frozen / "VERSION.json"
    if version_file.is_file():
        with contextlib.suppress(OSError, ValueError, TypeError):
            version = str(read_json(version_file).get("version_id") or "")
    destination = run / "trial.json"
    write_json(destination, record)
    env = {**os.environ, "APP_DIR": str(frozen), "PORT": str(args.port),
           "MODEL_PATH": str(args.model_dir.resolve()), "PYTHONDONTWRITEBYTECODE": "1"}
    server = None
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("trial terminated")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (run / "server.log").open("w") as output:
            server = subprocess.Popen(["bash", "serve.sh"], cwd=frozen, env=env,
                                      stdout=output, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        record["server_pid"] = server.pid
        wait_ready(server, args.port, args.startup_timeout)
        missing = candidate_bundle.check_generated(frozen, delivery)
        if missing:
            raise RuntimeError('startup did not generate declared runtime resources: ' + ', '.join(missing))
        # Cheap cold-start parity evidence.  Both this development helper and
        # final Docker launch run serve.sh below a live supervisor/init,
        # with the same APP_DIR/PORT/MODEL_PATH contract.  No timed-path
        # instrumentation and no additional model request are introduced.
        startup = {
            "schema": "serving_evolution.startup_parity/1",
            "status": "passed", "task_id": args.task,
            "candidate_tree_sha": candidate_sha,
            "version_id": version or None,
            "source_snapshot": str(frozen),
            "measurement_status": "COMPLETE",
            "command": ["bash", "serve.sh"],
            "environment": {name: env[name]
                            for name in ("APP_DIR", "PORT", "MODEL_PATH")},
            "topology": "live supervisor -> bash serve.sh -> server/workers",
            "server_pid": server.pid,
            "server_parent_pid": os.getpid(),
            "health": f"http://127.0.0.1:{args.port}/health",
        }
        write_json(run / "startup-parity.json", startup)
        record["startup_parity"] = str(run / "startup-parity.json")
        write_json(destination, record)
        chosen = selected_stages(args.stage, args.through)
        for stage in chosen:
            stage_dir = run / stage
            command = [sys.executable, "-m", "final_verdict.replay_verify" if stage == "replay"
                       else "final_verdict.scenario", args.task,
                       "--endpoint", f"127.0.0.1:{args.port}",
                       "--model-dir", str(args.model_dir.resolve()), "--run-dir", str(stage_dir)]
            if stage != "replay":
                command += ["--target", "candidate"]
                command += ["--check-protocol-context"] if stage == "correctness" else [
                    "--measure-only", "--duration", "1800"]
                command += [
                    "--candidate-tree-sha", candidate_sha,
                    "--source-snapshot", str(frozen),
                ]
                if version:
                    command += ["--candidate-version", version]
            entry = {"stage": stage, "status": "running"}
            record["stages"].append(entry)
            write_json(destination, record)
            entry.update(execute(command, cwd=tests, env=env, log=run / f"{stage}.log",
                                 timeout={"replay": 660, "correctness": args.correctness_timeout,
                                          "full": args.full_timeout}[stage]))
            evidence_path = stage_dir / ("replay-verification.json" if stage == "replay"
                                        else "development-evidence.json")
            evidence = read_json(evidence_path) if evidence_path.is_file() else {}
            if evidence and stage != "replay":
                # The helper, not the endpoint evaluator, owns the immutable
                # snapshot and can therefore make the binding authoritative.
                # This also upgrades reports from an older compatible evaluator
                # without weakening the frozen-tree check below.
                evidence.update(
                    candidate_tree_sha=candidate_sha,
                    version_id=version or None,
                    source_snapshot=str(frozen),
                    measurement_status=evidence.get(
                        "measurement_status", "COMPLETE"))
                write_json(evidence_path, evidence)
            entry["evidence"] = str(evidence_path)
            entry["evidence_sha256"] = (hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                                        if evidence_path.is_file() else None)
            entry["status"] = "passed" if (
                not entry["timed_out"] and entry["exit_code"] == 0
                and server.poll() is None
                and stage_holds(stage, evidence, args.task, candidate_sha)
            ) else "failed"
            write_json(destination, record)
            if not snapshot_unchanged(frozen, before, tests, evaluator_before):
                raise RuntimeError("source/evaluator changed; refusing the next phase")
            if entry["status"] != "passed":
                raise RuntimeError(f"{stage} failed or incomplete; inspect {entry['log']}")
            if stage == "full":
                record["objective"] = evidence["objective"]
                record["objective_value"] = evidence["objective_value"]
        record["status"] = "passed"
        record["qualified_public"] = {"full", "correctness"}.issubset(chosen)
    except (OSError, ValueError, RuntimeError, TimeoutError, KeyboardInterrupt) as exc:
        record["status"] = "failed"
        record["why"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        if server is not None:
            stop(server)
        # Startup may build extensions/venvs. Existing source must not change;
        # generated products are recorded, never silently promoted to source.
        try:
            after = digest_tree(frozen)
            record["generated_files"] = sorted(set(after) - set(before))
            unchanged = snapshot_unchanged(frozen, before, tests, evaluator_before)
        except (OSError, ValueError):
            unchanged = False
        if not unchanged:
            record.update(status="stale", qualified_public=False,
                          why="candidate/source/evaluator changed during trial")
        if record["status"] != "passed":
            record["qualified_public"] = False
        write_json(destination, record)
    print(json.dumps({k: record[k] for k in ("status", "qualified_public", "final_verdict")}))
    print(destination)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
