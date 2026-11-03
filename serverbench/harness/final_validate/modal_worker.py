"""Trusted in-sandbox transport worker. All checks/metrics are shared code."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import candidate_bundle, private_correctness, scenario, spine
from .arms import spec_for

EVIDENCE = Path("/evaluation")
SOURCE = Path("/app")
HARNESS = Path(__file__).resolve().parents[2]


def server_command(task: str, phase: str, target: str = "candidate") -> tuple[list[str], dict]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if phase == "candidate" and target == "candidate":
        env.update(APP_DIR=str(SOURCE), MODEL_PATH="/models/target", PORT="8000")
        return ["su", "agent", "-p", "-s", "/bin/bash", "-c",
                "cd /app && export HOME=/home/agent && exec bash /app/serve.sh"], env
    contract = spine.load_contract(task)
    arm = "sglang" if phase == "reference" else target
    spec = spec_for(arm)
    parallel = spec["tp_flag"].format(tp=contract.devices) if contract.devices > 1 else ""
    command = shlex.split(spec["serve"].format(model="/models/target", port=8000, parallel=parallel))
    extra = (private_correctness.plain_reference_serve_extra(contract)
             if phase == "reference" else scenario._serve_extra(contract, target))
    command += shlex.split(extra)
    if arm == "sglang":
        env.update(PYTHONPATH=str(Path(__file__).parent / "sglang_compat"))
    if phase == "reference":
        env["SGLANG_LOGPROB_CHUNK_SIZE"] = "256"
    return command, env


def wait_ready(process, *, phase: str, target="candidate", timeout: float = 900):
    deadline = time.monotonic() + timeout
    spec = spec_for("sglang" if phase == "reference" else target)
    request_timeout = float(spec.get("ready_request_timeout_s", 5))
    ready_path = spec.get("ready_path", "/health")
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{phase} did not start: server exited {process.returncode}; see serverlog")
        try:
            remaining = max(1, deadline - time.monotonic())
            with urllib.request.urlopen("http://127.0.0.1:8000" + ready_path,
                                        timeout=min(request_timeout, remaining)) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    raise RuntimeError(f"{phase} did not start: /health timed out; see serverlog")


def finish_reference(task: str, *, tokenizer, evidence: Path = EVIDENCE, target="candidate"):
    """The same reference stage (answer accuracy and EOS gate) as Docker."""
    result = json.loads((evidence / "candidate-phase.json").read_text())
    row = result["row"]
    contract = spine.load_contract(task)
    answers_file = evidence / "reference-answers.json"
    answers = json.loads(answers_file.read_text()) if answers_file.is_file() else {}
    artifact_dir = evidence / "agentx" / target
    artifact_dir.mkdir(parents=True, exist_ok=True)
    row["correctness"] = private_correctness.reference_stage(
        contract, reference_url="http://127.0.0.1:8000",
        artifact_dir=artifact_dir, model_dir=Path("/models/target"),
        tokenizer=tokenizer, row=row, answers=answers)
    rate = float(scenario.objective_detail(task)["usd_per_hour_total"])
    result["verdict"] = scenario.decide(contract, [row], rate)
    result["backend"] = "modal"
    (evidence / f"scenario-{target}.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("candidate", "reference"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate-tree-sha", default="")
    parser.add_argument("--development-workload", action="store_true")
    parser.add_argument("--questions", default="",
                        help="the root-only held-out questions for this verdict")
    parser.add_argument("--target", default="candidate", choices=("candidate", *spine.ARMS))
    args = parser.parse_args(argv)
    EVIDENCE.mkdir(exist_ok=True)
    delivery = None
    if args.phase == "candidate" and args.target == "candidate":
        delivery = candidate_bundle.audit(SOURCE)
        if delivery["errors"]:
            raise RuntimeError("DELIVERY_FAILED: " + "; ".join(delivery["errors"]))
        if (args.candidate_tree_sha and
                candidate_bundle.fingerprint(SOURCE, delivery["files"]) != args.candidate_tree_sha):
            raise RuntimeError("SOURCE_CHANGED: uploaded delivery differs from the frozen source")
        subprocess.run(["chown", "-R", "agent:agent", str(SOURCE)], check=True)
    if args.phase == "candidate":
        from .agentx_pin import require
        try:
            require()
        except RuntimeError as exc:
            raise RuntimeError(f"INFRASTRUCTURE_FAILURE: benchmark pin verification: {exc}") from exc
    command, env = server_command(args.task, args.phase, args.target)
    (EVIDENCE / f"{args.phase}-launch.json").write_text(json.dumps({"command": command}, indent=2))
    with (EVIDENCE / f"{args.phase}.serverlog.txt").open("w") as log:
        process = subprocess.Popen(command, env=env, cwd=SOURCE if args.phase == "candidate" else HARNESS,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(process, phase=args.phase, target=args.target)
            (EVIDENCE / f"{args.phase}-ready.json").write_text('{"ready":true}')
            if delivery is not None:
                missing = candidate_bundle.check_generated(SOURCE, delivery)
                if missing:
                    raise RuntimeError("DELIVERY_FAILED: " + "; ".join(missing))
            if args.phase == "candidate":
                # The delivered tree, so a candidate-declared load is read
                # from its concurrency.json.
                command = [args.task, "--target", args.target, "--endpoint", "127.0.0.1:8000",
                           "--model-dir", "/models/target", "--run-dir", str(EVIDENCE),
                           "--candidate-tree-sha", args.candidate_tree_sha,
                           "--candidate-snapshot", str(SOURCE)]
                if args.development_workload:
                    command.append("--development-workload")
                if args.questions:
                    # Set only now: the candidate server has already been
                    # launched with its own environment.
                    os.environ[private_correctness.QUESTIONS_FILE_ENV] = args.questions
                code = scenario.main(command)
                if code:
                    raise RuntimeError(f"candidate measurement exited {code}")
                (EVIDENCE / f"scenario-{args.target}.json").replace(EVIDENCE / "candidate-phase.json")
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("/models/target", local_files_only=True)
                finish_reference(args.task, tokenizer=tokenizer, target=args.target)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
