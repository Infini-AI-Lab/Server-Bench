#!/usr/bin/env python3
"""Queue evolves of one agent configuration on free cards; record each as it ends.

    $SERVERBENCH_PY evaluation_progress/watch_evolve.py --campaign <name> \
        --task <task> [--task <task> ...] --agent-cli claude --model <model id> \
        --effort max [--agent-env serverbench/artifacts/secrets/<model>.env] \
        [--hours 8] [--gpus 0,1,2,3]

Each task is one job. A job waits until the production launcher's
quiet/claim/container checks find enough free cards for the task, starts
`evolve-task.sh` on them, and, when the run ends, writes its row through
`record_run.py`. It never kills a holder, mutates a live workspace, or restarts
a failed run. Waiting time is outside the evolve budget. Queue telemetry stays
in the campaign's state directory (`watcher.log`, `status.json`, one launcher
log per job); each task has one CSV, and a run contributes exactly one row.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class Job:
    def __init__(self, task, args):
        self.task, self.args = task, args
        self.process = None
        self.run = None
        self.gpus = ""
        self.state = "WAITING_GPU"
        self.result_written = False
        self.launch_attempt = 0
        self.setup_retries = 0
        self.last_setup_failure = ""
        self.log = None
        self.task_key = task.lower()
        self.space = REPO / "serverbench/evolve_space/evolve" / task.lower()
        self.before = set(self.space.glob("*"))
        self.lock = (args.state_dir / f"{self.task_key}.launch.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.csv_path = args.output_dir / f"{task}.csv"

    def discover(self):
        if self.run:
            return
        matches = [p for p in self.space.glob("*") if p not in self.before
                   and load(p / "identity.json").get("task_id") == self.task]
        if len(matches) > 1 and self.gpus:
            # Another watcher may have launched the same task meanwhile; the
            # cards a run was created on tell the two apart.
            matches = [p for p in matches if run_cards(p) == set(self.gpus.split(","))]
        if len(matches) == 1:
            self.run = matches[0]
        elif len(matches) > 1:
            raise RuntimeError(f"Ambiguous concurrent launches for {self.task}; refusing to mix evidence")

    def collect(self):
        self.discover()

    def write_final_result(self):
        """Delegate the one allowed CSV write to the final-evidence recorder."""
        if self.result_written or not self.run:
            return
        recorder = REPO / "evaluation_progress/record_run.py"
        got = subprocess.run(
            [sys.executable, str(recorder), "--run", str(self.run),
             "--output", str(self.csv_path)],
            cwd=REPO, capture_output=True, text=True, timeout=300, check=False)
        if got.returncode:
            raise RuntimeError((got.stderr or got.stdout or
                                "final result recorder failed").strip())
        self.result_written = True
        self.args.emit(f"{now()} {self.task} {(got.stdout or '').strip()}")

    def retryable_setup_failure(self):
        """Return the launch-race reason when no agent work ever started.

        ``pick_gpus`` deliberately samples before returning.  A different
        launcher can still reserve the selected devices in the tiny interval
        before lifecycle takes its authoritative lease.  That is queue
        contention, not an evolve result, and must never create a CSV row.
        """
        if not self.run:
            return ""
        journal = self.run / "journal.jsonl"
        try:
            events = [json.loads(line) for line in journal.read_text().splitlines()
                      if line.strip()]
        except (OSError, ValueError):
            return ""
        if any(event.get("event") in {"agent_launched", "agent_finished"}
               for event in events):
            return ""
        for event in events:
            why = str(event.get("why", ""))
            if (event.get("event") == "run_state"
                    and event.get("frm") == "CREATED"
                    and event.get("to") == "FAILED"
                    and event.get("classification") == "HARNESS"
                    and "reserved by running container" in why):
                return why
        return ""

    def requeue_setup_race(self, reason):
        self.setup_retries += 1
        self.last_setup_failure = reason
        self.args.emit(
            f"{now()} {self.task} launch race; requeueing without CSV row "
            f"({self.setup_retries}/{self.args.max_setup_retries}): {reason}")
        if self.log:
            self.log.close()
        if self.run:
            self.before.add(self.run)
        self.process = None
        self.run = None
        self.gpus = ""
        self.state = "WAITING_GPU"

    def launch(self, reserved):
        if self.state != "WAITING_GPU":
            return
        spec = yaml.safe_load((REPO / "serving_tasks" / self.task / "contract.yaml").read_text())
        excluded = {g for g in os.environ.get("EXCLUDE_GPUS", "").split(",") if g.isdigit()}
        if self.args.gpus:
            excluded |= {str(i) for i in range(64)} - set(self.args.gpus.split(","))
        excluded.update(reserved)
        env = {**os.environ, "HOURS": str(self.args.hours), "STOP": "timebox",
               "AGENT_CLI": self.args.agent_cli, "AGENT_MODEL": self.args.model,
               "AGENT_EFFORT": self.args.effort,
               "AGENT_ENV": str(self.args.agent_env or ""),
               "TASK_ROOT": str(REPO / "serving_tasks"),
               "EXCLUDE_GPUS": ",".join(sorted(excluded, key=int))}
        got = subprocess.run(["bash", "-c", '. "$1"; pick_gpus "$2"', "gpu-selector",
                              str(REPO / "serverbench/scripts/server_evolve/evolve-common.sh"),
                              str(spec["devices"])], env=env, capture_output=True,
                             text=True, timeout=180, check=False)
        if got.returncode:
            return
        chosen = got.stdout.strip()
        if len(chosen.split(",")) != spec["devices"] or any(
                not p.isdigit() or p in excluded for p in chosen.split(",")):
            raise RuntimeError(f"Invalid GPU selection: {chosen!r}")
        self.gpus = chosen
        env["GPU"] = chosen
        self.launch_attempt += 1
        suffix = "" if self.launch_attempt == 1 else f".attempt-{self.launch_attempt}"
        self.log_path = self.args.state_dir / f"{self.task_key}{suffix}.log"
        self.log = self.log_path.open("x")
        launcher = REPO / "serverbench/scripts/server_evolve/evolve-task.sh"
        self.process = subprocess.Popen(["bash", str(launcher), self.task],
                                        cwd=REPO / "serverbench", env=env,
                                        stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.state = "LAUNCHING"


def run_cards(run_root):
    """The cards a run was created on, from its `run_created` event."""
    try:
        lines = (run_root / "journal.jsonl").read_text().splitlines()
    except OSError:
        return set()
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "run_created":
            return {str(g) for g in event.get("gpus") or []}
    return set()


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--agent-env", type=Path, default=None,
                        help="per-run credentials file for a model behind an API key; "
                             "omit for a model on the operator's own login")
    parser.add_argument("--agent-cli", choices=("claude", "codex"), default="claude")
    parser.add_argument("--model", default="accounts/fireworks/models/deepseek-v4p1-flash")
    parser.add_argument("--effort", default="xhigh", help="the agent's reasoning effort")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--gpus", default="",
                        help="cards this campaign may use, e.g. 0,1,2,3; default: any "
                             "free card not in EXCLUDE_GPUS")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--max-setup-retries", type=int, default=8,
                        help="requeue authoritative GPU-container launch races")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--state-dir", type=Path,
                        help="runtime logs/status/locks; defaults under artifacts/operations")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.agent_env and not args.agent_env.is_file():
        parser.error("agent-env does not exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.state_dir = (args.state_dir or
                      REPO / "serverbench/artifacts/operations" / args.campaign)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    watcher_log = args.state_dir / "watcher.log"

    def emit(message):
        with watcher_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")
        print(message, flush=True)

    args.emit = emit
    jobs = [Job(task, args) for task in args.task]
    state_path = args.state_dir / "status.json"
    while True:
        for job in jobs:
            if job.process:
                try:
                    job.collect()
                except (OSError, RuntimeError, ValueError) as exc:
                    emit(f"{now()} {job.task} telemetry error: {exc}")
                code = job.process.poll()
                if code is not None and job.state not in {"FINISHED", "FAILED"}:
                    reason = job.retryable_setup_failure() if code else ""
                    if reason and job.setup_retries < args.max_setup_retries:
                        job.requeue_setup_race(reason)
                        continue
                    job.state = "FINISHED" if code == 0 else "FAILED"
                    try:
                        # A setup-only launch race is not an evolve result and
                        # therefore never belongs in the task CSV.
                        if not reason:
                            job.write_final_result()
                    except (OSError, RuntimeError,
                            subprocess.SubprocessError) as exc:
                        emit(f"{now()} {job.task} final CSV error: {exc}")
                elif code is None and job.run:
                    job.state = "RUNNING"
        reserved = {g for j in jobs if j.process and j.process.poll() is None for g in j.gpus.split(",")}
        for job in jobs:
            if job.state == "WAITING_GPU":
                try:
                    job.launch(reserved)
                except subprocess.TimeoutExpired:
                    continue
                if job.process:
                    reserved.update(job.gpus.split(","))
        state = {"observed_utc": now(), "campaign": args.campaign,
                 "jobs": [{"task": j.task, "status": j.state, "gpus": j.gpus,
                           "run": str(j.run or ""), "csv": str(j.csv_path),
                           "final_result_written": j.result_written,
                           "launch_attempt": j.launch_attempt,
                           "setup_retries": j.setup_retries,
                           "last_setup_failure": j.last_setup_failure,
                           "pid": j.process.pid if j.process else None} for j in jobs]}
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)
        emit(json.dumps(state))
        if all(j.state in {"FINISHED", "FAILED"} for j in jobs):
            return 0 if all(j.state == "FINISHED" and j.result_written
                            for j in jobs) else 1
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
