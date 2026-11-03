"""One offered load, measured end to end, judged by the task's own gates.

This is the unit every strategy is built from: stand the arm up on quiet
cards, offer the load for a full scored window, read the aggregate, and decide
against the frozen service levels. It returns a row and never raises for a
measurement outcome -- a gate miss and a failure to run are both results, and
telling them apart is the whole point of the `invalid` flag.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from final_validate.concurrency import REPO
from final_validate.concurrency.arm import (
    docker_device_spec,
    serve_extra_for,
    wait_for_quiet_cards,
)


def aiperf_command(task_id: str, *, url: str, tokenizer: Path,
                   artifact_dir: Path, duration_s: int,
                   concurrency: int,
                   random_seed: int | None = None) -> list[str]:
    """The task's own invocation at one offered load.

    Built from `module.invocation()` rather than retyped, so the corpus, the
    scenario, the loader, the stop rule and the export level cannot drift from
    what the verdict offers. The workload seed is a fresh development draw
    unless `random_seed` fixes it -- the way a repeat of one point is made
    comparable to another.
    """
    from final_validate import tasks as task_modules
    from final_validate.scenario import _pinned_aiperf_argv
    module = task_modules.python_module(task_id)
    # AgentX's offline workers resolve tokenizer cache identities, not mounted
    # snapshot paths. Keep calibration on the same pinned id as the verdict.
    tokenizer_identity = str(getattr(module, "MODEL_ID", "") or tokenizer)
    cmd = list(module.invocation(url, tokenizer_identity, str(artifact_dir),
                                 duration_s=duration_s,
                                 concurrency=concurrency))
    if random_seed is not None:
        cmd[cmd.index("--random-seed") + 1] = str(int(random_seed))
    return _pinned_aiperf_argv(cmd)


def aggregate(artifact_dir: Path):
    """aiperf's own aggregate for this point, or None if it never wrote one."""
    found = sorted(Path(artifact_dir).rglob("profile_export_aiperf.json"))
    if not found:
        return None
    try:
        return json.loads(found[-1].read_text("utf-8"))
    except (OSError, ValueError):
        return None


def measure(concurrency: int, *, plan, levels) -> dict:
    """One point: stand the arm up, offer the load, read the task's gates.

    `plan` is a `concurrency.Plan` or anything with the same attributes -- the
    CLI passes its argparse namespace straight through.
    """
    from serving_evolve.runtime import image_for

    from final_validate import deploy, service_levels
    from final_validate import tasks as task_modules
    from final_validate.scenario import _replay_fidelity

    run_dir = Path(plan.out_dir) / f"c{concurrency}"
    artifact_dir = run_dir / "agentx" / plan.arm
    # A POINT MUST NEVER READ AN AGGREGATE IT DID NOT JUST WRITE.
    #
    # `aggregate()` globs the artifact directory for profile_export_aiperf.json
    # and takes the last one. If a previous search left one there -- same task,
    # same arm, same offered load, DIFFERENT server configuration -- a run whose
    # server failed to start reads the old file and reports it as this run's
    # result, gate verdicts and all.
    #
    # So the point's artifact directory is emptied first. An arm that fails to
    # start reports `invalid` with nothing measured, which is the truth.
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    row: dict = {"concurrency": concurrency, "run_dir": str(run_dir)}
    task_dir = REPO / "serving_tasks" / plan.task

    try:
        blocked = wait_for_quiet_cards(plan.gpu)
        if blocked:
            row.update({"holds": False, "invalid": True, "why": blocked})
            return row
        serve_extra = (plan.serve_extra
                       or serve_extra_for(task_dir, plan.arm))
        from final_validate.deploy import relocate_draft_model
        serve_extra, draft_dir = relocate_draft_model(serve_extra)
        deploy.launch(
            plan.arm, image=image_for(plan.arm),
            model_dir=Path(plan.model_dir),
            device_spec=docker_device_spec(plan.gpu),
            port=plan.port,
            name=f"cal-{plan.arm}-c{concurrency}-{int(time.time())}",
            log=run_dir / "server.log", tp=len(str(plan.gpu).split(",")),
            draft_model_dir=draft_dir,
            serve_extra=serve_extra,
            ready_timeout_s=plan.ready_timeout)
        cmd = aiperf_command(plan.task, url=f"http://127.0.0.1:{plan.port}",
                             tokenizer=Path(plan.model_dir),
                             artifact_dir=artifact_dir,
                             duration_s=plan.duration,
                             concurrency=concurrency,
                             random_seed=getattr(plan, "random_seed", None))
        env = dict(os.environ)
        env.update(getattr(task_modules.python_module(plan.task),
                           "ENVIRONMENT", {}) or {})
        got = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True,
                             text=True,
                             timeout=max(3600.0, plan.duration * 4.0),
                             check=False)
        (run_dir / "aiperf.stdout.txt").write_text(
            (got.stdout or "") + "\n" + (got.stderr or ""), "utf-8")
        row["aiperf_exit"] = got.returncode
    except Exception as exc:                                   # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # `_release_all()` is the teardown; there is no `deploy.stop`. A point
        # that dies holding a container makes the next point wait on a card
        # nobody is watching.
        with contextlib.suppress(Exception):
            deploy._release_all()

    body = aggregate(artifact_dir)
    if body is None:
        # NOT A GATE MISS. An arm that never produced an aggregate did not
        # fail the bar -- it failed to run, and the two must never be recorded
        # the same way: a search that read a server which never started
        # (say, on a card a previous server is still releasing) as its first
        # miss would stop and report "max concurrency: None" as an answer.
        row.update({"holds": False, "invalid": True,
                    "why": (row.get("error") or
                            "no aiperf aggregate was written for this point, "
                            "so nothing was measured")})
        return row

    def stat(metric: str, statistic: str):
        entry = body.get(metric)
        return entry.get(statistic) if isinstance(entry, dict) else None

    measured = {
        "ttft_p90_ms": stat("time_to_first_token", "p90"),
        "ttft_p95_ms": stat("time_to_first_token", "p95"),
        "tpot_p90_ms": stat("inter_token_latency", "p90"),
        "tpot_p95_ms": stat("inter_token_latency", "p95"),
        "output_tokens_per_s": stat("output_token_throughput", "avg"),
        "request_count": stat("request_count", "avg"),
        "benchmark_duration_s": stat("benchmark_duration", "avg"),
        # This is HTTP request concurrency, not the offered trajectory count.
        # AgentX can fork several branches inside one trajectory, so it may
        # legitimately exceed --concurrency. It is diagnostic evidence, not a
        # test of whether the requested lane count was applied.
        "effective_concurrency_max": stat("effective_concurrency", "max"),
        "effective_concurrency_avg": stat("effective_concurrency", "avg"),
        "osl_mismatch_count": stat("osl_mismatch_count", "avg"),
        "error_request_count": stat("error_request_count", "avg"),
        # THE SAMPLE THIS POINT ACTUALLY DREW. Recorded per point because a
        # time box plus a heavy-tailed corpus means two loads do not complete
        # the same work, and a reader comparing throughput across points needs
        # to see the denominator move, even under one seed.
        "total_output_tokens": stat("total_output_tokens", "avg"),
        "total_input_tokens": stat("total_isl", "avg"),
        "mean_output_tokens_per_request": stat("output_sequence_length", "avg"),
        "mean_input_tokens_per_request": stat("input_sequence_length", "avg"),
        "prefix_cache_hit_pct": stat("theoretical_prefix_cache_hit", "avg"),
        "random_seed": (body.get("run_info") or {}).get("random_seed"),
    }
    checks = service_levels.check(
        levels,
        ttft_by_percentile={90: measured["ttft_p90_ms"],
                            95: measured["ttft_p95_ms"]},
        tpot_by_percentile={90: measured["tpot_p90_ms"],
                            95: measured["tpot_p95_ms"]})
    performance_holds = service_levels.holds(checks)
    fidelity = _replay_fidelity(body)
    mismatch_count = fidelity["osl_mismatch_count"] or 0
    error_count = fidelity["error_request_count"] or 0
    correctness_holds = (
        body.get("submission_valid") is not False
        and not body.get("was_cancelled")
        and fidelity["holds"])
    correctness_why = ""
    if not correctness_holds:
        correctness_why = (
            f"replay_fidelity missed: {mismatch_count:.0f} output-length "
            f"mismatch(es), {error_count:.0f} failed request(s), "
            f"submission_valid={body.get('submission_valid', True)}, "
            f"was_cancelled={bool(body.get('was_cancelled'))}")
    why = "; ".join(part for part in (
        service_levels.why(checks), correctness_why) if part)
    row.update({"measured": measured, "checks": checks,
                "replay_fidelity": fidelity,
                "performance_holds": performance_holds,
                "correctness_holds": correctness_holds,
                # Offered load is a pressure/performance choice. Keep a
                # reference server's output-protocol defect visible without
                # letting it erase the concurrency answer.
                "holds": performance_holds,
                "why": why,
                "seconds": round(time.time() - started, 1)})
    return row
