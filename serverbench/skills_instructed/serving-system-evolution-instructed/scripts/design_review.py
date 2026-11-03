#!/usr/bin/env python3
"""Check explicit design obligations and source-bound public evidence.

This checks finite experiments and declared numeric bounds, not a proof for all
inputs. Unknowns fail closed. Raw profiles are checked outside timed serving.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from profile_candidate import file_sha, gpu_events, sha
from run_trial import digest_tree, read_json, stage_holds, write_json

AXES = (
    "deployment",
    "semantics",
    "state_memory",
    "scheduling",
    "host_io",
    "device_compute",
    "topology_transfer",
)
SCOPE = (
    "model_revision",
    "dtype",
    "hardware",
    "workload",
    "context_limit",
    "request_semantics",
    "objective",
    "excluded_features",
)


def template():
    return {
        "schema": "serving_evolution.design/1",
        "hypothesis": "",
        "scope": dict.fromkeys(SCOPE, ""),
        "readings": {"vllm": "", "sglang": "", "working_systems": ""},
        "microbenchmarks": {
            kind: {"input_tokens": None, "output_tokens": None, "passed": False, "evidence": ""}
            for kind in ("short", "extreme")
        },
        "axes": {
            axis: {
                "impact": "",
                "invariant": "",
                "metrics": [
                    {"name": "", "before": None, "after": None, "unit": "", "evidence": "", "max": None}
                ],
            }
            for axis in AXES
        },
    }


def design_errors(body):
    errors = []
    coverage = body.get("microbenchmarks", {})
    context = body.get("scope", {}).get("context_limit")
    if type(context) is not int or context < 2:
        errors.append("scope.context_limit must be the task context limit in tokens")
    for kind in ("short", "extreme"):
        row = coverage.get(kind, {})
        if (
            type(row.get("input_tokens")) is not int
            or row.get("input_tokens", 0) < 1
            or type(row.get("output_tokens")) is not int
            or row.get("output_tokens", 0) < 1
            or row.get("passed") is not True
            or not Path(row.get("evidence", "")).is_file()
        ):
            errors.append(f"microbenchmarks.{kind}: need completed token counts and passing evidence")
    short, extreme = coverage.get("short", {}), coverage.get("extreme", {})
    if (
        type(context) is int
        and all(
            type(r.get("input_tokens")) is int and type(r.get("output_tokens")) is int
            for r in (short, extreme)
        )
        and not (
            short["input_tokens"] < extreme["input_tokens"]
            and extreme["input_tokens"] >= 0.9 * context
            and all(r["input_tokens"] + r["output_tokens"] <= context for r in (short, extreme))
        )
    ):
        errors.append(
            "microbenchmark coverage must include a short request and >=90% context input, within task limits"
        )
    if not body.get("hypothesis"):
        errors.append("missing falsifiable hypothesis")
    for key in SCOPE:
        if not (body.get("scope") or {}).get(key):
            errors.append(f"missing scope.{key}")
    for key in ("vllm", "sglang", "working_systems"):
        if not (body.get("readings") or {}).get(key):
            errors.append(f"missing reading lesson: {key}")
    if set(body.get("axes", {})) != set(AXES):
        errors.append("all seven responsibility axes must be addressed exactly once")
    for axis in AXES:
        row = body.get("axes", {}).get(axis, {})
        if not row.get("impact") or not row.get("invariant"):
            errors.append(f"{axis}: state impact (including unchanged) and invariant")
        if row.get("not_applicable"):
            if axis != "topology_transfer" or not row.get("reason"):
                errors.append(f"{axis}: only inter-device transfer may be inapplicable, with reason")
            continue
        if not row.get("metrics"):
            errors.append(f"{axis}: missing metrics")
        for metric in row.get("metrics", []):

            def numeric(v):
                return type(v) in (float, int) and math.isfinite(v)

            if not all(metric.get(k) for k in ("name", "unit", "evidence")):
                errors.append(f"{axis}: metric needs name, unit and evidence path")
            if not numeric(metric.get("after")) or not numeric(metric.get("before")):
                errors.append(f"{axis}: before/after must be measured finite numbers")
                continue
            for bound, sign in (("max", 1), ("min", -1)):
                if metric.get(bound) is not None and (
                    not numeric(metric[bound]) or sign * metric["after"] > sign * metric[bound]
                ):
                    errors.append(f"{axis}/{metric.get('name')}: violates {bound}")
            evidence = Path(str(metric.get("evidence", "")))
            if not evidence.is_file():
                errors.append(f"{axis}: metric evidence file missing")
    return errors


def inspect(candidate, task, design, profile, trial):
    files = digest_tree(candidate)
    fingerprint = sha(files)
    errors = design_errors(read_json(design))
    p, t = map(read_json, (profile, trial))
    if (
        p.get("schema") != "serving_evolution.profile/1"
        or p.get("status") != "passed"
        or p.get("candidate_tree_sha") != fingerprint
    ):
        errors.append("missing/failed/stale CUDA profile for candidate SHA")
    artifacts = p.get("artifacts", {})
    if not artifacts:
        errors.append("profile has no native artifacts")
    for path, digest in artifacts.items():
        if not Path(path).is_file() or file_sha(path) != digest:
            errors.append("profile artifact changed or missing: " + path)
    export = next(
        (
            Path(path)
            for path in artifacts
            if Path(path).suffix == {"torch": ".json", "nsys": ".sqlite", "ncu": ".csv"}.get(p.get("backend"))
        ),
        None,
    )
    if export is None or gpu_events(p["backend"], export) < 1:
        errors.append("profile contains no CUDA kernels; profiling an HTTP client is insufficient")
    stages = {s.get("stage") for s in t.get("stages", []) if s.get("status") == "passed"}
    if (
        t.get("schema") != "serving_evolution.trial/1"
        or t.get("task_id") != task
        or t.get("status") != "passed"
        or t.get("candidate_sha256") != fingerprint
        or "correctness" not in stages
        or not stages.intersection({"replay", "full"})
    ):
        errors.append("need cold-start trial + protocol/context + replay for this candidate SHA")
    for stage in t.get("stages", []):
        path = Path(stage.get("evidence", ""))
        if not path.is_file() or file_sha(path) != stage.get("evidence_sha256"):
            errors.append("trial gate evidence missing or changed")
        elif not stage_holds(stage.get("stage"), read_json(path), task, fingerprint):
            errors.append("trial raw gate does not pass for current task/source")
    startup = Path(t.get("startup_parity", ""))
    if not startup.is_file():
        errors.append("cold-start parity evidence missing")
    else:
        boot = read_json(startup)
        if (
            boot.get("status") != "passed"
            or boot.get("candidate_tree_sha") != fingerprint
            or boot.get("task_id") != task
        ):
            errors.append("cold-start parity evidence failed or stale")
    return {
        "schema": "serving_evolution.iteration/2",
        "task_id": task,
        "candidate_tree_sha": fingerprint,
        "files": files,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "evidence_paths": {
            key: str(path.resolve())
            for key, path in zip(("design", "profile", "trial"), (design, profile, trial), strict=True)
        },
        "inputs": {str(path.resolve()): file_sha(path) for path in (design, profile, trial)},
        "scope": "bounded public checks; not a formal proof or final verdict",
    }


def validate_receipt(path, candidate, task):
    receipt = read_json(path)
    if (
        receipt.get("schema") != "serving_evolution.iteration/2"
        or receipt.get("task_id") != task
        or receipt.get("status") != "passed"
        or receipt.get("candidate_tree_sha") != sha(digest_tree(candidate))
    ):
        raise ValueError("missing/failed/stale iteration evidence; run design_review.py check")
    for name, expected in receipt.get("inputs", {}).items():
        if file_sha(name) != expected:
            raise ValueError("iteration evidence input changed: " + name)
    if len(receipt.get("inputs", {})) != 3:
        raise ValueError("iteration receipt must bind all three evidence inputs")
    paths = receipt.get("evidence_paths") or {}
    checked = inspect(candidate, task, *(Path(paths[key]) for key in ("design", "profile", "trial")))
    if checked["status"] != "passed":
        raise ValueError("; ".join(checked["errors"]))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check")
    for arg in ("candidate-dir", "design", "profile", "trial", "output"):
        check.add_argument("--" + arg, type=Path, required=True)
    check.add_argument("--task", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("use a fresh output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "init":
        write_json(args.output, template())
        return 0
    try:
        report = inspect(args.candidate_dir.resolve(), args.task, args.design, args.profile, args.trial)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {
            "schema": "serving_evolution.iteration/2",
            "task_id": args.task,
            "status": "failed",
            "errors": [str(exc)],
        }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
