#!/usr/bin/env python3
"""Create a serving task, complete and consistent, in one command.

    "$SERVERBENCH_PY" serving_tasks/_manage/new_task.py --write \\
        --mode docker --task qwen3_5_9b_standard_2xL40 \\
        --model-id Qwen/Qwen3.5-9B --revision <sha> \\
        --usd-per-gpu-hour 0.82 --price-in 0.1044 --price-out 0.1855 \\
        --p90-ttft 10000 --p90-tpot 60 --reference-concurrency 4 \\
        --title "Qwen3.5 9B Standard" --question "<one sentence>"

Every task is created under the live policy: request scoring
(`request_p90_billable_tokens_v4`, the p90 row alone: bill only qualifying
requests' token value, divided by whole-window GPU cost), a
candidate-declared offered load (the starter declares none: the agent writes
and tunes `concurrency.json` itself; `--reference-concurrency` is the SGLang
operating point the instruction states as a reference), a fresh random workload seed on
every run, and the private correctness stage including the EOS gate.

Written: `contract.yaml` and `reference.yaml` (from `_template/shared/`),
`task.toml`, `MODE.md` and `environment/Dockerfile` (from `_template/<mode>/`),
`demo/` (copied from a
sibling task's starter), the harness module under
`serverbench/harness/final_validate/tasks/`, and the generated `instruction.md`,
`README.md` and packaged evaluator.

`--mode docker` runs in a local container on a host with the cards;
`--mode modal` runs in a Modal sandbox and its hardware suffix must be a card
Modal offers. Nothing is measured here: the SGLang operating point is measured
separately on the task's own hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

from profiles import profile_of

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TASKS = REPO / "serving_tasks"
TEMPLATES = TASKS / "_template"
MODULES = REPO / "serverbench" / "harness" / "final_validate" / "tasks"

#: The Harbor release every task image is composed with.
HARBOR_VERSION = "0.15.0"


def _task_hash(task: str) -> str:
    """`final_validate.spine.task_hash`: the id's SHA-256, first 8 hex digits."""
    return hashlib.sha256(task.encode("utf-8")).hexdigest()[:8]


def modal_cards() -> set[str]:
    """Cards Modal offers, read from `_modal/modalctl.py::KNOWN_GPUS`."""
    body = (TASKS / "_modal" / "modalctl.py").read_text("utf-8")
    found = re.search(r"KNOWN_GPUS\s*=\s*\(([^)]*)\)", body, re.DOTALL)
    if not found:
        return set()
    return {name.strip().strip("\"'").upper()
            for name in found.group(1).split(",") if name.strip()}


def agentx_commit() -> str:
    """The AgentX build the harness pins; the task image installs the same."""
    body = (REPO / "serverbench" / "harness" / "final_validate"
            / "agentx_pin.py").read_text("utf-8")
    return re.search(r'PINNED_COMMIT = "([0-9a-f]+)"', body).group(1)


def cache_rate_is(price_in: float, cached: float) -> str:
    """The contract's sentence on how cached input is priced."""
    share = "100%" if cached == price_in else f"{cached / price_in:.0%}"
    return (f"${cached:g}/M ({share} of the ${price_in:g}/M input rate), an "
            f"operator convention rather than a published price")


def hardware_of(task: str) -> tuple[str, int]:
    found = re.search(r"_(\d+)x([A-Za-z0-9-]+)$", task)
    if not found:
        raise SystemExit(f"{task} does not end in its hardware (e.g. `_2xL40S`)")
    return f"{found.group(1)}x{found.group(2)}", int(found.group(1))


def substitutions(args, hardware: str, devices: int) -> dict[str, str]:
    cached = args.price_cached if args.price_cached is not None else args.price_in
    derived = (f"  derived_from: {args.derived_from}\n"
               if args.derived_from else "")
    commit = agentx_commit()
    return {
        "TASK_ID": args.task,
        "TASK_HASH": _task_hash(args.task),
        "TITLE": args.title,
        "REQUEST_SCORING": ("request_scoring:\n"
                            "  kind: request_p90_billable_tokens_v4\n"
                            "  gross_objective: gross_dollars_per_dollar\n"
                            f"  reference_task: {args.task}\n" + derived +
                            "  attainment_basis: profiling_request_count\n"
                            "  aggregation: sum_token_value\n"
                            "  request_scores:\n"
                            "    p90_both: 1.0\n"
                            "    otherwise: 0.0\n"),
        "QUESTION": args.question,
        "HARDWARE": hardware,
        "DEVICES": str(devices),
        "USD_PER_GPU_HOUR": f"{args.usd_per_gpu_hour}",
        "PRICE_MODEL": args.model_id.lower(),
        "PRICE_IN": f"{args.price_in}",
        "PRICE_CACHED": f"{cached}",
        "PRICE_OUT": f"{args.price_out}",
        # JSON strings are valid YAML scalars; a bare one breaks on ": ".
        "CACHE_RATE_IS": json.dumps(cache_rate_is(args.price_in, cached)),
        "PRICE_SOURCE": json.dumps(args.price_source),
        "P90_TTFT_MS": f"{float(args.p90_ttft)}",
        "P90_TPOT_MS": f"{float(args.p90_tpot)}",
        "SGLANG_SERVE_EXTRA": json.dumps(args.sglang_serve_extra),
        "VLLM_SERVE_EXTRA": json.dumps(args.vllm_serve_extra),
        "TRTLLM_SERVE_EXTRA": json.dumps(args.trtllm_serve_extra),
        "HARBOR_VERSION": HARBOR_VERSION,
        "AGENTX_COMMIT": commit,
    }


def render(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    left = re.findall(r"\{\{([A-Z_]+)\}\}", text)
    if left:
        raise SystemExit(f"the template still carries {sorted(set(left))} "
                         f"after substitution")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("docker", "modal"))
    ap.add_argument("--profile", choices=("standard", "instructed"),
                    help="guidance template, inferred from the task id if omitted")
    ap.add_argument("--task", required=True,
                    help="<model-slug>_<profile>[_modal]_<NxCARD>, no ordinal")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--max-context-length", type=int, default=None,
                    help="whole-trace loader filter, not prompt truncation")
    ap.add_argument("--required-context-length", type=int, default=None,
                    help="required served context (does not filter the workload)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--question", required=True,
                    help="one sentence: what this task asks")
    ap.add_argument("--usd-per-gpu-hour", type=float, required=True)
    ap.add_argument("--price-in", type=float, required=True)
    ap.add_argument("--price-out", type=float, required=True)
    ap.add_argument("--price-cached", type=float, default=None,
                    help="defaults to the input rate (cache hits billed in full)")
    ap.add_argument("--price-source", default="Operator-supplied task tariff")
    ap.add_argument("--derived-from", default="",
                    help="the task whose bars this one shares, "
                         "recorded as request_scoring.derived_from")
    ap.add_argument("--p90-ttft", type=float, required=True)
    ap.add_argument("--p90-tpot", type=float, required=True)
    ap.add_argument("--reference-concurrency", type=int, default=None,
                    help="the SGLang operating point the instruction states "
                         "as a reference; the candidate chooses its own load")
    ap.add_argument("--sglang-serve-extra", default="")
    ap.add_argument("--vllm-serve-extra", default="")
    ap.add_argument("--trtllm-serve-extra", default="")
    ap.add_argument("--from-demo", default="",
                    help="a task whose demo/ to copy as this one's starter; "
                         "defaults to a task with the same hardware")
    ap.add_argument("--write", action="store_true",
                    help="create it; otherwise print what would be created")
    args = ap.parse_args(argv)

    if re.match(r"[Tt]\d+_", args.task):
        ap.error(f"{args.task}: a task id names its model, profile and "
                 f"hardware, not a position; drop the ordinal prefix")
    hardware, devices = hardware_of(args.task)
    try:
        profile_of(args.task, args.profile)
    except ValueError as exc:
        ap.error(str(exc))
    for name in ("max_context_length", "required_context_length"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.mode == "modal":
        card = hardware.split("x", 1)[1].upper()
        known = modal_cards()
        if known and card not in known:
            raise SystemExit(f"REFUSED: Modal does not offer {card!r}; a "
                             f"modal-mode task's hardware must be one of "
                             f"{sorted(known)}. For {card} use --mode docker.")

    mode, shared = TEMPLATES / args.mode, TEMPLATES / "shared"
    values = substitutions(args, hardware, devices)
    plan = {
        "contract.yaml": render((shared / "contract.yaml").read_text("utf-8"),
                                values),
        "reference.yaml": render((shared / "reference.yaml").read_text("utf-8"),
                                 values),
        "task.toml": render((mode / "task.toml").read_text("utf-8"), values),
        "MODE.md": (mode / "MODE.md").read_text("utf-8"),
        "environment/Dockerfile": render(
            (mode / "Dockerfile.in").read_text("utf-8"), values),
    }
    target = TASKS / args.task
    if not args.write:
        print(f"{args.task} ({args.mode} mode, {hardware}, {devices} device(s))")
        for name, body in plan.items():
            print(f"  would write {name} ({len(body)} B)")
        print("  would copy demo/ and generate instruction.md, README.md and "
              "tests/")
        return 0
    module = MODULES / f"{args.task.lower()}.py"
    if target.exists() or module.exists():
        raise SystemExit(f"REFUSED: {args.task} already exists")
    source = args.from_demo or _nearest_demo(args.task, hardware)
    if not (TASKS / source / "demo" / "serve.sh").is_file():
        raise SystemExit("--from-demo must name an existing task with demo/serve.sh")

    for name, body in plan.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, "utf-8")

    # The starter is source copied from a sibling, with its model pinned and
    # its declared load reset to the one request it serves at a time.
    shutil.copytree(TASKS / source / "demo", target / "demo")
    starter = target / "demo" / "server.py"
    if starter.is_file():
        body = starter.read_text("utf-8")
        for key, value in (("MODEL_ID", args.model_id), ("REVISION", args.revision)):
            body = re.sub(rf"^{key} = .+$", f"{key} = {json.dumps(value)}",
                          body, flags=re.MULTILINE)
        starter.write_text(body, "utf-8")
    # No starter load: the offered concurrency is the agent's decision.
    (target / "demo" / "concurrency.json").unlink(missing_ok=True)
    print(f"copied demo/ from {source}")

    module.write_text(_module_body(args, hardware, devices), "utf-8")
    print(f"wrote {module.relative_to(REPO)}")
    _generate(args.task)
    print(f"\n{args.task} is created and consistent."
          + ("" if args.reference_concurrency else
             " Measure SGLang on this hardware and record its operating point "
             "with `taskctl set --reference-concurrency N` so the instruction "
             "can state it."))
    return 0


def _nearest_demo(task: str, hardware: str) -> str:
    """A sibling's starter: same hardware if there is one, else any task."""
    siblings = sorted(p.name for p in TASKS.iterdir()
                      if not p.name.startswith(("_", "."))
                      and (p / "demo" / "serve.sh").is_file() and p.name != task)
    if not siblings:
        raise SystemExit("no existing task has a demo/ to copy; pass --from-demo")
    same = [s for s in siblings if s.endswith(hardware)]
    return (same or siblings)[-1]


def _module_body(args, hardware: str, devices: int) -> str:
    context = args.max_context_length
    required = args.required_context_length
    return f'''"""{args.task} -- {args.title}.

{textwrap.fill(args.question, 78)}

Generated by `serving_tasks/_manage/new_task.py`. Identity, scenario, corpus
and durations live here; the p90 bars, prices, devices and window are in
`contract.yaml`, and the SGLang evidence is in `reference.yaml`.
"""

from __future__ import annotations

from . import _agentx_mvp as shared

TASK_ID = "{args.task}"
MODEL_ID = "{args.model_id}"
MODEL_REVISION = "{args.revision}"

# BEGIN TASKCTL CONCURRENCY
# CANDIDATE-DECLARED. This task has no fixed offered load: the published tree
# declares one in `concurrency.json` ({{"offered_concurrency": N}}) and the
# verdict offers the scenario at N. REFERENCE_CONCURRENCY is the SGLang
# operating point on {hardware} that the instruction states as a reference and
# never imposes.
# END TASKCTL CONCURRENCY
OFFERED_CONCURRENCY: int | None = None
CONCURRENCY_POLICY = "candidate"
REFERENCE_CONCURRENCY: int | None = {args.reference_concurrency!r}

PROFILE = "{args.title}"

SCENARIO = shared.SCENARIO
CORPUS = shared.CORPUS
CORPUS_SHA256 = shared.CORPUS_SHA256
CORPUS_MAX_PROMPT_TOKENS = shared.CORPUS_MAX_PROMPT_TOKENS
ARRIVAL_PROCESS = shared.ARRIVAL_PROCESS
LOADER = shared.LOADER
CAPPED_SIBLING_NOT_USED = shared.CAPPED_SIBLING_NOT_USED
VERDICT_DURATION_S = shared.VERDICT_DURATION_S
EVOLVE_DURATION_S = shared.EVOLVE_DURATION_S
#: No fixed seed: every run draws a fresh workload seed (see
#: `_agentx_mvp.development_seed` / `final_seed`).
RANDOM_SEED = None
SCENARIO_DURATION_FLOOR_S = shared.SCENARIO_DURATION_FLOOR_S
ENVIRONMENT = shared.ENVIRONMENT
DEVICES = {devices}
MAX_CONTEXT_LENGTH = {context!r}
REQUIRED_CONTEXT_LENGTH = {required!r}


def invocation(url: str, tokenizer: str, artifact_dir: str,
               duration_s: int = shared.VERDICT_DURATION_S,
               concurrency: int | None = OFFERED_CONCURRENCY,
               max_context_length: int | None = MAX_CONTEXT_LENGTH) -> list[str]:
    """The aiperf command this task is measured with. The shared one."""
    return shared.invocation(
        url, tokenizer, artifact_dir, duration_s=duration_s,
        max_context_length=max_context_length,
        concurrency=concurrency, tokenizer_revision=MODEL_REVISION)


REFUSES = shared.refusal(TASK_ID)

MODULE = shared.module(
    task_id=TASK_ID,
    profile=PROFILE,
    reported_objective="dollars_per_dollar",
    workload_shape=(
        "AgentX " + shared.SCENARIO + " over the pinned corpus on "
        "{hardware}, at the offered concurrency the candidate declares in "
        "concurrency.json, with a fresh random workload seed on every run. "
        "The final verdict replays " + str(shared.VERDICT_DURATION_S)
        + " seconds; development windows are the agent's choice. The only ranked "
        "objective is dollars_per_dollar: the token value of completed profiling "
        "requests meeting both p90 bars divided by whole-window GPU cost, with "
        "replay fidelity and output correctness as hard eligibility gates."),
)
'''


def _generate(task: str) -> None:
    """The derived files: the starter header, two documents, the evaluator."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO / "serverbench" / "harness"))
    import taskctl

    taskctl.sync_starter_surface(task, write=True)
    taskctl.sync_sglang_settings(task, write=True)
    taskctl.sync_model_config(task, write=True)
    taskctl.render_generated_docs(task, write=True)
    taskctl.repackage(task)
    print("generated instruction.md, README.md and tests/")


if __name__ == "__main__":
    raise SystemExit(main())
