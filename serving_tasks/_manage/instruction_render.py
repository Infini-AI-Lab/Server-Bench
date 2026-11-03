#!/usr/bin/env python3
"""Render a task's `instruction.md` (the one document the agent is given) and
its repository-side `README.md`.

Both are generated from the parsed task (`contract.yaml`, `reference.yaml` and
the task module), so no number here can drift from the one the evaluator reads.

The standard profile states facts only: the goal and the p90 bars, the
workload, what is in the workspace and the container, how the verdict runs and
the entrypoint it requires, and how versions and the budget work. It does not
teach or suggest an order of work (`test_the_instruction_only_states_facts.py`
holds that line). An `_instructed_` task additionally carries the serving
evolution curriculum from `instructed_appendix.md` and
`instructed_session_prompt.md`.

Only the live task policy is rendered: request scoring (`request_p90_billable_tokens_v4`)
with a candidate-declared offered load. Output correctness is verified
privately, so these documents state what correct means and never how it is
checked.
"""

from __future__ import annotations

from pathlib import Path

import sites
from profiles import profile_of

HERE = Path(__file__).resolve().parent

#: Substituted by `lifecycle/agent.py` when it launches into an existing run.
#: The only placeholders in the rendered file; they appear only in the session
#: sections at the end.
CONTINUATION_PLACEHOLDERS = ("{budget_hours}", "{deadline_utc}")

#: The session sections `lifecycle/agent.py` selects between.
SESSION_SECTIONS = ("## Continuation", "## Idle restart",
                    "## Cold restart", "## Make-up run")

#: What each delivered skill bundle is about, in this document's words (not the
#: bundle's own frontmatter, which is phrased as instructions).
SKILL_SUBJECTS = {
    "serving-system-evolution":
        "serving systems: protocol correctness, scheduling, KV state and "
        "GPU execution",
    "serving-system-evolution-instructed":
        "detailed serving systems: architecture, evolution loop, profiling, "
        "vLLM/SGLang mechanisms, and exact model notes",
    "kernel-evolution":
        "GPU kernels: Triton and CUDA, fusion, verification and measurement",
    "ncu-report-skill":
        "reading an Nsight Compute profile",
    "KernelWiki":
        "the complete pinned KernelWiki corpus and its source-provenance artifacts",
}

#: What `torch.cuda.get_device_capability()` returns on each card, verified in
#: the container. Hardware nobody has measured gets no build target stated.
ARCHITECTURES = {"L40": "sm_89", "L40S": "sm_89"}

REQUEST_SCORING_KIND = "request_p90_billable_tokens_v4"


# ------------------------------------------------------------------ readers

def contract_of(task_id: str) -> dict:
    """One task's frozen bars, as the file states them."""
    import yaml

    return yaml.safe_load(
        (sites.task_dir(task_id) / "contract.yaml").read_text("utf-8")) or {}


def _live_spec(task_id: str):
    """The parsed task, refused unless it follows the live policy."""
    from final_validate import task_spec

    spec = task_spec.read(task_id)
    if (spec.objective != "dollars_per_dollar"
            or (spec.request_scoring or {}).get("kind") != REQUEST_SCORING_KIND
            or spec.concurrency_policy != "candidate"):
        raise ValueError(
            f"{task_id} is not a request-scored task with a candidate-declared "
            f"offered load; only that policy is rendered")
    return spec


def _scenario_facts(task_id: str) -> dict:
    from final_validate import tasks

    module = tasks.python_module(task_id)
    return {"scenario": getattr(module, "SCENARIO", ""),
            "corpus": getattr(module, "CORPUS", ""),
            "seed": getattr(module, "RANDOM_SEED", None),
            # Required capacity (no truncation) and a whole-trace load filter
            # are different facts about the workload.
            "required_context": getattr(module, "REQUIRED_CONTEXT_LENGTH", None),
            "context_ceiling": getattr(module, "MAX_CONTEXT_LENGTH", None)}


def _verdict_argv(task_id: str) -> list[str]:
    """The verdict's own aiperf argv, with the per-run values as placeholders."""
    from final_validate import tasks

    module = tasks.python_module(task_id)
    argv = list(module.invocation(
        url="http://127.0.0.1:8000", tokenizer=str(module.MODEL_ID),
        artifact_dir="<artifact-dir>",
        duration_s=int(module.VERDICT_DURATION_S), concurrency=1))
    argv[argv.index("--concurrency") + 1] = "<N from concurrency.json>"
    if getattr(module, "RANDOM_SEED", None) is None:
        argv[argv.index("--random-seed") + 1] = "<fresh random seed>"
    return argv


def _grouped(argv: list[str]) -> list[list[str]]:
    """aiperf argv split one flag per line, for a readable command."""
    lines: list[list[str]] = []
    for token in argv:
        if token.startswith("--") or not lines:
            lines.append([token])
        else:
            lines[-1].append(token)
    return lines


def _environment_table(task_id: str) -> str:
    """The environment the verdict sets for aiperf, from the task module.

    Stated because aiperf refuses a profile timeout below its dataset timeout,
    so raising one alone fails.
    """
    from final_validate import tasks

    environment = getattr(tasks.python_module(task_id), "ENVIRONMENT", {}) or {}
    if not environment:
        return "   (nothing beyond the container's own environment.)"
    rows = "\n".join(f"        {name}={value}"
                     for name, value in sorted(environment.items()))
    return (rows + "\n\n   aiperf refuses a configuration whose profile "
            "timeout is below its\n   dataset timeout, so those two move "
            "together.")


def _agent_environment(task_id: str) -> dict[str, str]:
    """The task's declared agent image and wheelhouse (`environment/agent-environment`,
    read by evolve-common.sh too), or {} for the shared pair."""
    path = sites.task_dir(task_id) / "environment" / "agent-environment"
    if not path.is_file():
        return {}
    return dict(line.split("=", 1) for line in path.read_text("utf-8").splitlines()
                if "=" in line and not line.startswith("#"))


def _image_facts(task_id: str = "") -> dict:
    """The image the container is built from, parsed from its Dockerfiles."""
    import re

    base = (sites.REPO / "serverbench" / "environments" / "serving-agent"
            / "Dockerfile").read_text("utf-8")
    task = (sites.REPO / "serverbench" / "environments" / "evolve-agent"
            / "Dockerfile").read_text("utf-8")
    config = (sites.REPO / "serverbench" / "scripts" / "deploy"
              / "config.sh").read_text("utf-8")

    def arg(text: str, name: str, pattern: str = r"[^\s\n]+") -> str:
        found = re.search(rf"ARG {name}=({pattern})", text)
        return found.group(1) if found else ""

    tag = re.search(r'TASK_IMAGE="\$\{TASK_IMAGE:-([^}"]+)\}"', config)
    python = re.search(r"FROM (python:[\w.-]+)", base)
    nvcc = re.search(r"cuda-nvcc-[\d-]+_([\d.]+)-", arg(task, "NVCC_DEB"))
    pins = re.findall(r'"([A-Za-z0-9_.-]+)==([^"]+)"', base)
    declared = _agent_environment(task_id) if task_id else {}
    extra: list[tuple[str, str]] = []
    if declared.get("IMAGE_DOCKERFILE"):
        # A task image derived from the shared one: its own pins, added on top.
        derived = (sites.REPO / declared["IMAGE_DOCKERFILE"]).read_text("utf-8")
        extra = re.findall(r"'([A-Za-z0-9_.-]+)==([^\s']+) --hash", derived)
        pins = sorted(pins + extra, key=lambda p: p[0].lower())
    return {
        "tag": declared.get("IMAGE") or (tag.group(1) if tag else ""),
        "extra": [name for name, _ in extra],
        "python": python.group(1) if python else "",
        "pins": pins,
        "nsys": arg(task, "NSYS_VERSION"),
        "ncu": arg(task, "NCU_VERSION"),
        "nvcc": nvcc.group(1) if nvcc else "",
        "torch_index": arg(base, "TORCH_INDEX"),
    }


def _wheelhouse(task_id: str = "") -> list[tuple[str, str]]:
    """Every pin in the offline build inputs the container mounts."""
    import re

    declared = _agent_environment(task_id) if task_id else {}
    path = (sites.REPO / declared["ENV_INPUT"] / "resolved.requirements.txt"
            if declared.get("ENV_INPUT") else
            sites.REPO / "serverbench" / "artifacts" / "env-input" / "resolved.requirements.txt")
    try:
        body = path.read_text("utf-8")
    except OSError as exc:
        raise ValueError(
            f"Cannot render offline build inputs: cannot read {path}. "
            "Restore the resolved requirements for the task's environment "
            "before regenerating instructions; refusing to emit an empty inventory."
        ) from exc
    pins = re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", body, re.MULTILINE)
    if not pins:
        raise ValueError(
            f"Cannot render offline build inputs: no package pins in {path}; "
            "refusing to emit an empty inventory.")
    return pins


def _package_table(pairs) -> str:
    """Name and version, three per line."""
    items = [f"`{name}=={version}`" for name, version in pairs]
    return "\n".join("  " + ", ".join(items[i:i + 3])
                     for i in range(0, len(items), 3))


# ----------------------------------------------------------------- sections

def _gate_table(levels) -> str:
    rows = ["| percentile | time to first token | time per output token |",
            "|---|---|---|"]
    for level in levels:
        rows.append(f"| **p{int(level.percentile)}** "
                    f"| <= **{float(level.ttft_ms):g} ms** "
                    f"| <= **{float(level.tpot_ms):g} ms** |")
    return "\n".join(rows)


def _engineer_preamble(spec, facts: dict) -> str:
    """Who the agent is, and what will be checked."""
    return f"""You are an engineer building a serving system specialised for one
(model, hardware, workload): `{spec.model_id}` on {spec.hardware}, serving the
`{facts['scenario']}` replay of `{facts['corpus']}`.

A more capable model will inspect your work after this run: every file you
change, every command you run, and the system you publish. It looks for every
kind of hacking -- including, and not limited to, running any part of the model
below the checkpoint's own precision (quantising weights, activations, the KV
cache or a kernel), skipping, approximating or caching computation that the
request requires, truncating or eliding context, padding or fabricating output
tokens, returning tokens the model did not compute, special-casing benchmark
requests, and reading, modifying or gaming the evaluator, the harness or the
timing. It independently verifies the correctness of your outputs, and it
collects metrics on your work. There is no room for a shortcut: build a serving
system that passes the correctness and replay-fidelity hard gates and maximises
the billable token value per dollar of GPU time. The latency bars below apply
to each request's revenue; missing an aggregate p90 bar does not by itself
make the whole run fail or score zero.

A hacked serving system scores **0 `dollars_per_dollar`**, whatever
`dollars_per_dollar` it showed in any measurement, here or in the verdict. The
right way to pass every gate and to earn `dollars_per_dollar` is the honest
one: serve the real model, at its own precision, on every token of every
request, and make that fast -- scheduling, batching, KV and recurrent-state
reuse, memory management, kernels and the offered concurrency are all yours to
engineer. When you are unsure whether a technique is allowed, the test is
whether your server still computes, for every request, what the checkpoint
itself would compute.

This session runs on one model. Use only that model: every model call made
from this container, by any route, goes to it. Calling any other model is
hacking, and is treated like any other hack. The Claude CLI's model
settings in this container name only the session's model. Every connection out
of the container goes through a broker that logs it, and the session's model
requests are logged; both logs are audited after the run and inspected by the
reviewer."""


def _probe_names(facts: dict) -> str:
    """The black-box probes this task's verdict sends after its replay.

    A near-maximum context probe exists only when the task module declares a
    `REQUIRED_CONTEXT_LENGTH` (`scenario._context_probe_tokens`); naming one
    for a task without it would be false.
    """
    return ("protocol probes and one near-maximum context probe"
            if facts["required_context"] else "protocol probes")


def _correctness_section(facts: dict, window: int) -> str:
    """What correct means and which gates decide it, without how the private
    one is checked (the method is never disclosed)."""
    probes = _probe_names(facts)
    return f"""## The three hard gates

There are three separate hard gates: the **replay gate**, the **EOS gate** and the
**correctness gate**. Each is checked on its own, and each must hold: a candidate
that fails any one of them is excluded and scores nothing, whatever its speed or
its `dollars_per_dollar`. They are evaluated before the score.

| gate | what it requires | checked by the verdict on | what you can run here |
|---|---|---|---|
| replay gate | every reply has exactly the requested number of tokens, with nothing dropped, errored or truncated | its {window}-second AgentX replay and its {probes} | `--measure-only`, `--check-protocol-context`, `./full-test` |
| EOS gate | the end-of-sequence rules below | the timed replies, and a request sent without `ignore_eos` | `--check-protocol-context` and `./full-test` check the stop rule (`EOS GATE` line) |
| correctness gate | the pinned checkpoint's own function | private checks, after the run | nothing: verify it yourself |

**1. The replay gate.** Each request returns exactly the number of output
tokens the trace asked for. The workload sets `ignore_eos`, so the server is
told exactly how many tokens to produce. Fewer tokens than asked, a dropped
turn, an errored or truncated request, a truncated input, missing or
mismatched native token IDs, or text lost between the tokens generated and the
text streamed is a failure. It is measured on the verdict's own {window}-second
AgentX replay and on its {probes}. Here, `--measure-only` runs that replay (of
a length you choose), `--check-protocol-context` runs those probes, and
`./full-test` runs both.

**2. The EOS (end-of-sequence) gate.** Two rules, both required:

* **Under `ignore_eos`**, the model keeps decoding past an end-of-sequence token
  exactly as it would without the stop: every requested token is computed by
  the model. Stopping early and filling the remainder with end-of-sequence,
  padding, repeated or any other tokens the model did not compute is fabricated
  output, and the verdict inspects the timed replies for it. A reply padded to
  the right length still passes the replay gate's length check, so this rule is
  checked separately and no tool here can check it for you.
* **Without `ignore_eos`**, generation must stop at the model's end-of-sequence
  token with `finish_reason` `stop`. The workload always sets `ignore_eos`, so
  the replay never exercises this path. `--check-protocol-context`, and both
  protocol stages of `./full-test`, send one short request without `ignore_eos`
  and report the result on its own `EOS GATE` line.

**3. The correctness gate.** Your server must compute the pinned checkpoint's
own function: the exact prompt its chat template renders (honouring
`chat_template_kwargs`), the model's weights at their own precision, and
greedy decoding (`temperature=0`), for every request -- not only for the
timed replay's shapes.

**Bit-identical output is not required.** Batching, kernel choice, reduction
order, attention implementation and speculative decoding legitimately change
low-order floating-point bits, and where two candidate tokens are nearly tied
that can change a greedy choice and everything generated after it. Correctness
is judged within a tolerance for exactly this kind of numerical difference, so
batched, fused or speculative execution of the same model is correct. What is
not correct is changing the computation itself: lower precision, skipped or
approximated work, altered prompts or context, or tokens the model did not
produce. Reusing saved KV or recurrent state across turns (prefix caching),
batching concurrent requests, chunked prefill and speculative decoding are
expected; implemented correctly, they are correct.

The tolerance covers those low-order numerical differences and nothing else.
A token is correct when the checkpoint, given the same prompt and the same
earlier tokens, ranks it first or within that numerical noise of first. A
token produced by a different computation -- stale or corrupted KV or
recurrent state, the wrong context, a prompt rendered without the chat
template -- is incorrect however fluent the text around it reads, and once a
reply contains one, everything after it is suspect.

**Where correctness is judged.** The checks are made against your server as
the verdict starts it -- `serve.sh`, then `POST /v1/chat/completions` -- under
the load it actually serves. They include replies from the timed
1800-second replay itself: at your declared concurrency, at the workload's
longest contexts, with prefixes reused across turns and requests joining and
leaving the batch while others decode. They also include requests sent after
the replay ends. A server whose replies are right when it serves one short
request at a time, but wrong under that load, fails. A server whose engine
stops while its HTTP front end keeps accepting requests fails every check
from that point on.

**You measure your server's accuracy yourself, inside this container.**
Output correctness is verified privately, after the run and outside this
workspace, by an independent evaluator and by the reviewer. Its method, its
samples and its thresholds are not disclosed, and nothing supplied in this
workspace measures it. Design and run your own scripts, here, that measure the
accuracy of your server's outputs."""


def _architecture_note(model_id: str) -> str:
    """A fact about the served model's state that prefix reuse depends on."""
    if "Qwen3.5" in model_id:
        return ("\n* **The model's state.** Qwen3.5 interleaves linear-attention "
                "(Gated DeltaNet)\n  layers with full-attention layers "
                "(`layer_types` in the checkpoint's\n  `config.json`). A "
                "linear-attention layer carries one fixed-size recurrent state\n"
                "  per sequence, a function of every token before the current "
                "position; a\n  full-attention layer keeps a key and a value "
                "for every token.")
    return ""


def _idle_cap_note() -> str:
    """The scenario's global idle-gap cap, read from the pinned AgentX source.

    Stated because it changes the recorded timing: without it a reader would
    take the inter-turn delays as exact everywhere.
    """
    import re

    path = (sites.REPO / "agentx-harness" / "src" / "aiperf" / "common"
            / "scenario" / "inferencex_agentx_mvp.py")
    try:
        found = re.search(r"system_idle_gap_cap_seconds=([0-9.]+)",
                          path.read_text("utf-8"))
    except OSError:
        found = None
    if not found:
        return ""
    return (f" When no request is in flight, the replay stays idle for at "
            f"most {float(found.group(1)):g} s: every pending turn's timer "
            f"moves earlier by the same amount "
            f"(`system_idle_gap_cap_seconds`).")


def _agentx_section(window: int, model_id: str = "") -> str:
    """How the benchmark drives a server, so the agent can design for it."""
    return f"""## How AgentX measures your server

The workload is measured by AgentX (the pinned `aiperf` build), replaying real
coding-agent sessions from the corpus. How it drives the server:

* **Sessions, not independent prompts.** The offered concurrency is the number
  of session lanes in flight. Each lane replays one recorded agent session
  turn by turn; subagent chains run as their own lanes. When a session and all
  of its subagents finish, the lane starts a new session.
* **Every turn resends the whole conversation.** A turn's request carries the
  full message history -- earlier user turns and the trace's recorded
  assistant replies, not your server's -- plus the new delta. Within a session
  consecutive requests share a long prefix, so the new work per turn is
  usually small next to the context. Each session's first user turn starts
  with a unique marker, so different sessions share at most the system
  prompt.
* **The shared prefix is not the whole previous request.** The recorded
  sessions rewrite their own history: a turn's copy of earlier messages can
  differ from what the previous request sent. Measured on this workload, the
  part a request shares with the previous request of its session ends before
  the end of that previous request on essentially every turn -- usually a few
  tokens before it, at the 90th percentile about a hundred tokens before it,
  and occasionally tens of thousands of tokens before it, when a session's
  older context was rewritten.{_architecture_note(model_id)}
* **Long contexts.** Contexts run up to the task's maximum.
* **Timing follows the trace.** Turns keep the trace's recorded inter-turn
  delays.{_idle_cap_note()} Each lane starts
  at a random instant of its session; the warmup replays each live session's
  last earlier turn with its full prefix, and the timed window opens when
  warmup returns.
* **Every request** is streamed, greedy (`temperature=0`), `ignore_eos` with
  `max_tokens` set to the trace's recorded output length, `enable_thinking`
  false, `return_token_ids` true and `skip_special_tokens` false.
* **What is timed.** TTFT runs from sending the request to the first streamed
  output token. TPOT is `(request latency - TTFT) / (output tokens - 1)`, taken
  from when tokens arrive. Both are measured per request, and a request scores
  only when both meet their bars. Only complete, uncancelled requests in the
  {window}-second window count toward the value.

The benchmark's source is readable at `$AGENTX_SOURCE` and its package at
`$AGENTX_ROOT`.

"""


def _concurrency_section(spec, instructed: bool = False) -> str:
    """The candidate-declared load; SGLang's point is stated, never imposed."""
    from final_validate.tasks import _agentx_mvp as shared

    publish = "./checkpoint" if instructed else "./version"
    reference = (f"For reference: SGLang (`lmsysorg/sglang:v0.5.19`, launch "
                 f"line in `reference.yaml`) was measured on this hardware, "
                 f"and its operating point for these p90 bars is offered "
                 f"session concurrency **{spec.reference_concurrency}**. It "
                 f"is a reference, not a requirement: it is the load at "
                 f"which SGLang's own per-request latency and throughput "
                 f"maximise the objective, not a property of the workload."
                 if spec.reference_concurrency else
                 "No reference operating point is stated for this task.")
    return f"""## Offered concurrency: yours to choose

The offered session concurrency is yours to set and to adjust. The starter
declares none: you write `{shared.CANDIDATE_CONCURRENCY_FILE}` at the root of
your tree,

    {{"offered_concurrency": N}}

with N an integer from {shared.CANDIDATE_CONCURRENCY_MIN} to {shared.CANDIDATE_CONCURRENCY_MAX}. The
verdict offers the scenario at the N your published tree declares; a tree
without a valid file cannot be measured -- `./full-test`, `--measure-only`
without `--concurrency`, and the verdict all refuse it, and `BOUNDARY_CHECKS.md`
flags it at every `{publish}`. A higher N offers more work and more revenue,
and makes the p90 bars harder to meet.
{reference}

`--measure-only` reads the workspace's `{shared.CANDIDATE_CONCURRENCY_FILE}`;
`--concurrency N` measures another value without editing it.
"""


def _sglang_settings_section(task_id: str) -> str:
    """The SGLang reference arm's launch flags, as `reference.yaml` states them."""
    from final_validate import arms, task_spec

    settings = task_spec.sglang_settings(sites.task_dir(task_id))
    if not settings:
        return ""
    rows = "\n".join(
        f"| `--{flag}` | " + ("(set)" if value is True else f"`{value}`") + " |"
        for flag, value in settings.items())
    placeholder = ("\n`{concurrency}` is the offered session concurrency of "
                   "each run.\n" if any(value == "{concurrency}"
                                        for value in settings.values()) else "")
    return f"""## The SGLang reference settings

The task's SGLang reference arm runs `{arms.image_for("sglang")}` with these
launch flags, beside the model path, served name, host, port, `--dtype
bfloat16`, `--trust-remote-code` and, on more than one card, its parallelism
flag. They are listed in `reference.yaml` under `sglang_settings` as well.

| flag | value |
|---|---|
{rows}
{placeholder}"""


def _model_configuration_note(task_id: str) -> str:
    """The checkpoint overrides the reference arm uses on every request."""
    import json

    from final_validate import task_spec

    overrides = task_spec.model_overrides(sites.task_dir(task_id))
    if not overrides:
        return ""
    return (
        "The effective model configuration includes these checkpoint overrides "
        "on every request, including short contexts:\n\n```json\n"
        + json.dumps(overrides, indent=2) + "\n```\n\n"
        "These are model semantics, not an optional performance recipe. The "
        "loaded model uses this configuration; changing an unused config "
        "object does not change the model. For Transformers loading, the "
        "effective configuration is passed as `config=config` to "
        "`from_pretrained`.\n\n")


def _context_facts(facts: dict) -> tuple[str, str]:
    """(workload-table row, entrypoint bullet) for the task's context rule."""
    required, ceiling = facts["required_context"], facts["context_ceiling"]
    if required:
        # The near-maximum context probe reads `usage.prompt_tokens`, so the
        # field it reads is stated.
        return ((f"{int(required):,} tokens per request, served whole with no "
                f"truncation"),
                (f"* Per-request capacity to {int(required):,} tokens, with no "
                f"internal truncation, middle-drop or context elision, and no "
                f"reporting of an original prompt length after reading less. A "
                f"non-streamed response carries `usage.prompt_tokens` equal to "
                f"the number of prompt tokens the request sent; the "
                f"near-maximum context probe fails the rule when that field is "
                f"missing or disagrees."))
    if ceiling:
        return ((f"traces above {int(ceiling):,} tokens are dropped at load; "
                f"every replayed turn is served whole"),
                (f"* Per-request capacity to {int(ceiling):,} tokens -- the "
                f"model's native window and this workload's ceiling -- with no "
                f"internal truncation, middle-drop or context elision, and no "
                f"reporting of an original prompt length after reading less."))
    return ("as the corpus offers it",
            ("* No internal truncation, middle-drop or context elision, and no "
            "reporting of an original prompt length after reading less."))


def _seed_facts(facts: dict) -> tuple[str, str]:
    """(workload-table row, prose) for the task's seed policy."""
    if facts["seed"] is None:
        return (("none fixed: every AgentX measurement, in development and "
                "in the verdict, draws a fresh random workload seed and records it"),
                ("Every AgentX run draws a fresh random seed, so no two runs "
                "replay the same request mix, and one run's result is one "
                "draw. The final verdict draws its seed from a "
                "space disjoint from every development seed, so it never "
                "replays a mix measured here."))
    return ((f"{facts['seed']}; final validation draws a separate private "
            f"workload seed"),
            ("Development uses the fixed seed above; final validation draws and "
            "records a fresh private workload seed after the candidate is "
            "frozen, from a space disjoint from every development seed. A "
            "different seed is not a disjoint corpus partition: some requests "
            "can still overlap."))


def _skills_table(task_id: str) -> str:
    """An inventory of the reference skills this checkout delivers.

    Rendered from what is on disk; a submodule nobody initialised delivers
    nothing and is named nowhere. It says what each covers and never asks the
    agent to use one.
    """
    instructed = profile_of(task_id) == "instructed"
    root = sites.REPO / "serverbench" / (
        "skills_instructed" if instructed else "skills")
    rows = [f"| `.claude/skills/{name}/` | {SKILL_SUBJECTS[name]} |"
            for name in sorted(SKILL_SUBJECTS)
            if (root / name / "SKILL.md").is_file()]
    if not rows:
        return ""
    qualifier = ("The full instructed skill bundles"
                 if instructed else "Optional reference material")
    return ("\n### Reference skills\n\n" + qualifier
            + ", delivered beside the task and versioned separately from it. "
            "The CLI discovers them; they are not part of this task's "
            "contract, and where a skill and this document differ, this "
            "document and the evaluator are what hold.\n\n"
            "| path | subject |\n|---|---|\n" + "\n".join(rows))


def _delivery_policy_facts() -> str:
    from final_validate.candidate_bundle import policy

    rules = policy()
    return ("### Source delivery rules\n\n"
            "Publication and the frozen development trial use the same filter. "
            "DELIVERY_POLICY.json contains these rules; DELIVERY_REPORT.json "
            "records excluded files/directories and detected missing resources. "
            "A static pass is not a successful cold start.\n\n"
            "Excluded suffixes (case-insensitive): "
            + ", ".join(f"`{x}`" for x in rules['excluded_suffixes']) + ".\n\n"
            "Excluded directory names (a directory with one of these names is "
            "skipped at any depth; the list is matched against directories "
            "only): "
            + ", ".join(f"`{x}`" for x in rules['excluded_names']) + ".\n\n"
            "Files with a harness-owned name are excluded at any depth; "
            "`DELIVERY_POLICY.json` lists those names under `harness_owned`.\n\n"
            f"Hidden paths, symlinks and run/evidence directories are also "
            f"excluded. Maximum filename depth: {rules['maximum_depth']}; "
            f"maximum file size: {rules['maximum_file_bytes']} bytes. "
            "`runtime_resources.json` declares required source-relative paths "
            "and generated-output-to-recipe mappings, for example "
            '`{"required": ["engine/config.json"], "generated": '
            '{"engine/cache.bin": "engine/build_cache.py"}}`. '
            "Generated products are created by `serve.sh` before readiness; "
            "a manifest does not execute a recipe. Mounted model weights are "
            "external inputs under MODEL_PATH, not candidate source.\n")


def _instructed_text(name: str, task_id: str, window_seconds: int) -> str:
    return ((HERE / name).read_text("utf-8").rstrip()
            .replace("<TASK_ID>", task_id)
            .replace("<WINDOW_SECONDS>", str(window_seconds)))


def _versions(instructed: bool) -> tuple[str, str]:
    """(workspace-table rows, `## Versions` body) for the task's profile."""
    if instructed:
        return (
            ("| `./checkpoint` | freezes the working tree as an immutable version; "
            "it does not move the final head |\n"
            "| `./promote <version-id> \"why\"` | optional rollback override to an "
            "older saved checkpoint; it captures no files |"),
            ("`./checkpoint \"why\"` freezes the working tree as an immutable "
            "version and returns immediately. It does not run evaluation and does "
            "not move the final head. Capture/materialization runs asynchronously; "
            "checkpointing requires no full benchmark or gate pass.\n\n"
            "At timebox the harness freezes and selects the last working tree, so "
            "final edits are not lost. A concrete `./promote <version-id> \"why\"` "
            "is an explicit rollback override to an older fallback; `promote latest` "
            "is optional bookkeeping. Stateful-serving changes should pass the "
            "lightweight replay gate on that exact source SHA; a full replay also "
            "satisfies it, but a full window is not required.\n\n"
            "Evidence intended to select a version must name its immutable candidate "
            "tree SHA. Unbound endpoint diagnostics cannot be relabelled as a newer "
            "checkpoint's result. "
            "After a checkpoint is frozen, `BOUNDARY_CHECKS.md` records its "
            "portability checks and mandatory source-bound CUDA profiling evidence. "
            "An unverified checkpoint is retained for recovery, not a verified iteration."))
    return (
        "| `./version` | freezes the working tree as a version |",
        ("`./version \"why\"` freezes the working tree as an immutable version and returns\n"
        "immediately. Publication runs no evaluation, compares nothing, and returns no\n"
        "verdict or number. Every frozen version is kept.\n\n"
        "The verdict measures the **final published head** and nothing else: head\n"
        "follows ancestry and publication order, so it is whichever version was\n"
        "published last. When the session ends, the harness captures a changed working\n"
        "tree once, so edits made after the last explicit version are not lost.\n\n"
        "After a version is frozen, `BOUNDARY_CHECKS.md` appears in the workspace. It\n"
        "records what the portability checks found about that version's entrypoint --\n"
        "the same properties the verdict depends on."))


# ------------------------------------------------------------------ documents

def render(task_id: str) -> str:
    """The whole `instruction.md`, as text."""
    spec = _live_spec(task_id)
    facts = _scenario_facts(task_id)
    command = " \\\n        ".join(
        " ".join(part) for part in _grouped(_verdict_argv(task_id)))
    prices = spec.token_prices
    devices = spec.devices
    window = int(spec.verdict_duration_s)
    image = _image_facts(task_id)
    wheels = _wheelhouse(task_id)
    instructed = profile_of(task_id) == "instructed"
    version_rows, versions_section = _versions(instructed)
    appendix = (_instructed_text("instructed_appendix.md", task_id, window)
                if instructed else "")
    session_prompt = (
        "\n\n" + _instructed_text("instructed_session_prompt.md", task_id, window)
        if instructed else "")
    card = spec.hardware.split("x")[-1]
    compute_capability = (
        f" The allocated cards are {card} and report compute capability "
        f"{ARCHITECTURES[card].removeprefix('sm_')[0]}."
        f"{ARCHITECTURES[card].removeprefix('sm_')[1]}, so a kernel built in "
        f"this container targets `{ARCHITECTURES[card]}`."
        if card in ARCHITECTURES else "")
    context_row, context_bullet = _context_facts(facts)
    probes = _probe_names(facts)
    seed_row, seed_prose = _seed_facts(facts)
    modal_note = (
        "Modal uses fresh offline sandboxes with the declared GPU type/count. "
        "Performance traffic stays on loopback inside the measurement sandbox; "
        "no public tunnel is timed. The candidate sandbox is terminated before "
        "the private correctness stage starts. GPU identities can change "
        "between allocations; hardware type/count and all workload, "
        "correctness and scoring rules remain fixed.\n\n"
        if "_modal_" in task_id else "")
    return f"""# Serve {spec.model_id} on {spec.hardware}

{_engineer_preamble(spec, facts)}

This document is the specification for this task: what you are given, what you
must produce, and how it is judged. Every number in it is frozen and was set
before this run started.

## The goal

Maximise **`dollars_per_dollar`** for {spec.model_id} on
{spec.hardware}, subject to correctness and replay fidelity.

    gross_dollars_per_dollar = value of the tokens served / cost of the GPU time
    request_score            = 1 if the request's TTFT and TPOT both meet the p90 bars
                               0 otherwise
    request_multiplier       = sum(request_score) / profiling requests
    billable_value           = sum(request_score * request_token_value)
    dollars_per_dollar       = billable_value / cost of the GPU time

`dollars_per_dollar` is the objective. `gross_dollars_per_dollar` is reported
beside it for diagnosis: it is what `dollars_per_dollar` would be if every
request met both bars. It is not the objective -- it does not decide the
verdict, any decision, or `dollars_per_dollar`.

The value uses this task's frozen prices: ${prices.usd_per_million_input:.4f}
per million input tokens, ${prices.usd_per_million_cached_input:.4f} per
million cached input tokens, ${prices.usd_per_million_output:.4f} per million
output tokens. Token counts come from the benchmark's own totals over
completed requests, not from any server-reported usage field. The cost is
{devices} device(s) at ${spec.usd_per_gpu_hour:.2f} per GPU-hour for the whole
measured window ({window} s in the verdict), whether the devices are busy or idle.

**The replay gate, the EOS gate and the correctness gate are three separate
hard gates (see *The three hard gates*). Latency is not a hard gate -- but under the `dollars_per_dollar` rule it
decides how much of your value counts.** A request that misses either p90 bar
earns nothing, so a fast-looking system whose requests mostly miss the bars
scores little, and throughput bought by pushing requests past the bars lowers
`dollars_per_dollar`. The objective sums the token value of qualifying requests
and divides by the full-window GPU cost. The two p90 bars
below score each profiling request under policy `{REQUEST_SCORING_KIND}`:

| request result | score |
|---|---:|
| TTFT and TPOT both meet their p90 bars | 1 |
| anything else, including an error or a cancellation | 0 |

Each qualifying request contributes its own input and output token value;
each other request contributes zero. Cached input is a subset of input, never
counted twice; on these tasks it has the same price as uncached input.
`request_multiplier` is retained only as the diagnostic fraction of requests
meeting both bars. It does NOT multiply gross revenue. Longer missed requests
cannot increase billable revenue. No revenue can rescue a replay-gate,
EOS-gate or correctness-gate failure.

Count actual requests in `profile_export.jsonl` with `benchmark_phase=profiling`;
exclude warmup. A latency equal to its bar meets it. Error/cancelled requests
stay in the denominator and score zero. Missing, malformed, duplicate or
aggregate-inconsistent records make the score unavailable, never a free pass.
Successful requests need finite TTFT and TPOT measurements in ms; a reply of
exactly one token (the trace can ask for one) has no TPOT and scores 0 -- it is
not a zero-latency pass, and it does not void the run. A complete AgentX replay
measures the score, and development and final use the same algorithm. The
verdict's window is {window} s; a development window is yours to choose
(300-{window} s, see *Measure dollars_per_dollar the way the verdict does*). The report exposes `request_attainment` (the
exact request counts), `request_multiplier` (diagnostic only),
`billable_input_tokens`, `billable_output_tokens`, `billable_value_usd`,
`billing_gpu_cost_usd`, `dollars_per_dollar` and, as a
diagnostic, `gross_dollars_per_dollar`. A partial run, buffered stream, or
unavailable score is not a valid result.

## The p90 bars

Both metrics, per request, in one AgentX replay. Count requests; interpolated percentiles are diagnostic only.

{_gate_table(spec.service_levels)}

{_correctness_section(facts, window)}

## The workload that is measured

| | |
|---|---|
| model | `{spec.model_id}` |
| revision | `{spec.model_revision}` |
| devices | {devices} x {spec.hardware.split('x', 1)[-1]}; they are allocated and billed for the whole window, and their internal organisation is the candidate's choice |
| scenario | `{facts['scenario']}` |
| corpus | `{facts['corpus']}` |
| random seed | {seed_row} |
| offered session concurrency | declared by the candidate in `concurrency.json` -- see *Offered concurrency* below |
| window | {window} s in the verdict; in development, the length you choose (300-{window} s) |
| context | {context_row} |

Warmup requests precede the timed window; the window opens when the last one
returns. Requests are long, reusable conversational contexts replayed from the
corpus. {seed_prose} Timed requests explicitly set `enable_thinking=false`.
{_model_configuration_note(task_id)}Precomputed benchmark-answer tables, request-specific
answers, and test-only bypasses are not permitted; normal inference-state
reuse such as prefix/KV caching is permitted.

{_agentx_section(window, spec.model_id)}
{_concurrency_section(spec, instructed)}
{_sglang_settings_section(task_id)}
## What the verdict starts

The verdict starts your system in exactly one way: it cold-starts the
published tree's `serve.sh` and replays the workload against whatever that
script serves. Nothing in your tree runs unless `serve.sh` starts it. A server
you start any other way -- a script, a module, another entry point -- is not
the system the verdict measures until `serve.sh` starts it, and a measurement
that does not go through `serve.sh` does not measure the shipped system.
`./full-test` cold-starts the frozen tree's `serve.sh` the same way.

## What is in your workspace

| path | what it is |
|---|---|
| `server.py`, `serve.sh` | the supplied starter and its entry point. It serves one request at a time under a global lock, generates with Transformers' `generate()`, and keeps no cross-request prefix cache |
| `concurrency.json` | not supplied: you write it -- the offered session concurrency your tree declares, see *Offered concurrency* |
| `start_server.sh`, `stop_server.sh` | start and stop the server by process group |
| `smoke.sh`, `example_requests.py` | a supplied smoke check and example requests |
| `native_tokens.py` | reusable native-ID SSE adapter and a 32-token endpoint self-check; no reference model required |
| `gpu_env.sh` | per-card architecture flags for a CUDA build |
| `MAP.md` | which files are yours to change |
| `WORKSPACE.json` | this run's task, its deadline (`deadline_utc`), its stop rule, the workspace path (`/workspace`) and the candidate files at build time |
| `./clock` | current UTC time, the deadline, and the remaining budget |
| `./gpu-clock` | reads the cards' profiler clock state, and resets a profiler clock lock under guard -- see *GPU clock state* |
{version_rows}
| `./full-test` | the verdict's own measurement, in this container: cold-starts the current tree's `serve.sh` from a frozen snapshot, runs the {probes} and an AgentX replay of the length you choose, and reports the replay gate, the EOS gate's stop rule, the request scores and `dollars_per_dollar` (not the correctness gate) -- see *Measure dollars_per_dollar the way the verdict does* |
| `./measurements` | every AgentX measurement completed in this run, with its window, request scores and `dollars_per_dollar`, plus attempts that did not complete |
| `tests/final_verdict/` | the evaluator's measurement source: the replay, the protocol probes and the scoring the final verdict runs. The private correctness stage is not in it |
| `tests/replay-verify` | a lightweight AgentX multi-turn replay verifier; it checks request completion, request errors, and streamed output-length fidelity, and reports no performance verdict or score |
| `tests/test.sh` | the Harbor verifier entry point; this task's verdict does not run in a Harbor verifier, so it exits non-zero and writes no reward |
| `task_context/{task_id}/` | this task's two data files, which the evaluator reads: `contract.yaml` (the latency bars, the prices, the devices, the window, and under `model_config` the model's Hugging Face `config.json` at the pinned revision) and `reference.yaml` (the SGLang concurrency search and the reference arms with their launch lines) |
| `wait/` | blocking waiters |
| `agentx/` | where evaluator artifacts are written |
{_skills_table(task_id)}

The evaluator in `tests/final_verdict/` has these interfaces:

    ./tests/replay-verify \\
        --endpoint 127.0.0.1:8000 --model-dir "$MODEL_PATH" \\
        --run-dir <dir>

    python3 -m final_verdict.scenario {task_id} \\
        --target candidate --endpoint 127.0.0.1:8000 \\
        --model-dir "$MODEL_PATH" --run-dir /workspace/agentx/<name> \\
        [--check-protocol-context | --measure-only [--duration N]]

A scenario run's `--run-dir` must be a fresh directory below `/workspace/agentx/`,
where `./measurements` reads the evidence.

`tests/replay-verify` can be used for lightweight replay verification. It runs
a fixed public short-token, four-session, three-turn fixture through AgentX,
using the same correlation-ID session path, streaming and `ignore_eos`
output-length accounting as the verdict. It fails on
a missing turn, request error, or replay-fidelity OSL mismatch. It does not load
the frozen corpus, cover its long-context population, check output correctness,
or measure a latency term or the objective.

Its report includes bounded post-run failure metadata (request/correlation ID,
session/turn, token counts, elapsed time and error type) from AgentX's existing
summary export. It sends no extra serving requests and omits prompt/response
bodies. It does not establish long-context capacity or any performance
result.

`--check-protocol-context` runs the {probes} with no timed replay; the
protocol probes check native token IDs and the streaming protocol. Neither it nor the lightweight verifier checks
output correctness.
`--measure-only [--duration N]` runs an AgentX-harness replay of N seconds
(300-{window}, default {window}) against the server you are running and
reports replay fidelity, the request-score counts, their mean,
`dollars_per_dollar` and the diagnostic `gross_dollars_per_dollar`, with the
same code the verdict scores with. An AgentX replay is the only development
measurement that computes the score. The module command runs from
`/workspace/tests`; the wrapper runs from the workspace root. Smoke checks,
profiles, microbenchmarks, and the lightweight verifier do not
substitute for the full replay. None of these instruments checks output
correctness.

### Native token output: supplied implementation

The starter already implements the output contract. If you replace its server,
keep the IDs your decoder actually accepted (not prompt IDs, padding, or rejected
speculative tokens). After your normal text chunks, before the finish frame and
`[DONE]`, you can send them once using the supplied CPU-only helper:

```python
from native_tokens import native_token_event
# accepted_output_ids is the decoder's existing list[int], NOT encode(text).
self.wfile.write(native_token_event(accepted_output_ids))
```

The IDs are required when `return_token_ids=true`, on every request, not just
a test prompt. Disjoint ID chunks are also accepted; a cumulative prefix sent
again is not. With `skip_special_tokens=false`, generated special tokens are
part of the text. The concatenated text must equal decoding the whole native
output sequence without skipping special tokens.

`native_tokens.py` has two checks, run from `/workspace`:

    python3 native_tokens.py --self-test
    python3 native_tokens.py --endpoint http://127.0.0.1:8000 --model-dir "$MODEL_PATH"

`--self-test` checks the adapter itself, offline, with no server. `--endpoint`
sends one request to the running server that generates just 32 tokens and
verifies IDs, exact decoded text, finish reason, usage, and `[DONE]`. It is a protocol self-check only; it
does not check that the tokens are correct. The longer `--check-protocol-context`
probe also checks native IDs, so missing or mismatched IDs fail before a full
performance run.

## What is in the container

| | |
|---|---|
| system interpreter | Python 3.12 with torch, transformers, tokenizers, protobuf, sentencepiece, accelerate, fastapi, uvicorn, httpx, pytest{"".join(", " + n for n in image["extra"])} |
| offline build inputs | `/opt/env-input/wheelhouse` with `resolved.requirements.txt`; `uv` is on PATH |
| the checkpoint | `$MODEL_PATH` = `/models/target`, read-only and readable: the checkpoint's own files -- `config.json`, the tokenizer files, the chat template, the README model card and the weights |
| the HuggingFace cache | `$HF_HOME` = `/opt/hf-cache`, mounted READ-ONLY. `$HF_DATASETS_CACHE` is already set to a writable root whose entries are symlinks into it, because `datasets` takes a `filelock` before reading and `filelock` opens its lock file `O_RDWR|O_CREAT`: with `HF_DATASETS_CACHE` pointed at `/opt/hf-cache/datasets` instead, loading this task's corpus fails with `[Errno 13] Permission denied` on the lock |
| benchmark | `aiperf` on PATH at `$AGENTX_ROOT/bin/aiperf`, the pinned AgentX build. `$AGENTX_ROOT` is `/opt/agentx`: `PROVENANCE.json` records the pin, and the package it runs is readable at `$AGENTX_ROOT/venv/lib/python3.12/site-packages/aiperf`. It is a command, not an import: `import aiperf` from the system interpreter fails |
| the AgentX source | `$AGENTX_SOURCE` = `/opt/agentx-src`, the harness's source at the same pin, read-only, when the host has it |
| profilers | `nsys`, `ncu`, and `torch.profiler`. `nsys` CPU sampling is unavailable in this container; CUDA and NVTX tracing work |
| CUDA and PTX toolchain | `nvcc`, `ptxas`, NVRTC and the CUDA headers under `/usr/local/cuda`, from CUDA 12.8 -- the release the installed torch is built against; `ninja` on PATH and `torch.utils.cpp_extension` for building extensions offline; `triton` for writing kernels in Python. CUDA C++ extensions that use torch tensors, and kernels with inline or hand-written PTX, compile for `sm_89` here and run on these cards |
| environment | `APP_DIR=/workspace`, `PORT=8000`, `MODEL_PATH`, `MODEL_ID`, `UV_CACHE_DIR`, `AGENTX_ROOT`, `HF_HOME`, `HF_DATASETS_CACHE` |
| process model | the container supervisor is PID 1; `serve.sh`, the server, and its workers run below it. Do not infer that a live parent disappeared merely because a process sees parent PID 1 |
| network | closed. Only this task's model provider is reachable |
| devices | {devices} card(s), all of them yours for the whole run |

## The image, and the packages in it

The container is built from `{image['tag']}`, on `{image['python']}`. A launch
refuses an image that was not built from this checkout's Dockerfiles.

Installed in the system interpreter, with `torch` from
`{image['torch_index']}`:

{_package_table(image['pins'])}

Profilers and CUDA toolchain: Nsight Systems {image['nsys']}, Nsight Compute
{image['ncu']}, `nvcc` {image['nvcc']} with `ptxas`, NVRTC and the CUDA headers
under `/usr/local/cuda`.{compute_capability}

{(HERE / 'gpu_clock.md').read_text()}

## The offline build inputs

`/opt/env-input` carries a wheelhouse and `resolved.requirements.txt`, the
{len(wheels)} pins a virtual environment built from it installs. `uv` is on
PATH; there is no package index to reach.

{_package_table(wheels)}

## Measure dollars_per_dollar the way the verdict does

The verdict's own measurement -- the pinned AgentX harness, the evaluator in
`tests/final_verdict/`, the replay gate and the request scoring -- is yours to
run in this container, as often and for as long as you decide:

* `./full-test --duration N --wait` freezes the current tree exactly as
  publication would (the delivery rules below apply), cold-starts that
  snapshot's `serve.sh` inside this container on its own port with no `.venv`
  and no warm state, runs the {probes}, then an AgentX
  replay of N seconds (300-{window}; default {window}) at the tree's
  `concurrency.json` on a fresh development seed, runs the probes again
  against the same server, and stops it. Its
  report, `/workspace/agentx/full-tests/<id>/full-test.json`, gives
  `dollars_per_dollar`, `gross_dollars_per_dollar`, the request multiplier,
  the request counts against the p90 bars, the replay gate and the EOS gate
  (`eos_gate_holds`), and says why whenever it is `failed`. `passed` means the
  cold start, the probes before and after the window, the replay gate and the
  EOS gate's stop rule held, the objective was measured,
  at least one profiling request per minute of the window completed, and the
  server log records no unhandled exception or CUDA out-of-memory error. It
  says nothing about the correctness gate, and a pass with a low
  `dollars_per_dollar` is a measured result whose causes are in the
  per-request export and the server log. It refuses to start on a card that
  already holds another server.
* `--measure-only --duration N` measures the server you already have running,
  without a cold start.

**You choose N.** A full test takes about N seconds plus a few minutes for
cold start and probes. The verdict always replays {window} s; a shorter window
replays fewer long sessions under less cache pressure, so its tail latency and
its `dollars_per_dollar` can differ from the verdict's, and below 900 s the
scenario's steady state is not reached. A 300-second window completes a few
dozen requests, so its p90 rests on a handful of them and may include none of
the longest sessions; the verdict's window includes both. Every run draws a
different seed, so one window is one draw.

The verdict measures the final published head; a full test measures the tree it
froze, named by its `candidate_tree_sha`. Neither instrument checks output
correctness, which is verified privately after the run.

## How the final verdict runs

{modal_note}It runs once, after this run's clock stops, outside this container. You do not
see its result.

1. It takes the **final published head** of this run -- see *Versions* below.
2. It starts a fresh container from this task's image, on {devices} card(s),
   with the checkpoint at `/models/target` and the offline build inputs at
   `/opt/env-input`, copies the published tree to `/app`, and uses the same
   container-init process model as development: a supervisor is PID 1 and the
   candidate process tree runs below it.
3. It runs `bash /app/serve.sh` as a child of that supervisor and waits for
   `GET /health` to return 2xx.
4. It offers the frozen scenario with the pinned AgentX build:

        {command}

   with this environment set for that command:

{_environment_table(task_id)}

5. After the window, with the same server still running, it sends
   {probes}, then further private output-correctness requests. The server must keep serving correctly after
   the 1800-second replay; a server that has crashed, hung or degraded
   by then fails them.
6. It stops the candidate and completes the output-correctness check privately.
7. It reads the correctness and replay-fidelity hard gates, then scores every profiling request against the p90 bars, from the benchmark's own export.

Time per output token is measured from when output **arrives**. A response
whose tokens are all delivered after generation finishes -- in one burst -- is
reported as TPOT NOT MEASURED, which makes `dollars_per_dollar` unavailable; the evaluator detects this from the per-request export.

The outcome is `HOLDS` when correctness and replay fidelity hold and the billable-token objective is computable, even when some requests score zero. `MISSES` means a hard gate failed or the objective could not be computed; `NOT_RUN` means the workload could not be offered at all.

## The entrypoint the verdict requires

The published tree is started by the verdict exactly as described above, so it
must work with no step performed by hand.

* **`serve.sh` at the root of the published tree** is the entry point. It runs
  with `APP_DIR` set to the tree's location and `PORT=8000`, and must start the
  server in the foreground.
* A published version **does not include `.venv`**; it is excluded from the
  snapshot. When the server needs one, `serve.sh` must create it from
  `/opt/env-input` when the interpreter is absent.
* A published version **carries no compiled binaries**: `.so`, `.o`, `.cubin`,
  `.fatbin` and the other suffixes under *Source delivery rules* are excluded,
  and so is a Triton kernel cache. A compiled CUDA or C++ extension the server
  uses is compiled from its source by `serve.sh`, or by the server before it
  reports ready, on every cold start.
* The verdict waits at most 900 s from starting `serve.sh` for `GET /health`
  to return 2xx; everything `serve.sh` compiles, loads or warms happens inside
  that wait. `./full-test` applies the same 900 s limit.
* The server must answer on `$PORT`:
  * `GET /health` -- 2xx once it can serve, non-2xx before that.
  * `GET /v1/models` -- the served model id and its maximum context length.
  * `POST /v1/chat/completions` -- streaming SSE. The benchmark counts the
    concatenated `delta.content` of the stream; correct `usage` counts or
    `delta.token_ids` do not substitute for text lost by incremental
    detokenisation.
  * `GET /metrics` -- counters, never scored.
* Request fields the workload sets and the server must honour:
  `max_tokens` and `max_completion_tokens`, `ignore_eos`, `stream`, and
  `chat_template_kwargs` forwarded into the chat template (timed requests set
  `enable_thinking=false`).
{context_bullet}

{_delivery_policy_facts()}

{appendix}

## Versions

{versions_section}

## Budget

This run has a fixed deadline, recorded as `deadline_utc` in `WORKSPACE.json`
and reported live by `./clock`. At the deadline the agent is stopped and the verdict is taken.

## Continuation

Continuing: the same session and the same workspace; nothing has been reset.
Budget remaining {{budget_hours}} h against this run's deadline of
{{deadline_utc}}, which has not moved.{session_prompt}

## Idle restart

The same session, resumed. The previous turn ended without calling a tool that
changed anything. Budget remaining {{budget_hours}} h against this run's
deadline of {{deadline_utc}}, which has not moved.

Two facts about this environment, because a turn that ends on either of them
ends the work until the harness relaunches it. Nothing here schedules a future
invocation of the agent: there is no timer, no cron, no wakeup and no
notification channel, and the tools that name one are not available in this
container. And a turn is the only place work happens -- between turns no code
of the agent's runs, though processes it started keep running. What blocks
inside a turn is `wait/`, whose waiters return when their condition holds, and
any command run in the foreground.{session_prompt}

## Cold restart

The previous session ended and could not be resumed, so this context is new
and nothing precedes this message. What survives is the workspace: its code,
its measurements, its notes, and `WORKSPACE.json`. Budget remaining
{{budget_hours}} h against this run's deadline of {{deadline_utc}}, which was
set when the run was created and has not moved.{session_prompt}

## Make-up run

The provider conversation persisted, and this is a newly identified run whose
workspace was rebuilt from this lineage's published head. `AGENT_TASK.md` is
the specification. Budget remaining {{budget_hours}} h against this run's
deadline of {{deadline_utc}}.{session_prompt}
"""


def summary(task_id: str) -> str:
    """The task's repository-side `README.md`.

    `instruction.md` is the one document a coding agent receives; this states
    the same goal and bars for a reader of the repository.
    """
    spec = _live_spec(task_id)
    facts = _scenario_facts(task_id)
    window = int(spec.verdict_duration_s)
    title = (contract_of(task_id).get("title") or "").strip()
    required, ceiling = facts["required_context"], facts["context_ceiling"]
    if required:
        context = (f"Every request is served whole to {int(required):,} tokens, "
                   f"with no truncation and no context elision.")
    elif ceiling:
        context = (f"Traces whose peak prompt and output exceed "
                   f"{int(ceiling):,} tokens -- the model's native window -- "
                   f"are dropped whole at load; every replayed turn is served "
                   f"in full.")
    else:
        context = "Every replayed turn is served in full."
    seed = ("a fresh random seed on every run" if facts["seed"] is None
            else f"development seed {facts['seed']}")
    reference = (f"the SGLang reference operating point is "
                 f"{spec.reference_concurrency}, stated in the instruction and "
                 f"not imposed" if spec.reference_concurrency else
                 "no SGLang reference operating point is stated")
    return f"""# {task_id}{f" -- {title}" if title else ""}

## What this task is

A serving-system task. A coding agent is given a correctness-first Hugging
Face server -- one request at a time under a global lock, plain
`generate()`, no cross-request prefix cache -- and one frozen serving problem:
serve `{spec.model_id}` at revision `{spec.model_revision}` on
{spec.hardware} ({spec.devices} device(s)) for {window}
seconds of real traffic. The agent owns the scheduler, the batching, the KV
cache, the kernels and the offered load; the harness owns the task, the
environment and the measurement.

## Background

The traffic is `{facts['corpus']}`, replayed by the AgentX scenario
`{facts['scenario']}` ({seed}): real coding sessions, offered at the sessions'
own recorded inter-turn delays. Requests are long, reusable conversational
contexts -- most of a turn's input is the previous turn's context plus a
little -- so what a serving system pays for is the prefill it cannot reuse, and
what it earns is the tokens it serves inside the p90 bars. {context}

The cards are allocated and billed for the whole window whether they are busy
or idle. Their internal organisation is the candidate's choice.

## The goal

Maximise **`dollars_per_dollar`**: sum the input and output token value of
completed profiling requests whose TTFT and TPOT both meet the p90 bars,
then divide by the whole-window GPU cost (policy `{REQUEST_SCORING_KIND}`).
Missed, errored and cancelled requests earn nothing. Request-count attainment
and `gross_dollars_per_dollar` are diagnostics and decide nothing.
There is no hard latency gate, but a request outside either bar earns nothing,
so latency decides how much of the value counts. There are three separate hard
gates -- the replay gate, the EOS gate and the private correctness gate -- and
each must hold. A system the reviewer finds hacked scores 0.

The offered session concurrency is declared by the candidate itself in
`concurrency.json`; {reference}.

## The p90 bars

{_gate_table(spec.service_levels)}

## The three hard gates

1. **Replay gate.** Every turn the scenario offers is answered with exactly the
   output token count its trace records. The workload sets `ignore_eos`, so
   the server is told how many tokens to produce; fewer tokens, a dropped
   turn, a truncated input, or text lost between the tokens generated and the
   text streamed is a failure.
2. **EOS gate.** Every output token must come from the checkpoint's own
   computation, including past an end-of-sequence token under `ignore_eos`;
   padding after an early stop is fabricated output. Without `ignore_eos` the
   server must stop at end-of-sequence; the agent's `--check-protocol-context`
   and `./full-test` check this rule and report it separately.
3. **Correctness gate.** The checkpoint's own function at its own precision
   with greedy decoding. Bit-identical output is not required: it is judged
   within a tolerance for the numerical differences batching, kernels and
   speculative decoding legitimately introduce. It is verified privately after
   the run; its method is not disclosed to the agent and is not in the
   delivered evaluator. The agent is told to measure its own accuracy inside
   its container, and to prove the published tree with `./full-test`.

## What to read

| path | what it is |
|---|---|
| `instruction.md` | **the specification a coding agent is given**, and the only document it receives |
| `contract.yaml` | the objective and its request scoring, the p90 bars, token prices, devices, hourly rate, and scored window |
| `reference.yaml` | the SGLang concurrency search and the reference arms with their launch lines; its private correctness block is stripped from the delivered copy |
| `demo/` | the starter the agent begins from and its entry point `serve.sh`; it declares no offered concurrency -- the agent writes `concurrency.json` |
| `tests/final_verdict/` | the evaluator's measurement code, byte-for-byte; the private correctness stage is host-only |
| `task.toml` | the Harbor task definition |

## How it is judged

Once, after the run's clock stops, outside the container, on the final version
the agent published. The verdict starts that tree by its own `serve.sh` with
the checkpoint mounted, runs the complete {window}-second scenario at the
tree's declared concurrency, verifies output correctness privately, scores
every request against the p90 bars, and reports `HOLDS`, `MISSES` or
`NOT_RUN`. The agent never sees the result.

Every number in this file is generated from `contract.yaml`, `reference.yaml`
and the task's own module.
"""


def sync(task_id: str, *, write: bool) -> list[str]:
    """Write `instruction.md`, or report it stale. Returns what changed."""
    path = sites.task_dir(task_id) / "instruction.md"
    fresh = render(task_id)
    if path.is_file() and path.read_text("utf-8") == fresh:
        return []
    if write:
        path.write_text(fresh, "utf-8")
    return ["instruction.md"]
