"""Build, inspect, change and check ServerBench serving tasks.

A task's data is two files -- `contract.yaml` (the p90 bars, token prices,
devices, hourly rate, window and request scoring) and `reference.yaml` (the SGLang
operating point and reference arms, plus the private correctness settings) --
and its harness module under `serverbench/harness/final_validate/tasks/`. Everything
else in a task directory is generated from those: `instruction.md`,
`README.md`, the starter's header, and the packaged evaluator
under `tests/final_verdict/`.

    taskctl show   <task>               the task as `task_spec` parses it
    taskctl check  [task]               every generated file in step, spec valid
    taskctl render [task]               regenerate documents and the evaluator copy
    taskctl set    <task> [params]      change a parameter, re-render, read back
    taskctl new    <task> [params]      create a task (forwards to new_task.py)
    taskctl rename <task> --to <id>     move a task, and its evidence, to a new id

`set` refuses to raise a p90 bar without `--reason`; a bar's history lives in
git (`git log -p --follow serving_tasks/<task>/contract.yaml`).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEMPLATES = REPO / "serving_tasks" / "_template"
for _path in (str(REPO / "serverbench" / "harness"),
              str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import blocks
import sites
from profiles import profile_of

# ------------------------------------------------------------------- gates

@dataclass
class Gates:
    """The p90 bar a task scores requests against."""

    p90_ttft: float
    p90_tpot: float

    def value_for(self, metric: str) -> float:
        return getattr(self, f"p90_{metric.lower()}")

    def describe(self) -> str:
        return f"p90 {self.p90_ttft:,.0f}/{self.p90_tpot:g} ms"

    def looser_than(self, other: Gates) -> list[str]:
        """Which bars this raises relative to `other`. A raise loosens."""
        return [f"p90 {metric.upper()} {other.value_for(metric):,g} -> "
                f"{self.value_for(metric):,g} ms"
                for metric in ("ttft", "tpot")
                if self.value_for(metric) > other.value_for(metric)]


def current_gates(task_id: str) -> Gates:
    """The p90 bar in force, from `contract.yaml`."""
    import yaml

    body = yaml.safe_load(
        (sites.task_dir(task_id) / "contract.yaml").read_text("utf-8")) or {}
    levels = {int(r["percentile"]): r
              for r in (body.get("service_levels") or [])}
    return Gates(p90_ttft=float(levels[90]["ttft_ms"]),
                 p90_tpot=float(levels[90]["tpot_ms"]))


def _write_gates(task_id: str, gates: Gates) -> list[str]:
    """The `service_levels` list in `contract.yaml` and the `task.toml` scalars."""
    touched: list[str] = []
    path = sites.task_dir(task_id) / "contract.yaml"
    text = path.read_text("utf-8")
    lines = text.splitlines(keepends=True)
    block = blocks.find(lines, "service_levels", indent=0, under=None)
    if block is not None:
        body = "".join(lines[block.body])
        fresh = _rewrite_level_items(body, gates)
        if fresh != body:
            blocks.write_checked(
                path, blocks.replace_body(text, block, fresh.splitlines()))
            touched.append("contract.yaml:service_levels")
    toml = sites.task_dir(task_id) / "task.toml"
    if toml.is_file():
        text = toml.read_text("utf-8")
        fresh = text
        for metric in ("ttft", "tpot"):
            key = f"gate_p90_{metric}_ms"
            value = sites.formatted(gates.value_for(metric), "")
            fresh = re.sub(rf"^{key}\s*=\s*[\d,._]+$", f"{key} = {value}",
                           fresh, flags=re.MULTILINE)
        if fresh != text:
            blocks.write_checked(toml, fresh)
            touched.append("task.toml:gate_p90_*")
    return touched


def _rewrite_level_items(body: str, gates: Gates) -> str:
    """Set ttft_ms/tpot_ms inside the `- percentile: 90` item."""
    out: list[str] = []
    in_p90 = False
    for line in body.splitlines(keepends=True):
        m = re.match(r"\s*-\s*percentile:\s*(\d+)\s*$", line.rstrip("\n"))
        if m:
            in_p90 = m.group(1) == "90"
            out.append(line)
            continue
        m = re.match(r"(?P<lead>\s*)(?P<key>ttft_ms|tpot_ms):\s*"
                     r"(?P<val>[\d,._]+)\s*$", line.rstrip("\n"))
        if m and in_p90:
            metric = "ttft" if m.group("key") == "ttft_ms" else "tpot"
            rendered = sites.formatted(gates.value_for(metric), m.group("val"))
            out.append(f"{m.group('lead')}{m.group('key')}: {rendered}\n")
            continue
        out.append(line)
    return "".join(out)


# ----------------------------------------------------------------- scalars

def _render(value, syntax: str) -> str:
    """A value as the target language reads it back.

    YAML 1.1 reads `2e-09` (no decimal point in the mantissa) as a string, so
    exponents are written with one.
    """
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "null" if syntax == "yaml" else "None"
    if isinstance(value, float):
        text = repr(value)
        if "e" in text or "E" in text:
            mantissa, _, exponent = text.partition("e")
            if "." not in mantissa:
                mantissa += ".0"
            return f"{mantissa}e{exponent}"
        return text
    return str(value)


def _set_scalars(task_id: str, name: str, value) -> list[str]:
    """One named parameter, everywhere `sites.SCALARS` says it lives."""
    touched = []
    for site in sites.SCALARS.get(name, ()):
        path = site.path(task_id)
        if not path.is_file():
            continue
        text = path.read_text("utf-8")
        rendered = _render(value, site.syntax)
        if site.syntax == "yaml":
            fresh, changed = blocks.set_scalar(
                text, site.key, rendered, indent=site.indent, under=site.under)
            if not changed and site.under and site.key not in text:
                fresh, changed = blocks.insert_scalar(
                    text, site.key, rendered, indent=site.indent,
                    under=site.under)
        else:
            # Keep an annotation if the module carries one.
            pattern = (rf"^{re.escape(site.key)}(?P<ann>\s*:[^=\n]*)?"
                       rf"\s*=\s*.*$")
            fresh, n = re.subn(
                pattern,
                lambda m, k=site.key, v=rendered:
                    f"{k}{m.group('ann') or ''} = {v}",
                text, flags=re.MULTILINE)
            changed = bool(n)
        if changed and fresh != text:
            blocks.write_checked(path, fresh)
            touched.append(f"{site.file}:{site.key}")
    return touched


# ---------------------------------------------------------- generated files

def repackage(task_id: str | None = None) -> str:
    """Regenerate each task's packaged evaluator under `tests/final_verdict/`."""
    from final_validate import package_grader

    tasks = [task_id] if task_id else sites.live_tasks()
    for task in tasks:
        package_grader.write(sites.task_dir(task))
    return (f"WROTE: {len(tasks)} grader copy(ies) -> serving_tasks\n"
            + "".join(f"    {task}\n" for task in tasks))


def render_generated_docs(task_id: str, *, write: bool) -> list[str]:
    """Every generated document of a task; returns the stale ones."""
    import instruction_render

    mode = "modal" if "_modal_" in task_id else "docker"
    generated = {
        "README.md": instruction_render.summary(task_id),
        "instruction.md": instruction_render.render(task_id),
        "MODE.md": (TEMPLATES / mode / "MODE.md").read_text("utf-8"),
    }
    if profile_of(task_id) == "instructed":
        # The team's two subagent definitions, delivered to `.claude/agents/`.
        for source in sorted((TEMPLATES / "instructed" / "agents").glob("*.md")):
            generated[f"agents/{source.name}"] = source.read_text("utf-8")
    stale = []
    for name, fresh in generated.items():
        path = sites.task_dir(task_id) / name
        if not path.is_file() or path.read_text("utf-8") != fresh:
            stale.append(name)
            if write:
                path.parent.mkdir(parents=True, exist_ok=True)
                blocks.write_checked(path, fresh)
    return stale


STARTER_HEADER = '''#!/usr/bin/env python3
"""Correctness-first starter for {task_id}.

This deliberately slow OpenAI-compatible server uses plain Transformers,
one global request lock, and no cross-request prefix cache. ``AGENT_TASK.md``
in the workspace is the task's specification. Model, hardware, gates, timebox,
correctness, and concurrency values do not live in this reusable
implementation's comments.
"""
'''


#: A full task id (`qwen3_5_2b_standard_1xL40`), as starter helpers copied
#: from a sibling write it.
TASK_ID_PATTERN = re.compile(r"\b[a-z][a-z0-9_]*_\d+x[A-Za-z0-9-]+\b")


def pinned_model_config(task_id: str) -> Path | None:
    """The task's model `config.json` at its pinned revision, if this host
    has that snapshot. Only the exact revision counts: a newer snapshot of
    the same repo can describe a different architecture."""
    from final_validate import model_identity, task_spec

    spec = task_spec.read(task_id)
    folder = "models--" + spec.model_id.replace("/", "--")
    for root in model_identity._hf_cache_roots():
        path = root / folder / "snapshots" / spec.model_revision / "config.json"
        if path.is_file():
            return path
    return None


def model_config_body(task_id: str, source: Path) -> list[str]:
    """`contract.yaml`'s `model_config` block: the Hugging Face `config.json`
    of the task's model at its pinned revision, verbatim, with where it came
    from. No digest: the delivered contract carries none, and `check`
    compares the content itself against the snapshot."""
    import json

    import yaml
    from final_validate import task_spec

    class Dumper(yaml.SafeDumper):
        pass

    def flow_scalars(dumper, data):
        flat = all(not isinstance(item, (dict, list)) for item in data)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data,
                                         flow_style=flat)

    Dumper.add_representer(list, flow_scalars)
    spec = task_spec.read(task_id)
    block = {"repo": spec.model_id, "revision": spec.model_revision,
             "file": "config.json",
             "config": json.loads(source.read_bytes())}
    text = yaml.dump(block, Dumper=Dumper, sort_keys=False,
                     default_flow_style=False, width=10_000,
                     allow_unicode=True)
    return ["  " + line for line in text.splitlines()]


MODEL_CONFIG_NOTE = ("# model_config: the model's Hugging Face config.json at the "
                     "pinned revision, verbatim;\n# written by `taskctl render` "
                     "from the local snapshot, never by hand.\n")


def sync_model_config(task_id: str, *, write: bool) -> list[str]:
    """Keep `model_config` in `contract.yaml` equal to the pinned config.json.

    Without the snapshot on this host the block cannot be re-derived; it is
    then checked against the task's pinned model id and revision only.
    """
    from final_validate import task_spec

    path = sites.task_dir(task_id) / "contract.yaml"
    text = path.read_text("utf-8")
    lines = text.splitlines()
    block = blocks.find(lines, "model_config", indent=0, under=None)
    source = pinned_model_config(task_id)
    if source is None:
        if block is None:
            return ["contract.yaml model_config (the pinned snapshot is not "
                    "on this host to write it from)"]
        spec = task_spec.read(task_id)
        stated = task_spec.contract(sites.task_dir(task_id))["model_config"]
        if (stated.get("repo"), stated.get("revision")) != (
                spec.model_id, spec.model_revision):
            return ["contract.yaml model_config (names another model or "
                    "revision than the task pins)"]
        return []
    body = model_config_body(task_id, source)
    if block is not None:
        if lines[block.body] == body:
            return []
        fresh = blocks.replace_body(text, block, body)
    else:
        fresh = (text.rstrip("\n") + "\n" + MODEL_CONFIG_NOTE
                 + "model_config:\n" + "".join(line + "\n" for line in body))
    if write:
        blocks.write_checked(path, fresh)
    return ["contract.yaml model_config"]


def sglang_settings_body(task_id: str) -> list[str]:
    """`reference.yaml`'s `sglang_settings` block: the pinned image and every
    flag of the SGLang arm's launch line, one per key."""
    import yaml
    from final_validate import arms, task_spec

    def scalar(value):
        if isinstance(value, str):
            for kind in (int, float):
                try:
                    return kind(value)
                except ValueError:
                    pass
        return value

    settings = {"image": arms.image_for("sglang")}
    settings.update({flag: scalar(value) for flag, value in
                     task_spec.sglang_settings(sites.task_dir(task_id)).items()})
    text = yaml.safe_dump(settings, sort_keys=False, default_flow_style=False,
                          width=10_000)
    return ["  " + line for line in text.splitlines()]


SGLANG_SETTINGS_NOTE = ("# sglang_settings: generated from arms[sglang].serve_extra "
                        "by `taskctl render`;\n# change the launch line, not this block.\n")


def sync_sglang_settings(task_id: str, *, write: bool) -> list[str]:
    """Keep `sglang_settings` in `reference.yaml` equal to the SGLang arm."""
    path = sites.task_dir(task_id) / "reference.yaml"
    text = path.read_text("utf-8")
    lines = text.splitlines()
    if not any(line.startswith("- framework: sglang") for line in lines):
        return []
    body = sglang_settings_body(task_id)
    block = blocks.find(lines, "sglang_settings", indent=0, under=None)
    if block is not None:
        if lines[block.body] == body:
            return []
        fresh = blocks.replace_body(text, block, body)
    else:
        anchor = blocks.find(lines, "arms", indent=0, under=None)
        if anchor is None:
            raise SystemExit(f"{task_id}: reference.yaml has no arms block")
        kept = text.splitlines(keepends=True)
        fresh = "".join(kept[:anchor.body_end] + [SGLANG_SETTINGS_NOTE,
                        "sglang_settings:\n"] + [line + "\n" for line in body]
                        + kept[anchor.body_end:])
    if write:
        blocks.write_checked(path, fresh)
    return ["reference.yaml sglang_settings"]


def sync_starter_surface(task_id: str, *, write: bool) -> list[str]:
    """Keep the starter's task-specific lines generated from the task.

    The starter is reusable source: its header, its effective model overrides
    (from the frozen reference launch line), the shared native-token adapter,
    and the one-line bar in `serve.sh` are the only task-specific parts.
    """
    from final_validate import task_spec

    spec = task_spec.read(task_id)
    root = sites.task_dir(task_id) / "demo"
    changed = []

    adapter = root / "native_tokens.py"
    adapter_text = (Path(task_spec.__file__).parent
                    / "native_tokens.py").read_text("utf-8")
    if not adapter.is_file() or adapter.read_text("utf-8") != adapter_text:
        changed.append("demo/native_tokens.py shared adapter and self-check")
        if write:
            blocks.write_checked(adapter, adapter_text)

    # A starter is copied from a sibling task, so the helpers name their task
    # by id; they must name this one, not the task they were copied from.
    for name in ("smoke.sh", "example_requests.py"):
        path = root / name
        if not path.is_file():
            continue
        before = path.read_text("utf-8")
        fresh = TASK_ID_PATTERN.sub(task_id, before)
        if fresh != before:
            changed.append(f"demo/{name} task id")
            if write:
                blocks.write_checked(path, fresh)

    server = root / "server.py"
    if server.is_file():
        before = server.read_text("utf-8")
        fresh = re.sub(r'\A#!/usr/bin/env python3\n""".*?"""\n',
                       STARTER_HEADER.format(task_id=task_id),
                       before, count=1, flags=re.DOTALL)
        overrides = "MODEL_CONFIG_OVERRIDES = " + repr(
            task_spec.model_overrides(sites.task_dir(task_id)))
        fresh = re.sub(r"^MODEL_CONFIG_OVERRIDES = .*", lambda _: overrides,
                       fresh, count=1, flags=re.MULTILINE)
        if fresh != before:
            changed.append("demo/server.py starter header")
            if write:
                blocks.write_checked(server, fresh)

    serve = root / "serve.sh"
    if serve.is_file():
        before = serve.read_text("utf-8")
        marker = "set -e\n"
        suffix = before[before.index(marker):] if marker in before else before
        levels = {level.percentile: level for level in spec.service_levels}
        fresh = (f"# Starter launcher for {task_id}.\n"
                 f"# Generated task bar: p90 TTFT <= {levels[90].ttft_ms:,g} ms; "
                 f"p90 TPOT <= {levels[90].tpot_ms:g} ms.\n"
                 "# AGENT_TASK.md is the task's specification.\n\n" + suffix)
        if fresh != before:
            changed.append("demo/serve.sh launcher header")
            if write:
                blocks.write_checked(serve, fresh)
    return changed


# ------------------------------------------------------------------ checks

def identity_drift(task_id: str) -> list[str]:
    """The directory, both data files and the module name one task."""
    import yaml

    root = sites.task_dir(task_id)
    bad: list[str] = []
    for name in ("contract.yaml", "reference.yaml"):
        path = root / name
        if not path.is_file():
            bad.append(f"{task_id} has no {name}, so it is not a task")
            continue
        body = yaml.safe_load(path.read_text("utf-8")) or {}
        if body.get("task_id") != task_id:
            bad.append(f"{task_id}/{name} names task_id "
                       f"{body.get('task_id')!r}")
    try:
        from final_validate import tasks as task_modules
        module = task_modules.python_module(task_id)
    except Exception as exc:                                   # noqa: BLE001
        return bad + [f"{task_id} has no readable module: {exc!r:.80}"]
    if getattr(module, "TASK_ID", "") != task_id:
        bad.append(f"{task_id}'s module names TASK_ID "
                   f"{getattr(module, 'TASK_ID', None)!r}; the verdict "
                   f"dispatches on it")
    return bad


def concurrency_drift(task_id: str) -> list[str]:
    """The candidate-declared load: the policy, the starter, the reference.

    The starter declares NO load -- the offered concurrency is the agent's
    decision -- and the stated SGLang
    reference must be the operating point `reference.yaml` records, when it
    records one.
    """
    from final_validate import task_spec
    from final_validate.tasks import _agentx_mvp as shared

    spec = task_spec.read(task_id)
    if spec.concurrency_policy != "candidate":
        return [(f"{task_id}: offers a fixed load; every live task declares "
                f"CONCURRENCY_POLICY = \"candidate\"")]
    bad: list[str] = []
    starter = (sites.task_dir(task_id) / "demo"
               / shared.CANDIDATE_CONCURRENCY_FILE)
    if starter.exists():
        bad.append(f"{task_id}: the starter declares an offered load in "
                   f"{starter.name}; the agent must choose its own -- delete it")
    search = task_spec.reference(sites.task_dir(task_id)).get(
        "concurrency_search") or {}
    adopted = search.get("adopted")
    if adopted is not None and adopted != spec.reference_concurrency:
        bad.append(f"{task_id}: REFERENCE_CONCURRENCY is "
                   f"{spec.reference_concurrency}, but reference.yaml's "
                   f"SGLang search adopted {adopted}")
    return bad


def agent_environment_drift(task_id: str) -> list[str]:
    """A task's declared agent image and wheelhouse must agree with each other.

    `environment/agent-environment` names a derived image (its Dockerfile) and a
    wheelhouse. The starter runs on the image's interpreter and a candidate
    installs from the wheelhouse, so every package the derived image adds must
    be pinned identically in the wheelhouse's resolved requirements, and the
    image must derive from the shared one.
    """
    path = sites.task_dir(task_id) / "environment" / "agent-environment"
    if not path.is_file():
        return []
    declared = dict(line.split("=", 1) for line in path.read_text("utf-8").splitlines()
                    if "=" in line and not line.startswith("#"))
    bad = [f"{task_id}/environment/agent-environment does not declare {key}"
           for key in ("IMAGE", "ENV_INPUT", "IMAGE_DOCKERFILE") if not declared.get(key)]
    if bad:
        return bad
    dockerfile = sites.REPO / declared["IMAGE_DOCKERFILE"]
    if not dockerfile.is_file():
        return [f"{task_id}: the declared image Dockerfile {dockerfile} is missing"]
    body = dockerfile.read_text("utf-8")
    if not re.search(r"^FROM serverbench/evolve-agent:cu128\s*$", body, re.MULTILINE):
        bad.append(f"{task_id}: {declared['IMAGE_DOCKERFILE']} does not derive from "
                   "serverbench/evolve-agent:cu128")
    if declared["IMAGE"] not in body:
        bad.append(f"{task_id}: {declared['IMAGE_DOCKERFILE']} never names the tag "
                   f"{declared['IMAGE']} it is built as")
    added = dict(re.findall(r"'([A-Za-z0-9_.-]+)==([^\s']+) --hash", body))
    resolved = sites.REPO / declared["ENV_INPUT"] / "resolved.requirements.txt"
    try:
        pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;]+)",
                               resolved.read_text("utf-8"), re.MULTILINE))
    except OSError:
        return bad + [f"{task_id}: cannot read the declared wheelhouse's {resolved}"]
    for name, version in added.items():
        if pins.get(name) != version:
            bad.append(f"{task_id}: the image adds {name}=={version} but the wheelhouse "
                       f"pins {pins.get(name)!r}")
    return bad


def image_drift(task_id: str) -> list[str]:
    """The task image must install the AgentX build the harness pins."""
    dockerfile = sites.task_dir(task_id) / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return []
    pinned = re.search(r"agentx-harness\.git@([0-9a-f]{7,40})",
                       dockerfile.read_text("utf-8"))
    if pinned is None:
        return [(f"{task_id}/environment/Dockerfile installs aiperf without "
                f"pinning a commit")]
    from final_validate import agentx_pin

    want = agentx_pin.PINNED_COMMIT
    if not (want.startswith(pinned.group(1))
            or pinned.group(1).startswith(want)):
        return [(f"{task_id}/environment/Dockerfile pins AgentX "
                f"{pinned.group(1)[:12]} and the harness pins {want[:12]}")]
    return []


def _correctness(spec) -> dict:
    """The task's private correctness block (host-side; never delivered)."""
    from final_validate import task_spec
    return task_spec.reference(sites.task_dir(spec.task_id)).get("correctness") or {}


#: The execution contract every live task shares; a task that differs on one
#: of these is not comparable with the others.
MUST_MATCH = (
    ("verdict_duration_s", lambda s: s.verdict_duration_s),
    ("evolve_duration_s", lambda s: s.evolve_duration_s),
    ("correctness_rule", lambda s: _correctness(s).get("rule")),
    ("max_token_regret", lambda s: _correctness(s).get("max_token_regret")),
    ("answer_sample_size", lambda s: _correctness(s).get("answer_sample_size")),
    ("max_absolute_accuracy_drop",
     lambda s: _correctness(s).get("max_absolute_accuracy_drop")),
    ("eos_audit", lambda s: _correctness(s).get("eos_audit")),
    ("request_scoring", lambda s: (s.request_scoring or {}).get("kind")),
)


def consistency() -> list[str]:
    """Where the live tasks disagree on something they must share."""
    from final_validate import task_spec

    specs = {}
    for task in sites.live_tasks():
        try:
            specs[task] = task_spec.read(task)
        except Exception as exc:                               # noqa: BLE001
            return [(f"{task}: unreadable, so consistency cannot be checked: "
                    f"{type(exc).__name__}: {exc}")]
    bad = []
    for name, get in MUST_MATCH:
        seen = {task: get(spec) for task, spec in specs.items()}
        if len(set(seen.values())) > 1:
            bad.append(f"the live tasks disagree on `{name}`, which is part of "
                       f"the shared execution contract: "
                       + ", ".join(f"{t}={v}" for t, v in sorted(seen.items())))
    for task, spec in specs.items():
        if spec.evolve_duration_s != spec.verdict_duration_s:
            bad.append(f"{task} develops at {spec.evolve_duration_s} s and is "
                       f"scored at {spec.verdict_duration_s} s")
    return bad


def problems_of(task_id: str) -> list[str]:
    """Everything wrong with one task. Empty is clean."""
    from final_validate import task_spec

    bad = identity_drift(task_id)
    if bad:
        return bad
    bad += [f"{task_id}: stale {part}; run `taskctl render {task_id}`"
            for part in (sync_starter_surface(task_id, write=False)
                         + sync_sglang_settings(task_id, write=False)
                         + sync_model_config(task_id, write=False))]
    try:
        bad += [f"{task_id}: stale {name}; run `taskctl render {task_id}`"
                for name in render_generated_docs(task_id, write=False)]
    except Exception as exc:                                   # noqa: BLE001
        bad.append(f"{task_id}: the generated documents could not be "
                   f"rendered: {type(exc).__name__}: {exc}")
    bad += concurrency_drift(task_id)
    bad += image_drift(task_id)
    bad += agent_environment_drift(task_id)
    bad += sites.verdict_drift(task_id)
    try:
        bad += task_spec.read(task_id).problems()
    except Exception as exc:                                   # noqa: BLE001
        bad.append(f"{task_id}: unreadable as a spec: "
                   f"{type(exc).__name__}: {exc}")
    return bad


# ------------------------------------------------------------------- verbs

def cmd_show(args) -> int:
    import dataclasses

    from final_validate import task_spec
    print(json.dumps(dataclasses.asdict(task_spec.read(args.task)), indent=1))
    return 0


def cmd_check(args) -> int:
    tasks = [args.task] if args.task else sites.live_tasks()
    problems: list[str] = [] if args.task else consistency()
    for task in tasks:
        problems += problems_of(task)
    if problems:
        print(f"{len(problems)} problem(s) across {len(tasks)} task(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{len(tasks)} task(s) clean: every generated file is in step with "
          f"the task's data")
    for task in tasks:
        print(f"  {task}")
    return 0


def cmd_render(args) -> int:
    """Regenerate every task's documents, starter header and evaluator copy."""
    tasks = [args.task] if args.task else sites.live_tasks()
    requested = getattr(args, "profile", None)
    if requested:
        tasks = [task for task in tasks if profile_of(task) == requested]
    for task in tasks:
        for changed in (sync_starter_surface(task, write=True)
                        + sync_sglang_settings(task, write=True)
                        + sync_model_config(task, write=True)):
            print(f"{task}: wrote {changed}")
        for name in render_generated_docs(task, write=True):
            print(f"{task}: wrote {name}")
        print(repackage(task).rstrip())
    return max((cmd_check(argparse.Namespace(task=task)) for task in tasks),
               default=0)


#: `set` flags that write one registered scalar each: (flag attribute, name).
SCALAR_FLAGS = (
    ("reference_concurrency", "reference_concurrency"),
    ("usd_per_gpu_hour", "usd_per_gpu_hour"),
    ("price_in", "usd_per_million_input"),
    ("price_out", "usd_per_million_output"),
    ("price_cached", "usd_per_million_cached_input"),
    ("price_model", "price_model"),
    ("price_source", "price_source"),
    ("max_token_regret", "max_token_regret"),
    ("answer_sample_size", "answer_sample_size"),
    ("max_absolute_accuracy_drop", "max_absolute_accuracy_drop"),
    ("reference_arm", "reference_arm"),
    ("max_context_length", "max_context_length"),
    ("required_context_length", "required_context_length"),
    ("model_id", "model_id"),
    ("model_revision", "model_revision"),
)


def _read_back(task_id: str, expect: dict) -> list[str]:
    """Compare what `set` asked for with what the task now parses to."""
    from final_validate import task_spec

    spec = task_spec.read(task_id)
    levels = {lv.percentile: lv for lv in spec.service_levels}
    got = {
        "p90_ttft": levels[90].ttft_ms, "p90_tpot": levels[90].tpot_ms,
        "reference_concurrency": spec.reference_concurrency,
        "usd_per_gpu_hour": spec.usd_per_gpu_hour,
        "usd_per_million_input": spec.token_prices.usd_per_million_input,
        "usd_per_million_output": spec.token_prices.usd_per_million_output,
        "usd_per_million_cached_input":
            spec.token_prices.usd_per_million_cached_input,
        "price_model": spec.token_prices.model,
        "price_source": spec.token_prices.source,
        "max_token_regret": _correctness(spec).get("max_token_regret"),
        "answer_sample_size": _correctness(spec).get("answer_sample_size"),
        "max_absolute_accuracy_drop": _correctness(spec).get("max_absolute_accuracy_drop"),
        "reference_arm": _correctness(spec).get("reference_arm"),
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
    }
    bad = []
    for key, value in expect.items():
        if key not in got:
            continue
        same = (str(got[key]) == value if isinstance(value, str)
                else got[key] is not None
                and abs(float(got[key]) - float(value)) < 1e-9)
        if not same:
            bad.append(f"asked for {key}={value} but the task reads back "
                       f"{key}={got[key]}")
    return bad + spec.problems()


def cmd_set(args) -> int:
    import new_task

    task = args.task
    if not (sites.task_dir(task) / "contract.yaml").is_file():
        raise SystemExit(f"{task} is not a task on disk")

    old = current_gates(task)
    new = Gates(
        p90_ttft=args.p90_ttft if args.p90_ttft is not None else old.p90_ttft,
        p90_tpot=args.p90_tpot if args.p90_tpot is not None else old.p90_tpot)
    loosened = new.looser_than(old)
    if loosened and not args.reason:
        raise SystemExit(
            "REFUSED: this raises a bar, which loosens it:\n  "
            + "\n  ".join(loosened)
            + "\n\nPass --reason saying why the bar is being revised, and "
              "put it in the commit message: a bar's history lives in git.")

    touched: list[str] = []
    expect: dict = {}
    if new != old:
        touched += _write_gates(task, new)
        expect |= {"p90_ttft": new.p90_ttft, "p90_tpot": new.p90_tpot}
        print(f"gates: {old.describe()} -> {new.describe()}"
              + (f"  (reason: {args.reason})" if args.reason else ""))

    if args.cache_fraction is not None:
        from final_validate import task_spec
        base = (args.price_in if args.price_in is not None
                else task_spec.read(task).token_prices.usd_per_million_input)
        args.price_cached = round(base * args.cache_fraction, 6)
    if args.model_id and not args.price_model:
        args.price_model = args.model_id.lower()
    for attr, name in SCALAR_FLAGS:
        value = getattr(args, attr, None)
        if value is None:
            continue
        touched += _set_scalars(task, name, value)
        expect[name] = value

    if any(v is not None for v in (args.price_in, args.price_cached)):
        from final_validate import task_spec
        prices = task_spec.read(task).token_prices
        touched += _set_scalars(task, "cache_rate_is", new_task.cache_rate_is(
            prices.usd_per_million_input, prices.usd_per_million_cached_input))

    touched += sync_starter_surface(task, write=True)
    touched += sync_sglang_settings(task, write=True)
    touched += sync_model_config(task, write=True)
    touched += render_generated_docs(task, write=True)
    print(f"{len(touched)} site(s) written:"
          + "".join(f"\n  {t}" for t in touched))
    print(repackage(task).rstrip())

    problems = _read_back(task, expect) + problems_of(task)
    if problems:
        print(f"\nREFUSED -- the write did not land cleanly "
              f"({len(problems)} problem(s)):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nverified: the task reads back exactly as asked")
    return 0


def cmd_new(args) -> int:
    """Create a task through `new_task.py`, the one creator."""
    import new_task

    forwarded: list[str] = ["--task", args.task]
    for flag, value in (
            ("--mode", args.mode),
            ("--profile", args.profile),
            ("--max-context-length", args.max_context_length),
            ("--required-context-length", args.required_context_length),
            ("--model-id", args.model_id),
            ("--revision", args.model_revision),
            ("--title", args.title),
            ("--question", args.question),
            ("--usd-per-gpu-hour", args.usd_per_gpu_hour),
            ("--price-in", args.price_in),
            ("--price-out", args.price_out),
            ("--price-cached", args.price_cached),
            ("--price-source", args.price_source),
            ("--p90-ttft", args.p90_ttft),
            ("--p90-tpot", args.p90_tpot),
            ("--reference-concurrency", args.reference_concurrency),
            ("--from-demo", args.from_demo),
            ("--sglang-serve-extra", args.sglang_serve_extra),
            ("--vllm-serve-extra", args.vllm_serve_extra),
            ("--trtllm-serve-extra", args.trtllm_serve_extra)):
        if value is not None and value != "":
            forwarded += [flag, str(value)]
    # `new` shares `set`'s value flags; the ones below are not forwarded, so a
    # value given for one would be dropped without a word. Refuse instead.
    dropped = [f"--{name.replace('_', '-')}" for name in (
        "cache_fraction", "price_model", "max_token_regret", "answer_sample_size",
        "max_absolute_accuracy_drop", "reference_arm")
        if getattr(args, name, None) not in (None, "")]
    if dropped:
        raise SystemExit(f"taskctl new does not apply {', '.join(dropped)}; create "
                         "the task, then change it with `taskctl set`")
    if args.write:
        forwarded.append("--write")
    return new_task.main(forwarded)


def cmd_rename(args) -> int:
    """Move a task to a new id and carry everything that names it.

    For a renumber (same model, same hardware): the task directory, its
    module, and its SGLang evidence under `_concurrency_search_archive/` move
    together, and every id inside them and in the referring code, tests and
    docs is rewritten. Recorded run evidence (`evolve_space/`,
    `server_prototypes/`, `evaluation_progress/`) is never rewritten: a run
    keeps the id it ran under. A model or hardware change is a new task, not a rename.
    """
    old, new = args.task, args.to
    src, dst = sites.task_dir(old), sites.task_dir(new)
    if not (src / "contract.yaml").is_file():
        raise SystemExit(f"{old} is not a task on disk")
    if dst.exists():
        raise SystemExit(f"REFUSED: {new} already exists")
    if old.rsplit("_", 1)[-1] != new.rsplit("_", 1)[-1]:
        raise SystemExit(f"REFUSED: {old} and {new} name different hardware; "
                         f"a hardware change is a new task")

    def move(a: Path, b: Path) -> None:
        subprocess.run(["git", "mv", str(a), str(b)], cwd=str(REPO),
                       check=False, capture_output=True)
        if a.exists():                         # untracked, or git unavailable
            shutil.move(str(a), str(b))

    def rewrite(path: Path) -> None:
        text = path.read_text("utf-8", errors="ignore")
        fresh = text.replace(old, new).replace(old.lower(), new.lower())
        if fresh != text:
            path.write_text(fresh, "utf-8")

    print(f"{old}\n  -> {new}")
    move(src, dst)
    old_mod, new_mod = sites.module_path(old), sites.module_path(new)
    if old_mod.is_file():
        move(old_mod, new_mod)
        rewrite(new_mod)
    archive = sites.TASKS / "_concurrency_search_archive"
    if (archive / old).is_dir():
        move(archive / old, archive / new)
        for path in (archive / new).rglob("*"):
            if path.is_file() and path.suffix in (".md", ".json"):
                rewrite(path)
    for path in sorted(dst.rglob("*")):
        if path.is_file() and path.suffix in sites.PROSE_SUFFIXES:
            rewrite(path)
    referrers = sites.references(old)
    for path in referrers:
        rewrite(path)
    print(f"  rewrote the id in {len(referrers)} referring file(s)")
    return cmd_render(argparse.Namespace(task=new, profile=None))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="taskctl", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    show = sub.add_parser("show", help="print one task as a spec")
    show.add_argument("task")
    show.set_defaults(fn=cmd_show)

    check = sub.add_parser("check", help="is every generated file in step?")
    check.add_argument("task", nargs="?")
    check.set_defaults(fn=cmd_check)

    render = sub.add_parser(
        "render", help="regenerate documents, starter header and evaluator copy")
    render.add_argument("task", nargs="?")
    render.add_argument("--profile", choices=("standard", "instructed"),
                        help="only tasks of this guidance profile")
    render.set_defaults(fn=cmd_render)

    def params(p: argparse.ArgumentParser) -> None:
        p.add_argument("--p90-ttft", type=float, help="p90 TTFT bar, ms")
        p.add_argument("--p90-tpot", type=float, help="p90 TPOT bar, ms")
        p.add_argument("--reference-concurrency", type=int,
                       help="the SGLang operating point the instruction states")
        p.add_argument("--usd-per-gpu-hour", type=float)
        p.add_argument("--price-in", type=float, help="$ per 1M input tokens")
        p.add_argument("--price-out", type=float, help="$ per 1M output tokens")
        p.add_argument("--price-cached", type=float,
                       help="$ per 1M cached input tokens")
        p.add_argument("--cache-fraction", type=float,
                       help="cached price as a fraction of input, e.g. 1.0")
        p.add_argument("--price-model",
                       help="the model whose market the prices are; follows "
                            "--model-id unless given")
        p.add_argument("--price-source", help="where the prices came from")
        p.add_argument("--max-token-regret", type=float,
                       help="private correctness setting")
        p.add_argument("--answer-sample-size", type=int,
                       help="private correctness setting")
        p.add_argument("--max-absolute-accuracy-drop", type=float,
                       help="private correctness setting")
        p.add_argument("--reference-arm", help="private correctness setting")
        p.add_argument("--max-context-length", type=int,
                       help="drop whole traces above this at load time")
        p.add_argument("--required-context-length", type=int,
                       help="required per-request capacity; no truncation")
        p.add_argument("--model-id")
        p.add_argument("--model-revision")
        p.add_argument("--reason",
                       help="why a bar is moving. Required to raise one")

    st = sub.add_parser("set", help="change parameters of an existing task")
    st.add_argument("task")
    params(st)
    st.set_defaults(fn=cmd_set)

    mv = sub.add_parser("rename",
                        help="move a task (and its evidence) to a new id")
    mv.add_argument("task", help="the task as it is now")
    mv.add_argument("--to", required=True, help="the new id")
    mv.set_defaults(fn=cmd_rename)

    n = sub.add_parser("new", help="build a task from the managed template")
    n.add_argument("task", help="the new task id, ending in its hardware")
    n.add_argument("--profile", choices=("standard", "instructed"))
    n.add_argument("--mode", choices=("docker", "modal"), required=True)
    n.add_argument("--title", help='short human name, e.g. "Qwen3.5 2B"')
    n.add_argument("--question", help="one sentence: what this task asks")
    n.add_argument("--from-demo", default="",
                   help="a task whose demo/ to copy as this one's starter")
    n.add_argument("--dry-run", dest="write", action="store_false",
                   help="print what would be created and write nothing")
    for arm in ("sglang", "vllm", "trtllm"):
        n.add_argument(f"--{arm}-serve-extra", default="",
                       help=f"the {arm} arm's extra serve flags in reference.yaml")
    params(n)
    n.set_defaults(fn=cmd_new, write=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
