"""``serverbench evolve`` — the harness command line.

Automation-friendly by construction: ``--json`` on every command, stable exit
codes, no prompts, no colour, no spinner. Importing this module or running
``--help`` loads no model, initialises no device, starts no thread and writes
nothing.

Exit codes are part of the interface:

===  ==========================================================================
0    the command did what it was asked
1    a real failure: a gate failed, the record did not verify, a check found drift
2    usage error — unknown command, missing argument, bad path
3    blocked by something outside the harness: a permission, a credential, a device
===  ==========================================================================

The evolve run itself is `serverbench/scripts/evolve`, not a verb here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 3


def _emit(payload: dict[str, Any], as_json: bool, lines: Sequence[str] | None = None) -> None:
    if as_json:
        print(json.dumps(payload, indent=1, sort_keys=True, default=str))
    else:
        for line in (lines or []):
            print(line)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def _devices() -> list[dict[str, str]]:
    """The visible GPUs, as `nvidia-smi` reports them; empty without one."""
    import subprocess

    try:
        got = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in got.stdout.splitlines() if got.returncode == 0 else []:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3:
            rows.append({"index": parts[0], "name": parts[1], "memory": parts[2]})
    return rows


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the environment before anything expensive is attempted."""
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("python", sys.version_info >= (3, 9),
          f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Importing the package must not have pulled in a framework.
    heavy = sorted({"torch", "transformers", "vllm", "sglang", "tensorrt_llm"}
                   & set(sys.modules))
    check("import is side-effect free", not heavy,
          "no heavy framework imported" if not heavy else f"imported: {heavy}")

    devices = _devices()
    check("device inventory", True,
          f"{len(devices)} device(s)" if devices
          else "no device query tool; CPU-only work is still possible")

    ok = all(c["ok"] for c in checks)
    _emit({"ok": ok, "checks": checks, "devices": devices}, args.json,
          [f"  {'ok  ' if c['ok'] else 'FAIL'}  {c['check']:<28}{c['detail']}"
           for c in checks] + ["", f"doctor: {'PASS' if ok else 'FAIL'}"])
    return EXIT_OK if ok else EXIT_FAILED


def cmd_curve(args: argparse.Namespace) -> int:
    """Per-version metrics, cost and verdict for one run.

    The one reader for a run's own evidence -- see `curve.py` for why this is a
    command rather than something rewritten by hand each time it is needed.
    """
    from . import curve

    report = curve.read(Path(args.run))
    _emit(report, args.json, curve.lines(report))
    return EXIT_OK


def cmd_golden(args: argparse.Namespace) -> int:
    """Run the harness's own golden suites. No device, no model, no network."""
    from .evaluator.golden import run_all
    from .evaluator.statistics import run_golden

    gates = run_all()
    stats = run_golden()
    ok = gates["ok"] and stats["ok"]
    lines = ["gate golden cases:"]
    lines += [f"  {'ok  ' if c['pass'] else 'MISS'}  {c['case']:<24}"
              f"{c['expected']} -> {c['observed']}" for c in gates["cases"]]
    lines += ["", "estimator golden populations:"]
    lines += [f"  {'ok  ' if p['pass'] else 'MISS'}  {p['population']:<26}"
              f"{p['expected']} -> {p['observed']}" for p in stats["populations"]]
    lines += ["", f"deterministic: {stats['deterministic']}",
              f"golden: {'PASS' if ok else 'FAIL'}"]
    _emit({"ok": ok, "gates": gates, "estimator": stats}, args.json, lines)
    return EXIT_OK if ok else EXIT_FAILED


def cmd_skill(args: argparse.Namespace) -> int:
    """Validate a distilled-knowledge skill bundle."""
    from .skill_bundle import validate_bundle

    result = validate_bundle(args.path)
    lines = [f"  {'ok  ' if c['ok'] else 'FAIL'}  {c['check']:<38}{c['detail']}"
             for c in result["checks"]]
    lines.append(f"skill: {'PASS' if result['ok'] else 'FAIL'}")
    _emit(result, args.json, lines)
    return EXIT_OK if result["ok"] else EXIT_FAILED


def cmd_kernel_list(args: argparse.Namespace) -> int:
    """What the kernel harness will check, and what it requires it can detect."""
    from kernel_evolve import verification as ladder
    from kernel_evolve.contracts.verification import REQUIRED_MUTANTS
    from kernel_evolve.verification import CATALOG

    plan = ladder.LadderPlan(plan_id="default")
    lines = ["the verification ladder, cheapest first", ""]
    for rung in plan.ordered():
        accepts = "may accept" if rung.can_accept else "REJECT ONLY"
        lines.append(f"  {rung.cost_rank:>4}  {rung.gate_id:<28}{accepts:<12}"
                     f"~{rung.estimated_device_seconds:>5.0f}s device")
        lines.append(f"        {rung.description}")
    lines += ["", "required corruption classes, and the rung that must kill each", ""]
    for mutant in sorted(REQUIRED_MUTANTS, key=lambda m: m.value):
        entry = CATALOG[mutant]
        lines.append(f"  {mutant.value:<26}{entry['expected_killer']}")
        lines.append(f"        {entry['why']}")
    payload = {"rungs": [r.to_dict() for r in plan.ordered()],
               "required_mutants": sorted(m.value for m in REQUIRED_MUTANTS)}
    _emit(payload, args.json, lines)
    return EXIT_OK


def cmd_kernel_doctor(args: argparse.Namespace) -> int:
    """Can this host run a kernel candidate, and is the exclusion still holding?"""
    from kernel_evolve import verification as exclusion
    from kernel_evolve.contracts.verification import REQUIRED_MUTANTS
    from kernel_evolve.verification import CATALOG

    from .contracts.environment import ExecutionEnvironment

    repo = Path(args.repo)
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    env = (ExecutionEnvironment.from_interpreter(args.interpreter,
                                                 environment_id="kernel-build")
           if args.interpreter else ExecutionEnvironment.current(
               environment_id="kernel-build"))
    try:
        identity = env.resolve()
        record("declared interpreter", True,
               f"{identity['interpreter']} ({identity['version']})")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        record("declared interpreter", False, str(exc))

    for module in ("torch", "triton"):
        present = env.has_module(module)
        record(f"{module} importable", present,
               "present" if present else
               f"absent from the declared environment; a kernel candidate needing "
               f"{module} would be INVALID here, not a failure")

    # The kernel knowledge corpus is part of the kernel-evolution skill, so
    # scanning `skills` covers it.
    surfaces = [("kernel harness", repo / "serverbench/harness/kernel_evolve"),
                ("skills", repo / "serverbench/skills")]
    report = exclusion.scan([(n, p) for n, p in surfaces if p.exists()])
    record("out-of-scope technique excluded", report.clean, report.summary())

    covered = {m for m in REQUIRED_MUTANTS if m in CATALOG}
    record("required controls have a catalogued killer",
           covered == set(REQUIRED_MUTANTS),
           f"{len(covered)} of {len(REQUIRED_MUTANTS)} corruption classes")

    ok = all(c["ok"] for c in checks)
    lines = [f"  {'ok  ' if c['ok'] else 'FAIL'}  {c['check']:<44}{c['detail']}"
             for c in checks]
    lines.append(f"kernel doctor: {'PASS' if ok else 'FAIL'}")
    _emit({"ok": ok, "checks": checks}, args.json, lines)
    return EXIT_OK if ok else EXIT_FAILED


def cmd_kernel_knowledge(args: argparse.Namespace) -> int:
    """Does every entry in the kernel knowledge catalog resolve to a file?"""
    import yaml

    root = Path(args.root)
    catalog = yaml.safe_load((root / "catalog.yaml").read_text("utf-8"))
    entries = catalog.get("entries", [])
    missing = [e["path"] for e in entries if not (root / e["path"]).exists()]
    listed = {e["path"] for e in entries}
    on_disk = {str(p.relative_to(root)) for p in root.rglob("*.md")}
    unlisted = sorted(on_disk - listed - {"README.md"})
    ok = not missing and not unlisted
    lines = [f"  {'ok  ' if not missing else 'FAIL'}  "
             f"{len(entries)} catalog entr(ies), {len(missing)} missing"]
    lines += [f"          missing: {m}" for m in missing]
    if unlisted:
        lines.append("  FAIL  file(s) present but not in the catalog, so nothing "
                     "routes to them:")
        lines += [f"          {u}" for u in unlisted]
    lines.append(f"kernel knowledge: {'PASS' if ok else 'FAIL'}")
    _emit({"ok": ok, "entries": len(entries), "missing": missing,
           "unlisted": unlisted}, args.json, lines)
    return EXIT_OK if ok else EXIT_FAILED


def cmd_kernels(args: argparse.Namespace) -> int:
    """Kernel inventory, research brief, catalog validation, and provisioning.

    The first two run in the harness, outside any sandbox. The last delivers the
    result to an agent that never had to do the research itself.
    """
    from .contracts.workload import ModelIdentity
    from .kernels import inventory as inventory_mod
    from .kernels import provision, research

    action = args.kernels_action

    if action == "inventory":
        result = inventory_mod.operators_for(args.architecture)
        lines = [f"architecture  {result.architecture}"
                 f"{'' if result.recognised else '  (NOT RECOGNISED)'}"]
        if result.note:
            lines.append(f"              {result.note}")
        lines.append("")
        for op in result.operators:
            share = ("    —" if op.typical_step_share is None
                     else f"{op.typical_step_share:>5.0%}")
            flag = "" if op.required else "  (optional)"
            lines.append(f"  {share}  {op.name:<22}{op.rationale[:70]}{flag}")
        lines.append("")
        lines.append("shares are typical for this shape of decoder and are NOT "
                     "measurements of any deployment")
        _emit(result.to_dict(), args.json, lines)
        return EXIT_OK

    if action == "brief":
        model = ModelIdentity(model_id=args.model, revision=args.revision or "unpinned",
                              tokenizer_id=args.model)
        result = research.brief(model, architecture=args.architecture,
                                hardware_target=args.hardware or "")
        lines = [f"research brief for {args.model}", ""]
        lines += [f"  {op['name']:<22}{op['search_for']}" for op in result["operators"]]
        lines += ["", "for each candidate, answer:"]
        lines += [f"  - {q}" for q in result["questions_per_candidate"]]
        lines += ["", "rules:"] + [f"  - {r}" for r in result["rules"]]
        _emit(result, args.json, lines)
        return EXIT_OK

    # `research.read` raises whatever the filesystem and the JSON parser raise.
    # Both are reported through EvolveError like every other verb, naming what
    # this verb wanted -- in particular when it is handed the YAML knowledge
    # catalog, which belongs to `kernels knowledge`.
    from .errors import ContractError
    try:
        catalog = research.read(args.catalog)
    except FileNotFoundError as exc:
        raise ContractError(
            f"no catalog at {args.catalog}",
            detail="`kernels validate` and `kernels provision` take a kernel "
                   "catalog in JSON. The knowledge catalog "
                   "`serverbench/skills/kernel-evolution/knowledge/catalog.yaml` is a different "
                   "thing and is checked by `serverbench evolve kernels "
                   "knowledge`.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(
            f"{args.catalog} is not a JSON kernel catalog",
            detail=f"{exc}. If this is `kernel-evolution/knowledge/catalog.yaml`, that is "
                   f"the knowledge catalog and is checked by `serverbench "
                   f"evolve kernels knowledge`.") from exc

    if action == "validate":
        result = research.validate(catalog)
        lines = [f"catalog   {catalog.catalog_id}",
                 f"model     {catalog.identity.model_id}",
                 f"frozen    {catalog.frozen}",
                 f"coverage  {result['coverage']['covered']}/"
                 f"{result['coverage']['required']} required operators",
                 f"evidence  {result['coverage']['evidence_levels']}"]
        lines += [f"  problem: {p}" for p in result["problems"]]
        lines.append(f"catalog: {'PASS' if result['ok'] else 'FAIL'}")
        _emit(result, args.json, lines)
        return EXIT_OK if result["ok"] else EXIT_FAILED

    if action == "provision":
        manifest = provision.install(catalog, args.workspace)
        _emit(manifest, args.json,
              [f"installed {catalog.catalog_id} -> "
               f"{args.workspace}/{manifest['installed_at']}",
               f"digest    {manifest['install_sha256']}",
               "", provision.briefing_note(manifest)])
        return EXIT_OK

    return EXIT_USAGE


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="serverbench evolve",
        description="Harness utilities: environment check, run curve, "
                    "golden gate cases, skill-bundle and kernel-catalog checks.")
    p.add_argument("--json", action="store_true",
                   help="emit machine-readable JSON instead of text")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    d = sub.add_parser("doctor", help="check the environment before spending anything")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("curve", help="what a run did, per version, in one table")
    c.add_argument("--run", required=True,
                   help="the run directory under evolve_space/evolve/<task>/")
    c.set_defaults(func=cmd_curve)

    g = sub.add_parser("golden", help="run the harness's own golden suites (CPU only)")
    g.set_defaults(func=cmd_golden)

    sk = sub.add_parser("skill", help="validate a distilled-knowledge skill bundle")
    sk.add_argument("path", help="path to the skill bundle directory")
    sk.set_defaults(func=cmd_skill)

    k = sub.add_parser("kernels", help="kernel inventory, research brief, catalogs")
    ka = k.add_subparsers(dest="kernels_action", metavar="<action>", required=True)

    ki = ka.add_parser("inventory", help="what operators an architecture needs")
    ki.add_argument("--architecture", required=True)
    ki.set_defaults(func=cmd_kernels)

    kb = ka.add_parser("brief", help="what to research, operator by operator")
    kb.add_argument("--model", required=True)
    kb.add_argument("--architecture", default="")
    kb.add_argument("--revision", default="")
    kb.add_argument("--hardware", default="")
    kb.set_defaults(func=cmd_kernels)

    kv = ka.add_parser("validate", help="is a catalog fit to hand to an agent")
    kv.add_argument("catalog")
    kv.set_defaults(func=cmd_kernels)

    kp = ka.add_parser("provision", help="deliver a catalog into a workspace")
    kp.add_argument("catalog")
    kp.add_argument("--workspace", required=True)
    kp.set_defaults(func=cmd_kernels)

    kl = ka.add_parser("list", help="the verification ladder and the required controls")
    kl.set_defaults(func=cmd_kernel_list)

    kd = ka.add_parser("doctor", help="can this host run a kernel candidate at all")
    kd.add_argument("--repo", default=".")
    kd.add_argument("--interpreter", default="",
                    help="absolute path to the interpreter a build would use; "
                         "defaults to the one running this process")
    kd.set_defaults(func=cmd_kernel_doctor)

    kk = ka.add_parser("knowledge", help="does the kernel knowledge catalog resolve")
    kk.add_argument("--root", default="serverbench/skills/kernel-evolution/knowledge")
    kk.set_defaults(func=cmd_kernel_knowledge)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    from .errors import EvolveError, ExternalBlocker

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args))
    except ExternalBlocker as exc:
        body = exc.as_dict()
        print(json.dumps(body, indent=1) if args.json
              else f"blocked: {exc.message}\nresume: {exc.resume_command or '(none given)'}",
              file=sys.stderr)
        return EXIT_BLOCKED
    except EvolveError as exc:
        print(json.dumps(exc.as_dict(), indent=1) if args.json
              else f"{exc.reason_code}: {exc.message}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
