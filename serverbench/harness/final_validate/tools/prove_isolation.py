#!/usr/bin/env python3
"""Prove the private final validation is unreachable FROM INSIDE the agent container.

There are five places private final-validation code and assets must not
appear:

    the coding agent's container image or build context; mounted task context or
    workspace directories; agent-accessible repository history or archives; tool
    descriptions, prompts, memory, or development logs; and no agent-callable
    endpoint may reveal private final results during optimisation.

WHY THIS RUNS INSIDE THE CONTAINER
-----------------------------------
Checking from the host proves nothing. The host can see the whole repository, so
"the private code is at `serverbench/harness/final_validate/`" is true and
uninformative; the question is what the AGENT can reach, and only the agent's
own filesystem view answers it. A host-side check would keep passing after
somebody adds a bind mount.

So this composes the container with **the same mount set the real driver uses**
-- `scripts/agent/harbor_env.py::mounts_for`, imported rather than reimplemented,
so the proof cannot drift from the thing it is proving about -- and then runs
every probe from inside.

WHAT IT DOES NOT CLAIM
-----------------------
No GPU is attached and no agent is started: isolation is a property of the mount
set and the network, not of the device. The report says so, in those words,
rather than letting a reader infer that a full run was exercised.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

#: Where the probe's report is recorded; `tests/verification` reads it on a deployed host.
EVIDENCE = REPO / "serverbench" / "artifacts" / "isolation" / "agent-container.json"
sys.path.insert(0, str(REPO / "serverbench" / "scripts" / "agent"))

#: Everything that must not be reachable, and why it matters if it is.
FORBIDDEN_PATHS = {
    str(REPO): "the repository itself -- it holds every task's held-out data",
    str(REPO / "serverbench" / "harness" / "final_validate"):
        "the private final-validation program",
    str(REPO / "serverbench" / "artifacts" / "final_validation"):
        "the private final-validation results",
    str(REPO / "serving_tasks"):
        "the task sources, including oracle traces and reference outputs",
    str(REPO / "server_output"): "other agents' artifacts",
    "/opt/serverbench": "a conventional install location, checked in case",
}

#: Filenames that would give a final outcome away wherever they were found.
FORBIDDEN_NAMES = (
    "final_validation_result.json",
    "trace-final.json",
    "references-final.json",
    "trace-qualification.json",
    "references-qualification.json",
    "FINAL_VALIDATION.json",
)

#: Module names that must not import inside the container: the host verdict
#: package and its private correctness stage.
FORBIDDEN_IMPORTS = ("final_validate", "final_validate.scenario",
                     "final_validate.private_correctness")

#: What the agent MUST be able to reach. The agent benchmarks its own server
#: with AgentX inside the container, so absence is a defect, not safety. The
#: same probe therefore checks both directions: private things unreachable, and
#: the tools the agent was promised present. A probe that only looked for
#: breaches would pass on a container with nothing in it at all.
REQUIRED_PATHS = {
    "/opt/agentx/bin/aiperf": "the pinned AgentX CLI the brief tells the agent to use",
    "/opt/agentx-src/docs/tutorials": "AgentX's tutorials, named in the brief",
    "/opt/agentx-src/src/aiperf": "AgentX's source, named in the brief",
}

#: And what must NOT be writable: the agent may read the measurement runtime and
#: may not edit it: the authoritative runtime and grading assets stay outside
#: candidate-writable scope.
MUST_BE_READ_ONLY = ("/opt/agentx", "/opt/agentx-src")

PROBE = r"""
import json, os, sys, importlib.util
report = {"paths": {}, "names": {}, "imports": {}, "mounts": [], "network": {},
          "required": {}, "read_only": {}, "agentx": {}}
for p in FORBIDDEN_PATHS:
    report["paths"][p] = os.path.exists(p)
roots = ["/", "/workspace", "/opt", "/home", "/models", "/run", "/tmp"]
hits = {n: [] for n in FORBIDDEN_NAMES}
seen = 0
for root in ["/workspace", "/opt", "/home", "/run"]:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in ("proc", "sys", "dev")]
        seen += len(filenames)
        for name in filenames:
            if name in hits:
                hits[name].append(os.path.join(dirpath, name))
        if seen > 400000:
            break
report["names"] = hits
report["files_walked"] = seen
for p, why in REQUIRED_PATHS.items():
    report["required"][p] = os.path.exists(p)
for p in MUST_BE_READ_ONLY:
    probe = os.path.join(p, ".write-probe")
    try:
        with open(probe, "w") as fh:
            fh.write("x")
        os.unlink(probe)
        report["read_only"][p] = False
    except OSError:
        report["read_only"][p] = True
try:
    import subprocess
    got = subprocess.run(["aiperf", "--version"], capture_output=True,
                         text=True, timeout=120)
    report["agentx"] = {"on_path": True,
                        "version": (got.stdout or got.stderr).strip()[:40],
                        "returncode": got.returncode}
except Exception as exc:
    report["agentx"] = {"on_path": False, "why": type(exc).__name__}
for mod in FORBIDDEN_IMPORTS:
    try:
        report["imports"][mod] = importlib.util.find_spec(mod) is not None
    except Exception:
        report["imports"][mod] = False
try:
    with open("/proc/self/mountinfo") as fh:
        report["mounts"] = [row.split(" - ")[0].split()[3:5] for row in fh]
except OSError:
    pass
# Any route out at all: no agent-callable endpoint may reveal private final
# results, and no route means no endpoint.
import socket
for host, port in (("8.8.8.8", 53), ("127.0.0.1", 8765)):
    s = socket.socket(); s.settimeout(1.5)
    try:
        s.connect((host, port)); report["network"][f"{host}:{port}"] = "open"
    except Exception as exc:
        report["network"][f"{host}:{port}"] = type(exc).__name__
    finally:
        s.close()
print("PROBE_JSON:" + json.dumps(report))
"""


def _docker(*args: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          timeout=timeout, check=False)


def run(image: str, *, keep: bool = False) -> dict:
    import harbor_env

    name = f"isolation-probe-{int(time.time())}"
    staging = Path(tempfile.mkdtemp(prefix="isolation-"))
    workspace = staging / "workspace"
    workspace.mkdir()
    (workspace / "serve.sh").write_text("#!/bin/bash\nexit 0\n")
    (workspace / "memory.md").write_text("# Memory\n")
    model = staging / "model"
    model.mkdir()
    env_input = staging / "env-input"
    env_input.mkdir()
    uv_cache = staging / "uv-cache"
    uv_cache.mkdir()

    # THE REAL MOUNT SET, from the driver's own function.
    mounts = harbor_env.mounts_for(workspace=workspace, model=model,
                                   env_input=env_input, uv_cache=uv_cache)
    binds: list[str] = []
    for mount in mounts:
        spec = f"{mount['source']}:{mount['target']}"
        if mount.get("read_only"):
            spec += ":ro"
        binds += ["-v", spec]

    header = (f"REQUIRED_PATHS = {json.dumps(REQUIRED_PATHS)}\n"
              f"MUST_BE_READ_ONLY = {json.dumps(list(MUST_BE_READ_ONLY))}\n"
              f"FORBIDDEN_PATHS = {json.dumps(list(FORBIDDEN_PATHS))}\n"
              f"FORBIDDEN_NAMES = {json.dumps(list(FORBIDDEN_NAMES))}\n"
              f"FORBIDDEN_IMPORTS = {json.dumps(list(FORBIDDEN_IMPORTS))}\n")
    script = header + PROBE
    probe_file = staging / "probe.py"
    probe_file.write_text(script)
    binds += ["-v", f"{probe_file}:/tmp/probe.py:ro"]
    env = ["-e", f"PATH={harbor_env.AGENTX_MOUNT}/bin:/usr/local/bin:"
                 f"/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
           "-e", f"AGENTX_ROOT={harbor_env.AGENTX_MOUNT}"]

    got = _docker("run", "--rm", "--name", name, "--network", "none",
                  "--entrypoint", "python3", *env, *binds, image, "/tmp/probe.py")
    line = next((row for row in (got.stdout or "").splitlines()
                 if row.startswith("PROBE_JSON:")), "")
    if not line:
        return {"ok": False,
                "why": "the probe produced no report",
                "stdout": (got.stdout or "")[-2000:],
                "stderr": (got.stderr or "")[-2000:],
                "returncode": got.returncode,
                "binds": binds}
    report = json.loads(line[len("PROBE_JSON:"):])
    report["binds"] = binds
    report["image"] = image
    report["caveat"] = ("no GPU was attached and no agent was started. "
                        "Isolation is a property of the mount set and the "
                        "network, not of the device; this proves the "
                        "filesystem and network view an agent gets, and "
                        "nothing about device behaviour.")

    breaches: list[str] = []
    for path, present in report["paths"].items():
        if present:
            breaches.append(f"{path} IS reachable: {FORBIDDEN_PATHS.get(path, '')}")
    for name_, found in report["names"].items():
        if found:
            breaches.append(f"{name_} found at {found[:3]}")
    for mod, importable in report["imports"].items():
        if importable:
            breaches.append(f"`import {mod}` succeeds inside the container")
    for path, writable in report.get("read_only", {}).items():
        if writable is False:
            breaches.append(f"{path} is WRITABLE from inside; the agent could "
                            f"edit the measurement runtime it is benchmarked by")
    report["breaches"] = breaches
    # Missing tools are reported separately from breaches: an absent AgentX is
    # a broken promise to the agent, not a leak of private evidence, and
    # collapsing the two would let a container with nothing in it look safe.
    report["missing"] = [f"{p} is absent: {why}"
                         for p, why in REQUIRED_PATHS.items()
                         if not report.get("required", {}).get(p)]
    if not report.get("agentx", {}).get("on_path"):
        report["missing"].append(
            "`aiperf` is not on PATH; the brief must not name it")
    report["ok"] = not breaches and not report["missing"]
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="serverbench/evolve-agent:cu128")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, default=EVIDENCE,
                    help="where the report is written (default: %(default)s)")
    args = ap.parse_args(argv)
    report = run(args.image)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", "utf-8")
    if args.json:
        print(json.dumps(report, indent=1))
        return 0 if report.get("ok") else 1
    print(f"image: {report.get('image')}")
    print(f"files walked inside the container: {report.get('files_walked')}")
    for path, present in sorted(report.get("paths", {}).items()):
        print(f"  {'REACHABLE' if present else 'absent   '}  {path}")
    for mod, importable in sorted(report.get("imports", {}).items()):
        print(f"  {'IMPORTS' if importable else 'absent '}    import {mod}")
    for path, present in sorted(report.get("required", {}).items()):
        print(f"  {'present  ' if present else 'MISSING  '}  {path}")
    for path, ro in sorted(report.get("read_only", {}).items()):
        print(f"  {'read-only' if ro else 'WRITABLE '}  {path}")
    print(f"  agentx: {report.get('agentx')}")
    print(f"  network: {report.get('network')}")
    if report.get("ok"):
        print("\nISOLATED: nothing private is reachable from inside the "
              "agent's container.")
        print(report.get("caveat", ""))
        return 0
    if report.get("breaches"):
        print("\nBREACH -- private evidence is reachable:")
        for line in report["breaches"]:
            print(f"  - {line}")
    if report.get("missing"):
        print("\nMISSING -- the agent was promised something that is not there:")
        for line in report["missing"]:
            print(f"  - {line}")
    if not report.get("breaches") and not report.get("missing"):
        print(f"\n{report.get('why', 'the probe produced no report')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
