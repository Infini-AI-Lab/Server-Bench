#!/usr/bin/env python3
"""Does a Modal sandbox give the agent what the docker container gives it?

The migration's failure mode is not a crash. It is a sandbox that starts, an
agent that works, and one thing quietly absent that the brief promised -- and
the cost lands forty minutes later, or in a verdict nobody can explain: a tool
installed but on no shell's PATH, or a mount the brief names that does not
exist.

So this asks the question directly, against a live sandbox, for every mount and
every environment variable `harbor_env.py` gives the docker container.

    parity.py --sandbox sb-XXXX          against a running run
    parity.py --task <task> --start      stand a throwaway one up and check it

The inventory below is READ FROM `harbor_env.py` where it can be, and stated
here where it cannot. Anything stated here is a thing that can drift, so each
entry says what it is for -- a checklist nobody can read is a checklist nobody
maintains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "serving_tasks" / "_modal"))

#: (path, what it is for, fatal, scope).
#:
#: `fatal` means the brief promises it or the agent cannot work without it.
#: `scope` says WHERE it comes from, and matters because a throwaway sandbox
#: legitimately has no workspace and no per-run Secret: reporting those as
#: defects would make this tool cry wolf on the run it is most useful for --
#: the one standing an image up before a run exists.
#:
#:   IMAGE  the image or a mounted volume; checkable any time
#:   RUN    staged by `modal_env.py` for one run; only checkable against it
PATHS = [
    ("/workspace", "the working tree the agent edits", True, "RUN"),
    ("/workspace/tests/final_verdict",
     "the evaluator the brief tells it to measure with", True, "RUN"),
    ("/workspace/task_context",
     "the frozen task contract it reads", True, "RUN"),
    ("/workspace/.claude/skills",
     "the installed skills the brief names", True, "RUN"),
    ("/models/target",
     "MODEL_PATH -- the checkpoint, at the path the verdict also serves from",
     True, "RUN"),
    ("/models/target/config.json",
     ("proof the snapshot's symlinks resolve; a snapshot without its blobs "
      "is a farm of dangling links"), True, "RUN"),
    ("/opt/agentx/bin/aiperf",
     "the benchmark binary the brief names", True, "IMAGE"),
    ("/opt/agentx-src",
     "AGENTX_SOURCE -- the readable AgentX source", False, "IMAGE"),
    ("/opt/env-input/wheelhouse",
     "the offline wheelhouse the brief names six times", True, "IMAGE"),
    ("/opt/env-input/resolved.requirements.txt",
     "what the wheelhouse pins", True, "IMAGE"),
    ("/usr/local/cuda/bin/nvcc",
     "the CUDA compiler the brief tells it to confirm with `command -v nvcc`",
     True, "IMAGE"),
    ("/opt/nvidia/nsight-compute",
     "ncu, which the brief advises for kernels", True, "IMAGE"),
    ("/opt/nvidia/nsight-systems",
     "nsys, which the brief advises for the whole step", True, "IMAGE"),
    ("/home/agent",
     "the agent user's home, which the CLI writes to", True, "IMAGE"),
]

#: (variable, what it is for, fatal, scope). All of these are handed to the
#: sandbox as a per-run Secret, so a throwaway has none of them.
ENVIRONMENT = [
    ("MODEL_PATH",
     "where the checkpoint is; the verdict passes the same", True, "RUN"),
    ("MODEL_ID",
     ("which model this is, from the task's pin; a container carrying "
      "the wrong one has three sources of truth"),
     True, "RUN"),
    ("PORT", "the port serve.sh is expected to listen on", True, "RUN"),
    ("AGENTX_ROOT", "where AgentX is", True, "IMAGE"),
    ("HF_HUB_CACHE",
     ("the hub cache a hub-id lookup resolves through, offline. HF_HOME is a "
      "PARENT -- huggingface_hub looks under $HF_HOME/hub -- so naming the "
      "hub directory with HF_HOME silently finds nothing"), True, "RUN"),
    ("HF_HOME", "a home for anything that wants one", False, "RUN"),
    ("UV_CACHE_DIR",
     "so `uv` does not rebuild its cache every run", False, "IMAGE"),
]

#: (label, command, what a PASS looks like, fatal, scope). As the AGENT USER,
#: because that is who runs them for real and `su` is where a difference hides.
COMMANDS = [
    ("aiperf on PATH", "command -v aiperf",
     "/opt/agentx/bin/aiperf", True, "IMAGE"),
    ("nvcc on PATH", "command -v nvcc", "nvcc", True, "IMAGE"),
    ("the evaluator imports",
     "cd /workspace/tests && python3 -c 'import final_verdict; print(\"ok\")'",
     "ok", True, "RUN"),
    ("torch sees the cards",
     "python3 -c 'import torch;print(torch.cuda.device_count())'",
     "", True, "IMAGE"),
    ("ncu runs", "ncu --version 2>&1 | head -1", "", False, "IMAGE"),
    ("nsys runs", "nsys --version 2>&1 | head -1", "", False, "IMAGE"),
    ("the workspace is writable",
     "touch /workspace/.parity && rm /workspace/.parity && echo ok",
     "ok", True, "RUN"),
]

AGENT_USER = "agent"


def _as_agent(command: str) -> str:
    import shlex
    return (f"su {AGENT_USER} -p -s /bin/bash -c "
            + shlex.quote(f"export HOME=/home/{AGENT_USER}; {command}"))


def audit(sandbox) -> dict:
    """Every check, run in one shell, as the agent user."""
    script_lines = ["set +e"]
    for path, _, _, _ in PATHS:
        script_lines.append(
            f'if [ -e "{path}" ]; then echo "PATH|{path}|yes"; '
            f'else echo "PATH|{path}|NO"; fi')
    for name, _, _, _ in ENVIRONMENT:
        script_lines.append(
            f'if [ -n "${{{name}:-}}" ]; then echo "ENV|{name}|${{{name}}}"; '
            f'else echo "ENV|{name}|UNSET"; fi')
    for label, command, _, _, _ in COMMANDS:
        script_lines.append(
            f'__o=$({command} 2>&1 | tail -1); '
            f'echo "CMD|{label}|${{__o:-<empty>}}"')
    script = "\n".join(script_lines)

    process = sandbox.exec("/bin/bash", "-lc", _as_agent(script), timeout=600)
    out = process.stdout.read() or ""
    err = process.stderr.read() or ""
    process.wait()

    got = {"PATH": {}, "ENV": {}, "CMD": {}}
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3 and parts[0] in got:
            got[parts[0]][parts[1]] = parts[2]
    return {"got": got, "stderr": err.strip()[-400:], "raw": out}


def report(result: dict, scope: str = "ALL") -> int:
    """`scope='IMAGE'` checks only what an image and its volumes can provide.

    A throwaway sandbox has no workspace and no per-run Secret, so asserting
    those would make this cry wolf exactly when it is most useful: standing an
    image up BEFORE a run exists.
    """
    got = result["got"]
    problems, warnings = [], []

    def skipped(entry_scope: str) -> bool:
        return scope == "IMAGE" and entry_scope == "RUN"

    print("PATHS")
    for path, why, fatal, entry_scope in PATHS:
        if skipped(entry_scope):
            print(f"  {'(run)':>8}  {path}")
            continue
        state = got["PATH"].get(path, "?")
        ok = state == "yes"
        print(f"  {'ok ' if ok else 'MISSING':>8}  {path}")
        if not ok:
            (problems if fatal else warnings).append(f"{path} -- {why}")

    print("\nENVIRONMENT")
    for name, why, fatal, entry_scope in ENVIRONMENT:
        if skipped(entry_scope):
            print(f"  {'(run)':>8}  {name}")
            continue
        value = got["ENV"].get(name, "?")
        ok = value not in ("UNSET", "?")
        shown = value if len(value) < 60 else value[:57] + "..."
        print(f"  {'ok ' if ok else 'UNSET':>8}  {name}={shown}")
        if not ok:
            (problems if fatal else warnings).append(f"{name} -- {why}")

    print("\nCOMMANDS (as the agent user)")
    for label, _, expect, fatal, entry_scope in COMMANDS:
        if skipped(entry_scope):
            print(f"  {'(run)':>8}  {label}")
            continue
        value = got["CMD"].get(label, "?")
        ok = bool(value) and value != "<empty>" and (not expect or expect in value)
        shown = value if len(value) < 60 else value[:57] + "..."
        print(f"  {'ok ' if ok else 'FAILED':>8}  {label}: {shown}")
        if not ok:
            (problems if fatal else warnings).append(f"{label} -> {value!r}")

    if warnings:
        print("\nNOT FATAL:")
        for line in warnings:
            print(f"  - {line}")
    if problems:
        print("\nTHE AGENT IS MISSING SOMETHING IT IS PROMISED:")
        for line in problems:
            print(f"  - {line}")
        return 1
    what = ("the image and its volumes carry everything they owe"
            if scope == "IMAGE"
            else "the sandbox gives the agent everything the container does")
    print(f"\nparity: {what}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", default=None, help="an existing sandbox id")
    ap.add_argument("--task", default=None,
                    help="stand a throwaway sandbox up for this task")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    import modal
    import modalctl

    if args.sandbox:
        sandbox, own = modal.Sandbox.from_id(args.sandbox), False
    elif args.task:
        # The same factory the agent's sandbox and the verdict's come from --
        # auditing anything else would audit a sandbox nothing else creates.
        # Half an hour, no network: an audit reads, and reaches nobody.
        sandbox = modalctl.create_sandbox(
            args.task, purpose="parity", hours=0.5, cpu=4.0, memory=32768,
            network="none")
        own = True
    else:
        print("give --sandbox or --task", file=sys.stderr)
        return 2

    try:
        result = audit(sandbox)
    finally:
        if own:
            sandbox.terminate()

    if args.json:
        print(json.dumps(result["got"], indent=1))
        return 0
    # A throwaway can only be held to what an image and its volumes provide.
    return report(result, scope="IMAGE" if own else "ALL")


if __name__ == "__main__":
    raise SystemExit(main())
