#!/usr/bin/env python3
"""Stand a run's environment up as a Modal GPU sandbox.

The Modal counterpart of `harbor_env.py`. It writes the same `container.json`
in the same shape, and names the container `modal:<sandbox id>` so
`lifecycle/backend.py` can tell which backend it belongs to. Nothing downstream
learns a second vocabulary.

## What is the same, deliberately

* the image is OURS, composed from our own Dockerfiles by
  `serving_tasks/_modal/docker_to_modal.py`, so torch, transformers, nvcc,
  Nsight, ninja and the agent CLIs are the versions this suite is validated
  against rather than whatever a package manager would choose;
* `MODEL_PATH` is `/models/target`, exactly as on docker -- the checkpoint is
  a Volume, and a symlink puts it where the brief already says it is;
* `/workspace` is the working tree and the agent runs as `agent`.

## What is different, and why

**The network.** There is no iptables allowlist inside a sandbox and no egress
broker in front of it. Modal's `outbound_cidr_allowlist` does the same job at
the platform edge: the sandbox reaches the model provider and nothing else
(github, pypi, huggingface and download.pytorch.org are unreachable while the
provider answers). The addresses are resolved FROM
INSIDE MODAL, because the host's DNS gives different ones and locking to those
would cut the agent off from its own model.

**The user.** Modal ignores the `USER agent` our Dockerfile sets and runs
everything as root, which the agent CLI refuses. Every exec goes through
`su agent` -- see `modal_exec.py`.

**The workspace arrives as a tar.** There is no bind mount. It is uploaded
once, here, and the run's own publishing pulls versions back out.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serving_tasks" / "_modal"))
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

import modalctl  # noqa: E402
from lifecycle import credentials as creds_mod  # noqa: E402

#: Where the brief already tells the agent the checkpoint is. Kept identical to
#: the docker path so nothing in the task text has to know where it ran.
MODEL_MOUNT = "/models/target"
WORKSPACE = "/workspace"
AGENT_USER = "agent"


def _tar(root: Path) -> Path:
    """The workspace as one archive. Thousands of small files over a
    per-file API would take tens of minutes; a tar takes seconds."""
    staged = Path(tempfile.mkdtemp(prefix="serverbench-ws-")) / "workspace.tar"
    with tarfile.open(staged, "w") as tar:
        for entry in sorted(root.iterdir()):
            tar.add(entry, arcname=entry.name)
    return staged


def _exec(sandbox, command: str, *, timeout: int = 900, check: bool = True):
    process = sandbox.exec("/bin/bash", "-lc", command, timeout=timeout)
    out, err = process.stdout.read(), process.stderr.read()
    code = process.wait()
    if check and code != 0:
        raise RuntimeError(
            f"{command.splitlines()[0][:90]!r} failed ({code}): "
            f"{(err or out).strip()[:400]}")
    return (out or "").strip()


def _agent_env(args) -> dict[str, str]:
    """What the agent's process needs, read from the operator's file.

    Never from this process's environment: a key that is not exported here
    cannot be leaked by anything here.
    """
    env: dict[str, str] = {}
    if args.agent_env and Path(args.agent_env).is_file():
        for line in Path(args.agent_env).read_text("utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    # ONE definition of what makes the checkpoint loadable offline, shared
    # with the verdict, so the server the agent checks loads its model the
    # same way the verdict's does.
    for key, value in modalctl.model_environment(args.task.resolve().name).items():
        env.setdefault(key, value)
    # WHICH MODEL THIS IS, read from the task rather than written here, so
    # `MODEL_ID`, the mounted weights and the brief cannot disagree. There is
    # no fallback: a guessed model id is worse than none.
    model_id = _declared_model_id(args.task)
    if model_id:
        env.setdefault("MODEL_ID", model_id)
    # The port `serve.sh` is expected to listen on, and the one every probe in
    # the brief uses.
    env.setdefault("PORT", "8000")
    # APP_DIR, for the same reason as on docker: every `start_server.sh`
    # defaults it to `/app`, the VERDICT's mount, and the agent's tree is at
    # /workspace. See `harbor_env.py`.
    env.setdefault("APP_DIR", WORKSPACE)
    return env


def _declared_model_id(task_dir) -> str:
    """The task's model id, read from its task module."""
    return modalctl.model_spec(Path(task_dir).name)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hours", default="2")
    ap.add_argument("--agent-cli", default="claude")
    ap.add_argument("--agent-model", default="")
    ap.add_argument("--agent-effort", default="")
    ap.add_argument("--agent-env", default="")
    ap.add_argument("--credentials", default=None)
    ap.add_argument("--print-name-only", action="store_true")
    args = ap.parse_args(argv)

    import modal

    task = args.task.resolve().name
    card, count = modalctl.gpu_spec(task)
    snapshot = modalctl.snapshot_path(task)

    state = modalctl.model_state(task)
    if not state.get("ready"):
        print(f"the checkpoint is not usable in the volume: {state.get('why')}\n"
              f"Run: serving_tasks/_modal/modalctl.py model --task {task}",
              file=sys.stderr)
        return 1

    wheels = modalctl.env_input_state()
    if not wheels.get("ready"):
        print(f"the wheelhouse the brief promises is not in its volume: "
              f"{wheels.get('why')}\n"
              f"Run: serving_tasks/_modal/modalctl.py env-input",
              file=sys.stderr)
        return 1

    # For the record only -- the factory resolves these itself. Read here so
    # what the journal reports is what the sandbox was actually given.
    cidrs = modalctl.resolve_provider_cidrs(
        modal.App.lookup(modalctl.APP_NAME, create_if_missing=True))
    env = _agent_env(args)

    began = time.time()
    # THE ONE FACTORY. Everything an agent's sandbox and a verdict's sandbox
    # share -- image, GPUs, both volumes, the checkpoint at /models/target, the
    # environment that makes it loadable -- lives there, so a difference has to
    # be chosen rather than drifted into.
    sandbox = modalctl.create_sandbox(
        # The agent's budget PLUS an hour: the sandbox must outlive the agent
        # long enough for the boundary to pull the tree back and the verdict to
        # read it. A sandbox that expires at the deadline loses the work.
        task, purpose="evolve", hours=float(args.hours) + 1.0,
        workdir=WORKSPACE, env=env, network="provider")
    container = f"modal:{sandbox.object_id}"
    try:
        # -- the working tree
        staged = _tar(args.workspace)
        sandbox.filesystem.copy_from_local(str(staged), "/tmp/workspace.tar")
        _exec(sandbox, f"cd {WORKSPACE} && tar xf /tmp/workspace.tar && "
                       f"rm -f /tmp/workspace.tar && "
                       f"chown -R {AGENT_USER}:{AGENT_USER} {WORKSPACE}",
              timeout=1800)
        staged.unlink(missing_ok=True)

        # -- the operator's credentials, as ONE read-only file, never a home
        # ONE definition of where this lands, shared with the driver. The
        # driver has to write here again on every host token refresh -- a
        # sandbox holds a copy, not a bind -- and a second spelling of the path
        # would mean the refresh wrote beside the file the agent reads.
        if args.credentials and Path(args.credentials).is_file():
            sandbox.filesystem.write_text(
                Path(args.credentials).read_text("utf-8"),
                creds_mod.IN_CONTAINER)
            _exec(sandbox,
                  f"chown -R {AGENT_USER}:{AGENT_USER} "
                  f"{Path(creds_mod.IN_CONTAINER).parent} && "
                  f"chmod 600 {creds_mod.IN_CONTAINER}")

        # -- what the run's record should be able to say about the hardware
        uuids = _exec(sandbox, "nvidia-smi --query-gpu=index,uuid "
                               "--format=csv,noheader")
        gpu = {line.split(",")[0].strip(): line.split(",")[1].strip()
               for line in uuids.splitlines() if "," in line}
        seen = _exec(sandbox, "nvidia-smi --query-gpu=name --format=csv,noheader")

        # -- and that the agent can actually work here
        probe = _exec(
            sandbox,
            f"su {AGENT_USER} -p -s /bin/bash -c "
            f"'export HOME=/home/{AGENT_USER}; whoami; "
            f"python3 -c \"import torch;print(torch.cuda.device_count())\"; "
            f"{args.agent_cli} --version'")
    except Exception as exc:                                    # noqa: BLE001
        # A half-built environment is not left running: it holds two GPUs.
        with contextlib.suppress(Exception):
            sandbox.terminate()
        print(f"could not stand the Modal environment up: {exc!r}",
              file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    record = json.dumps({
        "container": container, "ip": "127.0.0.1",
        "backend": "modal", "app": modalctl.APP_NAME,
        "sandbox": sandbox.object_id,
        "gpu_indices": sorted(gpu), "gpu_uuids": gpu,
        "gpu_names": seen.splitlines(),
        "gpu_request": f"{card}:{count}",
        "image": "composed from serverbench/environments/*/Dockerfile",
        "model_path": MODEL_MOUNT, "model_snapshot": snapshot,
        "agent_cli": args.agent_cli, "agent_model": args.agent_model,
        "reasoning_effort": args.agent_effort,
        "agent_preflight": probe.splitlines(),
        "egress": {
            "mechanism": "modal outbound_cidr_allowlist",
            "allowed": cidrs,
            "note": ("there is no iptables allowlist or egress broker inside a "
                     "sandbox; the platform edge does the same job: github, "
                     "pypi, huggingface and download.pytorch.org are "
                     "unreachable, the provider answers. Addresses resolved "
                     "from inside Modal, because the host's DNS returns "
                     "different ones")},
        "measurement": {
            "from": "inside the sandbox, 127.0.0.1",
            "why": ("a sandbox has no routable address; measuring over a "
                    "public tunnel would put a wide-area round trip inside "
                    "every TTFT and TPOT this task gates on. The verdict runs "
                    "once the agent's process is stopped")},
        "seconds_to_stand_up": round(time.time() - began, 1),
    }, indent=1, sort_keys=True) + "\n"
    # BOTH names. The driver reads `harbor_environment.json` for the container
    # and the audits read `container.json`; writing one would leave the other
    # looking at a run that never started.
    (args.out / "container.json").write_text(record, "utf-8")
    (args.out / "harbor_environment.json").write_text(record, "utf-8")

    if args.print_name_only:
        print(container)
    else:
        print(f"container   {container}")
        print(f"gpu         {card}:{count} -> {', '.join(seen.splitlines())}")
        print(f"model       {MODEL_MOUNT} -> {snapshot}")
        print(f"wheelhouse  {modalctl.ENV_INPUT_MOUNT} "
              f"({wheels.get('wheels')} wheels, {wheels.get('gigabytes')} GB)")
        print(f"egress      {len(cidrs)} allowed CIDR(s), everything else refused")
        print(f"agent       {' | '.join(probe.splitlines())}")
        print(f"stood up in {round(time.time() - began, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
