#!/usr/bin/env python3
"""Reach into a Modal sandbox, for a driver whose interpreter has no `modal`.

`lifecycle/backend.py` builds argvs that run this under an interpreter that
does have it. Three verbs, each the Modal equivalent of something the docker
path does with one `docker` invocation:

    exec       docker exec <c> bash -lc '<command>'
    put        docker exec -i <c> bash -c 'cat > <path>'
    terminate  docker rm -f <c>

## The two things that are not mechanical

**The agent user.** Modal containers always run as root and ignore the
`USER agent` our Dockerfile sets, and the agent CLI refuses
`--dangerously-skip-permissions` under root -- correctly, as our own Dockerfile
says. So every command is run through `su agent`, with `-s` naming a shell
explicitly because the login shell is whatever the image set.

**Streaming.** The driver supervises this process the way it supervises
`docker exec`: it reads the agent's stdout as it arrives and enforces the
deadline by signalling this pid. Output is therefore pumped through as it
comes, not collected and printed at the end, and the exit status is the
command's own.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

#: The unprivileged user our Dockerfile creates. Named here because Modal
#: ignores `USER` and something has to re-apply it.
AGENT_USER = "agent"
AGENT_HOME = "/home/agent"

#: The shell every command runs under, named explicitly rather than taken
#: from the image's login shell.
SHELL = "/bin/bash"


def _sandbox(sandbox_id: str):
    import modal
    return modal.Sandbox.from_id(sandbox_id)


def _as_agent(command: str, workdir: str) -> str:
    """Run `command` as the agent user, in `workdir`, with a usable HOME."""
    import shlex
    inner = f"cd {shlex.quote(workdir)} && {command}" if workdir else command
    return (f"su {AGENT_USER} -p -s {SHELL} -c "
            + shlex.quote(f"export HOME={AGENT_HOME}; {inner}"))


def _pump(stream, sink) -> None:
    try:
        for chunk in stream:
            sink.write(chunk)
            sink.flush()
    except Exception:                                    # noqa: BLE001
        # A closed stream ends the pump; it is not a failure of the command.
        pass


def cmd_exec(args) -> int:
    sandbox = _sandbox(args.sandbox)
    command = args.command
    if args.stdin:
        # The caller writes the command's input. Staging a file this way keeps
        # its contents off every command line between here and the sandbox.
        payload = sys.stdin.read()
        import shlex
        command = f"printf %s {shlex.quote(payload)} | {command}"
    process = sandbox.exec(SHELL, "-lc", _as_agent(command, args.workdir),
                           timeout=int(args.timeout))
    threads = [threading.Thread(target=_pump, args=(process.stdout, sys.stdout),
                                daemon=True),
               threading.Thread(target=_pump, args=(process.stderr, sys.stderr),
                                daemon=True)]
    for thread in threads:
        thread.start()
    returncode = process.wait()
    for thread in threads:
        thread.join(timeout=15)
    return int(returncode or 0)


def cmd_put(args) -> int:
    """Write stdin to a path in the sandbox, owned by the agent user."""
    sandbox = _sandbox(args.sandbox)
    parent = str(Path(args.path).parent)
    import shlex
    process = sandbox.exec(SHELL, "-lc", _as_agent("mkdir -p " + shlex.quote(parent), ""), timeout=120)
    if process.wait() != 0:
        raise RuntimeError(f"could not create report directory: {parent}")
    sandbox.filesystem.write_text(sys.stdin.read(), args.path)
    # Ownership and mode, because the file is written as root and is read by
    # the agent. 600 matches what the docker path sets.
    quoted = shlex.quote(args.path)
    process = sandbox.exec(
        SHELL, "-lc",
        f"chown {AGENT_USER}:{AGENT_USER} {quoted} && chmod 600 {quoted}",
        timeout=120)
    process.stdout.read()
    return int(process.wait() or 0)


#: Never pulled back. `.venv` is hundreds of megabytes and is rebuilt by
#: `serve.sh` wherever it runs; caches are noise. Everything else the agent
#: touched comes home, including its own measurement exports under `agentx/`,
#: which are the evidence for what it did.
PULL_EXCLUDES = (".venv", "__pycache__", ".uv-cache", ".pytest_cache",
                 ".git", "node_modules")


def cmd_pull(args) -> int:
    """Bring the agent's working tree back out of the sandbox.

    On docker the workspace is a BIND MOUNT, so the host directory is the
    agent's tree and there is nothing to do. A sandbox has no bind mounts: the
    tree was uploaded once and lives only there, so without this the harness
    captures whatever the host still holds -- which is the seed -- and the
    agent's work is lost when the sandbox is terminated.
    """
    import shutil
    import tarfile
    import tempfile

    sandbox = _sandbox(args.sandbox)
    excludes = " ".join(f"--exclude={name}" for name in PULL_EXCLUDES)
    # Tarred in the sandbox: a per-file API over thousands of files costs
    # minutes, and this runs at every version boundary.
    process = sandbox.exec(
        SHELL, "-lc",
        f"cd {args.source} && tar {excludes} -cf /tmp/pull.tar . && "
        f"stat -c %s /tmp/pull.tar",
        timeout=1800)
    size = (process.stdout.read() or "").strip().splitlines()
    code = process.wait()
    if code != 0:
        print(f"could not archive {args.source} in the sandbox", file=sys.stderr)
        return 1

    staged = Path(tempfile.mkdtemp(prefix="serverbench-pull-")) / "pull.tar"
    sandbox.filesystem.copy_to_local("/tmp/pull.tar", str(staged))
    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="modal-pull-verified-") as temporary:
        fresh = Path(temporary)
        with tarfile.open(staged, "r") as tar:
            tar.extractall(fresh, filter="data")
        # Reconcile source deletions as well as edits. Otherwise deleted code
        # from an earlier boundary silently reappears in the final delivery.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from final_validate.candidate_bundle import audit
        old_files = set(audit(destination)["files"])
        new_files = set(audit(fresh)["files"])
        for name in old_files - new_files:
            (destination / name).unlink(missing_ok=True)
        for path in fresh.rglob("*"):
            relative = path.relative_to(fresh)
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"workspace target escapes destination: {relative}")
            if path.is_symlink():
                # Delivery rejects symlinks; retain neither a link nor a stale
                # previous regular file at the same name.
                if target.is_file():
                    target.unlink()
                continue
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    staged.unlink(missing_ok=True)
    print(f"pulled {size[-1] if size else '?'} bytes into {destination}")
    return 0


def cmd_requests(args) -> int:
    sandbox = _sandbox(args.sandbox)
    script = """import json
from pathlib import Path
root = Path('/workspace/.versions')
files = sorted(root.glob('*.json'))
if len(files) > 10000: raise ValueError('too many request envelopes')
result = {}
for path in files:
    if path.is_symlink() or path.stat().st_size > 16384:
        raise ValueError('unsafe/oversized request envelope')
    result[path.name] = path.read_text()
print(json.dumps(result))
"""
    process = sandbox.exec("python3", "-c", script, timeout=20)
    output = process.stdout.read()
    code = process.wait()
    if code == 0:
        print(output)
    return int(code or 0)


def cmd_push(args) -> int:
    import shlex
    import tarfile
    import tempfile
    sandbox = _sandbox(args.sandbox)
    with tempfile.TemporaryDirectory(prefix="modal-public-evidence-") as temporary:
        archive = Path(temporary) / "evidence.tar"
        with tarfile.open(archive, "w") as handle:
            for path in Path(args.source).rglob("*"):
                if path.is_file() and not path.is_symlink():
                    handle.add(path, arcname=str(path.relative_to(args.source)), recursive=False)
        sandbox.filesystem.copy_from_local(str(archive), "/tmp/public-evidence.tar")
        dest = shlex.quote(args.dest)
        process = sandbox.exec("bash", "-c", f"mkdir -p {dest} && "
                               f"tar xf /tmp/public-evidence.tar -C {dest} && "
                               f"chown -R agent:agent {dest}", timeout=600)
        return int(process.wait() or 0)


def cmd_terminate(args) -> int:
    try:
        _sandbox(args.sandbox).terminate()
    except Exception as exc:                             # noqa: BLE001
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="verb", required=True)

    run = sub.add_parser("exec")
    run.add_argument("--sandbox", required=True)
    run.add_argument("--workdir", default="/workspace")
    run.add_argument("--command", required=True)
    run.add_argument("--stdin", action="store_true")
    run.add_argument("--timeout", default=86400)

    put = sub.add_parser("put")
    put.add_argument("--sandbox", required=True)
    put.add_argument("--path", required=True)

    pull = sub.add_parser("pull")
    pull.add_argument("--sandbox", required=True)
    pull.add_argument("--source", default="/workspace")
    pull.add_argument("--dest", required=True)

    requests = sub.add_parser("requests")
    requests.add_argument("--sandbox", required=True)

    push = sub.add_parser("push")
    push.add_argument("--sandbox", required=True)
    push.add_argument("--source", required=True)
    push.add_argument("--dest", required=True)

    stop = sub.add_parser("terminate")
    stop.add_argument("--sandbox", required=True)

    args = ap.parse_args(argv)
    return {"exec": cmd_exec, "put": cmd_put, "pull": cmd_pull,
            "requests": cmd_requests, "push": cmd_push,
            "terminate": cmd_terminate}[args.verb](args)


if __name__ == "__main__":
    raise SystemExit(main())
