#!/usr/bin/env python3
"""The boundary rules for the agent's container. A library, not an entrypoint.

Harbor composes and starts every run's environment
(`scripts/agent/harbor_env.py` -> `harbor.environments.docker.DockerEnvironment`),
and `harness/lifecycle/agent.py` owns the agent launch. This module holds the
rules about the BOUNDARY, imported by `harbor_env.py` rather than copied into
it, so each safety rule has one implementation:

* `resolve_exposures` -- every `--expose` is scanned with the workspace leakage
  vocabulary AND hashed against the task's held-out oracle files, and a path
  carrying any of it is REFUSED rather than trimmed. Refusing is the only safe
  answer: a partially-mounted oracle still answers the question the agent is
  supposed to be solving.
* `_held_out_digests` -- and it refuses when there is nothing to compare
  against, because "this task ships no oracle" is the ordinary state of a fresh
  clone and is not an audit.
* `blob_mounts` / `check_model_readable` -- a Hugging Face snapshot is a farm of
  relative symlinks into `../../blobs/<sha>`, so mounting the snapshot alone
  gives a container with no weights and no `tokenizer_config.json`.
* `device_uuid` -- a logical ordinal renumbers and a UUID does not.
* `harden_and_probe` -- installs the egress allowlist in the container's own
  network namespace and then PROVES it from inside with six probes, because
  Harbor's docker backend implements no allowlist of its own.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

#: The repository root, two directories above this script.
REPO = Path(__file__).resolve().parents[3]

#: What must not be reachable from inside the task network, each refusal probed
#: rather than asserted. NOTHING on the internet is reachable from the
#: container, the model endpoint included: the container's allowlist names the
#: docker bridge gateway -- the host -- and the agent's model traffic is relayed
#: by `egress_broker.py`, a process outside its network namespace that opens one
#: kind of connection and logs every attempt. So every probe expects BLOCKED,
#: and a direct connection to the API from inside the container is a finding.
EGRESS_PROBES = {
    "api.anthropic.com": False, "pypi.org": False,
    "files.pythonhosted.org": False, "github.com": False,
    "huggingface.co": False, "download.pytorch.org": False,
}

WORKSPACE = "/workspace"
#: Where the model is mounted, for the agent and for the verdict alike.
#: `final_validate/deploy.py` mounts the model here too and runs `serve.sh`
#: VERBATIM, so both contexts agree on the one path the script needs. The name
#: says nothing about which model it is -- the task says that, in one place.
MODEL_MOUNT = "/models/target"
ENV_INPUT_MOUNT = "/opt/env-input"
UV_CACHE_MOUNT = "/opt/uv-cache"


def docker(*args: str, timeout: float = 600.0, check: bool = True,
           capture: bool = True) -> subprocess.CompletedProcess:
    argv = ["docker", *args]
    result = subprocess.run(argv, capture_output=capture, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise SystemExit(f"docker {' '.join(args[:3])} failed ({result.returncode}): "
                         f"{(result.stderr or '')[:400]}")
    return result


def device_uuid(index: int) -> str:
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60)
    return out.stdout.strip()


#: Path fragments that leak by location, whatever the content.
LEAKAGE_PATHS = ("server_prototypes", "third_party")

#: File names that leak by identity: the answers to a held-out partition.
LEAKAGE_NAMES = (
    "references.json", "baseline.json", "DATASHEET.txt",
    "references-aa.json", "references-qualification.json", "references-final.json",
    "trace-aa.json", "trace-qualification.json", "trace-final.json",
)

#: Where an exposed path lands inside the container, when the caller does not say.
EXPOSE_ROOT = "/opt/visible"


def _held_out_digests(task_dir: Path | None) -> dict[str, str]:
    """Content hashes of everything the agent is not allowed to have.

    The held-out partitions and the cross-framework consensus, matched by hash
    rather than by name, because a leak that mattered would not be called
    `references-final.json`.
    """
    if task_dir is None:
        return {}
    import hashlib

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    out: dict[str, str] = {}
    oracle = Path(task_dir) / "oracle"
    for name in ("aa", "qualification", "final"):
        for kind in ("trace", "references"):
            path = oracle / f"{kind}-{name}.json"
            if path.exists():
                out[digest(path)] = f"{kind}-{name}"
    for path in sorted(oracle.glob("consensus-*.json")):
        out[digest(path)] = path.stem
    return out


def resolve_exposures(specs: list[str],
                      task_dir: Path | None) -> tuple[list[str], list[str]]:
    """Turn `--expose` specs into docker -v arguments, or refuse them.

    Returns the mount arguments and a human-readable record of what was exposed.
    Raises SystemExit on anything that would hand the agent held-out evidence --
    the run does not start, because a container that has already been given the
    answer key cannot be un-given it.
    """
    if not specs:
        return [], []

    import hashlib

    held_out = _held_out_digests(task_dir)
    if task_dir is None:
        raise SystemExit(
            "--expose needs --task: an exposed path is checked against that "
            "task's held-out partitions and consensus by content hash, and "
            "without the task there is nothing to check against. Refusing "
            "rather than exposing something unaudited.")
    # An EMPTY digest map is the same condition as no task at all: with
    # nothing to compare against, a byte-identical copy of the answer key under
    # another name would mount cleanly. It is also the normal state of a fresh
    # clone, whose traces and references are regenerated per host.
    if not held_out:
        raise SystemExit(
            f"--expose refuses: {task_dir} has no held-out files to check "
            f"against (no oracle/trace-*.json, references-*.json or "
            f"consensus-*.json). The content-hash audit is the only thing that "
            f"catches a renamed copy of the answer key, and with nothing to "
            f"compare against it is not an audit. Regenerate the task's oracle "
            f"first, or expose nothing.")

    mounts: list[str] = []
    record: list[str] = []
    for spec in specs:
        parts = spec.split(":")
        mode = "ro"
        if parts and parts[-1] in ("ro", "rw"):
            mode = parts.pop()
        host = Path(parts[0]).expanduser().resolve()
        dest = parts[1] if len(parts) > 1 else f"{EXPOSE_ROOT}/{host.name}"
        if not host.exists():
            raise SystemExit(f"--expose {spec}: {host} does not exist")

        # `rglob` does NOT descend a symlinked directory, and a directory
        # symlink is not `is_file()`, so one inside an exposed tree would be
        # neither walked nor hashed, while a relative link stays live in the
        # container (`oracle -> ../../oracle`). Refused rather than followed:
        # auditing a link means auditing wherever it points, and a mount whose
        # contents can change under the audit is not audited.
        walked = [host] if host.is_file() else list(host.rglob("*"))
        links = [p for p in walked if p.is_symlink()]
        if links:
            listed = "\n  ".join(str(p) for p in sorted(links)[:10])
            raise SystemExit(
                f"--expose {host} contains {len(links)} symlink(s):\n"
                f"  {listed}\n"
                f"Refused. A link is not walked by the content audit, so a "
                f"tree containing one cannot be called audited. Expose the "
                f"real directory, or a copy with the links resolved.")

        files = [p for p in walked if p.is_file()]
        problems: list[str] = []
        for path in files:
            relative = str(path)
            if any(fragment in relative for fragment in LEAKAGE_PATHS):
                problems.append(f"{path.name}: forbidden path fragment")
            if path.name in LEAKAGE_NAMES:
                problems.append(f"{path.name}: withheld by name")
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            if h in held_out:
                problems.append(f"{path.name}: held-out {held_out[h]} by "
                                f"content hash")
        if problems:
            listed = "\n  ".join(sorted(set(problems))[:10])
            raise SystemExit(
                f"--expose {host} carries evidence the agent must not see:\n"
                f"  {listed}\n"
                f"Refused. Expose a narrower directory -- the point of the flag "
                f"is to widen visibility deliberately, and a tree that holds the "
                f"answer key cannot be widened into safely.")

        mounts += ["-v", f"{host}:{dest}:{mode}"]
        record.append(f"{host} -> {dest} ({mode}, {len(files)} file(s), audited)")
    return mounts, record


def blob_mounts(model_dir: Path) -> list[str]:
    """Also mount whatever the model directory's symlinks point at.

    A HuggingFace snapshot directory is a farm of relative symlinks
    (`config.json -> ../../blobs/<sha>`); the bytes live in a sibling `blobs/`.
    Bind-mounting the snapshot alone therefore mounts the *links* and not their
    targets, and every file in the model directory dangles inside the container
    -- including `tokenizer_config.json`, without which the agent cannot apply
    the chat template or run any gate.

    The links are relative, so once the snapshot sits at MODEL_MOUNT the target
    directory has a determinate path inside the container; mount the real one
    there.
    """
    if not model_dir.is_dir():
        return []
    targets: dict[str, Path] = {}
    for entry in sorted(model_dir.iterdir()):
        if not entry.is_symlink():
            continue
        raw = os.readlink(entry)
        if os.path.isabs(raw) or os.path.dirname(raw) in ("", "."):
            continue                      # same-directory link: already mounted
        host = (model_dir / os.path.dirname(raw)).resolve()
        inside = os.path.normpath(os.path.join(MODEL_MOUNT, os.path.dirname(raw)))
        if host.is_dir():
            targets[inside] = host
    mounts: list[str] = []
    for inside, host in sorted(targets.items()):
        mounts += ["-v", f"{host}:{inside}:ro"]
    return mounts


def harden_and_probe(container: str, out: Path, allow: list[str]) -> dict:
    """Install the egress allowlist in a running container, then PROVE it.

    Everything here is keyed by container NAME and reached with `docker
    exec`, so it works on the container Harbor composed: the allowlist is a
    property of the container's network namespace, not of who started it.

    The comma join is `restrict-network.sh`'s documented contract
    (`ALLOWED_HOSTS=host1,host2`); a space-joined list would parse as one
    unresolvable hostname and allow nothing.
    """
    firewall = docker("exec", "--user", "root", "-e",
                      f"ALLOWED_HOSTS={','.join(allow)}", container,
                      "bash", "/usr/local/bin/restrict-network.sh",
                      timeout=300, check=False)
    (out / "firewall.log").write_text(
        (firewall.stdout or "") + (firewall.stderr or ""), "utf-8")

    # Proved from inside, rather than asserted. Each of these is a way the
    # agent could otherwise fetch an answer instead of building one.
    egress: dict[str, dict] = {}
    for host, should_reach in EGRESS_PROBES.items():
        probe = docker(
            "exec", container, "bash", "-lc",
            f"timeout 8 bash -c '</dev/tcp/{host}/443' 2>/dev/null && echo REACHABLE "
            f"|| echo BLOCKED", timeout=60, check=False)
        reached = "REACHABLE" in probe.stdout
        egress[host] = {"reachable": reached, "expected_reachable": should_reach,
                        "ok": reached == should_reach}
    (out / "egress.json").write_text(
        json.dumps(egress, indent=1, sort_keys=True) + "\n", "utf-8")
    for host, result in sorted(egress.items()):
        mark = "ok  " if result["ok"] else "FAIL"
        state = "reachable" if result["reachable"] else "blocked"
        print(f"  {mark}  {host:<28}{state}")
    return egress


def check_model_readable(model_dir: Path) -> None:
    """Refuse to launch on a model directory whose files do not resolve.

    Cheap on the host, and it turns a silent agent stall into a message naming
    the file.
    """
    broken = [p.name for p in sorted(model_dir.iterdir())
              if p.is_symlink() and not p.resolve().exists()]
    if broken:
        raise SystemExit(
            f"model directory {model_dir} has {len(broken)} dangling link(s) on "
            f"the HOST -- {', '.join(broken[:4])}. The cache is incomplete; "
            f"re-download the revision rather than mounting it.")



