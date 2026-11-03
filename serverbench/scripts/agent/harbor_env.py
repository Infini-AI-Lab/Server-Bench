#!/usr/bin/env python3
"""Stand the agent's container up through **Harbor**, and hand back its name.

WHAT HARBOR OWNS, AND WHAT IT DOES NOT
======================================

Harbor owns the environment lifecycle: it composes the container, starts it,
executes in it and stops it -- `DockerEnvironment.start/exec/stop`, which is a
real interface with a real implementation rather than a shell string this
repository maintains. The offline wheelhouse (`serverbench/artifacts/env-input/`)
is not a Harbor format: it is a plain directory of wheels, a `uv` binary and
`resolved.requirements.txt`, built by `agent/build_env_bundle.py` and
bind-mounted at `/opt/env-input:ro`.

Harbor does **not** own the boundary. The exposure audit, the held-out content
hashing, the credential mount and the egress probes
stay in `run_agent.py` and are IMPORTED here rather than reimplemented -- they
are the properties a measurement rests on, and each has one implementation.

RUN THIS UNDER HARBOR'S OWN INTERPRETER
=======================================

    harbor/.venv/bin/python serverbench/scripts/agent/harbor_env.py ...

Harbor needs pydantic, litellm, fastapi, supabase and twenty more. The harness
venv deliberately has none of them -- `tests/repo/test_the_core_imports_in_a_bare_image.py`
fails a module-scope third-party import in `harness/`, because the harness core
must import inside a bare image to verify a candidate there. So Harbor runs in
its own environment and is reached as a subprocess, from the driver.
`harbor/.venv` is not committed; `serverbench/scripts/deploy/00-envs.sh`
creates it.

THREE THINGS HARBOR CANNOT SAY ON ITS OWN, AND HOW THEY ARE SAID
================================================================

`EnvironmentConfig` has no field for any of these, and all three are load-bearing
for this task, so they go in one generated compose overlay:

* **the GPU.** `capabilities.gpus` is `False` for the local docker backend and
  `_validate_gpu_support` (`environments/base.py:650`) RAISES when effective
  gpus > 0 -- so the card is granted by compose `deploy.resources.reservations`
  and the guard is silenced with `override_gpus=0`, the pattern of
  `configs/gpu_overlay_nvidia.yaml`. Both keys are required: one grants
  nothing, the other refuses.
* **`cap_add: [NET_ADMIN]`**, without which `restrict-network.sh` cannot install
  the egress allowlist and the agent can fetch a serving framework.
* **`shm_size`**, without which torch's dataloader and NCCL fall over on the
  default 64 MB.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

#: The repository root, three directories above this script.
REPO = Path(__file__).resolve().parents[3]

# The boundary rules live in `run_agent.py` and are imported, never copied.
# It is stdlib-only at module scope, so it loads under Harbor's interpreter too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_agent  # noqa: E402

WORKSPACE = run_agent.WORKSPACE
MODEL_MOUNT = run_agent.MODEL_MOUNT

#: The one model path the agent and the verdict share.
CANONICAL_MODEL_MOUNT = MODEL_MOUNT
ENV_INPUT_MOUNT = run_agent.ENV_INPUT_MOUNT
UV_CACHE_MOUNT = run_agent.UV_CACHE_MOUNT

#: Where the pinned AgentX runtime is mounted, read-only, so the agent can USE
#: it for development benchmarking. It is mounted rather than baked into the
#: image: a read-only bind costs nothing per run and stays pinned to the
#: submodule.
AGENTX_MOUNT = "/opt/agentx"

#: The host AgentX runtime `scripts/deploy/00-envs.sh` builds; `SERVERBENCH_AGENTX_RUNTIME` overrides it.
AGENTX_RUNTIME_DEFAULT = REPO / "serverbench" / "artifacts" / "agentx-runtime"
# A SIBLING, not a child: a bind cannot be nested inside a read-only bind --
#: docker cannot create the mountpoint on a read-only rootfs layer.
AGENTX_SOURCE_MOUNT = "/opt/agentx-src"

#: The HuggingFace cache, read-only, so AgentX's tokenizer resolves OFFLINE.
#:
#: AgentX at the pin cannot take a local directory as `--tokenizer` when
#: `HF_HUB_OFFLINE=1`: `tokenizer.py::_from_pretrained_local` resolves the name
#: through `snapshot_download(local_files_only=True)`, which validates it as a
#: REPO ID and rejects a path with `HFValidationError: Repo id must be in the
#: form 'repo_name' or 'namespace/repo_name'`. Plain `transformers`
#: loads the same directory without complaint, so this is AgentX's own
#: resolution path rather than anything about the mount.
#:
#: With the cache mounted and `HF_HOME` pointed at it, `--tokenizer <repo id>`
#: resolves from disk with no network. Read-only, and it carries only the model
#: the task already gives the agent.
HF_CACHE_MOUNT = "/opt/hf-cache"
#: The full context window of every non-Claude model the Claude CLI drives
#: here, read from its provider (Fireworks `contextLength`, OpenRouter
#: `context_length`).
#:
#: Every agent runs at its model's full window. The CLI's model catalog gives
#: Claude models their native 1M window; for any other model it assumes 200k
#: and auto-compacts at ~167k tokens, each compaction replacing the agent's
#: history (the brief included) with a summary.
#: `CLAUDE_CODE_MAX_CONTEXT_TOKENS` is the CLI's own setting for an unknown
#: model's window. A model missing from this table is refused rather than
#: silently run at 200k. Models on the Codex CLI get theirs from
#: `lifecycle.agent.CODEX_CONTEXT_WINDOWS`.
AGENT_CONTEXT_WINDOWS = {
    "z-ai/glm-5.3": 1_048_576,
    "accounts/fireworks/models/kimi-k3": 1_048_576,
    "accounts/fireworks/models/deepseek-v4p1-flash": 1_048_576,
    "xiaomi/mimo-v2.6-pro": 1_048_576,
    "qwen/qwen3.8-max-0902": 1_000_000,
}


def context_window_for(env: dict, model: str) -> int | None:
    """The window the CLI must use for `model`; None for a Claude model (the CLI gives it its native 1M)."""
    if model.startswith("claude-"):
        return None
    declared = env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS") or AGENT_CONTEXT_WINDOWS.get(model)
    if not declared:
        raise SystemExit(
            f"REFUSED: the context window of {model!r} is unknown. The Claude CLI would run it "
            f"at its 200k default and auto-compact at ~167k tokens. Add its provider-verified "
            f"window to AGENT_CONTEXT_WINDOWS (or CLAUDE_CODE_MAX_CONTEXT_TOKENS in its env file).")
    return int(declared)


def pin_model_routes(env: dict, agent_model: str, agent_cli: str) -> str:
    """Point every Claude CLI model alias at the run's one model, give the CLI
    that model's real context window, and return the model."""
    pinned = env.get("ANTHROPIC_MODEL") or agent_model
    if not pinned or agent_cli != "claude":
        return ""
    for alias in SINGLE_MODEL_ALIASES:
        env[alias] = pinned
    window = context_window_for(env, pinned)
    if window:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(window)
        env.pop("CLAUDE_CODE_AUTO_COMPACT_WINDOW", None)   # it would take precedence
    return pinned


#: Claude CLI variables that select a model other than the session's own.
SINGLE_MODEL_ALIASES = ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
                        "ANTHROPIC_DEFAULT_OPUS_MODEL",
                        "ANTHROPIC_DEFAULT_SONNET_MODEL",
                        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                        "CLAUDE_CODE_SUBAGENT_MODEL")

#: Fallback repo id for `hf_cache_for`'s presence check, used only when the
#: staged model directory does not say which repo it came from. The task's own
#: repo id (`repo_id_for`) is always preferred: checking the cache for some
#: other model would let the brief's `--tokenizer` instructions silently drop.
DEFAULT_MODEL_REPO_ID = "Qwen/Qwen3-4B"

#: The CUDA toolchain in the agent image, and the three variables that have to
#: reach every process rather than only a login shell. See the PATH note in
#: `main`.
CUDA_HOME = "/usr/local/cuda"
CUDA_BIN = f"{CUDA_HOME}/bin"

#: Writable, and NOT under `/workspace`: the workspace filesystem can run close
#: to full, and a dataset lock that cannot be taken looks like a hung loader
#: rather than a disk problem.
HF_DATASETS_CACHE = "/tmp/hf-datasets"

CREDENTIAL_DEFAULTS = {
    "claude": "~/.claude/.credentials.json",
    "codex": "~/.codex/auth.json",
}


def repo_id_for(model: Path) -> str:
    """The Hugging Face repo id a staged model directory came from.

    A cache snapshot carries it in the path
    (`.../hub/models--Qwen--Qwen3.5-9B/snapshots/<sha>`), which is the layout
    the harness mounts. Anything else falls back, because a flat directory
    name says nothing reliable about which model it holds.
    """
    for parent in Path(model).resolve().parents:
        name = parent.name
        if name.startswith("models--"):
            return name[len("models--"):].replace("--", "/", 1)
    return DEFAULT_MODEL_REPO_ID


def agentx_interpreter_mounts(runtime: Path) -> list[str]:
    """The interpreter AgentX's venv links to, at the path it links to.

    The runtime's venv is created by `uv` on the host, so `venv/bin/python` is
    an absolute symlink into the operator's uv toolchain:

        /opt/agentx/venv/bin/python
          -> /home/<user>/.local/share/uv/python/cpython-3.12-.../bin/python3.12

    Mounting the runtime alone carries the link and not its target, and inside
    the container every script in `venv/bin` would fail:

        /opt/agentx/venv/bin/aiperf: exec: /opt/agentx/venv/bin/python: not found

    `aiperf` on PATH is a wrapper that avoids the venv's interpreter, but an
    agent that finds the venv should not find a broken interpreter in it.

    Derived, never hardcoded: the link is read and its interpreter directory
    bound read-only at exactly the path the link names, so the venv resolves
    unchanged. It is a Python install -- `bin/`, `lib/`, `include/` -- and
    nothing else of the user's home. A runtime whose interpreter already lies
    inside it, or whose link does not resolve on this host, needs nothing.
    """
    link = Path(runtime) / "venv" / "bin" / "python"
    if not link.is_symlink():
        return []
    target = Path(os.readlink(link))
    if not target.is_absolute():
        return []
    # .../cpython-3.12-linux-x86_64-gnu/bin/python3.12 -> the install root
    install = target.parent.parent
    resolved_runtime = str(Path(runtime).resolve())
    if str(install).startswith(resolved_runtime) or not install.exists():
        return []
    return [bind(install.resolve(), str(install), read_only=True)]


def agentx_runtime() -> Path | None:
    """The host directory holding the pinned AgentX runtime, or None.

    Absent is not an error here: a host that has not built it gets a container
    without AgentX and an agent brief that does not name it, which is honest.
    What must never happen is a brief naming an executable that is not there.
    """
    from os import environ

    override = environ.get("SERVERBENCH_AGENTX_RUNTIME", "").strip()
    root = Path(override) if override else AGENTX_RUNTIME_DEFAULT
    return root if (root / "venv").is_dir() and (root / "bin" / "aiperf").is_file() else None


def agentx_source() -> Path | None:
    """A usable AgentX source checkout, rather than an empty submodule dir.

    Git creates the submodule's directory before it initialises the submodule,
    so ``is_dir()`` alone would mount an empty directory and export
    ``AGENTX_SOURCE=/opt/agentx-src`` as if source and tutorials were present.
    The two markers below are deliberately about CONTENT: together they prove
    this is the AgentX Python project the brief means, not merely a directory
    which happens to have the right name.
    """
    from os import environ

    override = environ.get("SERVERBENCH_AGENTX_SOURCE", "").strip()
    root = Path(override).expanduser() if override else REPO / "agentx-harness"
    markers = (root / "pyproject.toml", root / "src" / "aiperf" / "__init__.py")
    return root if all(marker.is_file() for marker in markers) else None

#: The service Harbor's own compose files define. Everything addresses it by
#: this name; it is Harbor's constant, not ours.
MAIN_SERVICE = "main"


def harbor_python() -> Path:
    """Harbor's interpreter, which is not the harness's."""
    return REPO / "harbor" / ".venv" / "bin" / "python"


def harbor_available() -> tuple[bool, str]:
    """Whether Harbor can be driven on this host, and why not when it cannot.

    Reported rather than raised so the caller can say something useful. A
    missing `harbor/.venv` is the ordinary state of a fresh clone -- it is
    gitignored, and `serverbench/scripts/deploy/00-envs.sh` creates it.
    """
    python = harbor_python()
    if not python.is_file():
        return False, (f"no Harbor interpreter at {python}. `harbor/.venv` is "
                       f"not committed; create it with "
                       f"`bash serverbench/scripts/deploy/00-envs.sh`")
    probe = subprocess.run(
        [str(python), "-c", "import harbor; print(harbor.__version__)"],
        capture_output=True, text=True, check=False, timeout=120)
    if probe.returncode != 0:
        return False, f"harbor does not import: {probe.stderr.strip()[-300:]}"
    return True, f"harbor {probe.stdout.strip()}"


#: Environment names whose VALUE is a credential. Matched case-insensitively
#: on the name, never on the value: a key has no reliable shape, and a rule
#: that looked for one would miss the next provider's.
SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def redacted(text: str) -> str:
    """The composed overlay with every credential value replaced.

    A RUN'S EVIDENCE MUST NOT CARRY A KEY. `harbor_environment.json` records
    the overlay this run was composed from -- the only statement of what the
    container was actually given -- and the overlay carries the agent's
    provider credentials as environment values. Evidence is copied into published
    versions, read by reports, and passed around when a run is being
    explained, so a credential in it is a credential in all of those.

    What is kept is the NAME and the fact that a value was set, because "no
    credential reached the container" is the most common cause of an agent
    that cannot start and the evidence has to be able to say which one was
    given.
    """
    out = []
    for line in text.splitlines(keepends=True):
        name, sep, _ = line.partition(":")
        bare = name.strip().strip('"')
        if sep and bare.isupper() and any(
                marker in bare for marker in SECRET_ENV_MARKERS):
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f'{indent}{name.strip()}: "[REDACTED: value was set]"\n')
            continue
        out.append(line)
    return "".join(out)


def overlay_text(*, devices: list[int], shm_size: str, uid: int,
                 gid: int, env: dict[str, str] | None = None) -> str:
    """The compose overlay carrying what `EnvironmentConfig` cannot express.

    `device_ids` are strings on purpose: compose rejects integers here, and the
    failure is a schema error hundreds of lines into `docker compose config`
    rather than anything naming the card.
    """
    ids = ", ".join(f'"{d}"' for d in devices)
    # The CONTAINER's own environment, and this is the only channel that puts it
    # there. Harbor's `persistent_env` is merged into commands run through
    # HARBOR's `exec`, and `task_env_config.env` feeds compose variable
    # substitution -- neither reaches `Config.Env`. The driver starts the agent,
    # the reviewer and the candidate server with raw `docker exec` by name, and
    # without this `MODEL_PATH` would be absent, the seed would fall back to
    # `/models/model`, and transformers would read that as a hub repo id.
    #
    # `$` is doubled because compose substitutes it inside its own files: an
    # unescaped one becomes an empty string or somebody else's variable.
    lines = []
    for key, value in sorted((env or {}).items()):
        lines.append(f'      {key}: "{str(value).replace(chr(36), chr(36) * 2)}"')
    environment = ("    environment:\n" + "\n".join(lines) + "\n") if lines else ""
    return (
        "# Generated by serverbench/scripts/agent/harbor_env.py. Do not edit.\n"
        "#\n"
        "# Four settings Harbor's EnvironmentConfig has no field for, and all\n"
        "# four decide whether this run means anything:\n"
        "#   init              keeps a supervisor at PID 1, matching final\n"
        "#                     validation's candidate process model.\n"
        "#   deploy.resources  the card. `capabilities.gpus` is False for the\n"
        "#                     local docker backend, so this is the only way\n"
        "#                     one is granted -- paired with override_gpus=0,\n"
        "#                     which silences the guard that would refuse.\n"
        "#   cap_add           NET_ADMIN, without which restrict-network.sh\n"
        "#                     cannot install the egress allowlist.\n"
        "#   shm_size          torch on the default 64 MB does not survive.\n"
        "services:\n"
        f"  {MAIN_SERVICE}:\n"
        "    init: true\n"
        f"    user: \"{uid}:{gid}\"\n"
        f"    shm_size: {shm_size}\n"
        "    cap_add:\n"
        "      - NET_ADMIN\n"
        "    deploy:\n"
        "      resources:\n"
        "        reservations:\n"
        "          devices:\n"
        "            - driver: nvidia\n"
        f"              device_ids: [{ids}]\n"
        "              capabilities: [gpu]\n"
        + environment)


def bind(source: Path | str, target: str, *, read_only: bool = False) -> dict:
    """One bind mount in Harbor's `ServiceVolumeConfig` shape.

    `read_only` is `NotRequired[Literal[True]]` in that TypedDict -- the key is
    OMITTED for a writable mount rather than set to False, which the schema
    rejects.
    """
    entry = {"type": "bind", "source": str(source), "target": target}
    if read_only:
        entry["read_only"] = True
    return entry


def _broker_allowlist(agent_cli: str, additional: list[str]) -> set[str]:
    """Exact provider endpoints plus explicit host-side additions."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import egress_broker  # noqa: E402, PLC0415

    return set(egress_broker.PROVIDER_HOSTS[agent_cli]) | set(additional)


#: Environment names that carry a credential for SOME provider. Only the run's
#: own route keeps any (see `credential_route`).
_CREDENTIAL_NAME = re.compile(r"(API_KEY|AUTH_TOKEN|ACCESS_TOKEN|SECRET|PASSWORD)$")
#: What each route may carry.
ROUTE_CREDENTIALS = {"api_key": {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"},
                     "oauth": {"CLAUDE_CODE_OAUTH_TOKEN"}, "codex": set()}


def credential_route(env: dict, agent_cli: str) -> str:
    """Which single credential this run uses; strip every other one from `env`.

    * ``api_key`` -- a Claude CLI routed through ``ANTHROPIC_BASE_URL`` (a
      third-party model): its env file's key, and the OAuth file is NOT mounted.
    * ``oauth``   -- a Claude CLI on Anthropic itself (e.g. Opus): the OAuth
      token (its file and `CLAUDE_CODE_OAUTH_TOKEN`); any API key is removed.
    * ``codex``   -- the Codex CLI's own auth file.
    """
    if agent_cli == "codex":
        route = "codex"
    elif env.get("ANTHROPIC_BASE_URL"):
        route = "api_key"
        if not (env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY")):
            raise SystemExit("ANTHROPIC_BASE_URL is set but the env file carries "
                             "no key for it; refusing to start a run that cannot "
                             "reach its model")
    else:
        route = "oauth"
    for name in [n for n in env if _CREDENTIAL_NAME.search(n)]:
        if name not in ROUTE_CREDENTIALS[route]:
            del env[name]
            print(f"credentials: removed {name} (not this run's {route} route)")
    return route


#: Anthropic's own endpoints, for a Claude CLI run with no third-party route.
ANTHROPIC_HOSTS = frozenset({"api.anthropic.com", "statsig.anthropic.com",
                             "sentry.io"})


def narrow_to_provider(hosts: set[str], env: dict, agent_cli: str,
                       additional: list[str]) -> set[str]:
    """The broker relays to this run's ONE model provider, nothing else.

    The Claude CLI's static list names several providers; a session routed
    through one of them (`ANTHROPIC_BASE_URL`) must not reach the others, so
    the brief's "only this task's model provider is reachable" is true and an
    agent cannot route to another model.
    """
    if agent_cli != "claude":
        return set(hosts)
    base = env.get("ANTHROPIC_BASE_URL") or ""
    if base:
        from urllib.parse import urlparse
        host = urlparse(base).hostname
        if not host:
            raise SystemExit(f"ANTHROPIC_BASE_URL has no host: {base!r}")
        return {host} | set(additional)
    return set(ANTHROPIC_HOSTS) | set(additional)


def start_environment(environment) -> None:
    """Actually start it. Harbor's environment API is asynchronous.

    `DockerEnvironment.start` is a COROUTINE, as are `stop`, `exec`,
    `upload_dir` and `download_dir`. Calling one without awaiting it returns a
    coroutine object and does nothing at all -- Python says so with
    `RuntimeWarning: coroutine 'DockerEnvironment.start' was never awaited`,
    which is easy to lose in a subprocess's stderr, and the failure then
    surfaces two steps later as "no running container carries its labels".

    One `asyncio.run` for the whole start, and nothing else on this object is
    touched afterwards -- the driver addresses the container by name with
    docker. Harbor's own CLI warns that a second `asyncio.run` over the same
    objects strands loop-bound singletons, so a second one is deliberately not
    taken.
    """
    asyncio.run(environment.start(force_build=False))


def credential_mount(credentials: Path, agent_cli: str = "claude",
                     run_root: Path | None = None) -> list[dict]:
    """The operator's credentials, one file, read-only -- from the RUN's copy.

    A file and never a directory: mounting `~/.claude` would carry the
    operator's own transcripts into the agent's workspace.

    And the run's own copy rather than the operator's file, because a bind of a
    file follows its INODE and Claude Code refreshes its token by atomic
    replace -- so the container would hold a token the refresh had revoked.
    `lifecycle.credentials` carries the full account and the in-place refresh
    that keeps the copy current; this only chooses what to bind.

    Without `run_root` the operator's file is bound directly and the refresh
    race is live, so the driver always passes it.
    """
    source = (lifecycle_mod("credentials").stage(credentials, run_root)
              if run_root is not None else credentials)
    if source is None or not source.exists():
        return []
    target = ("/run/secrets/codex-auth.json" if agent_cli == "codex"
              else f"/home/agent/.claude/{credentials.name}")
    return [bind(source, target, read_only=True)]


def prepare_codex_home(container: str, *, uid: int, gid: int,
                       credentials_mounted: bool) -> int:
    """Create a writable, persistent Codex home without host transcripts."""
    install_auth = (
        f"install -m 600 -o {uid} -g {gid} /run/secrets/codex-auth.json "
        "/home/agent/.codex/auth.json; "
    ) if credentials_mounted else ""
    command = (
        "set -eu; "
        f"install -d -m 700 -o {uid} -g {gid} /home/agent/.codex; "
        "rm -rf /home/agent/.codex/sessions; "
        "ln -s /home/agent/codex-sessions /home/agent/.codex/sessions; "
        f"chown -h {uid}:{gid} /home/agent/.codex/sessions; "
        f"install -d -m 700 -o {uid} -g {gid} /home/agent/.codex/skills; "
        f"{install_auth}"
        "if [ -d /workspace/.claude/skills ]; then "
        "for skill in /workspace/.claude/skills/*; do "
        "[ -d \"$skill\" ] || continue; "
        "link=/home/agent/.codex/skills/\"$(basename \"$skill\")\"; "
        "ln -sfn \"$skill\" \"$link\"; "
        f"chown -h {uid}:{gid} \"$link\"; "
        "done; fi; "
        "find /home/agent/.codex/skills -mindepth 1 -maxdepth 1 "
        "-type l | wc -l"
    )
    got = run_agent.docker(
        "exec", "--user", "root", container, "bash", "-lc", command,
        timeout=60, check=False)
    if got.returncode != 0:
        raise RuntimeError(
            "could not prepare the ephemeral Codex home: "
            f"{((got.stdout or '') + (got.stderr or ''))[-300:]}")
    with contextlib.suppress(ValueError, IndexError):
        return int(got.stdout.strip().splitlines()[-1])
    return 0


def preflight_codex(container: str, *, model: str, effort: str) -> dict:
    """Prove Codex authentication and one tool execution before the timebox."""
    marker = "SERVERBENCH_AGENT_PREFLIGHT_OK"
    command = [
        "codex", "exec", "--ephemeral", "--ignore-user-config",
        "--strict-config", "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox", "--json",
    ]
    command += lifecycle_mod("agent").codex_config(model, effort)
    command += [
        "Use the shell to run `printf SERVERBENCH_AGENT_PREFLIGHT_OK`; "
        "then answer with the single word OK."]
    got = run_agent.docker(
        "exec", container, "bash", "-lc", shlex.join(command),
        timeout=240, check=False)
    events = []
    for line in (got.stdout or "").splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            events.append(json.loads(line))
    tool_marker = any(
        event.get("type") == "item.completed"
        and (event.get("item") or {}).get("type") == "command_execution"
        and marker in str((event.get("item") or {}).get("aggregated_output") or "")
        for event in events)
    ok = got.returncode == 0 and tool_marker
    detail = "Codex authenticated and executed a shell command" if ok else (
        ((got.stdout or "") + (got.stderr or ""))[-500:]
        or "Codex returned no preflight events")
    return {"ok": ok, "tool_marker": tool_marker,
            "exit_code": got.returncode, "detail": detail}


def lifecycle_mod(name: str):
    """`lifecycle.<name>`, imported lazily and by path.

    This script runs under the HARBOR interpreter, which has no ServerBench
    package installed, so the harness tree is put on the path here rather than
    assumed. Stdlib-only on the other side, so nothing else comes with it.
    """
    harness = str(Path(__file__).resolve().parents[2] / "harness")
    if harness not in sys.path:
        sys.path.insert(0, harness)
    return importlib.import_module(f"lifecycle.{name}")


def claim_agent_claude_home(container: str, *, uid: int, gid: int) -> str:
    """Give `~/.claude` back to the agent, and say what happened.

    ## Why

    `credential_mount` binds ONE FILE to
    `/home/agent/.claude/.credentials.json`. When a bind's parent directory
    does not exist in the image, **docker creates it as root:root**, and the
    agent runs unprivileged. So the container comes up with

        drwxr-xr-x root root /home/agent/.claude
        -rw------- agent users /home/agent/.claude/.credentials.json

    -- the credentials readable, every signal green, and the directory
    containing them not writable by the only user who will ever run there.

    Nothing needs `~/.claude` to be writable until the context fills. Then the
    CLI tries to compact, compaction writes a transcript under
    `~/.claude/projects/`, and the run dies with

        Prompt is too long . automatic compaction failed:
        EACCES: permission denied, mkdir '/home/agent/.claude/projects'

    which reads as a MODEL limitation and is a filesystem permission. A short
    run on a large context never has to compact, so only long runs hit it.

    Fixed at runtime rather than in the image: the ownership is created by the
    MOUNT, so a `chown` in the Dockerfile is undone by the next `docker run`.
    `chmod 700` because it holds the operator's credentials.
    """
    got = run_agent.docker(
        "exec", "--user", "root", container, "bash", "-lc",
        f"mkdir -p /home/agent/.claude && "
        f"chown {int(uid)}:{int(gid)} /home/agent/.claude && "
        f"chmod 700 /home/agent/.claude && "
        f"stat -c '%U:%G %a' /home/agent/.claude",
        check=False)
    owner = (got.stdout or "").strip().splitlines()[-1:] or [""]
    # Proven by WRITING, not by reading the mode back: the mode can look right
    # while something else -- a read-only layer, a stale mount -- refuses.
    probe = run_agent.docker(
        "exec", container, "bash", "-lc",
        "mkdir -p /home/agent/.claude/projects && "
        "test -w /home/agent/.claude/projects && echo WRITABLE",
        check=False)
    if "WRITABLE" not in (probe.stdout or ""):
        raise RuntimeError(
            "/home/agent/.claude is not writable by the agent, so the CLI "
            "cannot compact and this run will die of context exhaustion the "
            "moment it fills -- reporting as 'Prompt is too long'. "
            f"chown said: {(got.stdout or got.stderr or '').strip()[:200]}")
    return owner[0]


#: Where the container keeps a WRITABLE HuggingFace `datasets` cache.
DATASETS_CACHE_IN_CONTAINER = "/tmp/hf-datasets"


def writable_datasets_cache(container: str) -> int:
    """Give `datasets` somewhere to write its locks, without opening the cache.

    ## Why

    `HF_HOME` points at `/opt/hf-cache`, which is the operator's own cache
    mounted READ-ONLY -- correctly, because it is not the agent's to modify.
    The `datasets` library takes a `filelock` before reading a processed
    dataset, and `filelock` opens its lock `O_RDWR|O_CREAT`. On a read-only
    mount that fails outright:

        DatasetLoaderError: Failed to load HuggingFace dataset
        'semianalysisai/cc-traces-weka-062126-256k': [Errno 30]
        Read-only file system: '/opt/hf-cache/datasets/..._default_....lock'

    A task whose workload IS a HuggingFace dataset would then stop the agent
    measuring itself on the thing it is scored on.

    ## Why symlinks rather than a copy or a writable mount

    A writable mount would let a container corrupt the operator's cache, and
    the processed corpus is 1.8 GB, so copying it per run is neither fast nor
    free. Instead the writable root holds one symlink per dataset directory:
    the DATA is read through the read-only mount and shared by every run, and
    only the lock files are local to the container.

    The env var reaches the agent through `/etc/profile.d/`, because a
    container env set after `docker run` cannot reach a process that is
    already running, and Debian's `/etc/profile` ASSIGNS `PATH` rather than
    appending -- the same reason `agentx_profile_mount` exists.
    """
    script = (
        f"mkdir -p {DATASETS_CACHE_IN_CONTAINER} && "
        f"for d in {HF_CACHE_MOUNT}/datasets/*/; do "
        f"[ -d \"$d\" ] && ln -sfn \"$d\" "
        f"{DATASETS_CACHE_IN_CONTAINER}/$(basename \"$d\"); done; "
        f"touch {DATASETS_CACHE_IN_CONTAINER}/.probe && "
        f"rm -f {DATASETS_CACHE_IN_CONTAINER}/.probe && "
        f"ls {DATASETS_CACHE_IN_CONTAINER} | wc -l")
    got = run_agent.docker("exec", container, "bash", "-lc", script,
                           check=False)
    linked = 0
    with contextlib.suppress(ValueError, IndexError):
        linked = int((got.stdout or "0").strip().splitlines()[-1])
    if linked <= 0:
        # WHAT THE CONTAINER SAID, both streams and the exit code: when the
        # script's `&&` chain short-circuits it writes to neither stream.
        raise RuntimeError(
            f"could not prepare a writable datasets cache in {container}. "
            f"`datasets` will fail with 'Read-only file system' on any task "
            f"whose workload is a HuggingFace dataset.\n"
            f"  exit {getattr(got, 'returncode', '?')}\n"
            f"  stdout: {(got.stdout or '').strip()[:400]!r}\n"
            f"  stderr: {(got.stderr or '').strip()[:400]!r}\n"
            f"  script: {script}")
    # The variable has to reach a LOGIN shell: the agent is already running and
    # a container env cannot be changed under it.
    drop_in = (
        "# HF `datasets` needs a writable cache root for its lock files; the\n"
        "# operator's cache is mounted read-only on purpose. Entries here are\n"
        "# symlinks into that mount, so only the locks are local.\n"
        f"export HF_DATASETS_CACHE={DATASETS_CACHE_IN_CONTAINER}\n")
    # `run_agent.docker()` takes no `input=` -- it is a thin wrapper that
    # runs `docker` and only forwards argv -- and
    # the drop-in is small and fixed, so it goes in as a quoted heredoc rather
    # than over stdin. Quoted: nothing in it may be expanded by the shell.
    heredoc = ("cat > /etc/profile.d/zz-hf-datasets.sh <<'SBEOF'\n"
               + drop_in
               + "SBEOF\nchmod 644 /etc/profile.d/zz-hf-datasets.sh")
    run_agent.docker("exec", "--user", "root", container, "bash", "-c",
                     heredoc, check=False)
    return linked


def mounts_for(*, workspace: Path, model: Path, env_input: Path,
               uv_cache: Path,
               codex_sessions: Path | None = None,
               task: Path | None = None) -> list[dict]:
    """Every bind the container needs, in Harbor's vocabulary.

    The blob mounts exist because a Hugging Face snapshot is a farm of
    relative symlinks into `../../blobs/<sha>`, so mounting the snapshot alone
    mounts the links without their targets and the container has no weights
    and no `tokenizer_config.json`. `run_agent`'s
    `blob_mounts` computes the extra sources; it returns raw `-v` flags, which
    are translated here rather than recomputed.
    """
    out = [
        bind(workspace.resolve(), WORKSPACE),
        # ONE PATH, AND THE VERDICT USES THE SAME ONE: `/models/target` is
        # where `final_validate/deploy.py` mounts it too, and the verdict runs
        # `serve.sh` VERBATIM, so what the agent develops against is what the
        # verdict serves from.
        bind(model.resolve(), MODEL_MOUNT, read_only=True),
        bind(env_input.resolve(), ENV_INPUT_MOUNT, read_only=True),
        bind(uv_cache.resolve(), UV_CACHE_MOUNT),
    ]
    # `blob_mounts` yields ["-v", "src:dst:ro", ...]; take the pairs.
    flags = run_agent.blob_mounts(model.resolve())
    for spec in flags[1::2]:
        source, _, rest = spec.partition(":")
        target = rest.rsplit(":", 1)[0] if rest.endswith(":ro") else rest
        out.append(bind(source, target, read_only=True))
    if codex_sessions is not None:
        out.append(bind(codex_sessions.resolve(), "/home/agent/codex-sessions"))
    # AgentX, read-only. The agent benchmarks its own server with it; nothing
    # here is writable, so the frozen grading configuration cannot be edited
    # from inside: the authoritative runtime and grading assets stay outside
    # candidate-writable scope.
    runtime = agentx_runtime()
    if runtime is not None:
        out.append(bind(runtime.resolve(), AGENTX_MOUNT, read_only=True))
        out += agentx_interpreter_mounts(runtime)
        source = agentx_source()
        if source is not None:
            out.append(bind(source.resolve(), AGENTX_SOURCE_MOUNT,
                            read_only=True))
        # The tokenizer AgentX needs, in the only layout it can resolve
        # offline -- and ONLY the repository this task serves.
        out += hf_cache_mounts_for(model, task=task)
    return out


def hf_cache_for(model: Path) -> Path | None:
    """A Hugging Face cache holding this model, or None.

    WHY THIS IS NOT JUST THE MODEL DIRECTORY. AgentX resolves `--tokenizer`
    through huggingface_hub, and the containers are offline by construction:
    `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are both set, which sends
    `Tokenizer._load_from_hub` down its offline branch, where the argument is
    handed to the hub as a REPO ID. A staged directory then raises
    `HFValidationError: Repo id must use alphanumeric chars ... '/tok'` -- an
    error naming neither offline mode nor the path. So a local path cannot be
    passed, and a repo id needs a real cache to resolve from.

    Two shapes are accepted, in order:

    1. The model directory is already inside a cache (`.../huggingface/hub/
       models--Org--Name/snapshots/<sha>`) -- mount the cache root it is in.
    2. It is not: `--model-dir` is a flat directory of real bytes, staged so
       the seed server can read weights without the symlink farm. Fall back to
       the host's own cache, and ONLY when it actually carries the model --
       an empty cache mounted at `HF_HOME` turns a clear "no such directory"
       into an offline hub lookup that fails further from its cause.

    None means the brief must not tell the agent to pass a repo id, because
    doing so would name a resolution path that is not there.
    """
    from os import environ

    for parent in Path(model).resolve().parents:
        if parent.name == "hub" and parent.parent.name == "huggingface":
            return parent.parent

    home = environ.get("HF_HOME", "").strip()
    root = Path(home) if home else Path.home() / ".cache" / "huggingface"
    hub = root / "hub"
    if not hub.is_dir():
        return None
    # Present means: a snapshot with a tokenizer in it. A `models--*` directory
    # can exist holding nothing but a `.locks` entry or a half-fetched blob.
    wanted = f"models--{repo_id_for(model).replace('/', '--')}"
    for snapshot in (hub / wanted / "snapshots").glob("*"):
        if (snapshot / "tokenizer_config.json").exists():
            return root
    return None


def _task_module(task: Path | None):
    """One task's own module: identity, scenario, corpus, seed, durations.

    The facts this script needs are the module's, which is also what the
    verdict passes to aiperf.
    """
    if task is None:
        return None
    try:
        import sys
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        from final_validate import tasks as task_modules
        return task_modules.python_module(Path(task).name)
    except Exception:                                          # noqa: BLE001
        return None


def declared_datasets(task: Path | None) -> list[str]:
    """Dataset repo ids this task replays: its corpus, and a draft's if any.

    THE MOUNT DEPENDS ON THIS BEING RIGHT. `hf_cache_mounts_for` binds one
    directory per name here, and the writable-cache probe then counts what
    arrived: a task whose corpus is not named gets no dataset mounted, the
    probe counts zero and the run REFUSES to start.
    """
    module = _task_module(task)
    if module is None:
        return []
    names = [getattr(module, "CORPUS", "")]
    return sorted({str(n) for n in names if n})


def _declared_model_id(task: Path | None) -> str:
    """The model id this task freezes, for `MODEL_ID` in the container.

    Read from the task rather than hardcoded, so the environment variable, the
    brief and the mounted weights cannot disagree.
    """
    module = _task_module(task)
    return str(getattr(module, "MODEL_ID", "") or "") if module else ""


def hf_cache_mounts_for(model: Path, *, task: Path | None = None) -> list[dict]:
    """Mount the ONE repository this task serves, not the operator's whole cache.

    THE CACHE IS NOT A SAFE THING TO HAND OVER WHOLE.
    ------------------------------------------------
    `hf_cache_for` returns a cache ROOT, and binding that root at
    `/opt/hf-cache` gives the agent read access to every model the operator has
    ever pulled -- other people's fine-tunes and every other serving task's
    weights. An agent that went looking could serve weights the harness never
    chose, and then the harness would not know which weights its own
    measurement was about.

    So: one bind, for one repository, read-only. `huggingface_hub` resolves a
    repo id by looking for `<root>/hub/models--Org--Name`, and nothing about
    that lookup requires the siblings to exist. Everything else stays outside.

    The model directory's own bind still supplies the weights; this is only the
    layout the offline hub lookup needs.
    """
    root = hf_cache_for(model)
    if root is None:
        return []
    out: list[dict] = []
    slug = f"models--{repo_id_for(model).replace('/', '--')}"
    repository = root / "hub" / slug
    if repository.is_dir():
        out.append(bind(repository.resolve(),
                        f"{HF_CACHE_MOUNT}/hub/{slug}", read_only=True))
    # THE CORPUS, AND ONLY THE CORPUS THIS TASK FREEZES. The operator's cache
    # can hold other datasets, including a truncated variant of the task's own
    # corpus that an agent could replay cheaply and report against the real
    # one's gates. Mounted by EXACT slug, so a `-256k` sibling does not come
    # along.
    for dataset in declared_datasets(task):
        dslug = f"datasets--{dataset.replace('/', '--')}"
        hub_copy = root / "hub" / dslug
        if hub_copy.is_dir():
            out.append(bind(hub_copy.resolve(),
                            f"{HF_CACHE_MOUNT}/hub/{dslug}", read_only=True))
        # `datasets.load_dataset` reads the prepared copy, which lives beside
        # `hub/` under a different naming convention.
        prepared = root / "datasets" / dataset.replace("/", "___")
        if prepared.is_dir():
            out.append(bind(
                prepared.resolve(),
                f"{HF_CACHE_MOUNT}/datasets/{dataset.replace('/', '___')}",
                read_only=True))
    return out


def agentx_profile_mount(staging: Path) -> list[dict]:
    """Put `/opt/agentx/bin` back on PATH for LOGIN shells.

    A container env `PATH` reaches a non-login shell and is then thrown away by
    a login one. Debian's `/etc/profile` does not append -- it ASSIGNS:

        PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games"

    So `docker exec ... bash -c 'command -v aiperf'` finds it and
    `bash -lc 'command -v aiperf'` does not, on the same container. The brief
    says "it is on your PATH", and that has to hold for every way a shell can
    be started.

    A `/etc/profile.d/` drop-in is sourced by `/etc/profile` AFTER the
    assignment, so it survives it. Both mechanisms are kept: the env var for
    non-login shells, this for login ones. Neither is redundant.
    """
    if agentx_runtime() is None:
        return []
    drop_in = staging / "10-agentx.sh"
    drop_in.write_text(
        "# Written by serverbench/scripts/agent/harbor_env.py.\n"
        "# /etc/profile ASSIGNS PATH rather than appending, so a login shell\n"
        "# drops the container's PATH. This runs after that assignment.\n"
        f'case ":$PATH:" in\n'
        f'  *:{AGENTX_MOUNT}/bin:*) ;;\n'
        f'  *) PATH="{AGENTX_MOUNT}/bin:$PATH" ;;\n'
        "esac\n"
        "export PATH\n", "utf-8")
    drop_in.chmod(0o644)
    return [bind(drop_in, "/etc/profile.d/10-agentx.sh", read_only=True)]


def container_name_for(session_id: str) -> str:
    """Ask docker which container Harbor's compose project produced.

    Derived by QUERY rather than by reproducing compose's naming rule. The
    project is the sanitised session id and the service is `main`, and both are
    Harbor's to change; the labels are docker's and are stable.
    """
    from harbor.environments.docker.docker import (  # noqa: PLC0415
        _sanitize_docker_compose_project_name,
    )
    project = _sanitize_docker_compose_project_name(session_id)
    got = subprocess.run(
        ["docker", "ps", "--no-trunc",
         "--filter", f"label=com.docker.compose.project={project}",
         "--filter", f"label=com.docker.compose.service={MAIN_SERVICE}",
         "--format", "{{.Names}}"],
        capture_output=True, text=True, check=False, timeout=120)
    names = [n for n in got.stdout.split() if n]
    if not names:
        raise RuntimeError(
            f"Harbor started compose project {project!r} and no running "
            f"container carries its labels. Nothing can be exec'd into")
    return names[0]


def build_environment(*, environment_dir: Path, trial_dir: Path,
                      session_id: str, image: str, devices: list[int],
                      mounts: list[dict], env: dict[str, str],
                      allowed_hosts: list[str], overlay: Path,
                      keep: bool):
    """The Harbor `DockerEnvironment` this run will live in.

    `override_gpus=0` is not "no GPU". It is the documented way past
    `_validate_gpu_support`, which raises for the local docker backend whenever
    effective gpus > 0; the card is granted by the overlay instead. Passing the
    count here and no overlay yields a refusal, and an overlay with no
    `override_gpus` yields the same refusal -- both keys, every time.
    """
    from harbor.environments.docker.docker import DockerEnvironment  # noqa: PLC0415
    from harbor.models.task.config import (  # noqa: PLC0415
        EnvironmentConfig,
        NetworkMode,
        NetworkPolicy,
    )
    from harbor.models.trial.paths import TrialPaths  # noqa: PLC0415

    # PUBLIC, then narrowed from INSIDE by `restrict-network.sh`.
    #
    # Not a preference. Harbor has exactly the mode this task wants --
    # `NetworkMode.ALLOWLIST` -- and the local docker backend does not implement
    # it: `EnvironmentCapabilities.network_allowlist` defaults False
    # (`environments/capabilities.py:22`) and `_validate_network_policy_support`
    # (`environments/base.py:695`) raises "network_mode='allowlist' is not
    # supported by docker". `EnvironmentConfig` also refuses `allowed_hosts`
    # unless the mode is `allowlist`, so the list cannot even be carried as a
    # statement of intent.
    #
    # `no-network` IS supported and is not usable either: it sets
    # `network_mode: none`, and the container then has no route to the
    # egress broker, so the agent CLI could not reach its model at all.
    # (Every measurement during an evolve runs inside the container; only the
    # final verdict, after the deadline, runs on the host.)
    #
    # So the allowlist stays the image's iptables script, which is applied in
    # the container's own network namespace, inherited by every later exec, and
    # PROVED from inside by six egress probes rather than asserted. The
    # allowlist Harbor would not carry is recorded in
    # `harbor_environment.json` instead, so what was asked for is on the
    # record.
    config = EnvironmentConfig(
        docker_image=image,
        network_mode=NetworkMode.PUBLIC,
        env=dict(env),
        workdir=WORKSPACE,
    )
    return DockerEnvironment(
        environment_dir=environment_dir,
        environment_name=f"serverbench-{session_id[:12]}",
        session_id=session_id,
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=config,
        keep_containers=keep,
        persistent_env=dict(env),
        mounts=list(mounts),
        # Matches the config above for the same reason. `NetworkPolicy` accepts
        # `allowed_hosts` at any mode, but `_validate_network_policy_support`
        # only checks the MODE -- so passing hosts here with mode PUBLIC would
        # store a list nothing enforces, which is configuration with the shape
        # of authority and no effect. The hosts go in the evidence instead.
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC,
                                     allowed_hosts=[]),
        # See the docstring: this is the guard workaround, not a device count.
        override_gpus=0,
        extra_docker_compose=[overlay],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--env-input", type=Path, required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--gpu", required=True,
                    help="device index, or a comma-separated list")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", type=Path, default=None)
    ap.add_argument("--uv-cache", default="/tmp/serverbench-uv-cache")
    ap.add_argument("--shm-size", default="8g")
    ap.add_argument("--session-id", default="")
    ap.add_argument("--keep", action="store_true", default=True)
    ap.add_argument("--agent-cli", choices=("codex", "claude"),
                    default="claude")
    ap.add_argument("--agent-model", default="")
    ap.add_argument("--agent-effort", default="")
    ap.add_argument("--allow", action="append", default=[],
                    help="additional exact model/auth hostname; repeatable")
    ap.add_argument("--expose", action="append", default=[])
    ap.add_argument("--credentials", default=None)
    ap.add_argument("--agent-env", default="",
                    help="a per-run KEY=VALUE file merged into the container's "
                         "environment, after the host default. Used to point "
                         "one run at a different model backend without "
                         "repointing every other run.")
    ap.add_argument("--print-name-only", action="store_true",
                    help="write the container name to stdout and nothing else")
    args = ap.parse_args(argv)

    if args.credentials is None:
        args.credentials = CREDENTIAL_DEFAULTS[args.agent_cli]
    if args.agent_cli == "codex" and not args.agent_effort:
        print("Codex requires a pinned --agent-effort", file=sys.stderr)
        return 2
    broker_hosts = _broker_allowlist(args.agent_cli, args.allow)

    args.out.mkdir(parents=True, exist_ok=True)
    # THE IMAGE MUST BE BUILT FROM THIS CHECKOUT'S DOCKERFILES. Checked
    # before anything is composed or any card is touched. See
    # `image_freshness.py`.
    import image_freshness
    stale = image_freshness.problem(args.image)
    if stale:
        print(f"REFUSED: {stale}", file=sys.stderr)
        return 2

    devices = [int(g) for g in str(args.gpu).split(",") if g.strip()]

    ok, why = harbor_available()
    if not ok:
        print(f"harbor unavailable: {why}", file=sys.stderr)
        return 3
    print(f"harbor      {why}")

    # THE BOUNDARY CHECKS, imported rather than reimplemented. Every one of
    # these is a property a measurement rests on, and none of them is Harbor's
    # job: a model whose symlinks dangle, or an exposure carrying a held-out
    # digest, is refused here. Contention is refused by the driver before this
    # process starts
    # (`lifecycle.run.Lifecycle.claim_devices`); this records identities only.
    uuids = {index: run_agent.device_uuid(index) for index in devices}
    run_agent.check_model_readable(args.model.resolve())
    _, exposure_record = run_agent.resolve_exposures(args.expose, args.task)
    for line in exposure_record:
        print(f"exposed     {line}")

    uv_cache = Path(args.uv_cache).expanduser().resolve()
    uv_cache.mkdir(parents=True, exist_ok=True)

    session_id = args.session_id or f"sb{uuid.uuid4().hex[:16]}"
    env = {
        # MODEL_PATH points at the canonical mount, so the value the agent
        # develops against is the value the verdict passes.
        "MODEL_PATH": CANONICAL_MODEL_MOUNT,
        # MODEL_ID, read from the task, so the variable, the mounted weights
        # and the brief cannot disagree. No fallback: a task that declares no
        # model has nothing to put here, and a guess would be wrong.
        "MODEL_ID": _declared_model_id(getattr(args, "task", None)),
        "PORT": "8000",
        "UV_CACHE_DIR": UV_CACHE_MOUNT,
        # APP_DIR IS WHERE `serve.sh` LIVES, AND HERE THAT IS THE WORKSPACE.
        #
        # Every task's `start_server.sh` and `stop_server.sh` default it to
        # `/app`, which is right in the VERDICT's container -- the published
        # tree is mounted there -- and wrong in this one. Unset, the brief's
        # own first command, `bash start_server.sh && bash wait/wait_ready.sh`,
        # would fail with `bash: /app/serve.sh: No such file or directory`.
        "APP_DIR": WORKSPACE,
    }
    # AgentX on PATH, and only when it is actually mounted: the brief must
    # never name an executable that is not there.
    if agentx_runtime() is not None:
        env["AGENTX_ROOT"] = AGENTX_MOUNT
        if agentx_source() is not None:
            env["AGENTX_SOURCE"] = AGENTX_SOURCE_MOUNT
        # Only when there is something behind it. `HF_HOME` pointing at a
        # directory that was never mounted is worse than leaving it unset: the
        # hub reports a cache miss in offline mode instead of the filesystem
        # reporting a missing path, and the agent debugs the wrong layer.
        if hf_cache_for(Path(args.model)) is not None:
            env["HF_HOME"] = HF_CACHE_MOUNT
        # NO AgentX DATASET CACHE. AgentX keeps a content-addressed copy of every
        # tokenized dataset under ~/.cache/aiperf/dataset_mmap with no eviction
        # ("delete the directory to reclaim disk"). Every replay here draws a
        # fresh workload seed, so an entry is never reused and the cache only
        # fills the docker disk (several GB per replay). Disabled, AgentX
        # writes the same byte-identical mmap files per run and removes them
        # at shutdown.
        env["AIPERF_DATASET_MMAP_CACHE_ENABLED"] = "false"
        # CUDA STAYS ON PATH. This line is an OVERRIDE, not an addition: it
        # replaces the image's `ENV PATH`, which puts `/usr/local/cuda/bin`
        # first. Without it `command -v nvcc` fails in every non-login shell --
        # the agent's default, and the exact probe the brief tells it to run to
        # confirm the toolchain -- and an agent that trusts the probe records
        # "no nvcc" and retreats to Triton, although torch still finds nvcc
        # through `CUDA_HOME`.
        env["PATH"] = (f"{AGENTX_MOUNT}/bin:{CUDA_BIN}:/usr/local/bin:"
                       f"/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin")
        # AND THE THINGS THAT ONLY EXISTED IN A LOGIN SHELL.
        #
        # `/etc/profile.d` is read by `bash -l` and by nothing else. A
        # subprocess the agent's server spawns is not a login shell, so three
        # settings the brief states unconditionally are set here for every
        # process:
        #
        #   HF_DATASETS_CACHE  the brief says it "already points somewhere
        #                      writable, so the loader can take its lock"
        #   LD_LIBRARY_PATH    a `cpp_extension` build links libcudart from
        #                      the CUDA tree; without this the import fails
        #   CUDA_HOME          what torch uses to find nvcc at all
        #
        # `Config.Env` reaches every process in the container, which is what
        # these three need to be true of.
        env.setdefault("CUDA_HOME", CUDA_HOME)
        env.setdefault("LD_LIBRARY_PATH", f"{CUDA_HOME}/lib64")
        env.setdefault("HF_DATASETS_CACHE", HF_DATASETS_CACHE)

    # TWO SECRET FILES, and the second is why they are not one.
    #
    #   agent.env           the host default, used when nothing else is named
    #   --agent-env PATH    a per-RUN override
    #
    # The per-run channel exists because two evolves can run at once against
    # DIFFERENT model backends -- one on Anthropic, one on an OpenRouter-hosted
    # GLM -- and a single global file would silently repoint every future run
    # the moment it was written.
    #
    # Both are mode-600 gitignored files read by PATH and never passed as
    # `-e KEY=value`, which on a shared host puts the secret in every user's
    # `ps`. Neither is read into evidence.
    sources = [REPO / "serverbench/artifacts/secrets/agent.env"]
    if getattr(args, "agent_env", None):
        named = Path(args.agent_env).expanduser()
        # A FILE THE OPERATOR NAMED AND THAT IS NOT THERE IS A REFUSAL.
        #
        # The host default may legitimately be absent; a per-run override
        # cannot be, because naming it is a statement that THIS run needs
        # different credentials. Skipped silently, a typo would produce a
        # container with `apiKeySource: none`, an agent whose first call fails
        # with `404 model_not_found`, and a run report blaming the agent for a
        # defect in the command line.
        if not named.is_file():
            raise SystemExit(
                f"--agent-env names {named}, which does not exist. Refusing "
                f"rather than falling back to the host default: naming a "
                f"per-run credentials file is a statement that this run needs "
                f"different credentials, and running without them produces an "
                f"agent that cannot reach its model at all. Available:\n  "
                + "\n  ".join(
                    str(path) for path in sorted(
                        (REPO / "serverbench/artifacts/secrets").glob("*.env"))
                ))
        sources.append(named)
    for token_file in sources:
        if not token_file.is_file():
            continue
        named = []
        for line in token_file.read_text("utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() and not name.strip().startswith("#"):
                env[name.strip()] = value.strip()
                named.append(name.strip())
        try:
            shown = token_file.relative_to(REPO)
        except ValueError:
            shown = token_file
        # The NAMES are evidence -- which run used which backend is a fact
        # about the experiment. The VALUES never are.
        print(f"token supplied via {shown} -> {sorted(named)} "
              f"(values not read into evidence)")
    # ONE MODEL FOR EVERY ROUTE. The Claude CLI sends subagents and background
    # calls to its own default aliases (e.g. `claude-opus-5`), and behind a
    # third-party endpoint those names do not exist. Every alias is pinned to
    # the run's model.
    pinned = pin_model_routes(env, str(getattr(args, "agent_model", "") or ""),
                              args.agent_cli)
    if pinned:
        print(f"every Claude CLI model route pinned to {pinned}")
        if env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS"):
            print(f"context window for {pinned}: {env['CLAUDE_CODE_MAX_CONTEXT_TOKENS']} tokens "
                  f"(CLAUDE_CODE_MAX_CONTEXT_TOKENS; the CLI's default for an unknown model is 200k)")
    broker_hosts = narrow_to_provider(broker_hosts, env, args.agent_cli, args.allow)
    print(f"egress broker relays only to {sorted(broker_hosts)}")
    # ONE CREDENTIAL ROUTE. An API-key run gets only its env file's key and
    # never the operator's OAuth file; an OAuth run gets only the OAuth file
    # and no API key. Decided after the env files are merged.
    route = credential_route(env, args.agent_cli)
    credentials = Path(args.credentials).expanduser()
    creds = (credential_mount(credentials, agent_cli=args.agent_cli,
                              run_root=Path(args.out))
             if route in ("oauth", "codex") else [])
    if route == "api_key":
        print("credentials: the run's API key only; no OAuth file is mounted")
    elif creds:
        print(f"credentials mounted read-only from {credentials.name}, via "
              f"this run's own copy so a host token refresh can reach it")
    else:
        print(f"WARNING: no credentials at {args.credentials}; the agent will "
              f"start and immediately report that it is not logged in")

    staging = Path(tempfile.mkdtemp(prefix="serverbench-harbor-"))
    container = ""
    broker: subprocess.Popen | None = None
    try:
        environment_dir = staging / "environment"
        environment_dir.mkdir()
        # NOT in `staging`, which is deleted when this function returns. A bind
        # mount whose source is gone keeps working only until something
        # re-resolves the container's mounts; then docker recreates the missing
        # source as an EMPTY DIRECTORY -- a restarted container would then
        # mount a directory over /etc/profile.d/10-agentx.sh and lose AgentX
        # from a login shell's PATH. The run directory outlives the container,
        # which is what a bind source has to do.
        persistent = Path(args.out) / "container-files"
        persistent.mkdir(parents=True, exist_ok=True)
        profile_mounts = agentx_profile_mount(persistent)
        overlay = staging / "gpu-and-caps.yaml"
        overlay.write_text(overlay_text(devices=devices, shm_size=args.shm_size,
                                        uid=os.getuid(), gid=os.getgid(),
                                        env=env), "utf-8")
        codex_sessions = None
        if args.agent_cli == "codex":
            codex_sessions = args.out / "codex-sessions"
            codex_sessions.mkdir(parents=True, exist_ok=True)
        environment = build_environment(
            environment_dir=environment_dir, trial_dir=staging / "trial",
            session_id=session_id, image=args.image, devices=devices,
            mounts=[*mounts_for(workspace=args.workspace, model=args.model,
                                env_input=args.env_input, uv_cache=uv_cache,
                                codex_sessions=codex_sessions,
                                task=args.task),
                    *creds, *profile_mounts],
            env=env, allowed_hosts=sorted(broker_hosts), overlay=overlay,
            keep=True)
        # Every `exec` Harbor performs runs as this uid. Left unset it uses the
        # image's default, which is root here -- and the workspace is a bind
        # mount owned by the invoking uid, so anything root wrote into it would
        # be a file the agent cannot then modify. The container's own `user:`
        # is set in the overlay; this is the exec side of the same decision.
        environment.default_user = f"{os.getuid()}:{os.getgid()}"
        start_environment(environment)
        container = container_name_for(session_id)
        print(f"container   {container} (harbor DockerEnvironment)")
        print("device      " + ", ".join(f"GPU {d} = {uuids[d]}"
                                         for d in devices))

        codex_skills_linked = 0
        if args.agent_cli == "codex":
            codex_skills_linked = prepare_codex_home(
                container, uid=os.getuid(), gid=os.getgid(),
                credentials_mounted=bool(creds))
            print(f"codex       persistent session store; "
                  f"{codex_skills_linked} workspace skill(s) linked")

        # EVERY DECLARED VARIABLE IS ACTUALLY IN THE CONTAINER.
        #
        # Checked, not assumed, because declaring them to harbor is not enough:
        # `persistent_env` reaches only harbor's own `exec` and
        # `task_env_config.env` only compose substitution. A missing
        # `MODEL_PATH` would leave every other signal green and the candidate
        # server reading the seed's fallback `/models/model` as a hub repo id.
        #
        # Read back from the container rather than from the compose file: what
        # matters is what a `docker exec` will actually inherit.
        seen = run_agent.docker("exec", container, "env").stdout
        present = dict(
            line.split("=", 1) for line in seen.splitlines() if "=" in line)
        missing = {k: v for k, v in env.items() if present.get(k) != v}
        if missing:
            raise RuntimeError(
                f"the container is missing {len(missing)} declared "
                f"environment variable(s): {sorted(missing)}. Harbor's "
                f"persistent_env reaches only its own exec; the compose overlay "
                f"is what puts them in the container, and every later "
                f"`docker exec` inherits from there")
        print(f"environment {len(env)} declared variable(s) present in the "
              f"container")

        # `~/.claude` WRITABLE BY THE AGENT, checked by writing.
        #
        # Docker makes it root-owned when it mounts the credentials file into
        # it, which silently disables the CLI's automatic compaction. See
        # `claim_agent_claude_home`.
        if creds and args.agent_cli == "claude":
            owner = claim_agent_claude_home(
                container, uid=os.getuid(), gid=os.getgid())
            print(f"agent home  /home/agent/.claude {owner} -- compaction can "
                  f"write its transcripts")

        # A WRITABLE `datasets` CACHE, without opening the operator's.
        # See `writable_datasets_cache`: without it, any task whose workload is
        # a HuggingFace dataset fails with "Read-only file system" on the lock.
        if hf_cache_for(Path(args.model)) is not None:
            linked = writable_datasets_cache(container)
            print(f"datasets    {linked} dataset(s) linked into "
                  f"{DATASETS_CACHE_IN_CONTAINER}, writable for locks")

        # THE AGENT'S ONLY WAY OUT, and it is not the internet.
        #
        # The broker runs on the HOST, outside the container's network
        # namespace, and relays CONNECT to the model endpoint and nothing else.
        # The container's allowlist then names one address -- the docker bridge
        # gateway -- so there is no route to PyPI, GitHub, Hugging Face, the
        # PyTorch index, or `api.anthropic.com` itself. It outlives this script
        # and is owned by the driver, which recorded its pid.
        gateway = run_agent.docker(
            "inspect", "-f",
            "{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}",
            container).stdout.strip().split()
        gateway_ip = gateway[0] if gateway else ""
        if not gateway_ip:
            raise RuntimeError(
                "the container has no gateway address, so the broker cannot "
                "be reached and the agent would have no model at all")
        broker_log = args.out / "egress-broker.jsonl"
        broker_argv = [
            sys.executable,
            str(Path(__file__).resolve().parent / "egress_broker.py"),
            "--bind", gateway_ip, "--port", "0", "--print-port",
            "--log", str(broker_log),
        ]
        for host in sorted(broker_hosts):
            broker_argv += ["--allow-host", host]
        broker = subprocess.Popen(
            broker_argv,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            start_new_session=True)
        broker_port = int((broker.stdout.readline() or "0").strip() or 0)
        if not broker_port:
            broker.kill()
            raise RuntimeError("the egress broker did not come up")
        print(f"broker      {gateway_ip}:{broker_port} pid {broker.pid} "
              f"(the only route out; {broker_log.name} records every attempt)")
        # `bash -lc` is how every later exec runs, so /etc/profile.d is the one
        # place the proxy reaches the agent AND anything it starts. The
        # container is created before the gateway and port are knowable, so
        # this cannot be a `-e` at run time.
        proxy = f"http://{gateway_ip}:{broker_port}"
        run_agent.docker(
            "exec", "--user", "root", container, "bash", "-lc",
            "cat > /etc/profile.d/00-egress-broker.sh <<'EOF'\n"
            f"export HTTPS_PROXY={proxy}\nexport https_proxy={proxy}\n"
            f"export HTTP_PROXY={proxy}\nexport http_proxy={proxy}\n"
            "export NO_PROXY=localhost,127.0.0.1\n"
            "export no_proxy=localhost,127.0.0.1\nEOF\n"
            "chmod 0644 /etc/profile.d/00-egress-broker.sh", check=False)

        # THE BOUNDARY, applied to the container Harbor composed. Harbor's own
        # docker backend cannot express an allowlist -- see `build_environment`
        # -- so this is `run_agent`'s implementation, called rather than
        # copied, on a container addressed by name. The rules are installed in
        # the container's network namespace and inherited by every later exec.
        # The allowlist is the GATEWAY, not the API: the container may speak to
        # the host's broker and to nothing else.
        egress = run_agent.harden_and_probe(container, args.out, [gateway_ip])
        egress_ok = all(r["ok"] for r in egress.values())
        if not egress_ok:
            raise RuntimeError("the container egress policy does not match the request")
        agent_preflight = None
        if args.agent_cli == "codex":
            agent_preflight = preflight_codex(
                container, model=args.agent_model, effort=args.agent_effort)
            if not agent_preflight["ok"]:
                raise RuntimeError(
                    "Codex preflight failed before the run budget was spent: "
                    f"{agent_preflight['detail']}")
            print(f"agent       {agent_preflight['detail']}")
        ip = run_agent.docker(
            "inspect", "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            container).stdout.strip()
        # `container.json` in the shape the driver and the audits read, so
        # nothing downstream has to know which backend started the run.
        (args.out / "container.json").write_text(json.dumps({
            "container": container, "ip": ip, "gpu_index": devices[0],
            "gpu_indices": devices, "gpu_uuid": uuids[devices[0]],
            "gpu_uuids": {str(d): uuids[d] for d in devices},
            "image": args.image, "backend": "harbor",
            "agent_cli": args.agent_cli,
            "agent_model": args.agent_model,
            "reasoning_effort": args.agent_effort,
            "agent_preflight": agent_preflight,
            "codex_workspace_skills_linked": codex_skills_linked,
            "egress": egress, "egress_ok": egress_ok,
            "allowlist": [gateway_ip],
            "egress_broker": {
                "endpoint": proxy, "pid": broker.pid,
                "log": str(broker_log),
                "allowed_hosts": sorted(broker_hosts),
                "note": ("the container's only route out. It relays CONNECT to "
                         "these hosts and refuses every other, and records each "
                         "attempt -- so what the agent's model traffic was is a "
                         "fact about the run rather than an assumption")},
            "env_input": str(args.env_input.resolve()),
            "credentials_mounted": bool(creds),
            "exposed_paths": exposure_record,
            "isolation_limitation": (
                "The container reaches no internet host, the model endpoint "
                "included: every egress probe expects BLOCKED. Its one "
                "allowed destination is the docker bridge gateway, where a "
                "harness-owned CONNECT broker relays to the model API and "
                "refuses everything else. What remains outside the harness's "
                "control is that the broker is reachable FROM the container, "
                "so an agent can talk to it -- what it cannot do is reach a "
                "package index or a model hub through it. Harbor's docker "
                "backend implements no allowlist of its own "
                "(capabilities.network_allowlist is False) and its no-network "
                "mode leaves the container with no address, which the host "
                "needs in order to measure the candidate"),
        }, indent=1, sort_keys=True) + "\n", "utf-8")
        if not egress_ok:
            print("egress policy is not what was asked for; refusing this "
                  "environment", file=sys.stderr)
            return 1
        (args.out / "harbor_environment.json").write_text(
            json.dumps({
                "kind": "harbor_environment", "schema_version": 1,
                "driver": "harbor.environments.docker.DockerEnvironment",
                "harbor": why, "session_id": session_id,
                "container": container, "image": args.image,
                "agent_cli": args.agent_cli,
                "agent_model": args.agent_model,
                "reasoning_effort": args.agent_effort,
                "agent_preflight": agent_preflight,
                "codex_workspace_skills_linked": codex_skills_linked,
                "gpu_indices": devices,
                "gpu_uuids": {str(d): uuids[d] for d in devices},
                "overlay": overlay.name,
                "overlay_text": redacted(overlay.read_text("utf-8")),
                # The driver reads THIS file (not `container.json`) to claim
                # the broker's pid so teardown can stop it.
                "egress_broker": {"endpoint": proxy, "pid": broker.pid,
                                  "log": str(broker_log)},
                "exposed_paths": exposure_record,
                "note": ("Harbor owns the environment lifecycle -- compose, "
                         "start, exec, stop. It does not own the boundary: the "
                         "exposure audit, the held-out hashing, the egress "
                         "probes and the credential mount stay in run_agent.py "
                         "and are imported by this script, not copied"),
            }, indent=1, sort_keys=True) + "\n", "utf-8")
        if args.print_name_only:
            print(container)
        return 0
    except Exception as exc:                                    # noqa: BLE001
        if broker is not None and broker.poll() is None:
            broker.terminate()
        if container:
            run_agent.docker("rm", "-f", container, check=False)
        print(f"harbor could not stand the environment up: {exc!r}",
              file=sys.stderr)
        return 1
    finally:
        # The STAGING directory goes; the container stays. It is generated
        # input, and its content is copied into `harbor_environment.json`
        # above so the run can say what it was composed from.
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
