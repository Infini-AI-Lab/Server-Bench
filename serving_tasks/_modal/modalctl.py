#!/usr/bin/env python3
"""Stand up Modal for any task that declares GPUs, from the task's own contract.

Nothing here is task-specific. Everything it needs it reads from the task:

    the task module's MODEL_ID / MODEL_REVISION   which checkpoint, at which revision
    contract.yaml `devices`                       how many cards the task bills
    the task id's suffix                          which card  (`..._2xL40S` -> L40S:2)

so a new task works the moment it exists, and a task whose hardware suffix and
device count disagree is refused rather than run on the wrong shape.

    modalctl check    --task qwen3_5_9b_standard_modal_2xL40S
    modalctl model    --task qwen3_5_9b_standard_modal_2xL40S
    modalctl sandbox  --task qwen3_5_9b_standard_modal_2xL40S --hours 2
    modalctl teardown [--task ...]

## Why the checkpoint is fetched rather than uploaded

The weights are pulled from HuggingFace INSIDE Modal at the revision the task
pins, which is far faster than this host's uplink. A checkpoint is
content-addressed and the pinned revision is the same bytes wherever it comes
from; only the serving operating point and performance evidence must be
re-measured on new hardware.

Every fetch is verified from a DIFFERENT container than the one that wrote it,
by resolving the checkpoint offline the way the server will. A size measured
by the process that just wrote it proves nothing: without an explicit commit
the writer sees files that never persist to the volume.

## Why the standard cache layout

`cache_dir=` writes `blobs/` beside `snapshots/`, which is what a serving stack
expects. A snapshot directory alone is a farm of relative symlinks into blobs
that are not there, and it presents as "no weights, and not even a chat
template".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "serving_tasks"

#: One volume for every task's weights, keyed by the HF cache layout inside it.
MODEL_VOLUME = "serverbench-models"
MODEL_ROOT = "/models"
HF_CACHE = f"{MODEL_ROOT}/hf"
#: The PROCESSED datasets cache. `load_dataset` offline reads this,
#: not the hub snapshot -- which is why the docker path mounts the
#: host's `~/.cache/huggingface/datasets` and not just `hub/`.
DATASETS_CACHE = f"{MODEL_ROOT}/datasets"

#: The app every sandbox of this suite belongs to, so `modal app list` and
#: the dashboard show one place to look and teardown has one thing to sweep.
APP_NAME = "serverbench-evolve-new"

#: Cards Modal offers that this suite might ask for, from Modal's GPU guide.
#: The task id names one; a name that is not here (an L40, which Modal does not
#: offer) is refused rather than passed through to Modal, whose own error
#: arrives after the request is accepted and costs a minute.
#:
#: Modal also lists `H100!` and `B200+`. Neither is reachable here by
#: construction: a task id is parsed with `([A-Za-z0-9-]+)$`, so `!` and `+`
#: cannot appear in one. They are left out rather than added as dead entries.
KNOWN_GPUS = ("L40S", "A100", "A100-40GB", "A100-80GB", "H100", "H200",
              "B200", "B300", "RTX-PRO-6000", "L4", "A10", "T4")


def _local_python() -> str:
    """This interpreter's `major.minor`, for images that run pickled helpers."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


class SetupError(RuntimeError):
    """Something that must be fixed before a run, stated rather than guessed."""


# --- what the task says ----------------------------------------------------

def _contract(task: str) -> dict:
    """The task's frozen bars: gates, prices, devices, window.

    `contract.yaml` and `reference.yaml` are the whole of a task's data; the
    model pin lives in the task's own module (`model_spec`).
    """
    import yaml
    path = TASKS / task / "contract.yaml"
    if not path.is_file():
        raise SetupError(f"no such task: {path}")
    return yaml.safe_load(path.read_text("utf-8")) or {}


def _task_module(task: str):
    """The task's own harness module: identity, scenario, corpus, seed."""
    import importlib

    root = str(REPO / "serverbench" / "harness")
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return importlib.import_module(f"final_validate.tasks.{task.lower()}")
    except Exception as exc:
        raise SetupError(f"could not read {task}'s module: {exc!r}") from exc


def gpu_spec(task: str) -> tuple[str, int]:
    """(card, count) from the task id's suffix, cross-checked against the contract.

    The id is the authority for WHICH card because it is what everything else
    in the suite is named after; the contract's `devices` is the authority for HOW
    MANY because it is what the objective bills. They must agree, and a task
    that says `2xL40S` while billing one device is a task whose denominator
    does not describe its own hardware.
    """
    match = re.search(r"_(\d+)x([A-Za-z0-9-]+)$", task)
    if not match:
        raise SetupError(
            f"{task} does not end in its hardware (expected something like "
            f"`_2xL40S`), so there is nothing to derive a GPU request from")
    count, card = int(match.group(1)), match.group(2)
    if card.upper() not in {g.upper() for g in KNOWN_GPUS}:
        raise SetupError(
            f"{task} asks for {card!r}, which Modal does not offer under that "
            f"name. Known: {', '.join(KNOWN_GPUS)}")
    declared = _contract(task).get("devices")
    if declared is not None and int(declared) != count:
        raise SetupError(
            f"{task} bills {declared} device(s) but its id says {count}. One "
            f"of the two is wrong, and running on either would measure a shape "
            f"the task does not describe")
    return card, count


def model_spec(task: str) -> tuple[str, str]:
    """(model id, revision) from the task's own module, where the pin lives."""
    module = _task_module(task)
    model_id = getattr(module, "MODEL_ID", "")
    revision = getattr(module, "MODEL_REVISION", "")
    if not model_id or not revision:
        raise SetupError(
            f"{task}'s module does not pin both MODEL_ID and MODEL_REVISION; "
            f"a run against an unpinned checkpoint is not reproducible from "
            f"its own record")
    return str(model_id), str(revision)


def snapshot_path(task: str) -> str:
    """Where the checkpoint lands inside a sandbox, for MODEL_PATH."""
    model_id, revision = model_spec(task)
    folder = "models--" + model_id.replace("/", "--")
    return f"{HF_CACHE}/{folder}/snapshots/{revision}"


# --- what the brief promises the agent ---------------------------------------

#: The offline wheelhouse the docker path BIND MOUNTS at /opt/env-input. The
#: brief names it repeatedly, so a sandbox without it is a sandbox whose brief
#: is wrong, and the agent spends its first minutes hunting for wheels that
#: are not there.
#:
#: A volume rather than an image layer: 3.8 GB uploaded once and reused by
#: every run, instead of baked into an image that is rebuilt whenever a
#: Dockerfile changes.
ENV_INPUT_VOLUME = "serverbench-env-input"
ENV_INPUT_MOUNT = "/opt/env-input"


def env_input_source() -> Path:
    """Where the wheelhouse is on this host, as `evolve-common.sh` names it."""
    override = os.environ.get("SERVERBENCH_ENV_INPUT", "").strip()
    root = (Path(override) if override else
            REPO / "serverbench" / "artifacts" / "env-input")
    if not (root / "wheelhouse").is_dir():
        raise SetupError(
            f"no wheelhouse at {root}: the brief tells the agent it can "
            f"install from /opt/env-input offline, and a sandbox without it "
            f"makes that false. Set SERVERBENCH_ENV_INPUT")
    return root


def env_input_state() -> dict:
    """Is the wheelhouse already in the volume?"""
    import modal
    app = modal.App("serverbench-envinput-check")
    volume = modal.Volume.from_name(ENV_INPUT_VOLUME, create_if_missing=True)

    @app.function(image=modal.Image.debian_slim(python_version=_local_python()),
                  volumes={ENV_INPUT_MOUNT: volume}, timeout=600, cpu=2.0,
                  name="state", serialized=True)
    def state() -> dict:
        import os as _os
        volume.reload()
        wheels = _os.path.join(ENV_INPUT_MOUNT, "wheelhouse")
        if not _os.path.isdir(wheels):
            return {"ready": False, "why": "no wheelhouse/ in the volume"}
        names = [n for n in _os.listdir(wheels) if n.endswith(".whl")]
        total = sum(_os.path.getsize(_os.path.join(wheels, n)) for n in names)
        return {"ready": bool(names), "wheels": len(names),
                "gigabytes": round(total / 1e9, 2),
                "has_requirements": _os.path.isfile(
                    _os.path.join(ENV_INPUT_MOUNT, "resolved.requirements.txt"))}

    with app.run():
        return state.remote()


def cmd_env_input(args) -> int:
    """Upload the wheelhouse into its volume, once."""
    import modal

    source = env_input_source()
    if not args.force:
        state = env_input_state()
        if state.get("ready"):
            print(f"already present: {state['wheels']} wheel(s), "
                  f"{state['gigabytes']} GB")
            return 0

    volume = modal.Volume.from_name(ENV_INPUT_VOLUME, create_if_missing=True)
    files = [p for p in source.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"uploading {len(files)} file(s), {total / 1e9:.2f} GB from {source}")
    began = time.time()
    with volume.batch_upload(force=True) as batch:
        for path in files:
            batch.put_file(str(path), str(path.relative_to(source)))
    print(f"  uploaded in {round(time.time() - began, 1)}s")

    state = env_input_state()
    if not state.get("ready"):
        print(f"the upload did not leave a usable wheelhouse: "
              f"{state.get('why')}", file=sys.stderr)
        return 1
    print(f"  verified from a fresh container: {state['wheels']} wheel(s), "
          f"{state['gigabytes']} GB, "
          f"resolved.requirements.txt {'present' if state['has_requirements'] else 'MISSING'}")
    return 0


def model_environment(task: str) -> dict:
    """The environment that makes this task's checkpoint loadable, offline.

    ONE definition, shared by the readiness check and every sandbox, so the
    check cannot resolve the model in an environment the server never gets.

    `HF_HOME` is a PARENT: huggingface_hub looks for its cache at
    `$HF_HOME/hub`. Our volume holds the hub directory itself, so the variable
    that names it is `HF_HUB_CACHE`. Both are set -- HF_HOME for anything that
    wants a home, HF_HUB_CACHE for the lookup that actually matters -- with
    HF_HOME pointing one level up so the two agree.

    Offline is explicit rather than inherited: it is what turns a missing cache
    entry into a clear error instead of a network timeout, and the sandboxes
    have no egress to time out against.
    """
    model_id, revision = model_spec(task)
    return {
        "MODEL_PATH": "/models/target",
        "MODEL_ID": model_id,
        "MODEL_REVISION": revision,
        "HF_HUB_CACHE": HF_CACHE,
        "HF_HOME": MODEL_ROOT,
        "HF_DATASETS_CACHE": DATASETS_CACHE,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # The `datasets` library has its OWN offline flag; HF_HUB_OFFLINE does
        # not stop it reaching for repo metadata, which fails with
        # "Couldn't reach ... (OfflineModeIsEnabled)" on a corpus already in
        # the volume.
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


# --- the corpus the scenario replays ------------------------------------------

def corpus_spec(task: str) -> tuple[str, str]:
    """(dataset repo, sha256) this task's scenario replays.

    Read from the task's own harness module, which is where the pin lives --
    `_agentx_mvp.CORPUS` and `CORPUS_SHA256` -- so a task re-pointed at another
    corpus does not need this file edited.
    """
    # Imported as a PACKAGE module, not by file path: the task modules use
    # `from . import _agentx_mvp as shared`, and a relative import has no
    # parent package when the file is loaded standalone.
    import importlib

    root = str(REPO / "serverbench" / "harness")
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        probe = importlib.import_module(f"final_validate.tasks.{task.lower()}")
    except Exception as exc:
        raise SetupError(
            f"could not read {task}'s corpus pin: {exc!r}") from exc
    corpus = getattr(probe, "CORPUS", "") or ""
    digest = getattr(probe, "CORPUS_SHA256", "") or ""
    if not corpus:
        raise SetupError(
            f"{task}'s module names no CORPUS, so there is no workload to "
            f"replay and nothing to stage")
    return str(corpus), str(digest)


def corpus_state(task: str) -> dict:
    """Is the corpus in the volume, and can `datasets` actually open it?

    The real operation again: aiperf replays it through the `datasets`
    library, so "present" means loadable offline, not "files exist".
    """
    import modal

    corpus = corpus_spec(task)[0]
    app = modal.App("serverbench-corpuscheck")
    volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
    image = (modal.Image.debian_slim(python_version=_local_python())
             .pip_install("datasets", "huggingface_hub"))

    @app.function(image=image, volumes={MODEL_ROOT: volume}, timeout=1800,
                  cpu=4.0, memory=16384, name="corpus", serialized=True)
    def state(repo: str, env: dict) -> dict:
        import os
        volume.reload()
        os.environ.update(env)
        folder = os.path.join(HF_CACHE,
                              "datasets--" + repo.replace("/", "--"))
        if not os.path.isdir(folder):
            return {"ready": False, "why": f"no {folder} in the volume"}
        # NOT streaming. `streaming=True` resolves metadata over the network on
        # every call, so it can never succeed offline, even on a corpus that
        # is sitting in the volume. A plain load reads the cached snapshot, which
        # is what a replay does.
        try:
            from datasets import load_dataset
            got = load_dataset(repo, split="train")
            return {"ready": True, "rows": int(got.num_rows),
                    "columns": sorted(got.column_names)[:8]}
        except Exception as exc:                               # noqa: BLE001
            import glob
            files = glob.glob(os.path.join(folder, "snapshots", "*", "**", "*"),
                              recursive=True)
            return {"ready": False,
                    "why": f"present but not loadable offline: "
                           f"{type(exc).__name__}: {str(exc)[:240]}",
                    "files_in_snapshot": len(files)}

    with app.run():
        return state.remote(corpus, model_environment(task))


def cmd_corpus(args) -> int:
    """Pull the task's corpus into the volume, FROM HuggingFace, inside Modal."""
    import modal

    task = args.task
    corpus = corpus_spec(task)[0]
    if not args.force:
        state = corpus_state(task)
        if state.get("ready"):
            print(f"already present and loadable: {corpus}")
            return 0

    app = modal.App("serverbench-corpus-fetch")
    volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
    image = (modal.Image.debian_slim(python_version=_local_python())
             .pip_install("huggingface_hub[hf_transfer]", "datasets")
             .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))

    @app.function(image=image, volumes={MODEL_ROOT: volume}, timeout=7200,
                  cpu=8.0, memory=32768, name="fetch", serialized=True)
    def fetch(repo: str) -> dict:
        import os
        import time

        from huggingface_hub import snapshot_download
        began = time.time()
        path = snapshot_download(repo, repo_type="dataset",
                                 cache_dir=HF_CACHE, max_workers=8)
        # The refs, for the same reason the model needs them: a repo id with no
        # `refs/main` cannot be resolved offline.
        folder = os.path.join(HF_CACHE,
                              "datasets--" + repo.replace("/", "--"))
        head = os.path.basename(path.rstrip("/"))
        refs = os.path.join(folder, "refs")
        os.makedirs(refs, exist_ok=True)
        for name in ("main",):
            with open(os.path.join(refs, name), "w") as handle:
                handle.write(head)
        # AND THE PROCESSED CACHE, built here because this container has
        # network and the sandboxes that read it do not. A hub snapshot alone
        # is not enough: `load_dataset` offline reads the processed cache, and
        # without it every replay reports "Couldn't reach ... on the Hub".
        os.environ["HF_DATASETS_CACHE"] = DATASETS_CACHE
        os.makedirs(DATASETS_CACHE, exist_ok=True)
        from datasets import load_dataset
        built = load_dataset(repo, split="train")
        rows = int(built.num_rows)
        volume.commit()
        total = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(folder) for f in fs
                    if not os.path.islink(os.path.join(r, f)))
        return {"snapshot": path, "seconds": round(time.time() - began, 1),
                "gigabytes": round(total / 1e9, 2), "rows": rows}

    print(f"fetching dataset {corpus} into {MODEL_VOLUME} ...")
    with app.run():
        got = fetch.remote(corpus)
    print(f"  fetched {got['gigabytes']} GB in {got['seconds']}s, "
          f"{got.get('rows')} rows processed into the volume")

    state = corpus_state(task)
    if not state.get("ready"):
        print(f"\nthe corpus is not loadable: {state.get('why')}",
              file=sys.stderr)
        return 1
    print(f"  verified from a fresh container: loadable offline, "
          f"columns {state.get('columns')}")
    return 0


# --- the network the agent is allowed ----------------------------------------

#: Hosts the agent's CLI must reach. Named, never addressed: see below.
PROVIDER_HOSTS = ("api.anthropic.com", "api.fireworks.ai")

#: Addresses seen for those hosts, accumulated across runs.
#:
#: They rotate, sometimes to a different cloud within minutes. A single
#: resolution is a snapshot, and a
#: sandbox locked to it loses its model the moment DNS moves, hours into a run,
#: with no useful error. So the union is remembered. It widens the lock
#: slightly; the alternatives are whole cloud ranges, or no lock at all.
CIDR_MEMORY = (REPO / "serverbench" / "artifacts" / "modal"
               / "provider_cidrs.json")

_RESOLVE_PROBE = """
import json, socket
seen = {}
for host in __HOSTS__:
    addresses = set()
    for _ in range(8):
        try:
            _, _, ips = socket.gethostbyname_ex(host)
            addresses.update(ips)
        except OSError:
            pass
    try:
        for info in socket.getaddrinfo(host, 443, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    seen[host] = sorted(addresses)
print("RESOLVED " + json.dumps(seen))
"""


def _remember(seen: set) -> set:
    known = set()
    try:
        known = set(json.loads(CIDR_MEMORY.read_text("utf-8")).get("cidrs", []))
    except (OSError, ValueError):
        known = set()
    union = known | seen
    if union != known:
        try:
            CIDR_MEMORY.parent.mkdir(parents=True, exist_ok=True)
            CIDR_MEMORY.write_text(json.dumps(
                {"schema": "serverbench.modal_provider_cidrs/1",
                 "note": ("resolved from inside Modal; these rotate, so the "
                          "union is kept rather than the latest answer"),
                 "cidrs": sorted(union)}, indent=1) + "\n", "utf-8")
        except OSError:
            pass
    return union


def resolve_provider_cidrs(app, hosts: tuple = PROVIDER_HOSTS) -> list:
    """What the providers resolve to FROM MODAL, unioned with what we have seen.

    This host's answer is the wrong answer: a provider can resolve to one
    cloud from here and another from inside Modal. Locking a sandbox to the
    local view silently cuts the agent off from its own model.
    """
    import modal
    # NOT `create_sandbox`, and deliberately: this is the only sandbox that is
    # not a run. It takes no GPU, needs none of our image, and must reach the
    # open internet -- which is the very thing it exists to narrow. Routing it
    # through the factory would need an allowlist to build the allowlist.
    probe = modal.Sandbox.create(
        "sleep", "infinity", app=app,
        image=modal.Image.debian_slim(python_version=_local_python()),
        cpu=0.5, memory=1024, timeout=300)
    try:
        probe.filesystem.write_text(
            _RESOLVE_PROBE.replace("__HOSTS__", repr(list(hosts))),
            "/resolve.py")
        process = probe.exec("python3", "/resolve.py", timeout=120)
        out = process.stdout.read()
        process.wait()
    finally:
        probe.terminate()
    for line in out.splitlines():
        if line.startswith("RESOLVED "):
            resolved = json.loads(line.split(" ", 1)[1])
            seen = {f"{ip}/32" for ips in resolved.values() for ip in ips}
            if not seen:
                raise SetupError(
                    f"none of {hosts} resolved from inside Modal; locking the "
                    f"sandbox to an empty allowlist would leave the agent with "
                    f"no model at all")
            return sorted(_remember(seen))
    raise SetupError(f"the resolver probe produced no answer: {out[:400]!r}")


# --- the image -------------------------------------------------------------

def agent_image(task: str):
    """The sandbox image: CUDA, torch, the agent CLI, and a non-root user.

    Built from NVIDIA's CUDA image rather than `debian_slim` because the agent
    compiles kernels, and `nvcc` is the whole point of letting it. Modal caches
    this by its definition, so the first run pays for it and the rest do not.

    The non-root user is not optional: Modal always runs as root and ignores
    `USER`, and the CLI exits immediately with "--dangerously-skip-permissions
    cannot be used with root/sudo privileges", which a driver reads as a dead
    agent and cold-restarts for the whole budget.
    """
    import modal
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
        .apt_install("curl", "git", "ca-certificates", "ripgrep", "build-essential")
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
            "apt-get install -y nodejs",
            "npm install -g @anthropic-ai/claude-code",
            "useradd -m -u 1000 -s /bin/bash agent",
            "mkdir -p /home/agent/.claude && chown -R agent:agent /home/agent",
        )
        .pip_install("torch", "transformers", "accelerate", "safetensors",
                     "huggingface_hub", "numpy", "pyyaml", "aiohttp",
                     "fastapi", "uvicorn")
    )


# --- the one place a run sandbox is created ------------------------------------

#: What a sandbox is allowed to reach.
#:
#:   "provider"  the model API and nothing else -- what an agent needs
#:   "none"      nothing at all -- what the verdict needs, since it runs no
#:               agent and talks to no API
NETWORK_MODES = ("provider", "none")


def create_sandbox(task: str, *, purpose: str, hours: float = 2.0,
                   workdir: str = "/workspace", env: dict | None = None,
                   network: str = "provider", cpu: float = 8.0,
                   memory: int = 65536, ports: list | None = None,
                   image=None):
    """Create a sandbox for this task. THE only place that does.

    Every difference between an agent's sandbox and a verdict's is an argument
    here, so a difference has to be chosen rather than drifted into: the
    wheelhouse mount and the HF cache variables cannot diverge between them.

    What is always the same: our image, this task's GPUs, the model volume and
    the wheelhouse volume, the checkpoint at `/models/target`, and the
    environment that makes that checkpoint loadable offline.
    """
    import docker_to_modal
    import modal

    if network not in NETWORK_MODES:
        raise SetupError(f"network must be one of {NETWORK_MODES}, not "
                         f"{network!r}")
    card, count = gpu_spec(task)
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    allowed = (resolve_provider_cidrs(app) if network == "provider" else [])

    sandbox = modal.Sandbox.create(
        "sleep", "infinity", app=app,
        image=image if image is not None else docker_to_modal.modal_image(),
        gpu=f"{card}:{count}", cpu=cpu, memory=int(memory),
        # `hours` is the sandbox's whole life, not the caller's budget. No
        # slack is added here: a sandbox that outlives its purpose is a GPU
        # nobody is watching, so whoever knows what comes after the work --
        # collecting evidence, pulling the tree -- asks for that time itself.
        timeout=int(float(hours) * 3600),
        volumes={"/model-store": modal.Volume.from_name(MODEL_VOLUME).with_mount_options(read_only=True),
                 ENV_INPUT_MOUNT: modal.Volume.from_name(ENV_INPUT_VOLUME).with_mount_options(read_only=True)},
        workdir=workdir,
        secrets=[modal.Secret.from_dict({**model_environment(task),
                                         **(env or {})})],
        # `None` means ALL CIDRs in Modal's API, so an empty list must never
        # collapse to `None` for an offline sandbox. `block_network` is the platform's explicit
        # no-egress control; the allowlist is only meaningful for an agent.
        block_network=(network == "none"),
        outbound_cidr_allowlist=(allowed if network == "provider" else None),
        **({"encrypted_ports": ports} if ports else {}),
    )
    # The checkpoint where the brief and the verdict both say it is. Run here
    # rather than by each caller, because a sandbox without it is a sandbox
    # every later step fails in.
    try:
        process = sandbox.exec(
            "/bin/bash", "-lc",
            f"mkdir -p {workdir} /models && ln -sfn /model-store/hf /models/hf && "
            f"ln -sfn /model-store/datasets /models/datasets && "
            f"ln -sfn {snapshot_path(task)} /models/target && "
            f"test -f /models/target/config.json && echo staged",
            timeout=300)
        staged = (process.stdout.read() or "").strip()
        code = process.wait()
        if code != 0 or "staged" not in staged:
            raise SetupError(
                f"the checkpoint did not stage into a {purpose} sandbox for "
                f"{task}; /models/target has no config.json")
    except BaseException:
        sandbox.terminate()
        raise
    return sandbox


# --- commands --------------------------------------------------------------

def cmd_check(args) -> int:
    """Refuse before a run rather than during one."""
    task = args.task
    problems: list[str] = []
    card, count = gpu_spec(task)
    model_id, revision = model_spec(task)
    print(f"task      {task}")
    print(f"gpu       {card}:{count}")
    print(f"model     {model_id} @ {revision}")
    print(f"snapshot  {snapshot_path(task)}")

    try:
        import modal
    except ImportError:
        print("\nmodal is not importable by this interpreter", file=sys.stderr)
        return 2
    try:
        modal.App.lookup(APP_NAME, create_if_missing=True)
        print(f"auth      ok, app {APP_NAME}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"\nmodal is not usable: {exc}", file=sys.stderr)
        return 2

    state = model_state(task)
    if state.get("ready"):
        print(f"weights   ok, {state['weight_gigabytes']} GB in "
              f"{MODEL_VOLUME}, resolves offline")
    else:
        problems.append(
            f"the checkpoint is not in the {MODEL_VOLUME} volume "
            f"({state.get('why', 'unknown')}). Run: modalctl model --task {task}")

    # Qualification is NOT this script's business, but saying nothing about it
    # would let a hardware retarget look ready when its operating point came
    # from another machine.
    import yaml
    reference_path = TASKS / task / "reference.yaml"
    reference = (yaml.safe_load(reference_path.read_text("utf-8")) or {}
                 if reference_path.is_file() else {})
    accuracy = (reference.get("correctness") or {}).get("reference_accuracy")
    if not isinstance(accuracy, (float, int)) or not 0 <= accuracy <= 1:
        problems.append(
            f"{task} has no valid reference.yaml correctness.reference_accuracy, so the verdict's "
            f"answer-correctness rule can only be INCONCLUSIVE. It is a "
            f"model-and-reference-stack property and may be carried only from "
            f"the same pinned checkpoint and reference arm")
    archive = TASKS / "_concurrency_search_archive" / task
    if (not (archive / "index.json").is_file()
            or getattr(_task_module(task), "OFFERED_CONCURRENCY", None) is None):
        detail = "no measured offered load has been selected for this hardware"
        problems.append(
            f"{task} has no concurrency search of its own; {detail}")

    if problems:
        print("\nNOT READY:")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("\nready")
    return 0


def model_state(task: str) -> dict:
    """Is the pinned checkpoint in the volume, and is it the right bytes?

    Runs in a Modal container, because that is the only place that can see the
    volume, and a fresh one, because the writer's own view is not evidence.
    """
    import modal

    model_id, revision = model_spec(task)
    app = modal.App("serverbench-modelcheck")
    volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
    # THE LOCAL INTERPRETER'S VERSION, not a pinned one. A `serialized=True`
    # function is pickled here and unpickled there, so Modal refuses the pair
    # when the versions differ -- and the helper images exist only to read a
    # volume, so they have no opinion about which Python that is.
    image = modal.Image.debian_slim(python_version=_local_python())

    # `transformers` because the check RESOLVES the model the way the starter
    # and the verdict do. A file-existence check reports ready on a cache with
    # no `refs/main`, on which every server answers 503.
    image = image.pip_install("transformers", "huggingface_hub")

    @app.function(image=image, volumes={MODEL_ROOT: volume}, timeout=1800,
                  cpu=4.0, name="state", serialized=True)
    def state(folder: str, revision: str, hub_id: str, served: dict) -> dict:
        import glob
        import os
        volume.reload()
        root = f"{HF_CACHE}/{folder}/snapshots/{revision}"
        if not os.path.isdir(root):
            return {"ready": False, "why": f"no snapshot at {root}"}
        weights = sorted(glob.glob(os.path.join(root, "*.safetensors")))
        dangling = [os.path.basename(w) for w in weights
                    if not os.path.exists(os.path.realpath(w))]
        if not weights:
            return {"ready": False, "why": "no .safetensors in the snapshot"}
        if not os.path.isfile(os.path.join(
                f"{HF_CACHE}/{folder}", "refs", "main")):
            return {"ready": False,
                    "why": ("the cache has no refs/main, so a hub id cannot "
                            "be resolved offline -- which is how the starter "
                            "loads its checkpoint")}
        if dangling:
            return {"ready": False,
                    "why": f"{len(dangling)} weight symlink(s) dangle: "
                           f"{dangling[:3]}"}
        total = sum(os.path.getsize(os.path.realpath(w)) for w in weights)

        # THE REAL OPERATION. The starter loads its checkpoint by HUB ID, and
        # the verdict serves that starter, so "is this cache usable" means
        # "does a hub-id lookup resolve here, offline". Nothing about the file
        # listing above answers that: a cache can hold every blob and no
        # `refs/main`. THE SAME environment the server will get, passed in
        # rather than rebuilt here, so the check cannot configure itself
        # differently from the thing it is checking.
        os.environ.update({k: v for k, v in (served or {}).items()
                           if k.startswith(("HF_", "TRANSFORMERS_"))})
        resolves, why_not = False, ""
        try:
            from transformers import AutoConfig, AutoTokenizer
            AutoConfig.from_pretrained(hub_id)
            AutoTokenizer.from_pretrained(hub_id)
            resolves = True
        except Exception as exc:                          # noqa: BLE001
            why_not = f"{type(exc).__name__}: {str(exc)[:200]}"

        return {"ready": bool(resolves), "weight_files": len(weights),
                "weight_gigabytes": round(total / 1e9, 2),
                "snapshot": root,
                "hub_id_resolves_offline": resolves,
                "why": "" if resolves else (
                    f"the cache is present but a hub-id lookup fails offline, "
                    f"which is how the starter loads: {why_not}")}

    folder = "models--" + model_id.replace("/", "--")
    with app.run():
        return state.remote(folder, revision, model_id,
                            model_environment(task))


def cmd_model(args) -> int:
    """Fetch the task's pinned checkpoint into the shared volume."""
    import modal

    task = args.task
    model_id, revision = model_spec(task)
    if not args.force:
        state = model_state(task)
        if state.get("ready"):
            print(f"already present: {state['weight_gigabytes']} GB, "
                  f"resolves offline")
            print(state["snapshot"])
            return 0

    app = modal.App("serverbench-model-fetch")
    volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
    image = (modal.Image.debian_slim(python_version=_local_python())
             .pip_install("huggingface_hub[hf_transfer]")
             .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))

    @app.function(image=image, volumes={MODEL_ROOT: volume}, timeout=7200,
                  cpu=8.0, memory=32768, name="fetch", serialized=True)
    def fetch(model_id: str, revision: str) -> dict:
        import time

        from huggingface_hub import snapshot_download
        began = time.time()
        path = snapshot_download(model_id, revision=revision,
                                 cache_dir=HF_CACHE, max_workers=8)
        # THE REFS, which `snapshot_download` does not write when it is given
        # an explicit revision sha.
        #
        # Without them the cache has blobs and snapshots and no way to resolve
        # a BRANCH, so `from_pretrained("Qwen/Qwen3.5-9B")` offline reports
        # "couldn't find them in the cached files" -- and the starter resolves
        # its checkpoint by hub id, so the server would answer 503 forever on
        # a sandbox with no egress. The host cache the docker path mounts has
        # `refs/main`; this makes the two agree.
        import os as _os
        folder = _os.path.join(HF_CACHE,
                               "models--" + model_id.replace("/", "--"))
        refs = _os.path.join(folder, "refs")
        _os.makedirs(refs, exist_ok=True)
        for name in ("main", revision[:12]):
            with open(_os.path.join(refs, name), "w") as handle:
                handle.write(revision)
        # EXPLICIT. An implicit commit does not reliably persist the weight
        # files, and the failure is invisible until a different container looks.
        volume.commit()
        return {"snapshot": path, "seconds": round(time.time() - began, 1)}

    print(f"fetching {model_id} @ {revision} into {MODEL_VOLUME} ...")
    with app.run():
        got = fetch.remote(model_id, revision)
    print(f"  fetched in {got['seconds']}s")

    state = model_state(task)
    if not state.get("ready"):
        print(f"\nthe fetch did not leave a usable checkpoint: "
              f"{state.get('why')}", file=sys.stderr)
        return 1
    print(f"  verified from a fresh container: {state['weight_files']} weight "
          f"file(s), {state['weight_gigabytes']} GB, resolves offline")
    print(state["snapshot"])
    return 0


def cmd_sandbox(args) -> int:
    """Create one GPU sandbox shaped by the task, through the one factory."""
    state = model_state(args.task)
    if not state.get("ready"):
        print(f"refusing: {state.get('why')}\n"
              f"Run: modalctl model --task {args.task}", file=sys.stderr)
        return 2
    sandbox = create_sandbox(args.task, purpose="manual",
                             hours=float(args.hours), cpu=float(args.cpu),
                             memory=int(args.memory))
    card, count = gpu_spec(args.task)
    print(json.dumps({"sandbox": sandbox.object_id, "app": APP_NAME,
                      "gpu": f"{card}:{count}",
                      "model_path": "/models/target",
                      "hours": float(args.hours)}))
    return 0


def cmd_teardown(args) -> int:
    """Terminate this app's sandboxes. A 24 h idle sandbox is not free."""
    import modal
    try:
        app = modal.App.lookup(APP_NAME, create_if_missing=False)
    except Exception:                                         # noqa: BLE001
        print(f"no such app: {APP_NAME}")
        return 0
    stopped = 0
    for sandbox in modal.Sandbox.list(app_id=app.app_id):
        try:
            sandbox.terminate()
            stopped += 1
            print(f"  terminated {sandbox.object_id}")
        except Exception as exc:                              # noqa: BLE001
            print(f"  {sandbox.object_id}: {exc}", file=sys.stderr)
    print(f"{stopped} sandbox(es) terminated in {APP_NAME}")
    return 0


def cmd_setup(args) -> int:
    """Build everything a task needs on Modal, from nothing, in one command.

    THE POINT OF THIS COMMAND is that the volumes are disposable. They cost
    money per GB-month and nothing in them is authored here: the checkpoint
    comes from HuggingFace by its pinned revision, the corpus comes from
    HuggingFace by its pinned dataset id, and the wheelhouse is built from the
    frozen `resolved.requirements.txt`. So the right lifecycle is
    `setup` before a batch of runs and `purge` after it, rather than carrying
    24 GB between runs on the chance it gets used again.

    Each stage is idempotent and verifies by DOING THE REAL OPERATION -- the
    checkpoint is resolved offline and its config read, the corpus is loaded
    from the processed cache, the wheelhouse is checked for the markers a
    candidate's `serve.sh` installs from. A file-existence check can pass on
    an HF cache that cannot load the model, which is why none of these stages
    is a listing.
    """
    stages = [("model", cmd_model), ("corpus", cmd_corpus)]
    if not args.skip_env_input:
        stages.append(("env-input", cmd_env_input))
    began = time.time()
    for name, run in stages:
        print(f"\n=== {name}", flush=True)
        code = run(args)
        if code != 0:
            print(f"\n{name} failed; nothing further was built. The volumes "
                  f"keep whatever earlier stages wrote, so fixing the cause "
                  f"and re-running `setup` resumes rather than restarts.",
                  file=sys.stderr)
            return code
    print("\n=== image")
    # Built here rather than at first use, because a 12-minute image build
    # inside a run spends the agent's budget on our infrastructure.
    import docker_to_modal
    import modal
    image = docker_to_modal.modal_image()
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    probe = create_sandbox(args.task, purpose="setup-probe", hours=0.25,
                           network="none", cpu=2.0, memory=8192)
    try:
        proof = probe.exec("/bin/bash", "-lc",
                           "which aiperf && nvidia-smi --query-gpu=name "
                           "--format=csv,noheader && python3 -c "
                           "'import torch;print(torch.__version__, "
                           "torch.cuda.device_count())'", timeout=300)
        print((proof.stdout.read() or "").strip())
        proof.wait()
    finally:
        probe.terminate()
    del image, app
    print(f"\n=== ready in {time.time() - began:.0f}s")
    print(f"app      {APP_NAME}")
    print(f"volumes  {MODEL_VOLUME}, {ENV_INPUT_VOLUME}")
    print(f"next     bash serverbench/scripts/server_evolve/modal-evolve.sh "
          f"{args.task} --hours N")
    print(f"after    {Path(__file__).name} purge --yes   "
          f"(the volumes are rebuildable; keeping them is a bill)")
    return 0


def cmd_purge(args) -> int:
    """Terminate every sandbox AND delete the volumes. Destructive, on purpose.

    Nothing here is authored: every byte is re-fetchable by `setup` from a
    pinned revision. What deleting costs is time -- roughly 12 minutes to
    re-download 19 GB of weights and rebuild the wheelhouse -- and what
    keeping costs is storage for 24 GB nobody is using between runs.

    REFUSES WHILE A SANDBOX IS STILL RUNNING unless `--force`. Deleting a
    volume out from under a live run would take the checkpoint away from a
    server mid-measurement, and the failure would look like a candidate
    defect.
    """
    import modal
    if not args.yes:
        print("refusing without --yes: this deletes the model and wheelhouse "
              "volumes", file=sys.stderr)
        return 2

    live = []
    try:
        app = modal.App.lookup(APP_NAME, create_if_missing=False)
        live = list(modal.Sandbox.list(app_id=app.app_id))
    except Exception:                                         # noqa: BLE001
        app = None
    if live and not args.force:
        print(f"refusing: {len(live)} sandbox(es) are still running in "
              f"{APP_NAME}:", file=sys.stderr)
        for sandbox in live:
            print(f"  {sandbox.object_id}", file=sys.stderr)
        print("Deleting the volumes under a live run takes the checkpoint "
              "away from a server mid-measurement. Stop them first "
              "(`teardown`), or pass --force if you know they are yours.",
              file=sys.stderr)
        return 2
    for sandbox in live:
        try:
            sandbox.terminate()
            print(f"  terminated {sandbox.object_id}")
        except Exception as exc:                              # noqa: BLE001
            print(f"  {sandbox.object_id}: {exc}", file=sys.stderr)

    gone = 0
    for name in (MODEL_VOLUME, ENV_INPUT_VOLUME):
        try:
            # `modal.Volume` (1.5.0) exposes no class-level `delete`; the
            # object method is `_instance_delete`, and the supported public
            # route is the CLI's `modal volume delete`. Tried in that order so
            # this works whether or not the private name is promoted.
            volume = modal.Volume.from_name(name)
            deleter = getattr(modal.Volume, "delete", None)
            if callable(deleter):
                deleter(name)
            elif hasattr(volume, "_instance_delete"):
                volume._instance_delete()
            else:
                raise AttributeError(
                    "this modal version exposes no volume deletion API; "
                    f"run `modal volume delete {name}` by hand")
            gone += 1
            print(f"  deleted volume {name}")
        except Exception as exc:                              # noqa: BLE001
            print(f"  {name}: {type(exc).__name__}: {exc}")
    print(f"{len(live)} sandbox(es) stopped, {gone} volume(s) deleted")
    print(f"Rebuild with: {Path(__file__).name} setup --task <TASK>")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, helptext in (("check", "refuse before a run rather than during"),
                           ("model", "fetch the task's pinned checkpoint"),
                           ("sandbox", "create one GPU sandbox for the task")):
        one = sub.add_parser(name, help=helptext)
        one.add_argument("--task", required=True)
        if name == "model":
            one.add_argument("--force", action="store_true",
                             help="re-fetch even if the volume looks correct")
        if name == "sandbox":
            one.add_argument("--hours", default="2")
            one.add_argument("--cpu", default="8")
            one.add_argument("--memory", default="65536")

    corpus = sub.add_parser("corpus",
                            help="pull the task's replay corpus from HuggingFace")
    corpus.add_argument("--task", required=True)
    corpus.add_argument("--force", action="store_true")

    wheels = sub.add_parser("env-input",
                            help="upload the offline wheelhouse the brief promises")
    wheels.add_argument("--force", action="store_true")

    down = sub.add_parser("teardown", help="terminate this app's sandboxes")
    down.add_argument("--task", default=None, help="ignored; for symmetry")

    setup = sub.add_parser(
        "setup",
        help="build everything this task needs on Modal, from nothing")
    setup.add_argument("--task", required=True)
    setup.add_argument("--force", action="store_true",
                       help="re-fetch each stage even if it looks correct")
    setup.add_argument("--skip-env-input", action="store_true",
                       help="leave the wheelhouse volume alone")

    purge = sub.add_parser(
        "purge", help="stop every sandbox AND delete the volumes")
    purge.add_argument("--yes", action="store_true",
                       help="required: this deletes 24 GB that `setup` "
                            "rebuilds from pinned revisions")
    purge.add_argument("--force", action="store_true",
                       help="purge even while sandboxes are running")
    purge.add_argument("--task", default=None, help="ignored; for symmetry")

    args = ap.parse_args(argv)
    try:
        return {"check": cmd_check, "model": cmd_model,
                "sandbox": cmd_sandbox, "teardown": cmd_teardown,
                "env-input": cmd_env_input,
                "corpus": cmd_corpus, "setup": cmd_setup,
                "purge": cmd_purge}[args.cmd](args)
    except SetupError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
