# Containers: Harbor, and the agent inside it

**There is one container path, and Harbor is it.**
`harness/lifecycle/run.py::_start_through_harbor` composes the environment
through `harbor.environments.docker.DockerEnvironment`, driven as a subprocess
under `harbor/.venv/bin/python` by `scripts/agent/harbor_env.py`. A host that
cannot drive Harbor **refuses the run**; there is no fallback to a raw
`docker run`, because two paths that both stand up an environment drift apart
on mounts, capabilities and workspace ownership.
`scripts/deploy/01-preflight.sh` checks that Harbor can be driven on the host
before a run spends a GPU hour finding out.

Harbor has no field for three things a serving task needs, so `harbor_env.py`
writes them into one generated compose overlay: the GPU (granted through compose
`deploy.resources.reservations`, with `override_gpus=0` silencing Harbor's
local-docker GPU guard), `cap_add: [NET_ADMIN]` for the firewall, and `shm_size`
for torch and NCCL.

## Harbor owns the box; this repository owns the task

A ServerBench task is declared by `contract.yaml` + `reference.yaml` and a
harness module under `final_validate/tasks/`: the p90 bars, prices, devices and
window, the SGLang reference, the scenario and the seed policy. Its `task.toml`
is a Harbor task definition for the container only. Harbor has no notion of a
frozen threshold, a reference arm or a private correctness stage, and it is not
asked to: it supplies the box the agent works in, and the harness supplies
everything that decides whether what the agent built is good.

Harbor is vendored at `harbor/` (upstream `harbor-framework/harbor`, with agent
install-skip patches) as an editable install. Its documentation, including the
ATIF trajectory format that run evidence is recorded against, is at
[harborframework.com/docs](https://www.harborframework.com/docs). Where the site
and the vendored tree disagree, the tree is what runs.

## Networking: the agent reaches its model and nothing else

An offline container is the right environment for *measurement* (no route to
PyPI, GitHub or Hugging Face, so "build, do not retrieve" is a property of the
network), and the wrong one for an *agent*, which cannot work without its model
endpoint. Full egress is wrong the other way: it puts a serving framework one
`pip install` away. The design gives the agent exactly one route out:

- **The firewall.** `environments/serving-agent/restrict-network.sh`, baked into
  the image and run as root (the one privileged step) before the agent starts,
  allows only loopback, established flows, the DNS resolver and the hosts in
  `ALLOWED_HOSTS`. It applies three independent layers: package and source
  hosts are stripped from the allowlist before it is resolved; those hosts'
  current addresses are rejected in a chain evaluated first; and anything not
  explicitly allowed is rejected, with IPv6 dropped wholesale. A failure to
  apply the rules is fatal. Resolving the blocklist takes a minute or two on a
  cold DNS cache.
- **The egress broker.** `ALLOWED_HOSTS` names one address, the docker bridge
  gateway, where `scripts/agent/egress_broker.py` runs on the host, outside the
  container's network namespace. It is an HTTP CONNECT relay and nothing else:
  it relays only to the selected agent CLI's provider hosts
  (`egress_broker.PROVIDER_HOSTS`), never sees plaintext, and logs every
  connection to the run's `egress-broker.jsonl`.

The container is attached to a bridge network, not `--network none`: a container
with no network has no address for the verifier or the broker to reach.

**The agent does not run as root.** The agent CLI refuses
`--dangerously-skip-permissions` under root, so the image creates an `agent`
user whose UID and GID are build arguments (`AGENT_UID`, `AGENT_GID`) matching
the invoking user; the workspace is bind-mounted from the host, and a mismatch
would leave the agent unable to write it.

## The images

Built by `scripts/deploy/03-build-images.sh` (see
[`../../environments/README.md`](../../environments/README.md)):

| image | Dockerfile | what it adds |
|---|---|---|
| `serverbench/serving-agent:cu128` | `environments/serving-agent/Dockerfile` | Python 3.12, torch and the serving stack, the Claude Code CLI and the Codex runtime (copied in as static binaries: no node, no npm, no registry), `iptables`/`dnsutils` and the baked `restrict-network.sh` |
| `serverbench/evolve-agent:cu128` | `environments/evolve-agent/Dockerfile` | built `FROM` the base: `uv`, Nsight Systems/Compute, the CUDA toolchain, and an interpreter configuration under which the agent's own virtualenv cannot inherit the base image's site-packages |
| `serverbench/evolve-agent:cu128-int4` | `environments/evolve-agent-int4/Dockerfile` | built `FROM` the evolve image: `compressed-tensors` for the INT4 task's checkpoint, installed `--no-deps` |

A task that needs its own image and wheelhouse declares them in
`environment/agent-environment` (`IMAGE=`, `ENV_INPUT=`); the launcher refuses an
explicit `IMAGE` or `ENV_INPUT` that contradicts the declaration.

Everything the agent and the gates need is baked in, because at run time there
is nowhere to install from. The build stamps `serverbench.image_inputs_sha256`,
and `scripts/agent/image_freshness.py` refuses an image whose label does not
match the Dockerfiles in the checkout, so a Dockerfile change reaches a run only
through a rebuild. Pin versions by resolving them from the index and writing
them down, never from memory.

## GPU

The launcher picks genuinely free devices rather than assuming `0..n-1`
([`scripts/server_evolve/README.md`](../../scripts/server_evolve/README.md)). The
**first reserved GPU is always `cuda:0` inside the container** whatever its host
index, so a task addresses `cuda:0` and never a host index.

## The Modal verdict

`final_validate/modal_verdict.py` runs the verdict for a Modal task in fresh
sandboxes with no network route: one serves the frozen candidate with its own
`serve.sh` and measures the task's 1,800-second replay on loopback at the
declared load and a fresh final seed; after it is torn down, a second sandbox
runs the private correctness stage against the pinned SGLang reference. The
evaluator is made unwritable by the agent user before the candidate starts,
and only post-measurement evidence is downloaded. A candidate that cannot be
offered the workload is `NOT_RUN`, with the reason in the verdict's
`not_run_because`. Building and tearing down the Modal side is
[`serving_tasks/_modal/README.md`](../../../serving_tasks/_modal/README.md).
