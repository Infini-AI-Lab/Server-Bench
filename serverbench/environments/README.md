# `serverbench/environments/` — the agent container images

Three Dockerfiles, built in order by `scripts/deploy/03-build-images.sh`. The
build context is the repository root, narrowed by the root `.dockerignore`.

| directory | image | what it is |
|---|---|---|
| `serving-agent/` | `serverbench/serving-agent:cu128` | the base: Python 3.12, torch and the serving stack, the agent CLIs, and the egress firewall `restrict-network.sh` |
| `evolve-agent/` | `serverbench/evolve-agent:cu128` | the image every docker evolve runs in, built `FROM` the base: `uv`, Nsight Systems/Compute, the CUDA toolchain, and an interpreter that keeps the agent's own venv isolated. `toolchain_smoke.py` runs during the build and fails it if a kernel cannot be compiled |
| `evolve-agent-int4/` | `serverbench/evolve-agent:cu128-int4` | built `FROM` the evolve image for the INT4 task: adds `compressed-tensors` (and `loguru`), pinned by hash and installed `--no-deps`, so the quantized checkpoint loads; everything else is the evolve image's |

A task names a non-default image and wheelhouse in its
`environment/agent-environment`; see
[`docs/serving_evolve/containers.md`](../docs/serving_evolve/containers.md).

`_claude-bin`, `_codex-bin`, `_codex-code-mode-host-bin` and `_codex-rg-bin`
are staged here by the build script (Docker cannot `COPY` from outside its
context) and are gitignored.

**Do not edit a Dockerfile casually, not even a comment.** The build stamps
`serverbench.image_inputs_sha256`, a digest over the `serving-agent/` and
`evolve-agent/` Dockerfiles' paths and bytes (`image_freshness.IMAGE_INPUTS`), and `scripts/agent/image_freshness.py` refuses to launch an image whose
label does not match the checkout. Any change, including moving a file, means
rebuilding the images before the next docker evolve.

Modal tasks do not build these images locally:
`serving_tasks/_modal/docker_to_modal.py` composes the base and evolve
Dockerfiles into one Modal image definition.
