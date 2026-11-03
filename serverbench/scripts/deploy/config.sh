#!/usr/bin/env bash
# Everything the deployment needs to know about this machine, in one place.
#
# Every value can be overridden from the environment, so moving to another server
# is a matter of exporting two or three variables rather than editing scripts.
# The GPU is chosen explicitly: a run that silently lands on whichever device
# happened to be free produces timings nobody can compare.
#
# Sourcing this needs the harness interpreter (env.sh refuses without one), so
# 00-envs.sh, which creates it, does not source this file.
set -euo pipefail

# The repository root, derived from this file's own location so the scripts work
# from any working directory.
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

export TASK="${TASK:-qwen3_5_2b_standard_1xL40}"
export TASK_DIR="${TASK_DIR:-$REPO/serving_tasks/$TASK}"

. "$REPO/serverbench/scripts/env.sh"
export PY="$SERVERBENCH_PY"

# The model, read off the task's own module (`MODEL_ID`, `MODEL_REVISION`), and
# the local Hugging Face snapshot that holds exactly that revision.
_task_model() { "$PY" "$REPO/serverbench/scripts/model_for_task.py" "$TASK_DIR" "$@" 2>/dev/null; }
export MODEL_ID="${MODEL_ID:-$(_task_model --id)}"
export MODEL_REVISION="${MODEL_REVISION:-$(_task_model --revision)}"
export MODEL_SRC="${MODEL_SRC:-$(_task_model)}"

# The shared offline wheelhouse runs mount at /opt/env-input. A task that
# declares its own in `environment/agent-environment` gets that one instead.
export ENV_INPUT="${ENV_INPUT:-$REPO/serverbench/artifacts/env-input}"

# One GPU, chosen on purpose. `nvidia-smi -L` lists what is here.
export GPU="${GPU:-0}"

export BASE_IMAGE="${BASE_IMAGE:-serverbench/serving-agent:cu128}"
export TASK_IMAGE="${TASK_IMAGE:-serverbench/evolve-agent:cu128}"

# --- where large transient files go --------------------------------------------
#
# Not the root filesystem: build contexts, extraction temp and package caches go
# to scratch, which can be lost. Weights, wheelhouses and evidence stay under the
# repository, because losing those costs GPU-hours to rebuild.
export PROJECT_SCRATCH="${PROJECT_SCRATCH:-/scratch/$USER/serverbench}"
mkdir -p "$PROJECT_SCRATCH"/{tmp,docker-tmp,uv-cache,build} 2>/dev/null || true
export TMPDIR="${TMPDIR:-$PROJECT_SCRATCH/tmp}"
export DOCKER_TMPDIR="${DOCKER_TMPDIR:-$PROJECT_SCRATCH/docker-tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_SCRATCH/uv-cache}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m   %s\n' "$*"; }
warn() { printf '   \033[33mwarn\033[0m %s\n' "$*"; }
die()  { printf '   \033[31mfail\033[0m %s\n' "$*" >&2; exit 1; }
