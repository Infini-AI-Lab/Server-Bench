#!/usr/bin/env bash
# What the evolve launchers share: resolve this host's assets, pick a device
# that is genuinely free, and exec the ONE production entrypoint,
# `serverbench/scripts/evolve`. No lifecycle logic lives here.
#
# Every value is overridable from the environment for a deliberate variation.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$REPO/serverbench/scripts/server_evolve"

. "$REPO/serverbench/scripts/env.sh"
PY="$SERVERBENCH_PY"
ENTRYPOINT="$REPO/serverbench/scripts/evolve"

# Devices this host must never be given, e.g. EXCLUDE_GPUS="3,4". One value,
# shared with `env.sh`'s SERVERBENCH_EXCLUDED_GPUS and the driver's own fence;
# a standing list of card indices is a claim about a machine, so the default
# excludes nothing and occupancy decides.
EXCLUDE_GPUS="${EXCLUDE_GPUS:-${SERVERBENCH_EXCLUDED_GPUS:-}}"

# --- what a run is configured with -------------------------------------------
HOURS="${HOURS:-4}"
GPU="${GPU:-auto}"
AGENT_CLI="${AGENT_CLI:-claude}"
case "$AGENT_CLI" in
    codex)
        AGENT_MODEL="${AGENT_MODEL:-gpt-5.6-sol}"
        AGENT_EFFORT="${AGENT_EFFORT:-max}"
        ;;
    claude)
        AGENT_MODEL="${AGENT_MODEL:-claude-opus-5}"
        AGENT_EFFORT="${AGENT_EFFORT:-xhigh}"
        ;;
    *) echo "AGENT_CLI must be codex or claude, got '$AGENT_CLI'" >&2; exit 2 ;;
esac
# A per-run KEY=VALUE file naming a different model backend. Only the path is
# passed; the secrets are read inside harbor_env, because `ps` is world-readable.
AGENT_ENV="${AGENT_ENV:-}"
# What "done" means.
STOP="${STOP:-timebox}"
# The agent image and the offline wheelhouse default to the shared ones; a task
# that needs a different environment declares it (see agent_environment_for).
IMAGE="${IMAGE:-}"
DEFAULT_IMAGE="serverbench/evolve-agent:cu128"
LINEAGE_ROOT="${LINEAGE_ROOT:-$REPO/server_prototypes}"
# An explicit SEED always wins; otherwise a task starts from its own demo/.
SEED="${SEED:-}"
# The offline wheelhouse, host-local and shared by every task on this machine.
ENV_INPUT="${ENV_INPUT:-}"
DEFAULT_ENV_INPUT="$REPO/serverbench/artifacts/env-input"
# The model is per task: `model_for_task.py` reads the task's own declaration
# and finds exactly that revision in the local Hugging Face cache. An explicit
# MODEL_DIR (weights staged by hand) is never second-guessed.
MODEL_DIR="${MODEL_DIR:-}"

resolve_model_dir() {
    local task_dir="$1"
    if [ -n "$MODEL_DIR" ]; then
        echo "$MODEL_DIR"; return 0
    fi
    "$PY" "$REPO/serverbench/scripts/model_for_task.py" "$task_dir"
}

# How many devices the task is billed for, from its `contract.yaml`.
devices_for_task() {
    "$PY" - "$1" <<'PYEOF'
import sys, yaml, pathlib
spec = yaml.safe_load((pathlib.Path(sys.argv[1]) / "contract.yaml").read_text("utf-8")) or {}
print(int(spec["devices"]))
PYEOF
}

# --- the device ---------------------------------------------------------------
# Is this card held by any ownership mechanism? 0 = held, 1 = free.
#
#   * a non-numeric claim owner is a deliberate operator hold and is respected;
#   * a numeric owner is respected while that process is alive;
#   * the `with-gpu` flock is respected whatever the claim says;
#   * a container that reserves the card (Docker DeviceRequests) holds it even
#     while it shows no process and no memory between experiments.
#
# Nothing is signalled and nothing is reaped. A held card is declined.
gpu_is_held() {
    local index="$1"
    local claims="${SERVERBENCH_CARD_CLAIMS:-/tmp/serverbench-card-claims}"
    local locks="${GPU_LOCK_DIR:-/tmp/serverbench-gpu-locks}"
    local owner_file="$claims/gpu$index/owner" owner=""
    if [ -r "$owner_file" ]; then
        owner="$(cat "$owner_file" 2>/dev/null || true)"
        if [ -n "$owner" ]; then
            case "$owner" in
                *[!0-9]*) return 0 ;;                       # a named hold
                *) kill -0 "$owner" 2>/dev/null && return 0 ;;
            esac
        fi
    fi
    # The flock, tried non-blocking and released immediately.
    local lock="$locks/gpu$index.lock"
    if [ -e "$lock" ]; then
        if ! flock -n "$lock" true 2>/dev/null; then
            return 0
        fi
    fi
    gpu_is_reserved_by_container "$index" && return 0
    return 1
}

gpu_is_reserved_by_container() {
    # Exit 1 is the helper's one "provably unreserved" answer. Anything else --
    # held (0), or an interpreter that could not run it at all -- is held.
    local rc=0
    "$PY" "$REPO/serverbench/harness/final_validate/gpu_reservations.py" \
        --device "$1" >/dev/null 2>&1 || rc=$?
    [ "$rc" -ne 1 ]
}

# Is one card quiet RIGHT NOW: no compute process and no meaningful memory.
gpu_is_quiet() {
    local index="$1" used
    used="$(nvidia-smi --id="$index" --query-gpu=memory.used \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    [ -n "$used" ] || return 1
    [ "$(nvidia-smi --id="$index" --query-compute-apps=pid \
         --format=csv,noheader 2>/dev/null | grep -c . || true)" -eq 0 ] \
        && [ "$used" -lt 1000 ]
}

# A card is free only if it STAYS free: a neighbour that runs short iterations
# reads as idle between them, so a candidate is sampled over a minute.
: "${GPU_QUIET_SAMPLES:=4}"
: "${GPU_QUIET_INTERVAL_S:=15}"

gpu_stays_quiet() {
    local index="$1" i
    for i in $(seq 1 "$GPU_QUIET_SAMPLES"); do
        gpu_is_quiet "$index" || return 1
        [ "$i" -lt "$GPU_QUIET_SAMPLES" ] && sleep "$GPU_QUIET_INTERVAL_S"
    done
    return 0
}

# $1 how many are wanted. Prints a comma list, or fails having taken nothing.
pick_gpus() {
    local want="${1:-1}" excluded=",${EXCLUDE_GPUS},"
    local index used candidates=() got=()
    while read -r index used; do
        index="${index%,}"
        case "$excluded" in *",${index},"*) continue ;; esac
        gpu_is_held "$index" && continue
        if [ "$(nvidia-smi --id="$index" --query-compute-apps=pid \
                --format=csv,noheader | grep -c . || true)" -eq 0 ] \
           && [ "${used%% *}" -lt 1000 ]; then
            candidates+=("$index")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    [ "${#candidates[@]}" -ge "$want" ] || return 1
    for index in "${candidates[@]}"; do
        gpu_is_held "$index" && continue
        gpu_stays_quiet "$index" || continue
        got+=("$index")
        [ "${#got[@]}" -ge "$want" ] && break
    done
    [ "${#got[@]}" -ge "$want" ] || return 1
    (IFS=,; echo "${got[*]}")
}

# --- launch --------------------------------------------------------------------
# $1 the task directory name under serving_tasks/; the rest goes to the entrypoint.
# A task whose checkpoint needs packages the shared environment lacks declares
# its own agent image and wheelhouse in `environment/agent-environment`
# (`IMAGE=<tag>`, `ENV_INPUT=<path under the repo>`). Without one the shared
# pair is used. An explicit IMAGE/ENV_INPUT that contradicts the declaration is
# refused, because the wrong pair can mean a starter that cannot load its own
# checkpoint (the INT4 task's needs compressed-tensors).
agent_environment_for() {
    local task_dir="$1" declared="$1/environment/agent-environment"
    local want_image="$DEFAULT_IMAGE" want_input="$DEFAULT_ENV_INPUT" key value
    if [ -f "$declared" ]; then
        while IFS='=' read -r key value; do
            case "$key" in
                IMAGE) want_image="$value" ;;
                ENV_INPUT) want_input="$REPO/$value" ;;
            esac
        done < <(grep -E '^(IMAGE|ENV_INPUT)=' "$declared")
    fi
    if [ -n "$IMAGE" ] && [ "$IMAGE" != "$want_image" ]; then
        echo "IMAGE=$IMAGE, but $(basename "$task_dir") runs in $want_image" >&2
        return 2
    fi
    if [ -n "$ENV_INPUT" ] && [ "$(readlink -f "$ENV_INPUT")" != "$(readlink -f "$want_input")" ]; then
        echo "ENV_INPUT=$ENV_INPUT, but $(basename "$task_dir") installs from $want_input" >&2
        return 2
    fi
    IMAGE="$want_image"
    ENV_INPUT="$want_input"
}

launch_task() {
    local task_name="$1"; shift

    local task_dir="${TASK_ROOT:-$REPO/serving_tasks}/$task_name"
    [ -d "$task_dir" ] || { echo "no task directory for $task_name" >&2; return 2; }
    agent_environment_for "$task_dir" || return 2

    local seed="$SEED"
    if [ -z "$seed" ]; then
        seed="$task_dir/demo"
        [ -f "$seed/serve.sh" ] || {
            echo "$task_name has no demo/serve.sh to start from; pass SEED" >&2
            return 2
        }
    fi

    local want; want="$(devices_for_task "$task_dir")"
    [ -n "$want" ] || want=1

    local gpu="$GPU"
    case "$gpu" in
        ""|auto|any)
            gpu="$(pick_gpus "$want")" || {
                echo "no $want free GPU(s) for $task_name" \
                     "(excluding $EXCLUDE_GPUS). Nothing was started." >&2
                return 3
            } ;;
    esac
    case "$gpu" in
        *[!0-9,]*|"") echo "--gpu takes device indices, got '$gpu'" >&2; return 2 ;;
    esac
    local given; given="$(echo "$gpu" | tr ',' '\n' | grep -c .)"
    if [ "$given" != "$want" ]; then
        echo "$task_name is billed for $want device(s) and $given were given" \
             "(GPU=$gpu)." >&2
        return 2
    fi

    # The banner shows what the run will actually get: the caller's own flags
    # land after ours on the command line and override them.
    local shown_hours="$HOURS" shown_stop="$STOP" prev=""
    for arg in "$@"; do
        [ "$prev" = "--hours" ] && shown_hours="$arg"
        [ "$prev" = "--stop" ] && shown_stop="$arg"
        prev="$arg"
    done

    local resolved_model
    resolved_model="$(resolve_model_dir "$task_dir")" || {
        echo "$task_name: cannot determine which model to serve." \
             "Nothing was started and no card was touched." >&2
        return 2
    }

    echo "task      $task_name"
    echo "task dir  $task_dir"
    echo "model     $resolved_model"
    echo "device    GPU $gpu  (${want} declared by the task)"
    echo "budget    ${shown_hours} h; the AGENT decides when a version happens"
    echo "agent     $AGENT_CLI / $AGENT_MODEL / $AGENT_EFFORT"
    echo "seed      $seed"
    echo "stop      ${shown_stop}"
    echo "entry     serverbench/scripts/evolve"

    exec "$PY" "$ENTRYPOINT" \
        --task "$task_dir" \
        --gpu "$gpu" \
        --hours "$HOURS" \
        --model-dir "$resolved_model" \
        --env-input "$ENV_INPUT" \
        --lineage-root "$LINEAGE_ROOT" \
        --seed "$seed" \
        --image "$IMAGE" \
        --agent-cli "$AGENT_CLI" \
        --agent-model "$AGENT_MODEL" \
        --agent-effort "$AGENT_EFFORT" \
        ${AGENT_ENV:+--agent-env "$AGENT_ENV"} \
        --stop "$STOP" \
        "$@"
}
