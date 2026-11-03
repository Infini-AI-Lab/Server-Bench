# Helpers the experiment scripts share. Sourced, never run.
#
# DRY_RUN=1 makes every script print each command it would run, shell-quoted
# and one per line, and start nothing. Lines starting with `#` are notes.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS="$REPO/experiments"
. "$REPO/serverbench/scripts/env.sh"
. "$EXPERIMENTS/config.sh"
PY="$SERVERBENCH_PY"
DRY_RUN="${DRY_RUN:-0}"
EVOLVE_ROOT="${EVOLVE_ROOT:-$REPO/serverbench/evolve_space/evolve}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

die() { echo "$*" >&2; exit 2; }
note() { echo "# $*"; }

# Print a command; run it unless this is a dry run.
run() {
    printf '%q ' "$@"; echo
    [ "$DRY_RUN" = 1 ] || "$@"
}

# Look up a roster label: sets R_CLI, R_MODEL, R_ENV (a path or empty), R_EFFORTS.
roster() {
    local label="$1" entry l cli model env efforts
    for entry in "${ROSTER[@]}"; do
        IFS='|' read -r l cli model env efforts <<<"$entry"
        if [ "$l" = "$label" ]; then
            R_CLI="$cli"; R_MODEL="$model"; R_EFFORTS="$efforts"
            R_ENV="${env:+$SECRETS_DIR/$env}"
            return 0
        fi
    done
    die "no model '$label' in the roster (experiments/config.sh)"
}

roster_labels() {
    local entry
    for entry in "${ROSTER[@]}"; do echo "${entry%%|*}"; done
}

# A model reached through an API key needs its credentials file before anything
# starts; a dry run only says it is missing.
require_credentials() {
    local label="$1"
    roster "$label"
    [ -z "$R_ENV" ] || [ -f "$R_ENV" ] && return 0
    [ "$DRY_RUN" = 1 ] && { note "missing credentials for $label: $R_ENV"; return 0; }
    die "$label needs its credentials file $R_ENV (KEY=VALUE, see serverbench/artifacts/README.md)"
}

# Queue evolves of one agent configuration on every task given, detached, through
# evaluation_progress/watch_evolve.py. Each run records its own result row.
#   queue <campaign> <label> <effort> <hours> <task>... [-- <agent env override>]
queue() {
    local campaign="$1" label="$2" effort="$3" hours="$4"; shift 4
    local tasks=() env_file=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --) env_file="$2"; shift 2 ;;
            *) tasks+=("$1"); shift ;;
        esac
    done
    roster "$label"
    env_file="${env_file:-$R_ENV}"
    local cmd=("$PY" "$REPO/evaluation_progress/watch_evolve.py" --campaign "$campaign"
               --agent-cli "$R_CLI" --model "$R_MODEL" --effort "$effort" --hours "$hours")
    local t
    for t in "${tasks[@]}"; do cmd+=(--task "$t"); done
    [ -n "$env_file" ] && cmd+=(--agent-env "$env_file")
    [ -n "$GPUS" ] && cmd+=(--gpus "$GPUS")
    local state="$REPO/serverbench/artifacts/operations/$campaign"
    printf '%q ' "${cmd[@]}"; echo
    note "detached; queue state and logs in ${state#"$REPO"/}/"
    [ "$DRY_RUN" = 1 ] && return 0
    mkdir -p "$state"
    (cd "$REPO" && setsid nohup "${cmd[@]}" </dev/null >"$state/launcher.out" 2>&1 &)
}

# The production launcher's helpers: pick_gpus, resolve_model_dir,
# agent_environment_for, devices_for_task.
load_launcher_helpers() {
    . "$REPO/serverbench/scripts/server_evolve/evolve-common.sh"
}

# Cards for one measurement: GPU if set, else as many free ones as the task is
# billed for. A dry run names the first cards without sampling occupancy.
cards_for() {
    local task="$1" want
    want="$(devices_for_task "$REPO/serving_tasks/$task")"
    if [ -n "${GPU:-}" ] && [ "$GPU" != auto ]; then echo "$GPU"; return 0; fi
    if [ "$DRY_RUN" = 1 ]; then seq -s, 0 $((want - 1)); return 0; fi
    pick_gpus "$want" || die "no $want free GPU(s) for $task; nothing was started"
}

# The task's pinned model snapshot in the local Hugging Face cache. A dry run on
# a machine without it shows a placeholder.
model_dir_for() {
    local task="$1"
    [ "$DRY_RUN" = 1 ] || { resolve_model_dir "$REPO/serving_tasks/$task"; return; }
    resolve_model_dir "$REPO/serving_tasks/$task" 2>/dev/null || echo "<snapshot of $task's model>"
}
