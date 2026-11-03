#!/usr/bin/env bash
# Evolve a serving system for one task.
#
#   bash serverbench/scripts/server_evolve/evolve-task.sh qwen3_5_2b_standard_1xL40
#   bash serverbench/scripts/server_evolve/evolve-task.sh 769bf633 --hours 4
#
# A task resolves by its full name (any case) or its hashed id (the
# `task_hash` in its contract.yaml); `--help` lists them all.
#
# Anything after the task goes to the entrypoint; `--help` lists it all.
#
#   --hours 4                 a shorter budget
#   --stop beat-all-baselines a label recorded on the run (default: timebox);
#                             --hours is the only bound the driver enforces
#   --dry-run                 build and boot the workspace, launch no agent,
#                             and consume no production run id
#
# You do not set a version cadence. The agent asks for a version with
# `./version "why"` (`./checkpoint` on instructed tasks) whenever it chooses, and the harness
# freezes the tree at that moment -- a round is a unit of the agent's work, not
# a number of minutes.
#
# Configuration that belongs to the EXPERIMENT rather than to the invocation
# lives in evolve-common.sh, and every value used is recorded in the run's
# journal.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/evolve-common.sh"

usage() {
    echo "usage: evolve-task.sh <task> [entrypoint args...]" >&2
    echo "" >&2
    echo "tasks:" >&2
    for d in "$REPO"/serving_tasks/*/; do
        [ -f "$d/contract.yaml" ] || continue
        n="$(basename "$d")"
        printf '  %-8s  %-38s %s\n' "$(task_hash_of "$n")" "$n" "$(task_title "$n")" >&2
    done
}

# The task's own one-line title, read from the task rather than copied beside it.
task_title() {
    "$PY" - "$REPO/serving_tasks/$1" <<'PYEOF' 2>/dev/null || true
import sys, pathlib, yaml
spec = yaml.safe_load((pathlib.Path(sys.argv[1]) / "contract.yaml").read_text("utf-8")) or {}
print(spec.get("title") or "")
PYEOF
}

# The hashed id `contract.yaml` states (`final_validate.spine.task_hash`).
task_hash_of() {
    sed -n 's/^task_hash: *//p' "$REPO/serving_tasks/$1/contract.yaml" 2>/dev/null
}

# Accept the full name in any case, or the task's hash -- one task each way.
resolve_task() {
    local want="$1" hits=()
    if [ -f "$REPO/serving_tasks/$want/contract.yaml" ]; then echo "$want"; return 0; fi
    for d in "$REPO"/serving_tasks/*/; do
        [ -f "$d/contract.yaml" ] || continue
        local n; n="$(basename "$d")"
        if [ "${n,,}" = "${want,,}" ] || [ "$(task_hash_of "$n")" = "${want,,}" ]; then
            hits+=("$n")
        fi
    done
    case "${#hits[@]}" in
        1) echo "${hits[0]}"; return 0 ;;
        0) echo "no task matches '$want'" >&2; return 2 ;;
        *) echo "'$want' matches ${hits[*]}" >&2; return 2 ;;
    esac
}

[ $# -ge 1 ] || { usage; exit 2; }
case "$1" in -h|--help|help) usage; exit 0 ;; esac

TASK="$(resolve_task "$1")" || { usage; exit 2; }
shift
title="$(task_title "$TASK")"
[ -n "$title" ] && echo "task      $TASK — $title"
launch_task "$TASK" "$@"
