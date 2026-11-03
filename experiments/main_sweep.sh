#!/usr/bin/env bash
# Main comparison: every agent model in the roster evolves a server for every
# sweep task, at its first roster effort, for HOURS (default 8).
#
#   bash experiments/main_sweep.sh                 # queue everything
#   DRY_RUN=1 bash experiments/main_sweep.sh       # print the commands only
#   MODELS="opus-5-5 glm-5.3" TASKS=gemma4_e4b_it_standard_1xL40 bash experiments/main_sweep.sh
#
# One detached watcher per model queues its tasks on free cards; each run is
# recorded in evaluation_progress/<task>.csv when its verdict lands.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODELS="${MODELS:-$(roster_labels | tr '\n' ' ')}"
TASKS="${TASKS:-$SWEEP_TASKS}"

for label in $MODELS; do require_credentials "$label"; done
for label in $MODELS; do
    roster "$label"
    effort="${EFFORT:-${R_EFFORTS%% *}}"
    # shellcheck disable=SC2086
    queue "main-$label-$effort-${HOURS}h-$STAMP" "$label" "$effort" "$HOURS" $TASKS
done
