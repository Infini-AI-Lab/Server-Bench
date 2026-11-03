#!/usr/bin/env bash
# A3b, prospective budget: fresh evolves given less time, compared with the
# HOURS-long runs of the main sweep.
#
# BUDGETS lists hours:replicates pairs (default "4:2 2:1": two 4 h runs and one
# 2 h run per model).
#
#   bash experiments/ablation_budget_prospective.sh
#   DRY_RUN=1 bash experiments/ablation_budget_prospective.sh
#   BUDGETS="16:1" MODELS=sonnet-5-5 bash experiments/ablation_budget_prospective.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODELS="${MODELS:-sonnet-5-5 glm-5.3}"
TASKS="${TASKS:-gemma4_e4b_it_standard_1xL40}"
BUDGETS="${BUDGETS:-4:2 2:1}"

for label in $MODELS; do require_credentials "$label"; done
for label in $MODELS; do
    roster "$label"
    effort="${EFFORT:-${R_EFFORTS%% *}}"
    for budget in $BUDGETS; do
        hours="${budget%%:*}" replicates="${budget#*:}"
        for r in $(seq 1 "$replicates"); do
            # shellcheck disable=SC2086
            queue "a3b-$label-$effort-${hours}h-r$r-$STAMP" "$label" "$effort" "$hours" $TASKS
        done
    done
done
