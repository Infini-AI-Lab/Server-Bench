#!/usr/bin/env bash
# A1, repeated construction runs and reasoning effort: is a model's score
# robust across independent runs, and which effort setting is best?
#
# Every model runs every effort its roster entry lists, REPLICATES times
# (default 1), on TASKS. With the main sweep's runs this gives two independent
# runs at each model's first effort and one at every other effort.
#
#   bash experiments/ablation_repeats.sh
#   DRY_RUN=1 bash experiments/ablation_repeats.sh
#   MODELS=opus-5-5 EFFORTS=max REPLICATES=2 bash experiments/ablation_repeats.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODELS="${MODELS:-$(roster_labels | tr '\n' ' ')}"
TASKS="${TASKS:-qwen3_5_9b_standard_2xL40 gemma4_e4b_it_standard_1xL40}"
REPLICATES="${REPLICATES:-1}"

for label in $MODELS; do require_credentials "$label"; done
for label in $MODELS; do
    roster "$label"
    for effort in ${EFFORTS:-$R_EFFORTS}; do
        for r in $(seq 1 "$REPLICATES"); do
            # shellcheck disable=SC2086
            queue "a1-$label-$effort-r$r-$STAMP" "$label" "$effort" "$HOURS" $TASKS
        done
    done
done
