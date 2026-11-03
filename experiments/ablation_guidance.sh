#!/usr/bin/env bash
# A2, task guidance: does a richer brief change what agents build and how often
# it is eligible? The standard and instructed tasks share one contract (bars,
# workload, prices, window, correctness); only the brief differs.
#
#   bash experiments/ablation_guidance.sh
#   DRY_RUN=1 bash experiments/ablation_guidance.sh
#   ARMS=instructed bash experiments/ablation_guidance.sh     # reuse sweep runs as the control
#   STANDARD_TASK=qwen3_5_9b_standard_2xL40 INSTRUCTED_TASK=qwen3_5_9b_instructed_2xL40 bash ...
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODELS="${MODELS:-sonnet-5-5 glm-5.3}"
ARMS="${ARMS:-standard instructed}"
STANDARD_TASK="${STANDARD_TASK:-gemma4_e4b_it_standard_1xL40}"
INSTRUCTED_TASK="${INSTRUCTED_TASK:-gemma4_e4b_it_instructed_1xL40}"
REPLICATES="${REPLICATES:-2}"

tasks=()
for arm in $ARMS; do
    case "$arm" in
        standard) tasks+=("$STANDARD_TASK") ;;
        instructed) tasks+=("$INSTRUCTED_TASK") ;;
        *) die "ARMS takes standard and/or instructed, got '$arm'" ;;
    esac
done

for label in $MODELS; do require_credentials "$label"; done
for label in $MODELS; do
    roster "$label"
    effort="${EFFORT:-${R_EFFORTS%% *}}"
    for r in $(seq 1 "$REPLICATES"); do
        queue "a2-$label-$effort-r$r-$STAMP" "$label" "$effort" "$HOURS" "${tasks[@]}"
    done
done
