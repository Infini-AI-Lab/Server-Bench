#!/usr/bin/env bash
# Reference arms: where each baseline framework's operating point is on a task,
# and its value and p90s over several random seeds there. The worst p90 of the
# SGLang seed runs is the rule a task's SGLang-derived bars follow.
#
# For each task in TASKS and arm in ARMS, on the task's own hardware:
#   search  every offered load in SEARCH_POINTS, one 1,800 s window each
#   seeds   SEEDS runs at the operating point, each on a fresh development seed
# STAGE picks one of them (default: both). The operating point is CONCURRENCY
# if set, else the task's adopted SGLang point (reference.yaml) for sglang,
# else the search's recommendation for any other arm.
#
#   bash experiments/reference_bars.sh
#   DRY_RUN=1 bash experiments/reference_bars.sh
#   TASKS=gemma4_e4b_it_standard_1xL40 ARMS=vllm GPU=3 bash experiments/reference_bars.sh
#
# Measurements run one after another in the foreground; run disjoint TASKS in
# separate invocations, each on its own GPU, to go faster. Evidence lands under
# OUT_ROOT/<task>/<arm>/; read it with experiments/analysis/reference_bars.py.
# Adopting a bar or an operating point is a task change: `taskctl set`.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_launcher_helpers

TASKS="${TASKS:-qwen3_5_2b_standard_1xL40 qwen3_5_9b_standard_2xL40 gemma4_e4b_it_standard_1xL40 qwen3_8_27b_int4_standard_2xL40}"
ARMS="${ARMS:-sglang vllm}"
SEARCH_POINTS="${SEARCH_POINTS:-1,2,4,8,16}"
SEEDS="${SEEDS:-5}"
STAGE="${STAGE:-all}"
OUT_ROOT="${OUT_ROOT:-$REPO/serverbench/artifacts/concurrency/reference}"

adopted_point() {
    "$PY" - "$REPO/serving_tasks/$1/reference.yaml" <<'PYEOF'
import sys, yaml
spec = yaml.safe_load(open(sys.argv[1])) or {}
print((spec.get("concurrency_search") or {}).get("adopted") or "")
PYEOF
}

recommended_point() {
    "$PY" - "$1" <<'PYEOF'
import json, sys
try:
    print((json.load(open(sys.argv[1])).get("selection") or {}).get("recommended") or "")
except (OSError, ValueError):
    print("")
PYEOF
}

measure() {
    local task="$1" arm="$2" points="$3" out="$4" model_dir cards
    model_dir="$(model_dir_for "$task")"
    cards="$(cards_for "$task")"
    run "$PY" -m final_validate.concurrency --task "$task" --arm "$arm" \
        --model-dir "$model_dir" --gpu "$cards" --points "$points" --out-dir "$out"
}

for task in $TASKS; do
    [ -f "$REPO/serving_tasks/$task/contract.yaml" ] || die "no task $task"
    for arm in $ARMS; do
        out="$OUT_ROOT/$task/$arm"
        if [ "$STAGE" = all ] || [ "$STAGE" = search ]; then
            note "$task $arm: search over $SEARCH_POINTS"
            measure "$task" "$arm" "$SEARCH_POINTS" "$out/search"
        fi
        if [ "$STAGE" = all ] || [ "$STAGE" = seeds ]; then
            point="${CONCURRENCY:-}"
            [ -n "$point" ] || [ "$arm" != sglang ] || point="$(adopted_point "$task")"
            [ -n "$point" ] || point="$(recommended_point "$out/search/calibration.json")"
            if [ -z "$point" ]; then
                [ "$DRY_RUN" = 1 ] || die "$task $arm: no operating point; run STAGE=search or set CONCURRENCY"
                point=1
                note "$task $arm: the point comes from the search's recommendation (shown as 1)"
            fi
            note "$task $arm: $SEEDS seed runs at concurrency $point"
            for k in $(seq 1 "$SEEDS"); do
                measure "$task" "$arm" "$point" "$out/slo-c$point-s$k"
            done
        fi
    done
done
