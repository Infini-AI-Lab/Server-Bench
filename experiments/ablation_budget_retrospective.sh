#!/usr/bin/env bash
# A3a, retrospective budget: how good would each run's system have been had its
# budget ended at CUTOFFS hours (default 2,4,6)? Each run's accepted head at
# each cutoff is measured again through the final verdict: a full 1,800 s replay
# on a fresh odd seed plus the private correctness stage. No agent runs.
#
# RUNS names run roots; by default every run of TASKS at EFFORTS (default max)
# whose lineage has a final verdict. A cutoff before the first version is a
# recorded zero and needs no measurement; a version that is the head at several
# cutoffs is measured once, and the run's final head keeps its final verdict.
#
#   bash experiments/ablation_budget_retrospective.sh
#   DRY_RUN=1 bash experiments/ablation_budget_retrospective.sh
#   RUNS="serverbench/evolve_space/evolve/<task>/<run-id>" GPU=2 bash experiments/ablation_budget_retrospective.sh
#
# The version is measured from a copy under serverbench/artifacts/budget-versions/
# so the verdict never replaces its lineage's FINAL.json; the evidence lands
# under serverbench/artifacts/final_validation/budget/. Measurements run one
# after another in the foreground. Read them with experiments/analyse.sh.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_launcher_helpers

TASKS="${TASKS:-qwen3_5_9b_standard_2xL40 gemma4_e4b_it_standard_1xL40}"
EFFORTS="${EFFORTS:-max}"
CUTOFFS="${CUTOFFS:-2,4,6}"
COPIES="$REPO/serverbench/artifacts/budget-versions"
EVIDENCE="$REPO/serverbench/artifacts/final_validation/budget"

default_runs() {
    "$PY" - "$EVOLVE_ROOT" "$TASKS" "$EFFORTS" <<'PYEOF'
import json, sys
from pathlib import Path
root, tasks, efforts = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3].split()
for task in tasks:
    for run in sorted((root / task.lower()).glob("*/journal.jsonl")):
        events = [json.loads(l) for l in run.read_text().splitlines() if l.strip()]
        created = next((e for e in events if e.get("event") == "run_created"), {})
        dirs = [e["directory"] for e in events if e.get("event") == "version_registered" and e.get("directory")]
        if created.get("agent_effort") in efforts and dirs and (Path(dirs[-1]).parent / "FINAL.json").is_file():
            print(run.parent)
PYEOF
}

final_head() {
    "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("head_version_dir", ""))' \
        "$1/FINAL.json" 2>/dev/null || true
}

RUNS="${RUNS:-$(default_runs | tr '\n' ' ')}"
[ -n "${RUNS// /}" ] || { note "no finished run of $TASKS at effort $EFFORTS under ${EVOLVE_ROOT#"$REPO"/}"; exit 0; }

for run_root in $RUNS; do
    run_root="$(cd "$run_root" && pwd)"
    task="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_id"])' "$run_root/identity.json")"
    task_dir="$REPO/serving_tasks/$task"
    IMAGE="" ENV_INPUT=""
    agent_environment_for "$task_dir"
    model_dir="$(model_dir_for "$task")"
    seen=" "
    while IFS=$'\t' read -r cutoff version; do
        if [ "$version" = - ]; then
            note "$(basename "$run_root") at ${cutoff} h: no version published yet (scores 0)"
            continue
        fi
        case "$seen" in *" $version "*) continue ;; esac
        seen="$seen$version "
        lineage="$(basename "$(dirname "$version")")"
        if [ "$(basename "$version")" = "$(final_head "$(dirname "$version")")" ]; then
            note "$(basename "$run_root") head at ${cutoff} h is its final head; its final verdict stands"
            continue
        fi
        copy="$COPIES/$lineage/$(basename "$version")"
        evidence="$EVIDENCE/$lineage/$(basename "$version")"
        if [ -f "$evidence/MEASURED.json" ]; then
            note "$(basename "$version") already measured: ${evidence#"$REPO"/}"
            continue
        fi
        note "$(basename "$run_root") head at ${cutoff} h: $(basename "$version")"
        [ -d "$copy" ] || { run mkdir -p "$(dirname "$copy")"; run cp -a "$version" "$copy"; }
        cmd=("$PY" "$REPO/serverbench/harness/final_validate/tools/measure_published.py"
             --task "$task" --version "$copy" --image "$IMAGE" --model-dir "$model_dir"
             --arm candidate --run-dir "$evidence")
        [ -n "${GPU:-}" ] && [ "$GPU" != auto ] && cmd+=(--gpu "$GPU")
        [ -n "$EXCLUDE_GPUS" ] && cmd+=(--deny "$EXCLUDE_GPUS")
        run "${cmd[@]}"
    done < <("$PY" "$EXPERIMENTS/analysis/budget.py" versions --run "$run_root" --hours "$CUTOFFS")
done
