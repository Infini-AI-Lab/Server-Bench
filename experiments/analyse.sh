#!/usr/bin/env bash
# Analyse every run on this machine, CPU only: re-score each verdict, read each
# run's development trajectory, and write the tables to OUT (default
# experiments/analysis/out/, never committed).
#
#   bash experiments/analyse.sh
#   DRY_RUN=1 bash experiments/analyse.sh
#   OUT=<dir> EVOLVE_ROOT=<run roots> SERVER_OUTPUT=<lineages> bash experiments/analyse.sh
#
# Other inputs: REMEASURED (re-measured verdicts, default
# serverbench/artifacts/final_validation/budget), REFERENCE_DIR (reference seed
# runs, default serverbench/artifacts/concurrency/reference), CUTOFFS (2,4,6).
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

OUT="${OUT:-$EXPERIMENTS/analysis/out}"
SERVER_OUTPUT="${SERVER_OUTPUT:-$REPO/server_output}"
REMEASURED="${REMEASURED:-$REPO/serverbench/artifacts/final_validation/budget}"
REFERENCE_DIR="${REFERENCE_DIR:-$REPO/serverbench/artifacts/concurrency/reference}"
CUTOFFS="${CUTOFFS:-2,4,6}"
A="$EXPERIMENTS/analysis"

# Each later step reads final_rescored.csv; a step that finds nothing to say
# does not stop the others.
run "$PY" "$A/rescore_finals.py" --evolve "$EVOLVE_ROOT" --server-output "$SERVER_OUTPUT" \
    --remeasured "$REMEASURED" --out "$OUT"
run "$PY" "$A/dev_trajectories.py" --evolve "$EVOLVE_ROOT" --out "$OUT" || true
run "$PY" "$A/run_features.py" --evolve "$EVOLVE_ROOT" --out "$OUT" || true
run "$PY" "$A/component_timeline.py" --evolve "$EVOLVE_ROOT" --out "$OUT" || true
run "$PY" "$A/design_scan.py" --server-output "$SERVER_OUTPUT" --out "$OUT" || true
run "$PY" "$A/dev_vs_verdict.py" --out "$OUT" || true
run "$PY" "$A/budget.py" curve --hours "$CUTOFFS" --evolve "$EVOLVE_ROOT" --out "$OUT" || true
run "$PY" "$A/reference_bars.py" --dir "$REFERENCE_DIR" --out "$OUT" || true
