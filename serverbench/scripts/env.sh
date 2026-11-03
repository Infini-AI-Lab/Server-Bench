#!/usr/bin/env bash
# The two interpreters this project runs on. Source this; do not guess.
#
#   . "$REPO/serverbench/scripts/env.sh"
#   "$SERVERBENCH_PY"          -m pytest serverbench/tests -q
#   "$SERVERBENCH_PY"          -m final_validate.concurrency --task <T> ...
#
# WHY THERE ARE TWO
#
#   SERVERBENCH_PY           the HARNESS: stdlib + yaml + pytest + ruff + the
#                            pinned AgentX, and deliberately no torch or
#                            transformers: the harness core must import inside a
#                            bare image, and `tests/repo` fails a module-scope
#                            import of either.
#   SERVERBENCH_BASELINE_PY  MEASUREMENT: transformers for the tokenizer, httpx
#                            for the client. Also no torch -- SGLang, vLLM and
#                            TensorRT-LLM run in their own container images, so
#                            the orchestrator never imports a framework. Defaults
#                            to SERVERBENCH_PY.
#
# There is no fallback to `python3` from PATH: an interpreter below the 3.10
# floor fails much later and confusingly, so this resolves or it refuses.
# `serverbench/scripts/deploy/00-envs.sh` creates the checkout-local venv.
#
# Resolution order, first hit wins:
#   1. $SERVERBENCH_PY / $SERVERBENCH_BASELINE_PY   -- an operator override
#   2. serverbench/scripts/env.local.sh             -- this machine, gitignored
#   3. $REPO/serverbench/.venv                      -- a checkout-local venv
set -u

_env_sh_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVERBENCH_ROOT="$(dirname "$_env_sh_dir")"
SERVERBENCH_REPO="$(dirname "$SERVERBENCH_ROOT")"

# 2. host-local override, if the machine has one.
[ -f "$_env_sh_dir/env.local.sh" ] && . "$_env_sh_dir/env.local.sh"

# 3. checkout-local environments. A shared or fleet venv belongs in
#    env.local.sh; committed code must not name one person's checkout.
: "${SERVERBENCH_PY:=$SERVERBENCH_ROOT/.venv/bin/python}"
: "${SERVERBENCH_BASELINE_PY:=$SERVERBENCH_PY}"

serverbench_env_refuse() {
    printf '\nthis project does not run on the interpreter from PATH.\n  %s\n\n' "$1" >&2
    printf 'Set SERVERBENCH_PY (and SERVERBENCH_BASELINE_PY) or write\n' >&2
    printf '  %s/env.local.sh\n naming this machine'"'"'s environments.\n' "$_env_sh_dir" >&2
    exit 1
}

[ -x "$SERVERBENCH_PY" ] || serverbench_env_refuse "no harness interpreter at $SERVERBENCH_PY"
[ -x "$SERVERBENCH_BASELINE_PY" ] || \
    serverbench_env_refuse "no baseline interpreter at $SERVERBENCH_BASELINE_PY"

# The floor, checked rather than assumed: cheap here, expensive inside a run.
"$SERVERBENCH_PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' \
    || serverbench_env_refuse "$SERVERBENCH_PY is older than the 3.10 floor the harness promises"

export SERVERBENCH_PY SERVERBENCH_BASELINE_PY SERVERBENCH_ROOT SERVERBENCH_REPO
# A shared virtualenv may contain editable-install metadata for another clone.
# Put this checkout first so importing lifecycle/final_validate always reads
# the code beside the script the operator sourced. `harness/` is the one
# package root: lifecycle, serving_evolve, kernel_evolve and final_validate.
_serverbench_pythonpath="$SERVERBENCH_ROOT/harness:$SERVERBENCH_REPO"
export PYTHONPATH="$_serverbench_pythonpath${PYTHONPATH:+:$PYTHONPATH}"

# NOTHING here exports a generic `PY`: the two interpreters are not
# interchangeable. A caller says which one it wants, by name:
#     PY="$SERVERBENCH_PY"           # harness, tests, gates
#     PY="$SERVERBENCH_BASELINE_PY"  # the measurement path

# ---------------------------------------------------------------------------
# THE DEPLOYMENT-LOCAL GPU FENCE.
#
# Cards the lease (`final_validate/lease.py`) never takes on this machine. A
# machine-specific index does not belong in library code, so it lives here and
# defaults to empty. Set it to a comma list to fence cards that are not ours:
#
#     export SERVERBENCH_EXCLUDED_GPUS=0,1
#
# `SERVERBENCH_LEASE_DENY` composes with it for a single run.
: "${SERVERBENCH_EXCLUDED_GPUS:=}"
export SERVERBENCH_EXCLUDED_GPUS
