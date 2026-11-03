#!/usr/bin/env bash
# Evolve a task on Modal's GPUs, in our own image.
#
#   bash serverbench/scripts/server_evolve/modal-evolve.sh \
#       gemma4_e4b_it_standard_modal_1xL40S --hours 1
#
# The agent, its workspace, its server and the cards are all in one Modal
# sandbox -- the same arrangement as the docker path, on someone else's
# hardware. The driver stays here and orchestrates.
#
# WHAT THIS DOES NOT DO: pick a local GPU. The task's id says which card and
# how many (`_1xL40S` -> L40S:1) and Modal allocates them, so `GPU` is a
# placeholder that satisfies the entrypoint's arity check and nothing more.
# `claim_devices` and the holder processes are skipped for this backend --
# claiming local indices would reserve a neighbour's cards on THIS host for a
# run that never touches them.
#
# The provider key goes to the SANDBOX and nowhere else: the driver reads it
# from the file named below and hands it to Modal as a per-sandbox Secret.
# `env -u` removes both ANTHROPIC variables from this shell, because an API key
# in the environment takes precedence over an OAuth token and must never be
# what an opus run authenticates with.
set -u

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO"

TASK="${1:?usage: modal-evolve.sh <task> [--hours N] [entrypoint args...]}"
shift

HOURS="1"
if [ "${1:-}" = "--hours" ]; then HOURS="$2"; shift 2; fi

AGENT_ENV="${AGENT_ENV:-$REPO/serverbench/artifacts/secrets/minimax-m3-fireworks.env}"
MODEL="${MODEL:-accounts/fireworks/models/minimax-m3}"
MODAL_PY="${MODAL_PY:-$REPO/harbor/.venv/bin/python}"

[ -d "$REPO/serving_tasks/$TASK" ] || { echo "no such task: $TASK" >&2; exit 2; }
[ -f "$AGENT_ENV" ] || { echo "no provider credentials at $AGENT_ENV" >&2; exit 2; }
[ -x "$MODAL_PY" ] || { echo "no interpreter with modal at $MODAL_PY" >&2; exit 2; }

# Refuse before anything is allocated. Each of these arrives much later and
# much more expensively if it is left to the run to discover.
"$MODAL_PY" -c "import modal" 2>/dev/null || {
    echo "$MODAL_PY cannot import modal" >&2; exit 2; }
timeout 120 "$MODAL_PY" -m modal app list >/dev/null 2>&1 || {
    echo "modal is not authenticated: run 'modal token set ...'" >&2; exit 2; }
# BUILD WHAT IS MISSING, rather than refusing because the last run cleaned up
# after itself. Every stage is idempotent and verifies by performing the real
# operation, so this is a no-op when the volumes are already correct and a
# rebuild when they are not.
timeout 3600 "$MODAL_PY" "$REPO/serving_tasks/_modal/modalctl.py" \
    setup --task "$TASK" || {
    echo "could not build the Modal environment for $TASK (see above)" >&2
    exit 2; }
timeout 900 "$MODAL_PY" "$REPO/serving_tasks/_modal/modalctl.py" \
    check --task "$TASK" || {
    echo "the task is not ready for a Modal run (see above)" >&2; exit 2; }

# The device count the task declares; the entrypoint checks arity against it.
DEVICES="$("$MODAL_PY" - "$TASK" <<'PY'
import sys, pathlib, yaml
task = sys.argv[1]
spec = yaml.safe_load(
    (pathlib.Path("serving_tasks") / task / "contract.yaml").read_text("utf-8"))
print(int(spec["devices"]))
PY
)"
PLACEHOLDER="$(seq -s, 0 $((DEVICES - 1)))"

printf '%s %s on Modal, %s h, %s (%s device placeholder)\n' \
    "$(date -u '+%FT%TZ')" "$TASK" "$HOURS" "$MODEL" "$PLACEHOLDER"

env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
    SERVERBENCH_BACKEND=modal \
    SERVERBENCH_MODAL_PY="$MODAL_PY" \
    GPU="$PLACEHOLDER" TASK_ROOT="$REPO/serving_tasks" \
    AGENT_ENV="$AGENT_ENV" AGENT_MODEL="$MODEL" \
    bash serverbench/scripts/server_evolve/evolve-task.sh "$TASK" \
        --hours "$HOURS" --stop timebox "$@"
rc=$?
printf '%s run finished rc=%s\n' "$(date -u '+%FT%TZ')" "$rc"

# The lifecycle and verifier release their own recorded sandboxes. Never sweep
# the whole app: another evolve can be running there. Shared storage remains
# billed until the operator explicitly requests cleanup. Even then, purge must
# refuse live sandboxes and must not fall back to terminating somebody's run.
if [ "${SERVERBENCH_KEEP_MODAL_STORAGE:-1}" = "1" ]; then
    printf 'keeping the volumes: SERVERBENCH_KEEP_MODAL_STORAGE=1\n'
else
    timeout 600 "$MODAL_PY" "$REPO/serving_tasks/_modal/modalctl.py" \
        purge --yes || {
        printf 'purge refused or failed; other runs and their sandboxes are left untouched\n' >&2
    }
fi
exit "$rc"
