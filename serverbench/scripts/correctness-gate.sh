#!/usr/bin/env bash
# The CPU-only correctness gate for the specialized evolve path.
#
# One entry point, no device, no model, no network. It runs every check that can
# be run without spending a GPU-hour, and prints
#
#     SERVING_EVOLVE_SPECIALIZED_CORRECTNESS_READY
#
# only when all of them hold. The token is the point: "the tests passed" is a
# claim somebody has to go and verify, and a run that is about to spend a device
# hour should be able to check one string instead.
#
# It fails on the FIRST failing check and says which, because a gate that runs
# everything and prints a summary invites reading the summary.
#
#   bash serverbench/scripts/correctness-gate.sh          # gate
#   bash serverbench/scripts/correctness-gate.sh --quick  # skip the full suite
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVERBENCH="$(dirname "$HERE")"
REPO="$(dirname "$SERVERBENCH")"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# THIS checkout's interpreter and lint, never whatever is on PATH: the harness
# pins ruff>=0.16, the first release that reports syntax needing a newer
# interpreter than `target-version`, so only that ruff's green is evidence the
# tree loads on 3.10.
. "$SERVERBENCH/scripts/env.sh"
PY_BIN="$SERVERBENCH_PY"

fail() { printf '\nNOT READY: %s\n' "$1" >&2; exit 1; }
step() { printf '  %-46s' "$1"; }
ok()   { printf 'ok%s\n' "${1:+  $1}"; }

[ -x "$PY_BIN" ] || { printf '\nNOT READY: no interpreter at %s\n' "$PY_BIN" >&2; exit 1; }
"$PY_BIN" -m ruff --version >/dev/null 2>&1 || {
    printf '\nNOT READY: the venv has no ruff; ruff from PATH is not a substitute.\n' >&2; exit 1; }

printf 'serving-evolve specialized correctness gate\n'
printf '  repo   %s\n\n' "$REPO"

# ---------------------------------------------------------------- 1. style ---
step "ruff (harness, driver, contract tests)"
if ! out=$(cd "$SERVERBENCH" && "$PY_BIN" -m ruff check harness/serving_evolve scripts/server_evolve \
            tests/contracts 2>&1); then
    printf 'FAILED\n%s\n' "$out" >&2
    fail "ruff. A lint error in this path is usually an unfinished edit."
fi
ok

# ------------------------------------------------------- 2. it must import ---
# Discovered, not listed, so a renamed or removed module cannot leave the
# gate checking a list that no longer matches the tree.
step "every module imports"
if ! out=$(cd "$SERVERBENCH/harness" && "$PY_BIN" -c '
import importlib
import pkgutil, sys
bad = []
for pkg in ("serving_evolve", "kernel_evolve", "lifecycle"):
    root = importlib.import_module(pkg)
    for found in pkgutil.walk_packages(root.__path__, pkg + "."):
        try:
            importlib.import_module(found.name)
        except Exception as exc:
            bad.append("%s: %s: %s" % (found.name, type(exc).__name__, exc))
if bad:
    print("\n".join(bad)); sys.exit(1)
print("ok")' 2>&1); then
    printf 'FAILED\n%s\n' "$out" >&2
    fail "an import. Nothing downstream can run."
fi
ok

# ------------------------------------------------------ 3. the path itself ---
# End to end on artifacts this repository actually contains. A unit test can
# pass against a shape nobody writes; this cannot.
#
# The measurement and verification path: scenario offer,
# aggregate read, gate judgement, verdict separation, and the packaged verdict
# being byte-identical to its source.
step "end to end on the measurement path"
if ! out=$(cd "$SERVERBENCH/tests" \
        && "$PY_BIN" -m pytest measurement/ verification/ -q 2>&1); then
    printf 'FAILED\n%s\n' "$(printf '%s' "$out" | tail -25)" >&2
    fail "the end-to-end path. See above for the first check that broke."
fi
ok "$(printf '%s' "$out" | tail -1 | tr -d '\n')"

# The run's SPINE, executed rather than read: `tests/evolve/` drives what the
# agent is told, the run journal and the request file. Kept out of the --quick
# skip on purpose: this is the part a run cannot afford to have wrong.
step "the run spine (capture, verdict, publish)"
if ! out=$(cd "$SERVERBENCH/tests" && "$PY_BIN" -m pytest evolve/ -q 2>&1); then
    printf 'FAILED\n%s\n' "$(printf '%s' "$out" | tail -25)" >&2
    fail "the spine. A candidate cannot become a version."
fi
ok "$(printf '%s' "$out" | tail -1 | tr -d '\n')"

# ------------------------------------------------------------- 4. the suite --
if [ "$QUICK" = 1 ]; then
    step "full test suite"; ok "SKIPPED (--quick)"
else
    step "contract suite"
    if ! out=$(cd "$SERVERBENCH/tests" && "$PY_BIN" -m pytest contracts/ -q 2>&1); then
        printf 'FAILED\n%s\n' "$(printf '%s' "$out" | tail -25)" >&2
        fail "the contract suite."
    fi
    ok "$(printf '%s' "$out" | tail -1 | tr -d '\n')"

    step "full test suite"
    if ! out=$(cd "$SERVERBENCH/tests" && "$PY_BIN" -m pytest -q 2>&1); then
        printf 'FAILED\n%s\n' "$(printf '%s' "$out" | tail -25)" >&2
        fail "the full suite."
    fi
    ok "$(printf '%s' "$out" | tail -1 | tr -d '\n')"
fi

printf '\nSERVING_EVOLVE_SPECIALIZED_CORRECTNESS_READY\n'
