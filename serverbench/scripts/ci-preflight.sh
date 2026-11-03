#!/usr/bin/env bash
# Run CI's checks against what a push would actually carry.
#
# Tests pass here and fail in CI whenever one of them reads a file this machine
# has and a checkout does not: traces, references and other generated data are
# large, host-dependent and regenerated per machine, so they are deliberately
# not in Git.
#
# So this assembles a tree from `git ls-files` plus untracked-but-not-ignored --
# precisely the set a commit would contain -- and runs CI's three checks in it.
# Anything depending on generated data fails HERE, before the push.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/ci-preflight-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
LIST="$WORK/files.txt"
{ git ls-files; git ls-files --others --exclude-standard; } | sort -u > "$LIST"
echo "a push would carry $(wc -l < "$LIST") file(s)"

mkdir -p "$WORK/tree"
tar cf - -T "$LIST" | (cd "$WORK/tree" && tar xf -)

# CI checks out a repository, and at least one test asks Git what it ignores.
# A bare directory of files would fail that for a reason unrelated to the change.
( cd "$WORK/tree" && git init -q . && git add -A >/dev/null 2>&1 \
  && git -c user.email=ci@local -c user.name=ci commit -qm preflight >/dev/null 2>&1 )

cd "$WORK/tree/serverbench"
status=0

# CI installs the package before it lints or type-checks, and that changes what
# `ty` resolves. So the preflight installs too -- into a throwaway venv, because
# an editable install into the ambient environment would point it at $WORK,
# which is deleted on exit.
echo "== install (as CI runs it, in an isolated venv) =="
python3 -m venv "$WORK/venv"
PY="$WORK/venv/bin/python"
"$PY" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
"$PY" -m pip install -q -e '.[dev]' >/dev/null 2>&1 \
    || { echo "install failed; skipping the checks that depend on it" >&2; status=1; }
"$PY" -m pip install -q ty >/dev/null 2>&1 || true
echo; echo "== ruff (as CI runs it) =="
"$PY" -m ruff check harness/serverbench tests || status=1
echo; echo "== ty (advisory: version-sensitive) =="
# Advisory, not fatal: `ty`'s verdict can depend on which `transformers` the
# resolver picks (the overload set in its stubs differs between versions), so
# a local disagreement may not be one CI has. It prints the versions that
# produced the verdict and does not vote.
"$PY" -c "
import importlib
for name in ('torch', 'transformers'):
    try:
        print(f'   resolved {name} {importlib.import_module(name).__version__}')
    except Exception as exc:
        print(f'   resolved {name}: not importable ({type(exc).__name__})')
" 2>/dev/null || true
"$PY" -m ty check harness/serving_evolve || echo \
    "   ty disagrees here. Compare against the last CI run before changing code:
   if the flagged lines are unchanged from a commit CI accepted, this is the
   resolver, not the code."
echo; echo "== pytest (as CI runs it) =="
"$PY" -m pytest tests -q || status=1

echo
if [ "$status" = 0 ]; then
    echo "PREFLIGHT PASSED -- these checks will pass on the pushed commit"
else
    echo "PREFLIGHT FAILED -- fix it here rather than discovering it in CI." >&2
    echo "  A failure here that passes in the working tree means a test is" >&2
    echo "  reading generated data that is not in Git. Give it a fixture." >&2
fi
exit "$status"
