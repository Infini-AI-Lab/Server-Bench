#!/usr/bin/env bash
# Create the host environments every later step runs on.
#
#   serverbench/.venv   the harness interpreter: `serverbench[dev]` (editable)
#                       plus the pinned AgentX that the final verdict measures
#                       with, verified by `python -m final_validate.agentx_pin`.
#   harbor/.venv        Harbor, which composes every evolve container.
#   serverbench/artifacts/agentx-runtime
#                       the AgentX the agent measures with, mounted at /opt/agentx.
#
# Idempotent: an environment that already exists is updated in place and
# re-verified. Needs network and `uv` on PATH.
#
# This step runs before `config.sh`, because `config.sh` sources `env.sh`, which
# refuses to start without the interpreter this step creates.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
VENV_DIR="$REPO/serverbench/.venv"
PY="$VENV_DIR/bin/python"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m   %s\n' "$*"; }
die()  { printf '   \033[31mfail\033[0m %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv is not on PATH; install it from https://docs.astral.sh/uv/"

# The AgentX commit comes from the module that checks it, so the install and
# the check cannot name different commits.
pin_file="$REPO/serverbench/harness/final_validate/agentx_pin.py"
agentx_commit="$(sed -n 's/^PINNED_COMMIT = "\([0-9a-f]\{40\}\)"$/\1/p' "$pin_file")"
agentx_remote="$(sed -n 's/^PINNED_REMOTE = "\(.*\)"$/\1/p' "$pin_file")"
[ -n "$agentx_commit" ] && [ -n "$agentx_remote" ] \
    || die "could not read PINNED_COMMIT / PINNED_REMOTE from $pin_file"

say "harness interpreter: $VENV_DIR"
if [ -x "$PY" ]; then
    ok "exists: $("$PY" -V)"
else
    uv venv "$VENV_DIR" --python 3.12
fi
uv pip install --python "$PY" -e "$REPO/serverbench[dev]" \
    "aiperf @ git+$agentx_remote@$agentx_commit"
ok "serverbench[dev] and aiperf @ ${agentx_commit:0:12}"

# Verified from `serverbench/harness`, the package root, with nothing on
# PYTHONPATH that could shadow the installed AgentX.
(cd "$REPO/serverbench/harness" && env -u PYTHONPATH "$PY" -m final_validate.agentx_pin >/dev/null) \
    || die "the installed AgentX is not the pinned one; run: (cd serverbench/harness && $PY -m final_validate.agentx_pin)"
ok "AgentX matches the pin"

# The AgentX runtime the agent's container mounts read-only at /opt/agentx, so
# the agent measures its own server with the same pinned AgentX. A relocatable
# venv on a uv-managed interpreter: `harbor_env.agentx_interpreter_mounts`
# mounts that interpreter's install root at the path the venv links to.
runtime="${SERVERBENCH_AGENTX_RUNTIME:-$REPO/serverbench/artifacts/agentx-runtime}"
say "AgentX runtime: $runtime"
if [ ! -x "$runtime/venv/bin/python" ]; then
    uv venv "$runtime/venv" --python 3.12 --managed-python --relocatable
fi
uv pip install --python "$runtime/venv/bin/python" "aiperf @ git+$agentx_remote@$agentx_commit"
mkdir -p "$runtime/bin"
cat > "$runtime/bin/aiperf" <<'SH'
#!/bin/sh
# `aiperf` on PATH inside the agent container: the pinned AgentX in ../venv.
exec "$(dirname "$0")/../venv/bin/python" -m aiperf "$@"
SH
chmod +x "$runtime/bin/aiperf"
"$runtime/bin/aiperf" --help >/dev/null || die "the AgentX runtime at $runtime does not start"
ok "aiperf @ ${agentx_commit:0:12} at $runtime/bin/aiperf"

say "Harbor: $REPO/harbor/.venv"
(cd "$REPO/harbor" && uv sync --all-extras --dev)
"$REPO/harbor/.venv/bin/python" -c 'import harbor' \
    || die "harbor/.venv exists but 'import harbor' fails"
ok "harbor $("$REPO/harbor/.venv/bin/python" -c 'import harbor; print(harbor.__version__)')"

say "environments ready"
