#!/bin/bash
# Block until a delegated subagent finishes, and return the moment it does.
#
#   wait/await_agent.sh                      [--deadline 7200] [--lines 60]
#   wait/await_agent.sh <agentId>            [--deadline 7200] [--lines 60]
set -u
DEADLINE=7200
LINES=60
AGENT=""
SINCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --deadline) DEADLINE="$2"; shift 2 ;;
    --lines)    LINES="$2"; shift 2 ;;
    --since)    SINCE="$2"; shift 2 ;;
    -*) echo "await_agent: unknown argument $1" >&2; exit 2 ;;
    *)  AGENT="$1"; shift ;;
  esac
done

# GENERATION BINDING. An agent id is unique per invocation, so naming one is the
# reliable form and the one the brief teaches. The bare form -- newest output
# file -- is a convenience, and on a resume it can find a COMPLETED file from an
# earlier generation and return instantly with someone else's report. `--since`
# is the guard: a file whose last write predates the launch is not this launch's.
# Pass `--since "$(date +%s)"` immediately before delegating.

# The session directory is the CLI's, not ours, so it is discovered rather than
# assumed: one `tasks/` directory per session under the scratchpad root.
# AWAIT_AGENT_ROOT exists so this is testable somewhere other than /tmp; nothing
# in a run sets it.
ROOT="${AWAIT_AGENT_ROOT:-/tmp/claude-*}"
output_for() {
  local name="${1:-*}"
  # Unquoted on purpose: ROOT and name are globs.
  ls -t $ROOT/*/*/tasks/$name.output 2>/dev/null | head -1
}

start=$SECONDS
file=""
while :; do
  [ -n "$file" ] || file="$(output_for "$AGENT")"
  if [ -n "$file" ] && [ -f "$file" ]; then
    # The marker is written last, so seeing it means the file is complete.
    code=$(tail -c 200 "$file" | sed -n 's/.*\[exited with code \([0-9]*\)\].*/\1/p' | tail -1)
    if [ -n "$code" ] && [ -n "$SINCE" ]; then
      # Complete, but is it OURS? A finished file older than the launch belongs
      # to an earlier generation; ignore it and keep waiting for a real one.
      if [ "$(stat -c %Y "$file" 2>/dev/null || echo 0)" -lt "$SINCE" ]; then
        file=""; code=""
      fi
    fi
    if [ -n "$code" ]; then
      echo "subagent ${AGENT:-$(basename "$file" .output)} finished after $((SECONDS-start))s, exit $code"
      echo "--- last $LINES line(s) of its output ---"
      # The completion marker itself is not part of what it reported.
      grep -v '^\[exited with code' "$file" | tail -n "$LINES"
      exit 0
    fi
  fi
  if [ $((SECONDS - start)) -ge "$DEADLINE" ]; then
    if [ -z "$file" ]; then
      echo "await_agent: no subagent output file found after ${DEADLINE}s." >&2
      echo "await_agent: is one actually running? Delegate first, then await." >&2
    else
      echo "await_agent: $file has not finished after ${DEADLINE}s" >&2
      tail -5 "$file" >&2
    fi
    exit 1
  fi
  sleep 2
done
