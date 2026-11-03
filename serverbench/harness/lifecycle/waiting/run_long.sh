#!/bin/bash
# Start work that might outlive its tool call, and get a handle you can block on.
#
#   bash wait/run_long.sh --tag sweep -- python tools/prefix_check.py --all
#   bash wait/run_long.sh --tag build --first-wait 30 -- bash rebuild.sh
set -u
TAG="job"; FIRST=45; DIR="${TMPDIR:-/tmp}/long"
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)        TAG="$2"; shift 2 ;;
    --first-wait) FIRST="$2"; shift 2 ;;
    --dir)        DIR="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "run_long: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [ $# -eq 0 ]; then
  echo "run_long: nothing to run. Put the command after --" >&2; exit 2
fi

mkdir -p "$DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG="$DIR/$TAG.$STAMP.log"
DONE="$DIR/$TAG.$STAMP.done"
PIDF="$DIR/$TAG.$STAMP.pid"

# `setsid` so it survives the tool call's process group being torn down, which
# is the whole failure this addresses. The exit code goes into the done-file,
# because "it finished" and "it worked" are different questions.
setsid bash -c '"$@" > "'"$LOG"'" 2>&1; echo $? > "'"$DONE"'"' _ "$@" &
CHILD=$!
echo "$CHILD" > "$PIDF"

WAITED=0
while [ "$WAITED" -lt "$FIRST" ]; do
  [ -f "$DONE" ] && break
  sleep 1
  WAITED=$((WAITED + 1))
done

if [ -f "$DONE" ]; then
  CODE="$(cat "$DONE")"
  cat "$LOG"
  echo "[run_long] $TAG finished in ${WAITED}s, exit $CODE"
  exit "$CODE"
fi

cat <<EOF
[run_long] $TAG is still running after ${FIRST}s. It is detached and will finish
on its own. Do NOT sleep for it -- block on it, and get the moment it ends:

    bash wait/wait_for.sh --file $DONE --deadline 3600 && cat $LOG

  log   $LOG
  done  $DONE   (contains the exit code once it finishes)
  pid   $CHILD
EOF
exit 0
