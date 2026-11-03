#!/bin/bash
# Block until a background job is actually done, and return the moment it is.
#
#   wait/wait_for.sh --file /tmp/done            [--deadline 600]
#   wait/wait_for.sh --pid 12345                 [--deadline 600]
#   wait/wait_for.sh --match 'DONE' --in /tmp/x.log
DEADLINE=600
MODE="" TARGET="" HAYSTACK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --file)     MODE=file; TARGET="$2"; shift 2 ;;
    --pid)      MODE=pid;  TARGET="$2"; shift 2 ;;
    --match)    MODE=match; TARGET="$2"; shift 2 ;;
    --in)       HAYSTACK="$2"; shift 2 ;;
    --deadline) DEADLINE="$2"; shift 2 ;;
    *) echo "wait_for: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODE" ] || { echo "wait_for: need --file, --pid or --match" >&2; exit 2; }
[ "$MODE" != match ] || [ -n "$HAYSTACK" ] || {
  echo "wait_for: --match needs --in <file>" >&2; exit 2; }

# A ZOMBIE IS EXITED. `kill -0` succeeds on a process that has terminated but
# whose parent has not reaped it, so a plain `kill -0` loop waits forever for a
# job that finished immediately -- which is precisely the hang this script
# exists to remove, reintroduced one layer down. A job launched with `nohup ...
# &` from a shell that then exits is reparented and reaped, so it does not hit
# this; a job whose parent is still alive does, every time.
pid_done() {
  kill -0 "$1" 2>/dev/null || return 0
  local state
  state=$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)
  [ "$state" = "Z" ]
}

start=$SECONDS
while :; do
  case "$MODE" in
    file)  [ -e "$TARGET" ] && { echo "$TARGET appeared after $((SECONDS-start))s"; exit 0; } ;;
    pid)   pid_done "$TARGET" && { echo "pid $TARGET exited after $((SECONDS-start))s"; exit 0; } ;;
    match) grep -qE -- "$TARGET" "$HAYSTACK" 2>/dev/null && {
             echo "matched /$TARGET/ in $HAYSTACK after $((SECONDS-start))s"; exit 0; } ;;
  esac
  if [ $((SECONDS - start)) -ge "$DEADLINE" ]; then
    echo "wait_for: condition ($MODE $TARGET) still false after ${DEADLINE}s" >&2
    [ -n "$HAYSTACK" ] && [ -f "$HAYSTACK" ] && tail -5 "$HAYSTACK" >&2
    exit 1
  fi
  sleep 0.5
done
