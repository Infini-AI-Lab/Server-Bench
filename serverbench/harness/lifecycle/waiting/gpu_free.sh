#!/bin/bash
# Wait for the device to actually free -- and if it does not, say WHO is holding
# it, immediately, instead of watching a number that is never going to move.
#
#   wait/gpu_free.sh [min_free_MiB] [deadline_s]
MIN="${1:-120000}"
DEADLINE="${2:-90}"
start=$SECONDS
stalled_at=""
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -ge "$MIN" ]; then
    echo "device free: ${free} MiB after $((SECONDS - start))s"
    exit 0
  fi
  # Unchanged for 30s is the signal that this is a leak rather than a teardown
  # in progress: a context actually being released shrinks while you watch.
  if [ "$free" = "$stalled_at" ] && [ $((SECONDS - start)) -ge 30 ]; then
    break
  fi
  [ "$free" != "$stalled_at" ] && { stalled_at="$free"; start=$SECONDS; }
  [ $((SECONDS - start)) -ge "$DEADLINE" ] && break
  sleep 2
done

echo "device NOT free: ${free} MiB, and the number stopped moving." >&2
echo "This is a leaked process, not a slow teardown. It is holding:" >&2
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
echo >&2
echo "Each pid above, and whether it is yours:" >&2
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  ps -o pid=,etime=,args= -p "$pid" 2>/dev/null | cut -c1-160 >&2
done
echo >&2
echo "Kill it by PROCESS GROUP -- 'kill -TERM -<pgid>' -- or with" >&2
echo "'bash stop_server.sh'. NEVER 'pkill -f' on task text: the command line" >&2
echo "that launched this run contains it and you will kill your own session." >&2
exit 1
