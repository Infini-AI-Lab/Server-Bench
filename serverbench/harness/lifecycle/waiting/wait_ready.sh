#!/bin/bash
# Block until the server answers, and return the moment it does.
#
#   bash wait/wait_ready.sh [port] [deadline_s] [path]
PORT="${1:-8000}"
DEADLINE="${2:-110}"
ROUTE="${3:-/health}"
exec python3 - "$PORT" "$DEADLINE" "$ROUTE" <<'PY'
import sys, time, urllib.request, urllib.error

port, deadline, route = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
url = f"http://127.0.0.1:{port}{route}"
started = time.time()
last = ""
while time.time() - started < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if 200 <= response.status < 300:
                print(f"ready in {time.time() - started:.1f}s ({url})")
                sys.exit(0)
            last = f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        last = f"HTTP {exc.code}"
    except Exception as exc:                     # connection refused while loading
        last = type(exc).__name__
    time.sleep(0.25)
print(f"not ready after {deadline:.0f}s: last saw {last} at {url}",
      file=sys.stderr)
print("  the server log is the next thing to read; a server that never binds "
      "the port has died during load and its traceback is there", file=sys.stderr)
sys.exit(1)
PY
