# Starter launcher for qwen3_5_9b_instructed_2xL40.
# Generated task bar: p90 TTFT <= 4,100.28 ms; p90 TPOT <= 23.4974 ms.
# AGENT_TASK.md is the task's specification.

set -e
APP_DIR="${APP_DIR:-/app}"
[ -r "$APP_DIR/gpu_env.sh" ] && . "$APP_DIR/gpu_env.sh"
exec python3 "$APP_DIR/server.py"
