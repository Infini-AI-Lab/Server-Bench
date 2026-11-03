# Starter launcher for qwen3_8_27b_int4_standard_2xL40.
# Generated task bar: p90 TTFT <= 10,298.8 ms; p90 TPOT <= 48.3769 ms.
# AGENT_TASK.md is the task's specification.

set -e
APP_DIR="${APP_DIR:-/app}"
[ -r "$APP_DIR/gpu_env.sh" ] && . "$APP_DIR/gpu_env.sh"
exec python3 "$APP_DIR/server.py"
