# Starter launcher for gemma4_e4b_it_standard_modal_1xL40S.
# Generated task bar: p90 TTFT <= 3,000 ms; p90 TPOT <= 80 ms.
# AGENT_TASK.md is the task's specification.

set -e
APP_DIR="${APP_DIR:-/app}"
[ -r "$APP_DIR/gpu_env.sh" ] && . "$APP_DIR/gpu_env.sh"
exec python3 "$APP_DIR/server.py"
