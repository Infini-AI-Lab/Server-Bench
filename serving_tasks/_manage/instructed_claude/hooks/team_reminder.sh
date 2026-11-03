#!/bin/sh
# Re-states the team workflow inside long sessions: after any tool call, at
# most once every 20 minutes. Delivered to instructed workspaces only.
stamp=/tmp/.serverbench-team-reminder
now=$(date +%s)
last=$(cat "$stamp" 2>/dev/null || echo 0)
[ $((now - last)) -ge 1200 ] || exit 0
echo "$now" > "$stamp"
cat <<'JSON'
{"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "Team workflow reminder. Cycle: plan, request, build, integrate, verify, decide; keep kernel-engineer and runtime-engineer working in the background, each in its own git worktree, and send each its next assignment when it returns. A cycle ends only at a verified tree: started cold by serve.sh (the only thing the verdict starts), correct against an independent reference through the HTTP endpoint at the workload's shapes (longest contexts, prefix reuse, forks, concurrent joins and leaves, long decodes, ignore_eos continuation and a natural stop), and measured by ./full-test with every request completing and every surprise explained. Choose the offered load by measuring several; trust repeated windows, not the best one. Every GPU job runs under `flock /tmp/gpu.lock` for its own lifetime, one agent at a time; CPU work overlaps freely; the lead balances GPU time."}}
JSON
