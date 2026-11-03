# The agent roster and shared settings every experiment script reads.
#
# Sourced, never run. Every value can be overridden from the environment
# (`MODELS="opus-5-5 glm-5.3" HOURS=4 bash experiments/main_sweep.sh`), and the
# roster itself by pointing ROSTER_FILE at a file that sets ROSTER.

# --- the agent roster -----------------------------------------------------------
# One entry per agent model:  label|agent CLI|model id|credentials file|efforts
#
#   label        a short name: selects the model in MODELS and names its campaigns
#   agent CLI    `claude` (Claude Code) or `codex` (Codex CLI)
#   model id     what the CLI is asked to run. A `vendor/model` id is reached
#                through an API key, a bare id on the operator's own login
#   credentials  a KEY=VALUE file under $SECRETS_DIR for an API-key model,
#                empty for a login (OAuth) model. The harness reads it by path
#   efforts      the reasoning efforts the effort ablation runs for this model;
#                the first is the one every other experiment uses
#
# Every agent runs at its model's full context window. Claude models get their
# native 1M from the Claude CLI; any other model needs its full window in
# `AGENT_CONTEXT_WINDOWS` (serverbench/scripts/agent/harbor_env.py, or
# CLAUDE_CODE_MAX_CONTEXT_TOKENS in its credentials file) or, on Codex, in
# `CODEX_CONTEXT_WINDOWS` (serverbench/harness/lifecycle/agent.py); otherwise
# the run is refused.
ROSTER=(
    "opus-5-5|claude|claude-opus-5-5||max xhigh high"
    "sonnet-5-5|claude|claude-sonnet-5-5||max xhigh high"
    "opus-5|claude|claude-opus-5||max xhigh"
    "gpt-6-astra|codex|gpt-6-astra||max xhigh ultra"
    "gpt-6.1-sol|codex|gpt-6.1-sol||max xhigh ultra"
    "gpt-6-sol|codex|gpt-6-sol||max xhigh"
    "glm-5.3|claude|z-ai/glm-5.3|glm-5.3.env|max"
    "kimi-k3|claude|accounts/fireworks/models/kimi-k3|kimi-k3.env|max"
    "deepseek-v4p1-flash|claude|accounts/fireworks/models/deepseek-v4p1-flash|deepseek-v4p1-flash.env|max"
    "mimo-v2.6-pro|claude|xiaomi/mimo-v2.6-pro|mimo-v2.6-pro.env|max"
    "qwen3.8-max|claude|qwen/qwen3.8-max-0902|qwen3.8-max.env|max"
)
[ -n "${ROSTER_FILE:-}" ] && . "$ROSTER_FILE"

SECRETS_DIR="${SECRETS_DIR:-$REPO/serverbench/artifacts/secrets}"

# --- what every evolve shares ---------------------------------------------------
HOURS="${HOURS:-8}"                 # the agent's wall-clock budget
# Cards the campaigns may use, e.g. "0,1,2,3"; empty means any free card not in
# EXCLUDE_GPUS. Occupancy is checked before every launch either way.
GPUS="${GPUS:-}"

# The tasks of the main comparison: one per model family and hardware shape.
SWEEP_TASKS="${SWEEP_TASKS:-qwen3_5_9b_standard_2xL40 gemma4_e4b_it_standard_1xL40 qwen3_8_27b_int4_standard_2xL40}"
