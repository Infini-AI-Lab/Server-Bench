"""Every Claude CLI model route goes to the run's one model.

The CLI sends subagents and background calls to its own default aliases, which
an OpenAI-compatible endpoint serving one open model cannot route. An agent
uses only the model the run names, so every alias is pinned to it, and the
brief says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "scripts" / "agent"))

import harbor_env  # noqa: E402

MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"


def test_every_alias_is_pinned_to_the_run_model():
    env = {"ANTHROPIC_MODEL": MODEL, "ANTHROPIC_BASE_URL": "https://x"}
    assert harbor_env.pin_model_routes(env, "", "claude") == MODEL
    for alias in ("CLAUDE_CODE_SUBAGENT_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
                  "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                  "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
        assert env[alias] == MODEL, alias


def test_the_named_model_is_used_when_the_env_file_names_none():
    env = {}
    assert harbor_env.pin_model_routes(env, "claude-opus-5-5", "claude") == "claude-opus-5-5"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "claude-opus-5-5"


def test_an_open_model_gets_its_real_context_window_not_the_cli_default():
    # Left to itself the CLI assumes a 200k window and auto-compacts well below
    # it; the open models this harness runs serve ~1M.
    for model, window in harbor_env.AGENT_CONTEXT_WINDOWS.items():
        env = {"ANTHROPIC_MODEL": model, "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000"}
        assert harbor_env.pin_model_routes(env, "", "claude") == model
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == str(window)
        assert window >= 1_000_000
        assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env


def test_a_claude_model_keeps_the_clis_own_window():
    env = {}
    harbor_env.pin_model_routes(env, "claude-opus-5-5", "claude")
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env


def test_an_open_model_with_an_unknown_window_is_refused():
    import pytest
    with pytest.raises(SystemExit, match="context window"):
        harbor_env.pin_model_routes({"ANTHROPIC_MODEL": "some/new-model"}, "", "claude")
    env = {"ANTHROPIC_MODEL": "some/new-model", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "262144"}
    harbor_env.pin_model_routes(env, "", "claude")
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"


def test_other_clis_are_untouched():
    env = {"ANTHROPIC_MODEL": MODEL}
    assert harbor_env.pin_model_routes(env, MODEL, "codex") == ""
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in env


def test_the_brief_tells_the_agent_to_use_only_its_model():
    for task in sorted((REPO / "serving_tasks").glob("*")):
        brief = task / "instruction.md"
        if brief.is_file():
            assert "Use only that model" in brief.read_text("utf-8"), task.name


def test_the_broker_reaches_only_the_runs_provider():
    """The brief says only the task's model provider is reachable; the broker
    must make that true, not relay to every provider the CLI knows."""
    static = {"api.anthropic.com", "statsig.anthropic.com", "sentry.io",
              "openrouter.ai", "api.fireworks.ai"}
    env = {"ANTHROPIC_BASE_URL": "https://api.fireworks.ai/inference"}
    assert harbor_env.narrow_to_provider(static, env, "claude", []) == {"api.fireworks.ai"}
    assert harbor_env.narrow_to_provider(static, {}, "claude", []) == set(harbor_env.ANTHROPIC_HOSTS)
    assert "openrouter.ai" not in harbor_env.narrow_to_provider(static, {}, "claude", [])
    assert harbor_env.narrow_to_provider({"api.openai.com"}, {}, "codex", []) == {"api.openai.com"}


def test_a_third_party_run_carries_only_its_own_key():
    """GLM/DeepSeek on Fireworks: their key only -- never the operator's
    OAuth file, never another provider's key."""
    env = {"ANTHROPIC_BASE_URL": "https://api.fireworks.ai/inference",
           "ANTHROPIC_AUTH_TOKEN": "k", "ANTHROPIC_API_KEY": "k",
           "OPENROUTER_API_KEY": "other", "OPENAI_API_KEY": "other",
           "MODEL_PATH": "/models/target"}
    assert harbor_env.credential_route(env, "claude") == "api_key"
    assert "OPENROUTER_API_KEY" not in env and "OPENAI_API_KEY" not in env
    assert env["ANTHROPIC_AUTH_TOKEN"] == "k" and env["MODEL_PATH"] == "/models/target"


def test_an_anthropic_run_uses_the_oauth_token_only():
    """Opus: the OAuth file, and no API key that would take precedence."""
    env = {"ANTHROPIC_API_KEY": "stray", "ANTHROPIC_AUTH_TOKEN": "stray",
           "OPENROUTER_API_KEY": "other", "CLAUDE_CODE_OAUTH_TOKEN": "oauth"}
    assert harbor_env.credential_route(env, "claude") == "oauth"
    assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "oauth"}


def test_an_api_key_run_never_carries_the_oauth_token():
    env = {"ANTHROPIC_BASE_URL": "https://api.fireworks.ai/inference",
           "ANTHROPIC_AUTH_TOKEN": "k", "CLAUDE_CODE_OAUTH_TOKEN": "oauth"}
    assert harbor_env.credential_route(env, "claude") == "api_key"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_a_base_url_without_a_key_is_refused():
    import pytest
    with pytest.raises(SystemExit):
        harbor_env.credential_route({"ANTHROPIC_BASE_URL": "https://x"}, "claude")


def test_the_oauth_file_is_mounted_only_on_the_oauth_route():
    source = (REPO / "serverbench/scripts/agent/harbor_env.py").read_text()
    assert 'if route in ("oauth", "codex") else []' in source
