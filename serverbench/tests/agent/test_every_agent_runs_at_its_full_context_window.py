"""Every agent runs at its model's full context window.

Left to themselves the CLIs run a model they do not fully know at a smaller
default window and compact the agent's history below it: the Claude CLI at
200k for any non-Claude model, Codex at its catalog's default (272k or 500k)
rather than the model's `max_context_window`. The harness states each model's
full window and refuses a model whose window it does not know.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "scripts" / "agent"))

import harbor_env  # noqa: E402
from lifecycle import agent  # noqa: E402


def _roster() -> list[tuple[str, str]]:
    text = (REPO / "experiments" / "config.sh").read_text("utf-8")
    entries = re.findall(r'^\s*"([^"|]+)\|(claude|codex)\|([^|]+)\|', text, re.M)
    assert entries, "experiments/config.sh lists no roster"
    return [(cli, model) for _, cli, model in entries]


def test_a_codex_model_runs_at_its_full_window():
    for model, window in agent.CODEX_CONTEXT_WINDOWS.items():
        config = agent.codex_config(model, "max")
        assert config[:2] == ["--model", model]
        assert f"model_context_window={window}" in config
        assert 'model_reasoning_effort="max"' in config


def test_a_codex_model_with_an_unknown_window_is_refused():
    with pytest.raises(SystemExit, match="context window"):
        agent.codex_config("gpt-unknown")


@pytest.mark.parametrize("cli,model", _roster())
def test_every_roster_model_has_a_known_full_window(cli: str, model: str):
    if cli == "codex":
        assert agent.CODEX_CONTEXT_WINDOWS.get(model), model
    elif not model.startswith("claude-"):
        assert harbor_env.AGENT_CONTEXT_WINDOWS.get(model, 0) >= 1_000_000, model


def test_the_default_codex_model_has_a_known_full_window():
    text = (REPO / "serverbench" / "harness" / "lifecycle" / "run.py").read_text("utf-8")
    default = re.search(r'"(gpt-[^"]+)" if args\.agent_cli == "codex"', text).group(1)
    assert default in agent.CODEX_CONTEXT_WINDOWS
