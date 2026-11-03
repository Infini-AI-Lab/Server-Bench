"""A run records what its container was given. Not the key it was given.

`harbor_environment.json` records the compose overlay the environment was
built from, which is the only statement of what the container actually got --
and the overlay carries the agent's provider credentials as environment
values. Evidence is copied into published prototypes, read by reports and
passed around when a run is being explained, so a credential in it would be a
credential in all of those. Values are redacted.

What must survive redaction is the NAME. "No credential reached the container"
is the most common reason an agent cannot start, and the evidence has to be
able to say which variable was set.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "serverbench" / "harness"), str(REPO / "serverbench" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(REPO / "serverbench" / "scripts" / "agent"))
import harbor_env  # noqa: E402

OVERLAY = '''services:
  main:
    environment:
      ANTHROPIC_API_KEY: "fw_TESTKEYnotarealcredential0"
      ANTHROPIC_AUTH_TOKEN: "fw_TESTKEYnotarealcredential0"
      ANTHROPIC_BASE_URL: "https://api.fireworks.ai/inference"
      MODEL_PATH: "/models/target"
      PORT: "8000"
'''


def test_the_key_does_not_survive():
    got = harbor_env.redacted(OVERLAY)
    assert "fw_TESTKEYnotarealcredential0" not in got


def test_the_names_do_survive():
    got = harbor_env.redacted(OVERLAY)
    assert "ANTHROPIC_API_KEY" in got and "ANTHROPIC_AUTH_TOKEN" in got
    assert "REDACTED" in got


def test_everything_that_is_not_a_credential_is_untouched():
    got = harbor_env.redacted(OVERLAY)
    assert '"/models/target"' in got
    assert '"8000"' in got
    # A base URL is not a credential and is load-bearing for diagnosis: it is
    # what says which provider a run went to.
    assert "https://api.fireworks.ai/inference" in got


def test_the_structure_still_parses():
    import yaml

    body = yaml.safe_load(harbor_env.redacted(OVERLAY))
    env = body["services"]["main"]["environment"]
    assert env["PORT"] == "8000"
    assert "REDACTED" in env["ANTHROPIC_API_KEY"]
