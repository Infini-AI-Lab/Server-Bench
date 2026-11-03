"""Running out of money is a PAUSE, not a verdict.

## Why this exists

A run whose key hits its spend limit sees

    403 Failed to authenticate. API Error: 403 Key limit exceeded (total limit).

`agent.terminal_failure()` is right that retrying cannot help -- every attempt
fails exactly like the first -- so the driver stops asking. It must NOT end the
RUN, because the one thing that does fix this is an operator raising the limit
or dropping in a new key, and that takes a minute. Ending the run would throw
away the workspace, the lineage position, and every remaining GPU-hour on cards
already paid for.

So a billing stop parks the run instead. The agent is not relaunched into a
wall; the driver waits, says so in the journal, and picks the agent up again the
moment the key works.

An operator-requested pause follows the same accounting rule. Its resume event
is `operator_resumed`, with the observed parked interval in `paused_seconds`.
That keeps an intentional pause from silently consuming the agent's timebox.

## What counts as "the key works again"

Two independent signals, because neither alone covers the cases:

  * THE BALANCE. An OpenRouter key can be asked directly
    (`/api/v1/key` -> `limit_remaining`), so a limit raised on the dashboard is
    noticed without anyone touching this host.
  * THE CREDENTIALS FILE. A REPLACED key is a different token, and the balance
    of the OLD key never recovers. So the staged credentials are fingerprinted,
    and any change to them ends the pause -- that is the "replace the key" half,
    and without it the wait would sit forever on a dead key while a live one sat
    beside it on disk.

Neither signal is trusted to be available: a key that is not an `sk-or-` one
cannot be asked for its balance at all, and for those the file is the only
signal. That is stated rather than silently degraded.

## What a pause must not cost

THE PAUSE IS NOT THE AGENT'S TIME. `--hours` bounds the agent's work, and an
8-hour run parked for 2 hours waiting for an operator has not had 8 hours of
agent. `paused_seconds()` totals what the journal recorded, and
`agent.remaining_seconds()` takes it as `paused_s` so the deadline moves with
it. Without that, pausing would merely be a slower way to lose the run.

The wait is bounded by `MAX_PAUSE_S` so a run cannot hold two cards forever
over a key nobody is coming back to fix.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

#: Longest a run will hold its cards waiting for someone to fix the key.
#: Eight hours: long enough to survive a night, short enough that an abandoned
#: run gives the hardware back.
MAX_PAUSE_S = 8 * 3600.0

#: How often the balance and the credentials file are re-checked.
POLL_S = 60.0

#: OpenRouter's own view of a key. The only balance endpoint this fleet uses.
KEY_ENDPOINT = "https://openrouter.ai/api/v1/key"

#: Below this the key is treated as spent. Not zero: a key with half a cent on
#: it buys one request and fails again, and a pause that ends into an immediate
#: second pause is a loop, not a recovery.
MIN_USABLE_USD = 0.25


def is_billing(why: str) -> bool:
    """Whether `agent.terminal_failure()`'s reason was money.

    Reads the sentence that function already writes, rather than re-scanning
    the log for markers -- one place decides what a billing failure is, and it
    is `agent.BILLING_MARKERS`.
    """
    return bool(why) and "BILLING limit" in why


def _token(secrets_dir: Path | str) -> str:
    """The staged OpenRouter token, or "" when the run is not on one."""
    root = Path(secrets_dir)
    if not root.is_dir():
        return ""
    for candidate in sorted(root.glob("*.env")):
        try:
            for line in candidate.read_text("utf-8").splitlines():
                name, _, value = line.partition("=")
                if name.strip() in ("ANTHROPIC_AUTH_TOKEN",
                                    "OPENROUTER_API_KEY"):
                    token = value.strip()
                    if token.startswith("sk-or-"):
                        return token
        except OSError:
            continue
    return ""


def fingerprint(secrets_dir: Path | str) -> str:
    """A digest of the staged credentials, for noticing a REPLACED key.

    The token itself is never returned or logged; only whether it changed.
    """
    root = Path(secrets_dir)
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = sorted(root.rglob("*"))
    else:
        return ""
    digest = hashlib.sha256()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            digest.update(candidate.name.encode("utf-8"))
            digest.update(candidate.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()[:16]


def remaining_usd(token: str, *, timeout: float = 20.0) -> float | None:
    """What the key has left, or None when it cannot be asked.

    None is not zero. A network blip, a non-OpenRouter key or a changed API
    shape all produce None, and treating that as "still broke" would keep a
    working run parked.
    """
    if not token.startswith("sk-or-"):
        return None
    request = urllib.request.Request(
        KEY_ENDPOINT, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    data = body.get("data") or {}
    left = data.get("limit_remaining")
    if isinstance(left, bool) or not isinstance(left, (int, float)):
        # A key with NO limit set reports null, which means unlimited rather
        # than empty. Unlimited keys do not hit this path in the first place,
        # but reading null as 0 would park one forever if they did.
        return None if left is None else None
    return float(left)


def wait_for_credit(*, secrets_dir: Path | str, journal=None,
                    poll_s: float = POLL_S, max_pause_s: float = MAX_PAUSE_S,
                    sleep=time.sleep, now=time.monotonic) -> dict:
    """Park until the key works again. Returns what happened, and how long.

    Ends on ANY of: the balance recovering above `MIN_USABLE_USD`, the staged
    credentials changing at all, or `max_pause_s` elapsing.
    """
    started = now()
    before = fingerprint(secrets_dir)
    token = _token(secrets_dir)
    opening = remaining_usd(token)
    watched = ("the key's balance and the staged credentials file"
               if token else
               "the staged credentials file only -- this run's key is not an "
               "OpenRouter key, so its balance cannot be queried")
    if journal is not None:
        journal.append(
            "billing_paused", watching=watched,
            remaining_usd=opening, poll_seconds=poll_s,
            max_pause_seconds=max_pause_s,
            why=("the key is out of budget. The run is PARKED rather than "
                 "ended: raise the limit or replace the credentials file and "
                 "it picks up where it left off. The agent's deadline is "
                 "extended by however long this takes"))

    while True:
        waited = now() - started
        if waited >= max_pause_s:
            outcome = {"resumed": False, "why": "max_pause",
                       "paused_seconds": round(waited, 1)}
            break
        sleep(min(poll_s, max_pause_s - waited))
        if fingerprint(secrets_dir) != before:
            outcome = {"resumed": True, "why": "credentials_changed",
                       "paused_seconds": round(now() - started, 1)}
            break
        token = _token(secrets_dir)
        left = remaining_usd(token)
        if left is not None and left >= MIN_USABLE_USD:
            outcome = {"resumed": True, "why": "balance_restored",
                       "remaining_usd": left,
                       "paused_seconds": round(now() - started, 1)}
            break

    if journal is not None:
        journal.append("billing_resumed" if outcome["resumed"]
                       else "billing_gave_up", **outcome)
    return outcome


def wait_out(seconds: float, *, journal=None, event: str,
             why: str, secrets_dir: Path | str | None = None,
             poll_s: float = 30.0, sleep=time.sleep,
             now=time.monotonic) -> float:
    """Park for a known duration, then record it so the deadline can grow.

    `wait_for_credit` waits on a CONDITION that might never come. This waits
    on a CLOCK that will. A replacement credentials file also ends the wait:
    its quota belongs to a different OAuth credential and may already be live,
    so sleeping until the old account's reset wastes both agent and GPU time.
    Kept here beside `wait_for_credit` so both pauses are accounted the same
    way -- `paused_seconds` reads the event this writes, and
    `agent.remaining_seconds` adds it back.
    """
    started = now()
    opening_fingerprint = (fingerprint(secrets_dir)
                           if secrets_dir is not None else "")
    credentials_changed = False
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        step = min(poll_s, remaining)
        sleep(step)
        remaining -= step
        if (secrets_dir is not None
                and fingerprint(secrets_dir) != opening_fingerprint):
            credentials_changed = True
            break
    waited = now() - started
    if journal is not None:
        journal.append(event, paused_seconds=round(waited, 1), why=why,
                       credentials_changed=credentials_changed)
    return waited


def paused_seconds(journal) -> float:
    """Total time this run has spent parked on billing or a session limit.

    Added to the deadline by `agent.remaining_seconds`, so a pause costs the
    operator wall-clock and costs the AGENT nothing.
    """
    total = 0.0
    for event in ("billing_resumed", "billing_gave_up",
                  "session_limit_resumed", "operator_resumed"):
        for record in journal.events(event):
            value = record.get("paused_seconds")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += float(value)
    return total
