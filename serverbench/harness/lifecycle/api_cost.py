"""What each published version cost in API spend.

Each published version carries its API cost, taken from the agent's own record
where it has one and otherwise computed from the model and the usage over time.

The agent DOES record it, and that record is used in preference to anything
computed here. `agent.stream.jsonl` carries two things worth having:

* every `assistant` event carries an exact `usage` block -- `input_tokens`,
  `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`,
  and `output_tokens_details.thinking_tokens`;
* the final `result` event carries `total_cost_usd` for the whole session,
  computed by the CLI from the real rates it was billed at.

## Why both, and which wins

A version is published DURING the run, so the run total does not exist yet when
its record is written. Token usage up to that moment does. So:

    per-version usage      exact, from the stream, at publication time
    per-version cost_usd   apportioned from that usage
    run total_cost_usd     the agent's own figure, when the run has ended

and the apportioning is then **rescaled so the versions sum to the agent's
reported total**. The price table below therefore only decides how the total is
SPLIT between versions, never what the total is -- which matters, because a
published price for a model can be wrong or can change and the agent's own
billing figure cannot.

`reconciliation` records both numbers and their difference. A table that has
drifted shows up there instead of quietly moving every version's cost.

## The price table is explicitly a ratio, not an authority

`RATES` exists to weight output tokens against input tokens, because a token
count alone would misweight them by roughly 5x and put the split badly wrong.
Each entry says whether it was verified against a published price list. An
unverified entry still produces a defensible SPLIT -- the ratios between the
four token classes are stable across the Claude family even when the absolute
prices are not -- and the absolute total still comes from the agent.

A model with no entry at all falls back to `DEFAULT_RATES` and is recorded as
having done so, so nobody reads such a split as precise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Dollars per million tokens, by class. Used to WEIGHT a split, not to state a
#: price -- see the module docstring. `verified` says whether the numbers were
#: checked against a published list; where they were not, the ratios are what
#: is being relied on and the absolute total comes from the agent's own record.
RATES: dict[str, dict[str, Any]] = {
    "claude-opus-4": {
        "input": 15.0, "output": 75.0,
        "cache_write": 18.75, "cache_read": 1.50,
        "verified": True,
        "source": "Anthropic published pricing for Claude Opus 4 (input $15/Mtok, "
                  "output $75/Mtok, cache write 1.25x input, cache read 0.1x input)",
    },
    # The default agent model. The ratios are the Opus tier's and are what the
    # split depends on; the absolute total still comes from the agent's own
    # `total_cost_usd` where it reports one.
    "claude-opus-5": {
        "input": 15.0, "output": 75.0,
        "cache_write": 18.75, "cache_read": 1.50,
        "verified": False,
        "source": ("Opus-tier ratios (5x output to input, 1.25x cache write, "
                   "0.1x cache read). Not checked against a published list for "
                   "this model specifically, which is why `verified` is False "
                   "-- the SPLIT depends only on the ratios and the TOTAL "
                   "comes from the agent"),
    },
    # An OpenRouter-hosted GLM, for the evolve that runs on one. VERIFIED
    # against OpenRouter's own /api/v1/models.
    #
    # It matters more here than elsewhere that this is right. A Claude Code
    # talking to OpenRouter may not report a `total_cost_usd` at all, and when
    # it does not, `attribute` falls back to computing the total FROM these
    # numbers -- so for a GLM run they are the absolute price rather than only
    # a ratio. Weighting a $0.15/M model with Opus's $15/M would overstate the
    # run's cost by a hundredfold.
    "z-ai/glm-5.3-flash": {
        "input": 0.15, "output": 0.50,
        "cache_write": 0.1875, "cache_read": 0.015,
        "verified": True,
        "source": ("OpenRouter /api/v1/models: prompt "
                   "$0.15/Mtok, completion $0.50/Mtok. Cache rates are the "
                   "usual 1.25x write / 0.1x read of input and are NOT "
                   "separately published, so they are an assumption about "
                   "cache pricing rather than a quoted figure"),
    },
    "z-ai/glm-5.3": {
        "input": 1.40, "output": 4.40,
        "cache_write": 1.75, "cache_read": 0.14,
        "verified": True,
        "source": "OpenRouter /api/v1/models",
    },
    "claude-sonnet-4": {
        "input": 3.0, "output": 15.0,
        "cache_write": 3.75, "cache_read": 0.30,
        "verified": True,
        "source": "Anthropic published pricing for Claude Sonnet 4",
    },
}

#: What an unrecognised model is weighted by. The Opus ratios, because the
#: default agent is Opus and the RATIOS are what matter; they are the same
#: across the Claude tiers (5x output to input, 1.25x cache write, 0.1x cache
#: read).
DEFAULT_RATES = {
    "input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 1.50,
    "verified": False,
    "source": ("no published rate on file for this model. These are the Opus "
               "RATIOS -- 5x output to input, 1.25x cache write, 0.1x cache "
               "read -- which is all the split depends on. The absolute total "
               "comes from the agent's own `total_cost_usd`"),
}

ZERO = {"input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0,
        "cache_read_tokens": 0, "thinking_tokens": 0, "messages": 0}


#: HOW FAR THE STREAM'S OWN ARITHMETIC IS FROM THE BILL, measured.
#:
#: Summing the per-message `usage` blocks does NOT reproduce what the run was
#: charged, and the gap is systematic rather than noisy: across finished
#: claude-opus-5 runs that recorded both numbers, reported/computed falls in
#: 0.22-0.28. The stream over-states the bill by about FOUR TIMES, because
#: its blocks do not reconcile: more than one thing is being counted, and not
#: consistently.
#:
#: That is harmless while a run finishes: `attribute` uses the reported total
#: and the computed one only decides the SPLIT. It stops being harmless when a
#: run does NOT finish -- a killed agent, an exhausted context, a run still in
#: flight -- because then there is no reported total and the computed one is
#: what gets published.
#:
#: So an unreported total is still reported -- absent would be worse -- but it
#: is labelled as an upper bound and a calibrated figure is offered beside it.
#: The factor is applied only to models whose stream it was measured on.
STREAM_OVERSTATES_BY = 4.0
CALIBRATION_EVIDENCE = (
    "median reported/computed = 0.252 across five finished claude-opus-5 runs, "
    "range 0.223-0.280")


def calibrated_for(model: str) -> float | None:
    """The factor to apply to a computed total, or None if none was measured.

    Only Anthropic-served models: the measurement above is of the Claude CLI's
    own stream. An OpenRouter-served model is a different case entirely and is
    handled by `reported_total_is_trustworthy`.
    """
    return (1.0 / STREAM_OVERSTATES_BY
            if str(model or "").startswith("claude-") else None)


def reported_total_is_trustworthy(model: str) -> bool:
    """Whether the CLI's own `total_cost_usd` is this run's actual bill.

    It is, for a model Anthropic serves and bills. It is NOT for a model
    reached through OpenRouter: the CLI prices what it sees with Anthropic's
    card, so a GLM run comes back costed as though it were Opus: overstated
    several-fold against its own published rates.

    For those, the rate table and `recover_openrouter` are the authorities and
    the CLI's figure is discarded rather than averaged in. Two wrong numbers do
    not make a right one.
    """
    return str(model or "").startswith("claude-")


def rates_for(model: str) -> dict[str, Any]:
    """The weights for this model, and whether they were verified."""
    for name, table in RATES.items():
        if model.startswith(name):
            return {**table, "matched": name}
    return {**DEFAULT_RATES, "matched": ""}


def _add(into: dict[str, int], usage: dict[str, Any]) -> None:
    into["input_tokens"] += int(usage.get("input_tokens") or 0)
    into["output_tokens"] += int(usage.get("output_tokens") or 0)
    into["cache_write_tokens"] += int(
        usage.get("cache_creation_input_tokens") or 0)
    into["cache_read_tokens"] += int(usage.get("cache_read_input_tokens") or 0)
    into["thinking_tokens"] += int(
        (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
    into["messages"] += 1


#: Where an OpenRouter-hosted run's real numbers come from.
#:
#: Claude Code talking to OpenRouter emits the Anthropic-shaped `usage` block
#: with EVERY FIELD ZERO, and no `result.total_cost_usd` at all.
#:
#: A cost of exactly zero is the most dangerous possible answer here: it is
#: plausible, it is silent, and it makes an expensive run look free. So rather
#: than price a GLM run from the local rate table, this asks OpenRouter what it
#: actually billed. Each assistant message carries its generation id, and
#: `/api/v1/generation?id=` returns `total_cost` and the token counts.
OPENROUTER_GENERATION = "https://openrouter.ai/api/v1/generation"


#: Wall-clock budget for one billing sweep, and how many lookups run at once.
#:
#: `attribute()` is called every time a version is published, so this sits on
#: the run's critical path. A long run carries hundreds of distinct
#: generations; sequentially that is minutes per publish. Bounded and parallel
#: it is a few seconds, and a sweep that runs out of
#: budget reports what it got plus how much it missed -- a partial recovery is
#: still far better evidence than the stream's frozen counters, PROVIDED the
#: shortfall is stated rather than hidden.
OPENROUTER_SWEEP_SECONDS = 90.0
OPENROUTER_SWEEP_WORKERS = 8

#: Generation bodies already fetched, keyed by id. A run publishes many
#: versions and re-reads the same stream each time; without this, every publish
#: re-asks the billing API for every generation the run has ever made.
_GENERATION_CACHE: dict[str, dict[str, Any]] = {}


def _openrouter_usage(ids: list[str], token: str,
                      timeout: float = 10.0) -> dict[str, Any] | None:
    """Real usage and cost for a run, from OpenRouter's own billing.

    Returns None when it cannot be had -- no token, no ids, or the endpoint
    refusing -- because an absent cost must stay absent. A zero here would be
    indistinguishable from a free run.
    """
    import json as _json
    import time as _time
    import urllib.error
    import urllib.request
    from concurrent import futures

    if not ids or not token:
        return None
    # DEDUPED, and that is not only an optimisation.
    #
    # One generation id appears once per assistant message that streamed from
    # it, so a run carries far fewer distinct generations than messages.
    # Asking for the same id twice in the same burst is what the endpoint
    # rejects, so a de-duplicated sweep succeeds where the raw list does not.
    #
    # `dict.fromkeys` rather than `set` so the order is the run's own, which
    # keeps the cumulative timeline below meaningful.
    unique = list(dict.fromkeys(ids))
    deadline = _time.monotonic() + OPENROUTER_SWEEP_SECONDS

    def fetch(gen_id: str) -> tuple[str, dict[str, Any] | None]:
        if gen_id in _GENERATION_CACHE:
            return gen_id, _GENERATION_CACHE[gen_id]
        # A billing endpoint is allowed to rate-limit, and a cost that is
        # absent because we asked too quickly is a cost nobody can recover
        # later -- the run is over by then. Three tries with a widening gap,
        # and never past the sweep's deadline.
        for attempt in range(3):
            if _time.monotonic() > deadline:
                return gen_id, None
            request = urllib.request.Request(
                f"{OPENROUTER_GENERATION}?id={gen_id}",
                headers={"Authorization": f"Bearer {token}"})
            try:
                with urllib.request.urlopen(request,
                                            timeout=timeout) as response:
                    body = (_json.loads(response.read().decode("utf-8"))
                            or {}).get("data") or {}
                _GENERATION_CACHE[gen_id] = body
                return gen_id, body
            except (urllib.error.URLError, ValueError, OSError, TimeoutError):
                if attempt < 2:
                    _time.sleep(0.4 * (attempt + 1))
        return gen_id, None

    total = dict(ZERO)
    cost = 0.0
    seen = failed = 0
    with futures.ThreadPoolExecutor(
            max_workers=OPENROUTER_SWEEP_WORKERS) as pool:
        for _gen_id, body in pool.map(fetch, unique):
            if body is None:
                failed += 1
                continue
            total["input_tokens"] += int(body.get("native_tokens_prompt") or 0)
            total["output_tokens"] += int(
                body.get("native_tokens_completion") or 0)
            total["cache_read_tokens"] += int(
                body.get("native_tokens_cached") or 0)
            total["thinking_tokens"] += int(
                body.get("native_tokens_reasoning") or 0)
            total["messages"] += 1
            cost += float(body.get("total_cost") or 0.0)
            seen += 1
    if not seen:
        return None
    # A PARTIAL SWEEP IS A LOWER BOUND, AND MUST SAY SO.
    #
    # Reporting some of the generations as if it were the bill understates the
    # run and nothing in the number shows it. The shortfall rides along with
    # the figure so a reader sees both.
    partial = failed > 0
    return {"total": total, "total_cost_usd": round(cost, 6),
            "generations": seen, "generations_unreadable": failed,
            "is_lower_bound": partial,
            "source": (
                f"OpenRouter /api/v1/generation, {seen} generation(s)"
                + (f" of {len(unique)}; {failed} could not be read within the "
                   f"{OPENROUTER_SWEEP_SECONDS:.0f}s sweep budget, so this "
                   f"total is a LOWER BOUND" if partial else ""))}


def read_stream(path: Path) -> dict[str, Any]:
    """Usage over time, the model, and the agent's own total, from one stream.

    Returns `events` as `(unix_seconds, cumulative_usage)` so a version
    captured at time T can be charged for what had been spent by T. The stream
    carries a `t` field per line -- the harness's own timestamp -- which is
    what makes that possible; the API's own events carry no wall clock.
    """
    path = Path(path)
    out: dict[str, Any] = {
        "stream": str(path),
        "model": "",
        "reported_total_cost_usd": None,
        "reported_usage": None,
        "duration_api_ms": None,
        "timeline": [],
        "total": dict(ZERO),
    }
    if not path.is_file():
        out["unavailable_because"] = (
            f"{path} does not exist, so no usage was recorded and no cost can "
            f"be attributed. Absent, never zero: a run whose stream is missing "
            f"did not cost nothing")
        return out

    running = dict(ZERO)
    for line in path.read_text("utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        event = row.get("event") or {}
        when = row.get("t")
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            out["model"] = str(event.get("model") or "")
        elif kind == "assistant":
            message = event.get("message") or {}
            usage = message.get("usage") or {}
            # The generation id, kept whether or not this usage block is
            # populated: it is what makes an OpenRouter run's real numbers
            # recoverable when the block comes back zeroed.
            if str(message.get("id") or "").startswith("gen-"):
                out.setdefault("generation_ids", []).append(message["id"])
            if usage:
                _add(running, usage)
                out["timeline"].append((float(when or 0.0), dict(running)))
        elif kind == "result":
            # THE AGENT'S OWN FIGURE. Authoritative for the total.
            if event.get("total_cost_usd") is not None:
                out["reported_total_cost_usd"] = float(event["total_cost_usd"])
            out["reported_usage"] = event.get("usage")
            out["duration_api_ms"] = event.get("duration_api_ms")
    out["total"] = running

    return out


def recover_openrouter(read: dict[str, Any], *,
                       token: str = "", force: bool = False) -> dict[str, Any]:
    """Fill in a zeroed OpenRouter run's real usage and cost, in place.

    A SEPARATE STEP from `read_stream` on purpose. `read_stream` parses a file
    and touches no network, which is what lets it run anywhere and be tested
    without one; this reaches out to a billing API and can fail for reasons
    that have nothing to do with the stream. Folding the two together would
    make a network failure look like an unreadable stream.

    Why it is needed at all: Claude Code against OpenRouter emits the
    Anthropic-shaped usage block with EVERY FIELD ZERO and no
    `result.total_cost_usd`. An unmodified read reports `$0.00`, which is the
    most dangerous possible answer: plausible, silent, and it makes an
    expensive run look free.

    Returns the same dict, annotated. Absent stays absent: a run whose cost
    could not be recovered records WHY, never a zero.
    """
    ids = read.get("generation_ids") or []
    running = read.get("total") or {}
    if not ids:
        return read
    if not force and any(running.get(k) for k in
                         ("input_tokens", "output_tokens", "cache_read_tokens")):
        return read                      # the stream already had real numbers
    # `force` is for a stream that reported real numbers AND THEN STOPPED.
    #
    # The bail-out above is right for a healthy stream and wrong for a stalled
    # one, which carries genuine usage for a while and then freezes its
    # counters with most of the run still to come. The billing API is the only
    # source for the rest.

    # THE RUN'S OWN KEY, NOT THE AMBIENT ONE.
    #
    # A generation is only visible to the account that created it, so the token
    # that can price this run is the one the RUN used -- the per-run secrets
    # file named by `--agent-env`. The process environment may export an
    # `OPENROUTER_API_KEY` belonging to another account, which would ask about
    # generations it does not own: every request 404s, indistinguishable from
    # a network failure.
    #
    # So: the caller's explicit token first, then the secrets tree, and the
    # ambient environment LAST and only when it is the same account.
    if not token:
        token = _token_from_secrets()
    recovered = _openrouter_usage(ids, token)
    if not recovered:
        read["usage_unavailable_because"] = (
            f"the stream carries {len(set(ids))} distinct OpenRouter "
            f"generation id(s) and a zeroed usage block, and OpenRouter could "
            f"not be asked what it billed. The cost is ABSENT rather than "
            f"zero: a run that consumed tokens did not cost nothing")
        return read
    read["total"] = recovered["total"]
    read["reported_total_cost_usd"] = recovered["total_cost_usd"]
    read["usage_recovered_from"] = recovered["source"]
    # THE TIMELINE IS KEPT, because it is what splits the total by version.
    #
    # Collapsing it to a single point at t=0 would put every penny on the
    # FIRST version: `attribute` differences consecutive capture times, so
    # with all usage at t=0 the earliest version absorbs everything and the
    # rest difference to zero.
    #
    # OpenRouter's per-generation records carry no timestamps we can align to
    # version captures, so the stream's own timeline is the only shape
    # available -- and it is a good one: `attribute` uses it for PROPORTIONS
    # and scales those to the recovered total, so the split is the stream's and
    # the total is OpenRouter's. Both halves come from the source that actually
    # knows them.
    if not read.get("timeline"):
        read["timeline"] = [(0.0, dict(recovered["total"]))]
    read["timeline_note"] = (
        "shape from the agent stream, total from OpenRouter's billing record")
    return read


def _token_from_secrets() -> str:
    """The OpenRouter token from the gitignored secrets tree, or "".

    Read here rather than passed in because `read_stream` is called from
    several places and none of them should have to know about credentials.
    Never logged, never returned in any body this module produces.
    """
    root = Path(__file__).resolve().parents[3] / "serverbench/artifacts/secrets"
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


def weighted(usage: dict[str, int], rates: dict[str, Any]) -> float:
    """The usage in dollars under these weights. A relative quantity."""
    return (
        usage["input_tokens"] / 1e6 * float(rates["input"])
        + usage["output_tokens"] / 1e6 * float(rates["output"])
        + usage["cache_write_tokens"] / 1e6 * float(rates["cache_write"])
        + usage["cache_read_tokens"] / 1e6 * float(rates["cache_read"]))


def _delta(later: dict[str, int], earlier: dict[str, int]) -> dict[str, int]:
    return {k: later.get(k, 0) - earlier.get(k, 0) for k in ZERO}


def _token_sum(entry: dict[str, Any]) -> int:
    """Every token field of one cumulative usage snapshot."""
    return sum(int(entry.get(k) or 0) for k in
               ("input_tokens", "output_tokens", "cache_write_tokens",
                "cache_read_tokens"))


def stalled_share(read: dict[str, Any]) -> float:
    """What fraction of a run's messages arrive after its last token update.

    Near zero for a stream reporting usage normally. `STALL_LIMIT` is where it
    stops being evidence for a per-version split.
    """
    timeline = read.get("timeline") or []
    if not timeline:
        return 0.0
    last = timeline[-1][1]
    total_messages = int(last.get("messages") or 0)
    if not _token_sum(last) or not total_messages:
        return 0.0
    grew_at, seen = 0, 0
    for _when, entry in timeline:
        current = _token_sum(entry)
        if current > seen:
            seen, grew_at = current, int(entry.get("messages") or 0)
    return max(0.0, (total_messages - grew_at) / total_messages)


def _timeline_stalled(read: dict[str, Any]) -> bool:
    return stalled_share(read) > STALL_LIMIT


#: How much of a run may fall after the last token update before the stream's
#: token split stops being evidence. A provider reporting usage normally puts
#: essentially every message before its final increase; one whose counters
#: freeze part-way would charge every version past the freeze $0.0000. See
#: `attribute()`.
STALL_LIMIT = 0.10


def attribute(stream: Path, captured_at: list[tuple[str, float]]
              ) -> dict[str, Any]:
    """Per-version API cost, from a stream and each version's capture time.

    `captured_at` is `[(version_id, unix_seconds)]` in any order. Each version
    is charged for the usage between the previous capture and its own -- the
    work that produced it -- so the versions partition the run rather than each
    carrying the run's running total.

    When the agent reported a session total, the per-version figures are
    RESCALED to sum to it, and both numbers are recorded. That way the price
    table decides only the split and the agent's own billing decides the
    amount.
    """
    read = read_stream(stream)
    if "unavailable_because" in read:
        return {"available": False, **read}
    # An OpenRouter-hosted run reports every usage count as zero. Ask the
    # billing API what it actually was, before anything is apportioned from it.
    #
    # `force` covers the other shape: a stream that reported usage for a while
    # and then stopped, which the zero-check below cannot see because the
    # numbers it finds are real -- just badly incomplete. See `STALL_LIMIT`.
    read = recover_openrouter(read, force=_timeline_stalled(read))

    rates = rates_for(read["model"])
    ordered = sorted(captured_at, key=lambda pair: pair[1])

    def cumulative(at: float) -> dict[str, int]:
        found = dict(ZERO)
        for when, running in read["timeline"]:
            if when <= at:
                found = running
            else:
                break
        return found

    per_version: dict[str, dict[str, Any]] = {}
    previous = dict(ZERO)
    for version_id, when in ordered:
        upto = cumulative(when)
        spent = _delta(upto, previous)
        per_version[version_id] = {
            "captured_unix": when,
            "usage": spent,
            "cumulative_usage": dict(upto),
            "cost_usd_unscaled": round(weighted(spent, rates), 6),
        }
        previous = upto

    computed_total = round(weighted(read["total"], rates), 6)

    # WHEN THE TOKENS ARE ALL ZERO, SPLIT BY MESSAGES.
    #
    # OpenRouter returns a zeroed `usage` block on every streamed message, so
    # for those runs the timeline carries `messages: 1..N` and nothing else.
    # `weighted()` over all-zero usage is 0, every version's slice is 0, and
    # the recovered total -- which IS accurate -- has nothing to be spread
    # across.
    #
    # Messages are a real signal and the only one these runs carry: they are
    # spread across the run, and the count between two captures is proportional
    # to the work done for that version in a way that "all of it" and "none of
    # it" are not. Recorded as the basis, because a proportion derived from
    # message counts is a coarser thing than one derived from tokens and a
    # reader should know which they have.
    # THE CONDITION IS ABOUT THE TIMELINE, and getting it wrong costs either
    # every Claude run or every OpenRouter one.
    #
    # Not "the total is zero": `recover_openrouter` fills the total from the
    # billing API, so it is non-zero even when every timeline point is zeroed.
    # Not "the slices are empty" either -- that is also true whenever the
    # captures happen to slice nothing, and would wipe the cost off every
    # Claude run.
    #
    # What is actually being detected is a provider that reports no token
    # usage per message. That shows up in the TIMELINE and nowhere else: its
    # final cumulative entry has messages and no tokens.
    # A TIMELINE CAN ALSO STALL PART-WAY, WHICH IS WORSE THAN CARRYING NOTHING.
    #
    # `tokens_in_timeline == 0` catches a provider that reports no usage at
    # all. Some providers report usage for a while and then stop, while the
    # messages keep coming. The timeline is then not EMPTY, so that fallback
    # never fires, and because the split is a difference of cumulative totals,
    # every version after the freeze would be charged exactly $0.0000.
    #
    # `$0.00` is the most dangerous answer available here: it is plausible, it
    # is silent, and it makes an expensive run look free. So the test is not
    # "are there tokens" but "do the tokens cover the run" -- measured as the
    # share of messages that arrive after the last time any token counter
    # moved. A healthy stream puts that near zero.
    timeline = read.get("timeline") or []
    last = timeline[-1][1] if timeline else {}
    tokens_in_timeline = _token_sum(last)
    stall = stalled_share(read)

    split_basis = "weighted token usage from the stream's own usage blocks"
    #: True when `cost_usd_unscaled` holds a PROPORTION (a message count)
    #: rather than a currency amount. Such a split is meaningless until it is
    #: rescaled to a total that came from somewhere else, and if no such total
    #: exists the cost is ABSENT -- never the raw proportion, which would
    #: report an N-message run as costing $N.
    proportional_only = False
    if stall > STALL_LIMIT:
        split_basis = (
            f"message count -- the provider's token counters stopped advancing "
            f"with {stall:.0%} of the run's messages still to come, so a split "
            f"by token deltas would charge those versions nothing")
        proportional_only = True
        for body in per_version.values():
            body["cost_usd_unscaled"] = float(
                (body["usage"] or {}).get("messages") or 0)
        computed_total = sum(body["cost_usd_unscaled"]
                             for body in per_version.values())
    if tokens_in_timeline == 0 and last.get("messages"):
        split_basis = ("message count -- the provider returned no token usage, "
                       "so the split is by how many messages fell between one "
                       "version and the next")
        for body in per_version.values():
            body["cost_usd_unscaled"] = float(
                (body["usage"] or {}).get("messages") or 0)
        # The denominator is the SUM OF THE SLICES, not the run's message
        # total. Those differ -- the slices sum to the messages that fell
        # between the first and last capture, and the run's total also counts
        # everything before the first version and after the last -- and using
        # the larger one would make the versions sum to MORE than the bill.
        # Dividing by what was actually sliced
        # makes the versions sum to the reported total exactly, which is the
        # only arithmetic a reader can check.
        computed_total = sum(body["cost_usd_unscaled"]
                             for body in per_version.values())
        proportional_only = True

    reported = read["reported_total_cost_usd"]

    # THE CLI'S OWN FIGURE IS NOT ALWAYS THIS RUN'S BILL.
    #
    # It is for a model Anthropic serves. It is not for one reached through
    # OpenRouter, where the CLI prices what it sees with Anthropic's card --
    # see `reported_total_is_trustworthy`. Discarded rather than averaged in.
    # A RECOVERED total is exempt: `recover_openrouter` replaces the CLI's
    # figure with what OpenRouter says it actually billed, which is the most
    # accurate number available for those models and the whole reason that
    # recovery exists.
    recovered = bool(read.get("usage_recovered_from"))
    discarded = None
    if (reported is not None and not recovered
            and not reported_total_is_trustworthy(read["model"])):
        discarded, reported = reported, None

    scale = 1.0
    if reported is not None and computed_total > 0:
        scale = reported / computed_total

    # An unfinished run has no reported total, and the computed one overstates
    # by about four times. Still reported -- absent is worse than approximate
    # -- but labelled, with the calibrated figure beside it.
    factor = calibrated_for(read["model"]) if reported is None else None
    calibrated = (round(computed_total * factor, 6)
                  if factor and computed_total else None)
    if reported is None and calibrated is not None and computed_total > 0:
        # The SPLIT is unaffected -- it is proportional -- but each version's
        # `cost_usd` should carry the calibrated scale so the per-version
        # numbers sum to the calibrated total rather than the raw one.
        scale = calibrated / computed_total
    # A PROPORTION IS NOT A PRICE.
    #
    # When the split fell back to message counts, `cost_usd_unscaled` holds a
    # message COUNT. It only becomes money once multiplied by a real total, and
    # the only real totals are the agent's own reported figure or OpenRouter's
    # billing record. Without one of those there is nothing to scale by, and
    # emitting the raw count would report a message count as dollars.
    #
    # So the cost goes ABSENT, with the reason attached. That is the same rule
    # `recover_openrouter` already follows for an unrecoverable total, and for
    # the same reason: a wrong number is worse than a missing one, because only
    # the missing one prompts anybody to look.
    no_total_to_scale_by = proportional_only and reported is None
    if no_total_to_scale_by:
        why = (
            f"the provider stopped reporting token usage {stall:.0%} of the "
            f"way through the run, so the per-version split could only be made "
            f"by message count -- a proportion. Scaling it into dollars needs a "
            f"run total, and none is available: the agent recorded none, and "
            f"OpenRouter's billing record could not be read "
            f"({read.get('usage_unavailable_because') or 'lookup did not run'}). "
            f"This run DID cost money; that amount is not knowable from the "
            f"evidence here")
        for body in per_version.values():
            body["cost_usd"] = None
            body["cost_unavailable_because"] = why
    else:
        for body in per_version.values():
            body["cost_usd"] = round(body["cost_usd_unscaled"] * scale, 6)

    return {
        "available": True,
        "model": read["model"],
        "rates": {k: v for k, v in rates.items() if k != "source"},
        "rates_source": rates["source"],
        "rates_note": (
            "these weights decide how the total is SPLIT between versions, "
            "never what the total is. The total is the agent's own "
            "`total_cost_usd` where it recorded one"),
        "run_total_cost_usd": (
            None if no_total_to_scale_by else
            (reported if reported is not None
             else (calibrated if calibrated is not None else computed_total))),
        "stalled_share": round(stall, 4),
        "cost_unavailable_because": (why if no_total_to_scale_by else ""),
        "run_total_from": (
            (f"OpenRouter's own billing record ({read['usage_recovered_from']})"
             if recovered else "the agent's own result event")
            if reported is not None
            else (f"the stream's usage, calibrated by "
                  f"{1.0 / STREAM_OVERSTATES_BY:.2f} because the run recorded "
                  f"no total -- an unfinished or crashed run. "
                  f"{CALIBRATION_EVIDENCE}"
                  if calibrated is not None else
                  "computed from usage and the weights above, because the run "
                  "recorded no total -- which is what an unfinished or crashed "
                  "run looks like")),
        "run_total_is_exact": reported is not None,
        "run_total_is_upper_bound": reported is None and calibrated is None,
        "calibration": {
            "applied": factor,
            "raw_computed_total_usd": computed_total,
            "calibrated_total_usd": calibrated,
            "evidence": CALIBRATION_EVIDENCE if factor else "",
            "note": ("the stream's per-message usage blocks do not reconcile "
                     "to the bill; summing them overstates it about fourfold. "
                     "Where the run reported its own total that is used "
                     "instead and this is ignored"),
        },
        "reported_total_discarded_usd": discarded,
        "reported_total_discarded_why": (
            f"the CLI reported ${discarded} for {read['model']}, which "
            f"OpenRouter serves -- the CLI prices every model with Anthropic's "
            f"card, so that figure is this model's tokens at Opus rates. The "
            f"rate table and `recover_openrouter` are the authorities here"
            if discarded is not None else ""),
        "reconciliation": {
            "reported_total_cost_usd": reported,
            "computed_total_cost_usd": computed_total,
            "scale_applied": round(scale, 6),
            "note": ("both are recorded so a drifted price table shows up here "
                     "rather than quietly moving every version's cost. A scale "
                     "far from 1.0 means the table needs updating"),
        },
        "run_usage": read["total"],
        "split_is_approximate": True,
        "split_caveat": (
            "The TOTAL is exact -- it is the agent's own billing figure. The "
            "SPLIT between versions is proportional and approximate, because "
            "the per-message `usage` blocks in the stream do not reconcile to "
            "the `result` event's totals. More than one "
            "`assistant` event per turn carries a usage block, and a streamed "
            "one carries a partial count. What the series IS reliably is "
            "monotone in time, which is what a proportional split needs. Read "
            "a version's cost as its share of the run, not as an invoice."),
        # Computed above: token-weighted normally, message-count when the
        # provider returned no token usage at all. A reader needs to know
        # which of the two produced the numbers beside it.
        "split_basis": split_basis + (
            ", differenced between consecutive captures, then rescaled so the "
            "versions sum to the reported total"),
        "duration_api_ms": read["duration_api_ms"],
        "versions": per_version,
        "unattributed_note": (
            "usage recorded before the first capture is charged to the first "
            "version: it is the work that produced it. Usage after the last "
            "capture is not charged to any version, because no version came "
            "out of it"),
    }
