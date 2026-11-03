"""`dollars_per_dollar` is scored per request; eligibility gates stay independent.

Each profiling request scores 1 when its TTFT and TPOT both meet the p90 bars,
otherwise 0 (errors and cancellations included). `dollars_per_dollar` is
the qualifying requests' token value divided by full-window GPU cost.
The gross figure and request-count fraction are diagnostics only.
"""
from __future__ import annotations

import copy
import json
import math

import pytest
from final_validate import objective, scenario, spine

CONFIG = {
    "kind": "request_p90_billable_tokens_v4",
    "gross_objective": "gross_dollars_per_dollar",
    "attainment_basis": "profiling_request_count",
    "aggregation": "sum_token_value",
    "request_scores": {"p90_both": 1.0, "otherwise": 0.0},
}
TASK = "qwen3_5_9b_instructed_2xL40"
PRICES = {"token_prices": {"usd_per_input_token": 1e-6,
                            "usd_per_output_token": 1e-6,
                            "usd_per_cached_input_token": 1e-6}}


def _checks():
    return [{"percentile": 90, "ttft_ms": 50, "ttft_ms_gate": 100,
             "tpot_ms": 5, "tpot_ms_gate": 10}]


def _records(n=100):
    return [{"metadata": {"benchmark_phase": "profiling", "x_request_id": str(i)},
             "metrics": {"time_to_first_token": {"value": 50, "unit": "ms"},
                         "inter_token_latency": {"value": 5, "unit": "ms"},
                         "input_sequence_length": {"value": 10000, "unit": "tokens"},
                         "output_sequence_length": {"value": 10000, "unit": "tokens"}}}
            for i in range(n)]


def _write(tmp_path, records):
    (tmp_path / "profile_export.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    return str(tmp_path)


def _score(tmp_path, records, **kwargs):
    errors = kwargs.pop("expected_errors", 0)
    for name in ("input", "output"):
        total = sum(objective._count(r.get("metrics", {}).get(
                        f"{name}_sequence_length", {}).get("value")) or 0
                    for r in records if not r.get("error")
                    and not r.get("metadata", {}).get("was_cancelled")
                    and r.get("metadata", {}).get("benchmark_phase") == "profiling")
        kwargs.setdefault(f"expected_{name}_tokens", total)
    kwargs.setdefault("prices", PRICES)
    kwargs.setdefault("duration_s", 1800)
    kwargs.setdefault("usd_per_hour", 2)
    return objective.request_scored(
        kwargs.pop("gross", 2.0), kwargs.pop("checks", _checks()),
        kwargs.pop("config", CONFIG), artifact_dir=_write(tmp_path, records),
        expected_requests=kwargs.pop("expected_requests", len(records) - (errors or 0)),
        expected_errors=errors, **kwargs)


def test_the_names():
    assert objective.OBJECTIVE == "dollars_per_dollar"
    assert objective.GROSS == "gross_dollars_per_dollar"


def test_only_qualifying_requests_own_tokens_are_billed(tmp_path):
    rows = _records(2)
    rows[0]["metrics"]["input_sequence_length"]["value"] = 100
    rows[0]["metrics"]["output_sequence_length"]["value"] = 20
    rows[1]["metrics"]["time_to_first_token"]["value"] = 101
    prices = {"token_prices": {"usd_per_input_token": .001,
                               "usd_per_output_token": .01,
                               "usd_per_cached_input_token": .001}}
    first = _score(tmp_path, rows, prices=prices)
    assert first["billable_value_usd"] == pytest.approx(.3)
    assert first["dollars_per_dollar"] == .3  # $1 whole-window GPU cost
    assert first["request_multiplier"] == .5  # diagnostic, not revenue weight
    rows[1]["metrics"]["input_sequence_length"]["value"] = 10**9
    rows[1]["metrics"]["output_sequence_length"]["value"] = 10**8
    inflated = _score(tmp_path, rows, prices=prices, gross=10**9)
    assert inflated["dollars_per_dollar"] == first["dollars_per_dollar"]
    assert inflated["billable_input_tokens"] == 100
    assert inflated["billable_output_tokens"] == 20
    assert _score(tmp_path, rows, prices=prices, duration_s=3600)["dollars_per_dollar"] == .15


def test_revenue_is_summed_before_rounding_and_server_usage_is_ignored(tmp_path):
    rows = _records(100)
    for row in rows:
        row["metrics"]["input_sequence_length"]["value"] = 1
        row["metrics"]["output_sequence_length"]["value"] = 2
        row["metrics"]["input_token_count"] = {"value": 10**12, "unit": "tokens"}
        row["usage"] = {"prompt_tokens": 10**12, "completion_tokens": 10**12}
    rates = {"token_prices": {"usd_per_input_token": 1e-8,
                              "usd_per_output_token": 1e-8,
                              "usd_per_cached_input_token": 1e-8}}
    got = _score(tmp_path, rows, prices=rates, usd_per_hour=.000002)
    assert got["billable_value_usd"] == pytest.approx(3e-6)
    assert got["dollars_per_dollar"] == 3


@pytest.mark.parametrize("name", ["input", "output"])
@pytest.mark.parametrize("bad", [None, -1, 1.5, True, "12", math.nan, math.inf])
def test_bad_client_token_counts_fail_closed(tmp_path, name, bad):
    rows = _records(2)
    rows[0]["metrics"][f"{name}_sequence_length"]["value"] = bad
    assert _score(tmp_path, rows)["dollars_per_dollar"] is None


@pytest.mark.parametrize("field", ["expected_input_tokens", "expected_output_tokens"])
def test_client_token_totals_must_match_aggregate(tmp_path, field):
    assert _score(tmp_path, _records(2), **{field: 1})["dollars_per_dollar"] is None


def test_discounted_cache_price_requires_per_request_evidence(tmp_path):
    prices = copy.deepcopy(PRICES)
    prices["token_prices"]["usd_per_cached_input_token"] *= .1
    got = _score(tmp_path, _records(), prices=prices)
    assert got["dollars_per_dollar"] is None
    assert "per-request cache evidence" in got["request_scoring_why"]


@pytest.mark.parametrize("field", ["duration_s", "usd_per_hour"])
@pytest.mark.parametrize("bad", [None, 0, -1, math.nan, math.inf, True])
def test_invalid_gpu_cost_cannot_score(tmp_path, field, bad):
    assert _score(tmp_path, _records(), **{field: bad})["dollars_per_dollar"] is None


@pytest.mark.parametrize("met,multiplier", [(100, 1), (0, 0), (58, .58)])
def test_equal_value_requests_match_the_count_fraction(tmp_path, met, multiplier):
    rows = _records()
    for row in rows[met:]:
        row["metrics"]["time_to_first_token"]["value"] = 150
    got = _score(tmp_path, rows)
    assert got["request_multiplier"] == multiplier
    assert got["dollars_per_dollar"] == round(2 * multiplier, 4)
    assert got["request_attainment"]["request_score_counts"] == {
        "p90_both": met, "otherwise": 100 - met}
    assert "diagnostic only" in got["request_scoring_basis"]


def test_both_metrics_must_meet_their_bar_on_the_same_request(tmp_path):
    rows = _records(2)
    rows[0]["metrics"]["inter_token_latency"]["value"] = 15     # TPOT misses
    rows[1]["metrics"]["time_to_first_token"]["value"] = 250    # TTFT misses
    got = _score(tmp_path, rows)
    assert got["request_multiplier"] == 0
    assert got["request_attainment"]["request_score_counts"] == {
        "p90_both": 0, "otherwise": 2}


def test_exact_request_scores_override_interpolated_percentiles(tmp_path):
    rows = _records(118)
    for row in rows[:6]:
        row["metrics"]["time_to_first_token"]["value"] = 250
    got = _score(tmp_path, rows)  # _checks deliberately reports passing aggregates
    assert got["request_multiplier"] == 112 / 118
    assert got["request_attainment"]["terms"]["p90_ttft"]["met_requests"] == 112


def test_equal_threshold_meets_target_and_warmup_is_excluded(tmp_path):
    rows = _records()
    for row in rows:
        row["metrics"]["time_to_first_token"]["value"] = 100
        row["metrics"]["inter_token_latency"]["value"] = 10
    rows.append({"metadata": {"benchmark_phase": "warmup"}})
    got = _score(tmp_path, rows, expected_requests=100)
    assert got["request_multiplier"] == 1
    assert got["request_attainment"]["request_count"] == 100


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_requests_stay_in_denominator(tmp_path, cancelled):
    rows = _records()
    for row in rows[:17]:
        row.pop("metrics")
        if cancelled:
            row["metadata"]["was_cancelled"] = True
        else:
            row["error"] = {"type": "TimeoutError"}
    got = _score(tmp_path, rows, expected_errors=0 if cancelled else 17)
    assert got["request_multiplier"] == .83
    assert got["request_attainment"]["terms"]["p90_ttft"]["met_requests"] == 83


def test_aiperf_sparse_error_count_and_separate_success_count(tmp_path):
    assert _score(tmp_path, _records(), expected_errors=None)["request_multiplier"] == 1
    rows = _records(94)
    for row in rows[:65]:
        row["error"] = {"type": "TimeoutError"}
    got = _score(tmp_path, rows, expected_requests=29, expected_errors=65)
    assert got["request_multiplier"] == 29 / 94
    assert got["request_attainment"]["request_count"] == 94


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -1, True, "5"])
def test_invalid_latency_is_unavailable_not_zero_credit(tmp_path, value):
    rows = _records()
    rows[0]["metrics"]["inter_token_latency"]["value"] = value
    got = _score(tmp_path, rows)
    assert got["dollars_per_dollar"] is None
    assert got["request_multiplier"] is None
    assert got["request_scoring_why"]


@pytest.mark.parametrize("fault", ["duplicate", "phase", "missing_tpot", "unit", "skip",
                                   "truncated", "error_count", "empty"])
def test_bad_exports_fail_closed(tmp_path, fault):
    rows = _records()
    kwargs = {}
    if fault == "duplicate":
        rows[0]["metadata"]["x_request_id"] = "1"
    elif fault == "phase":
        del rows[0]["metadata"]["benchmark_phase"]
    elif fault == "missing_tpot":
        del rows[0]["metrics"]["inter_token_latency"]
    elif fault == "unit":
        rows[0]["metrics"]["time_to_first_token"]["unit"] = "s"
    elif fault == "skip":
        rows[0]["metadata"]["context_overflow_skip"] = True
    elif fault == "truncated":
        kwargs["expected_requests"] = 101
    elif fault == "error_count":
        kwargs["expected_errors"] = 1
    elif fault == "empty":
        rows = []
    assert _score(tmp_path, rows, **kwargs)["dollars_per_dollar"] is None


def test_missing_and_malformed_exports_fail_closed(tmp_path):
    kwargs = {"artifact_dir": tmp_path, "expected_requests": 100, "expected_errors": 0}
    assert objective.request_scored(2, _checks(), CONFIG, **kwargs)["request_scoring_why"]
    (tmp_path / "profile_export.jsonl").write_text('{"metadata":')
    assert objective.request_scored(2, _checks(), CONFIG, **kwargs)["request_scoring_why"]


def test_a_p95_row_is_refused(tmp_path):
    checks = _checks() + [{"percentile": 95, "ttft_ms_gate": 200, "tpot_ms_gate": 20}]
    assert _score(tmp_path, _records(3), checks=checks)["dollars_per_dollar"] is None


@pytest.mark.parametrize("field,value", [("kind", "request_latency_multiplier_v2"),
                                         ("gross_objective", "dollars_per_dollar"),
                                         ("aggregation", "mean"),
                                         ("attainment_basis", "percentile"),
                                         ("request_scores", {})])
def test_unknown_policy_is_not_silently_accepted(tmp_path, field, value):
    config = {**CONFIG, field: value}
    assert objective.request_scoring_error(config)
    assert _score(tmp_path, _records(), config=config)["dollars_per_dollar"] is None


def test_request_score_configuration_is_versioned():
    assert objective.request_scoring_error(CONFIG) == ""
    for field, value in [("p90_both", .9), ("otherwise", .1)]:
        config = copy.deepcopy(CONFIG)
        config["request_scores"][field] = value
        assert objective.request_scoring_error(config)
    config = copy.deepcopy(CONFIG)
    config["request_scores"]["p95_both"] = .5
    assert objective.request_scoring_error(config)


def _decision_row(tmp_path):
    rows = _records()
    for row in rows:
        row["metrics"]["time_to_first_token"]["value"] = 60_000
        row["metrics"]["inter_token_latency"]["value"] = 140
    return {
        "arm": "candidate", "artifact_dir": _write(tmp_path, rows),
        "correctness": {"holds": True},
        "stream_timing": {"buffered": False},
        "submission_valid": True, "was_cancelled": False,
        "command": ["--benchmark-duration", "1800"],
        "aggregate": {
            "benchmark_duration_s": 1800, "request_count": 100, "error_request_count": 0,
            "total_input_tokens": 1_000_000, "total_output_tokens": 1_000_000,
            "theoretical_prefix_cache_hit": {"count": 1, "sum": 0},
            "ttft_p90_ms": 60_000, "tpot_p90_ms": 140,
        },
    }


def test_development_and_final_use_the_same_score(tmp_path, capsys):
    contract = spine.load_contract(TASK)
    row = _decision_row(tmp_path)
    verdict = scenario.decide(contract, [row], 1.64)
    decision = verdict["arms"][0]
    report = scenario.development_report(contract, row, usd_per_hour=1.64, mode="full")
    assert decision["outcome"] == "HOLDS"
    assert decision["request_multiplier"] == 0
    assert decision["dollars_per_dollar"] == 0
    assert decision["gross_dollars_per_dollar"] > 0
    assert verdict["objective"] == "dollars_per_dollar"
    assert verdict["best_objective_value"] == 0
    assert report["objective"] == "dollars_per_dollar"
    assert report["objective_value"] == decision["dollars_per_dollar"]
    assert report["gross_dollars_per_dollar"] == decision["gross_dollars_per_dollar"]
    assert report["request_attainment"] == decision["request_attainment"]
    assert not report["hard_gates_hold"]
    scenario._print_request_scores(report)
    output = capsys.readouterr().out
    assert "both p90 bars met: 0/100" in output and "otherwise: 100/100" in output
    assert "0.000000" in output and "<- the objective" in output
    assert "decides nothing" in output


@pytest.mark.parametrize("task", [
    "qwen3_5_2b_standard_1xL40", "qwen3_5_9b_standard_2xL40",
    "gemma4_e4b_it_standard_1xL40", "qwen3_5_9b_standard_modal_2xL40S",
    "gemma4_e4b_it_standard_modal_1xL40S", "qwen3_5_9b_instructed_2xL40",
    "gemma4_e4b_it_instructed_1xL40",
])
def test_all_tasks_bill_a_mixed_population_identically_in_development_and_final(tmp_path, task):
    row = _decision_row(tmp_path)
    rows = _records(2)
    rows[0]["metrics"]["input_sequence_length"]["value"] = 100
    rows[0]["metrics"]["output_sequence_length"]["value"] = 20
    rows[1]["metrics"]["input_sequence_length"]["value"] = 1000000
    rows[1]["metrics"]["output_sequence_length"]["value"] = 100000
    rows[1]["metrics"]["time_to_first_token"]["value"] = 10**9
    row["artifact_dir"] = _write(tmp_path, rows)
    row["aggregate"].update(request_count=2, total_input_tokens=1000100,
                            total_output_tokens=100020)
    row["correctness"] = {"replay_fidelity": {"holds": True},
                           "output_correctness": {"evaluable": True, "holds": True}}
    detail = scenario.objective_detail(task)
    hourly = detail["usd_per_hour_total"]
    prices = detail["token_prices"]
    revenue = 100 * prices["usd_per_input_token"] + 20 * prices["usd_per_output_token"]
    contract = spine.load_contract(task)
    final = scenario.decide(contract, [row], hourly)["arms"][0]
    dev_row = {**row, "correctness": {"holds": True}}
    dev = scenario.development_report(contract, dev_row, usd_per_hour=hourly, mode="full")
    assert final["outcome"] == "HOLDS"
    assert final["dollars_per_dollar"] == dev["objective_value"] == round(revenue / (hourly / 2), 4)
    assert final["billable_value_usd"] == dev["billable_value_usd"] == revenue
    assert final["billable_input_tokens"] == dev["billable_input_tokens"] == 100
    assert final["billable_output_tokens"] == dev["billable_output_tokens"] == 20
    assert final["request_multiplier"] == .5
    assert final["dollars_per_dollar"] != round(final["gross_dollars_per_dollar"] * .5, 4)


def test_the_gross_figure_never_decides(tmp_path):
    """A higher gross figure with a lower score loses: only the score ranks."""
    contract = spine.load_contract(TASK)
    (tmp_path / "fast").mkdir()
    (tmp_path / "slow").mkdir()
    fast = _decision_row(tmp_path / "slow")
    fast["artifact_dir"] = _write(tmp_path / "fast", _records())  # meets the bars
    slow = _decision_row(tmp_path / "slow")                       # meets none
    slow["arm"] = "sglang"
    slow["aggregate"] = {**slow["aggregate"], "total_output_tokens": 4_000_000}
    slow_rows = _records()
    for r in slow_rows:
        r["metrics"]["output_sequence_length"]["value"] = 40000
        r["metrics"]["time_to_first_token"]["value"] = 60000
    _write(tmp_path / "slow", slow_rows)
    verdict = scenario.decide(contract, [fast, slow], 1.64)
    arms = {a["arm"]: a for a in verdict["arms"]}
    assert (arms["sglang"]["gross_dollars_per_dollar"]
            > arms["candidate"]["gross_dollars_per_dollar"])
    assert arms["sglang"]["dollars_per_dollar"] < arms["candidate"]["dollars_per_dollar"]
    assert verdict["best_holding_arm"] == "candidate"


@pytest.mark.parametrize("failed_gate", ["replay_fidelity", "output_correctness"])
def test_eligibility_failures_never_become_selected_scores(tmp_path, failed_gate):
    row = _decision_row(tmp_path)
    row["correctness"] = {
        "holds": True,  # stale conjunction must not override the failed child
        "replay_fidelity": {"holds": True, "evaluable": True},
        "output_correctness": {"holds": True, "evaluable": True},
    }
    row["correctness"][failed_gate]["holds"] = False
    verdict = scenario.decide(spine.load_contract(TASK), [row], 1.64)
    assert verdict["arms"][0]["outcome"] == "MISSES"
    assert verdict["best_objective_value"] is None
    assert verdict["arms"][0]["dollars_per_dollar"] is None


@pytest.mark.parametrize("fault", ["buffered", "partial", "cancelled", "invalid"])
def test_invalid_runs_cannot_earn_a_score(tmp_path, fault):
    row = _decision_row(tmp_path)
    if fault == "buffered":
        row["stream_timing"]["buffered"] = True
    elif fault == "partial":
        row["evidence_is_partial"] = True
    elif fault == "cancelled":
        row["was_cancelled"] = True
    else:
        row["submission_valid"] = False
    d = scenario.decide(spine.load_contract(TASK), [row], 1.64)["arms"][0]
    assert d["outcome"] == "MISSES"
    assert d["dollars_per_dollar"] is None


def _meeting_row(tmp_path):
    """A decision row whose every request meets the bars: a nonzero score."""
    row = _decision_row(tmp_path)
    row["artifact_dir"] = _write(tmp_path, _records())
    return row


@pytest.mark.parametrize("failed_gate", ["replay_fidelity", "output_correctness"])
def test_a_misses_decision_carries_no_objective(tmp_path, failed_gate):
    """A verdict that fails a hard gate scores nothing; the measured number is
    kept only under a diagnostic name no ranking reads."""
    row = _meeting_row(tmp_path)
    row["correctness"] = {
        "replay_fidelity": {"holds": True, "evaluable": True},
        "output_correctness": {"holds": True, "evaluable": True}}
    row["correctness"][failed_gate].update(holds=False, why="failed")
    d = scenario.decide(spine.load_contract(TASK), [row], 1.64)["arms"][0]
    assert d["outcome"] == "MISSES"
    assert d["dollars_per_dollar"] is None
    assert d["dollars_per_dollar_if_eligible"] > 0
    assert d["gross_dollars_per_dollar"] > 0          # a diagnostic, kept


def test_a_failed_replay_withholds_the_development_score(tmp_path, capsys):
    row = _meeting_row(tmp_path)
    row["correctness"] = {"holds": False, "why": "replay failed"}
    contract = spine.load_contract(TASK)
    report = scenario.development_report(contract, row, usd_per_hour=1.64,
                                         mode="full")
    assert report["full_window_complete"] is True
    assert report["objective_value"] is None
    assert report["objective_value_if_eligible"] > 0
    scenario._print_request_scores(report)
    assert "WITHHELD" in capsys.readouterr().out
    held = scenario.decide(contract, [_meeting_row(tmp_path)], 1.64)["arms"][0]
    assert held["outcome"] == "HOLDS" and held["dollars_per_dollar"] > 0
    assert "dollars_per_dollar_if_eligible" not in held


def test_submission_valid_is_read_where_agentx_writes_it(tmp_path):
    """AgentX writes the stamp under `metadata`, not at the top level."""
    body = {"metadata": {"scenario": "inferencex-agentx-mvp",
                         "submission_valid": False,
                         "submission_invalid_reasons": ["unsafe_override"]}}
    assert scenario.submission_valid_of(body) is False
    assert scenario.submission_invalid_reasons_of(body) == ["unsafe_override"]
    assert scenario.submission_valid_of({"submission_valid": False}) is False
    assert scenario.submission_valid_of(
        {"metadata": {"submission_valid": True}, "submission_valid": False}) is True
    assert scenario.submission_valid_of({}) is None
    (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(body))
    with pytest.raises(SystemExit, match="unsafe_override"):
        scenario.read_aggregate(tmp_path)
    assert scenario.read_aggregate(tmp_path, allow_diagnostic=True) == body


@pytest.mark.parametrize("private", [False, True])
def test_measure_one_records_the_metadata_stamp(tmp_path, monkeypatch, private):
    from types import SimpleNamespace

    module = SimpleNamespace(
        MODEL_ID="m", ENVIRONMENT={},
        invocation=lambda **_: ["aiperf", "profile", "--random-seed", "2",
                                "--benchmark-duration", "1800"])
    monkeypatch.setattr(scenario, "_module_of", lambda _: module)
    monkeypatch.setattr(scenario, "_pinned_aiperf_argv", lambda a: a)
    monkeypatch.setattr(scenario, "_run_watching_warmup",
                        lambda *a, **k: (SimpleNamespace(returncode=0), None))
    monkeypatch.setattr(scenario, "read_aggregate", lambda *_a, **_k: {
        "metadata": {"submission_valid": False,
                     "submission_invalid_reasons": ["context_overflow_rate"]},
        "request_count": {"avg": 8}})
    monkeypatch.setattr(scenario.stream_timing, "assess", lambda _d: {})
    row = scenario.measure_one("test", "candidate", url="http://unused",
                               tokenizer=tmp_path, run_dir=tmp_path,
                               duration_s=1800, private_workload=private)
    assert row["submission_valid"] is False
    assert row["submission_invalid_reasons"] == ["context_overflow_rate"]
    assert not scenario.submission_acceptable(row)


@pytest.mark.parametrize("window,override,role,scores", [
    (600, True, "development", True),           # our own short window
    (600, True, "final_validation", False),     # never in a verdict
    (1200, True, "development", False),         # an override we never add
    (600, False, "development", False),         # invalid for another reason
    (1800, False, "final_validation", False),   # the verdict, invalid
])
def test_only_a_short_development_override_window_still_scores(
        tmp_path, window, override, role, scores):
    row = _meeting_row(tmp_path)
    row["submission_valid"] = False
    row["command"] = (["aiperf", "profile", "--benchmark-duration", str(window)]
                      + (["--unsafe-override"] if override else []))
    row["workload_selection"] = {"role": role}
    contract = spine.load_contract(TASK)
    d = scenario.decide(contract, [row], 1.64)["arms"][0]
    assert (d["dollars_per_dollar"] is not None) is scores
    assert (d["outcome"] == "HOLDS") is scores
    if not scores:
        assert "submission_valid=false" in d["why_misses"]
    report = scenario.development_report(contract, row, usd_per_hour=1.64,
                                         mode="full")
    assert (report["objective_value"] is not None) is scores


def test_a_one_token_reply_scores_zero_and_does_not_void_the_run(tmp_path):
    """The trace can ask for exactly one output token; that reply has no TPOT.
    One such turn in a replay of hundreds must not make the whole run's
    dollars_per_dollar unavailable, in development or in the verdict."""
    rows = _records()
    del rows[0]["metrics"]["inter_token_latency"]
    rows[0]["metrics"]["output_sequence_length"] = {"value": 1, "unit": "tokens"}
    got = _score(tmp_path, rows)
    assert got["request_multiplier"] == pytest.approx(0.99)
    assert got["dollars_per_dollar"] == pytest.approx(2.0 * 0.99)
    assert got["request_attainment"]["single_token_request_count"] == 1


def test_a_missing_tpot_on_a_longer_reply_still_voids_the_score(tmp_path):
    rows = _records()
    del rows[0]["metrics"]["inter_token_latency"]
    rows[0]["metrics"]["output_sequence_length"] = {"value": 40, "unit": "tokens"}
    got = _score(tmp_path, rows)
    assert got["dollars_per_dollar"] is None
