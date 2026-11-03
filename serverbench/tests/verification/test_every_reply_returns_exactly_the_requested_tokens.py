"""The replay gate is exact: under ignore_eos every committed reply carries
exactly the requested number of native token ids.

AgentX's own output-length check tolerates min(5% of max_tokens, 50) tokens,
so a server could stop a little early on every request and still pass the
aggregate. The gate reads the raw export instead, in the packaged evaluator,
so a development measurement and the verdict apply the same rule.
"""
from __future__ import annotations

import json

import pytest
from final_validate import native_tokens, package_grader, replay_verify, scenario

CLEAN = {"request_count": {"avg": 3}, "osl_mismatch_count": {"avg": 0},
         "error_request_count": {"avg": 0}}


def _record(request_id, ids, *, wanted=None, alias=False, phase="profiling",
            ignore_eos=True, **metadata):
    wanted = len(ids) if wanted is None else wanted
    payload = {"ignore_eos": ignore_eos,
               ("max_completion_tokens" if alias else "max_tokens"): wanted}
    delta = {"content": "x" * len(ids or [])}
    if ids is not None:
        delta["token_ids"] = ids
    return {"metadata": {"x_request_id": request_id, "benchmark_phase": phase,
                         "session_num": 3, "turn_index": 1, **metadata},
            "payload": payload,
            "responses": [{"packets": [{"name": "data", "value": json.dumps(
                {"choices": [{"index": 0, "delta": delta}]})}]}]}


def _export(tmp_path, records):
    (tmp_path / "profile_export_raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), "utf-8")
    return tmp_path


def test_exact_replies_hold(tmp_path):
    run = _export(tmp_path, [
        _record("a", [1, 2, 3]), _record("b", [4] * 40, alias=True),
        # Not committed profiling replies: never counted here.
        _record("warm", [1], wanted=9, phase="warmup"),
        _record("cancel", [1], wanted=9, was_cancelled=True)])
    got = native_tokens.exact_output_lengths(run)
    assert got["holds"] is True and got["checked"] == 2
    assert scenario.replay_fidelity_of_run(CLEAN, run)["holds"] is True


def test_one_reply_one_token_short_fails_and_is_named(tmp_path):
    """Well inside AgentX's tolerance, and still a failure."""
    run = _export(tmp_path, [_record("a", [1] * 100),
                             _record("short-one", [1] * 99, wanted=100)])
    got = native_tokens.exact_output_lengths(run)
    assert got["holds"] is False and got["mismatched"] == 1
    replay = scenario.replay_fidelity_of_run(CLEAN, run)
    assert replay["holds"] is False
    assert "request short-one" in replay["why"]
    assert "returned 99 native token ids" in replay["why"]
    assert "exactly 100" in replay["why"]
    assert replay["exact_output_length"]["examples"][0]["x_request_id"] == "short-one"


def test_a_reply_with_no_native_ids_fails(tmp_path):
    run = _export(tmp_path, [_record("bare", None, wanted=12)])
    got = native_tokens.exact_output_lengths(run)
    assert got["holds"] is False and got["missing_native_ids"] == 1
    assert "carried no native token ids" in got["why"]
    assert scenario.replay_fidelity_of_run(CLEAN, run)["holds"] is False


def test_a_run_without_a_raw_export_cannot_pass(tmp_path):
    got = native_tokens.exact_output_lengths(tmp_path)
    assert got["holds"] is False and "profile_export_raw.jsonl" in got["why"]
    assert scenario.replay_fidelity_of_run(CLEAN, tmp_path)["holds"] is False


def test_requests_without_ignore_eos_are_not_length_checked(tmp_path):
    run = _export(tmp_path, [_record("free", [1, 2], wanted=50,
                                     ignore_eos=False)])
    assert native_tokens.exact_output_lengths(run)["checked"] == 0


def test_development_preflight_and_verdict_share_the_rule(tmp_path):
    run = _export(tmp_path, [_record("short", [1] * 9, wanted=10)])
    lengths = native_tokens.exact_output_lengths(run)
    preflight = replay_verify.assess(
        {**CLEAN, "request_count": {"avg": replay_verify.REQUEST_COUNT}},
        lengths=lengths)
    assert preflight["holds"] is False and "request short" in preflight["why"]


@pytest.mark.parametrize("path", ["measure_one", "partial"])
def test_measure_one_and_partial_evidence_read_the_raw_export(
        tmp_path, monkeypatch, path):
    from types import SimpleNamespace

    run_dir = tmp_path / "run"
    artifact = run_dir / "agentx" / "candidate"
    artifact.mkdir(parents=True)
    _export(artifact, [_record("short", [1] * 9, wanted=10)])
    if path == "partial":
        monkeypatch.setattr(scenario, "read_aggregate",
                            lambda *_a, **_k: dict(CLEAN))
        got = scenario._partial_evidence(artifact)["correctness"]
    else:
        module = SimpleNamespace(
            MODEL_ID="m", ENVIRONMENT={},
            invocation=lambda **_: ["aiperf", "profile", "--random-seed", "2",
                                    "--benchmark-duration", "1800"])
        monkeypatch.setattr(scenario, "_module_of", lambda _: module)
        monkeypatch.setattr(scenario, "_pinned_aiperf_argv", lambda a: a)
        monkeypatch.setattr(scenario, "_run_watching_warmup",
                            lambda *a, **k: (SimpleNamespace(returncode=0), None))
        monkeypatch.setattr(scenario, "read_aggregate",
                            lambda *_a, **_k: dict(CLEAN))
        monkeypatch.setattr(scenario.stream_timing, "assess", lambda _d: {})
        got = scenario.measure_one("test", "candidate", url="http://unused",
                                   tokenizer=tmp_path, run_dir=run_dir,
                                   duration_s=1800)["correctness"]
    assert got["holds"] is False and "request short" in got["why"]


def test_the_reader_is_packaged_and_the_private_reader_is_not_imported():
    assert "native_tokens.py" in package_grader.SHARED_MODULES
    source = (package_grader.PACKAGE / "scenario.py").read_text("utf-8")
    assert "agentx_reference" not in source
