"""The private EOS gate and the two seed spaces.

Request scoring itself is covered by test_request_scoring_objective.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from final_validate import private_correctness

# ------------------------------------------------------------------ EOS gate

EOS = {7, 9}


@pytest.mark.parametrize("tokens,suspect", [
    ([1, 2, 3, 4, 5, 6], False),                # no EOS at all
    ([1, 2, 3, 4, 5, 7], False),                # EOS is the last token
    ([1, 2, 3, 7, 4, 5, 6], True),              # anything after EOS is audited
    ([1, 2, 7, 7, 7, 7, 7], True),              # stopped, then padded
    ([1, 2, 9, 9, 9, 9], True),                 # pad id
    ([1, 7, 2, 7, 3, 7, 4], True),              # EOS recurs, never a run
    ([5] * private_correctness.REPEAT_RUN, True),
    ([5] * (private_correctness.REPEAT_RUN - 1), False),
])
def test_padding_suspicion(tokens, suspect):
    assert bool(private_correctness._suspicion(tokens, EOS)) is suspect


def test_eos_then_anything_is_a_suspect():
    """Padding after an early EOS with ordinary, varied, non-repeating tokens
    is never a long EOS run or a repeat run, and must still be audited."""
    padded = [11, 12, 7] + [100 + (i * 37) % 997 for i in range(500)]
    assert len(set(padded[3:])) > 400                 # no repeat run
    why = private_correctness._suspicion(padded, EOS)
    assert why and "follow the first end-of-sequence" in why
    assert not private_correctness._suspicion([11, 12, 13, 7], EOS)


def test_an_unaudited_suspect_names_why_it_was_not_audited():
    got = private_correctness.judge_eos_gate(
        candidate_stop=_stop("stop", 2), reference_stop=_stop("stop", 2),
        scan={"scanned": 5, "missing_native_ids": 0, "suspects": {"a": "pad"}},
        audit=None)
    assert got["holds"] is False
    assert not got["why"].rstrip().endswith(":")
    assert "could not be audited: no audit result was produced" in got["why"]


def _raw_record(request_id, ids, text, *, phase="profiling"):
    import json as _json
    packets = [{"name": "data", "value": _json.dumps({"choices": [{
        "index": 0, "delta": {"content": text, "token_ids": ids}}]})}]
    return {"metadata": {"x_request_id": request_id, "benchmark_phase": phase},
            "payload": {"ignore_eos": True, "max_tokens": len(ids)},
            "responses": [{"packets": packets}]}


class _Decoder:
    """Decodes token id n to the letter chr(n)."""

    def decode(self, ids, **_kwargs):
        return "".join(chr(i) for i in ids)


def test_the_private_scan_requires_ids_to_decode_to_the_streamed_text(tmp_path):
    import json as _json

    records = [_raw_record("good", [104, 105], "hi"),
               _raw_record("bad", [104, 105], "ho")]
    (tmp_path / "profile_export_raw.jsonl").write_text(
        "".join(_json.dumps(r) + "\n" for r in records))
    scan = private_correctness.padding_suspects(tmp_path, EOS,
                                                tokenizer=_Decoder())
    assert scan["decode_checked"] == 2
    assert scan["decode_mismatches"] == 1
    assert scan["decode_mismatch_examples"] == ["bad"]
    got = private_correctness.judge_eos_gate(
        candidate_stop=_stop("stop", 2), reference_stop=_stop("stop", 2),
        scan=scan, audit=None)
    assert got["holds"] is False and "request bad" in got["why"]
    clean = dict(scan, decode_mismatches=0, decode_mismatch_examples=[])
    assert private_correctness.judge_eos_gate(
        candidate_stop=_stop("stop", 2), reference_stop=_stop("stop", 2),
        scan=clean, audit=None)["holds"] is True


def _stop(finish, n):
    return {"finish_reason": finish, "token_ids": list(range(n))}


def test_eos_gate_holds_for_a_server_that_stops_and_does_not_pad():
    got = private_correctness.judge_eos_gate(
        candidate_stop=_stop("stop", 2), reference_stop=_stop("stop", 2),
        scan={"scanned": 5, "missing_native_ids": 0, "suspects": {}},
        audit=None)
    assert got["holds"] is True


def test_eos_gate_fails_a_server_that_ignores_the_stop():
    got = private_correctness.judge_eos_gate(
        candidate_stop=_stop("length", private_correctness.STOP_PROBE_MAX_TOKENS),
        reference_stop=_stop("stop", 2),
        scan={"scanned": 5, "missing_native_ids": 0, "suspects": {}},
        audit=None)
    assert got["holds"] is False


def test_stop_probe_is_inconclusive_when_the_reference_does_not_stop():
    got = private_correctness.judge_eos_gate(
        candidate_stop=_stop("length", 64), reference_stop=_stop("length", 64),
        scan={"scanned": 5, "missing_native_ids": 0, "suspects": {}},
        audit=None)
    assert got["holds"] is True


@pytest.mark.parametrize("audit,holds", [
    ({"evaluable": True, "holds": True}, True),
    ({"evaluable": True, "holds": False}, False),
    ({"evaluable": False, "holds": False}, False),
    (None, False),
])
def test_suspects_must_pass_the_reference_audit(audit, holds):
    got = private_correctness.judge_eos_gate(
        candidate_stop=_stop("stop", 2), reference_stop=_stop("stop", 2),
        scan={"scanned": 5, "missing_native_ids": 0, "suspects": {"a": "pad"}},
        audit=audit)
    assert got["holds"] is holds


def test_a_reply_without_native_ids_fails_the_gate():
    got = private_correctness.judge_eos_gate(
        candidate_stop=_stop("stop", 2), reference_stop=_stop("stop", 2),
        scan={"scanned": 4, "missing_native_ids": 1, "suspects": {}},
        audit=None)
    assert got["holds"] is False


def test_the_eos_gate_joins_answer_correctness():
    replay = {"holds": True}
    answer = {"holds": True, "evaluable": True}
    ok = private_correctness.combined_correctness(replay, answer, {"holds": True})
    bad = private_correctness.combined_correctness(
        replay, answer, {"holds": False, "why": "padded"})
    assert ok["holds"] is True and bad["holds"] is False
    assert bad["output_correctness"]["eos_gate"]["why"] == "padded"


# ------------------------------------------------------------ seed spaces

def test_final_and_development_seeds_are_disjoint():
    from final_validate.tasks import _agentx_mvp as shared

    dev = {shared.development_seed() for _ in range(2000)}
    final = {shared.final_seed() for _ in range(2000)}
    assert all(s % 2 == 0 for s in dev)
    assert all(s % 2 == 1 for s in final)
    assert not dev & final


def test_the_verdict_draws_a_final_seed_and_the_workspace_a_development_one(
        tmp_path, monkeypatch):
    from final_validate import scenario

    seen = {}

    def fake_run(argv, environment, log, timeout):
        seen["seed"] = int(argv[argv.index("--random-seed") + 1])
        raise RuntimeError("stop here")

    monkeypatch.setattr(scenario, "_run_watching_warmup", fake_run)
    for private in (True, False):
        with pytest.raises(RuntimeError):
            scenario.measure_one(
                "qwen3_5_2b_standard_1xL40", "candidate",
                url="http://127.0.0.1:9", tokenizer=tmp_path,
                run_dir=tmp_path, duration_s=1800, private_workload=private,
                concurrency=4)
        assert seen["seed"] % 2 == (1 if private else 0)


def test_the_host_verdict_is_not_mistaken_for_the_packaged_copy():
    """Run as a script (how the verdict runs it), the host copy is not IN_HARBOR."""
    import subprocess
    import sys
    from pathlib import Path

    from final_validate import scenario

    host = Path(scenario.__file__)
    probe = ("import runpy,sys; sys.argv=['scenario']; "
             f"g=runpy.run_path({str(host)!r}, run_name='__main__probe'); "
             "print(g['IN_HARBOR'])")
    got = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True, cwd=host.parent.parent)
    assert got.stdout.strip().splitlines()[-1] == "False"
    packaged = (Path(__file__).resolve().parents[3] / "serving_tasks"
                / "qwen3_5_2b_standard_1xL40" / "tests"
                / "final_verdict" / "scenario.py")
    assert packaged.parent.name == "final_verdict"


def test_an_unanswered_stop_probe_is_reported_as_unanswered():
    got = private_correctness.judge_eos_gate(
        candidate_stop={"error": "TimeoutError: timed out"},
        reference_stop=_stop("stop", 2),
        scan={"scanned": 5, "missing_native_ids": 0, "suspects": {}},
        audit=None)
    assert got["holds"] is False
    assert "did not answer" in got["why"] and "TimeoutError" in got["why"]


def test_the_private_settings_are_validated_where_they_live(tmp_path):
    """The answer sample, its absolute accuracy allowance and the suspect-audit
    regret limit live in reference.yaml's private block."""
    from types import SimpleNamespace

    def contract(block):
        (tmp_path / "reference.yaml").write_text(
            "correctness:\n" + "".join(f"  {k}: {v}\n" for k, v in block.items()))
        return SimpleNamespace(task_dir=tmp_path)

    got = contract({"answer_sample_size": 20, "max_absolute_accuracy_drop": 0.1,
                    "max_token_regret": 0.05})
    assert private_correctness.answer_settings(got) == (20, 0.1)
    assert private_correctness.suspect_regret(got) == 0.05
    with pytest.raises(SystemExit):
        private_correctness.suspect_regret(contract({"max_token_regret": ".nan"}))
    with pytest.raises(SystemExit):
        private_correctness.answer_settings(contract({"answer_sample_size": 0}))
    with pytest.raises(SystemExit):
        private_correctness.answer_settings(
            contract({"max_absolute_accuracy_drop": 1.5}))
    with pytest.raises(SystemExit, match="obsolete"):
        private_correctness.answer_settings(contract({"min_relative_accuracy": 0.9}))


def test_the_workspace_copy_never_draws_a_final_seed(monkeypatch):
    """Inside the container every replay draws a fresh even seed, whatever
    mode the copy is run in; only the verdict outside draws an odd one."""
    from final_validate import scenario
    from final_validate.tasks import _agentx_mvp
    assert _agentx_mvp.RANDOM_SEED is None
    seeds = {_agentx_mvp.development_seed() for _ in range(64)}
    finals = {_agentx_mvp.final_seed() for _ in range(64)}
    assert len(seeds) == 64 and len(finals) == 64          # random, not fixed
    assert all(s % 2 == 0 for s in seeds) and all(s % 2 == 1 for s in finals)
    source = (Path(scenario.__file__)).read_text("utf-8")
    assert "if private_workload and not IN_HARBOR:" in source
