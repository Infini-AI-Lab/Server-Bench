"""The private answer-accuracy gate.

After the timed replay a seeded draw of held-out questions with known numeric
answers goes to the candidate, and the same questions to the plain checkpoint
after the candidate is gone. Allow 0.10 absolute accuracy loss, i.e. two fewer
correct answers out of 20. None of this is packaged or described to the agent.
"""
from __future__ import annotations

import json

import pytest
from final_validate import package_grader
from final_validate import private_correctness as pc


@pytest.mark.parametrize("text,want", [
    ("so 3 + 4 = 7\n#### 7", 7.0),
    ("#### $1,234.50", 1234.5),
    ("The answer is 18 dollars.", 18.0),
    ("#### 12 apples, from 3 boxes", 12.0),
    ("first 5, then #### -3", -3.0),
    ("no number here", None),
    ("", None),
])
def test_the_final_number_is_read(text, want):
    assert pc.extracted_answer(text) == want


def test_the_expected_answer_is_the_one_after_the_marker():
    item = {"reference_answer": "2/2=<<2/2=1>>1 bolt\n2+1=<<2+1=3>>3\n#### 3"}
    assert pc.expected_answer(item) == 3.0


def _probe(correct: int, total: int, *, errors: int = 0):
    questions = [{"id": f"q{i}", "question": "?"} for i in range(total)]
    replies = []
    for i in range(total):
        if i < errors:
            replies.append({"id": f"q{i}", "error": "TimeoutError"})
        else:
            replies.append({"id": f"q{i}", "text": f"#### {1 if i < correct else 0}"})
    return {"questions": questions, "replies": replies}


ANSWERS = {f"q{i}": 1.0 for i in range(20)}


@pytest.mark.parametrize("candidate,reference,holds", [
    (14, 14, True),      # identical accuracy
    (13, 14, True),
    (12, 14, True),      # 14 - 2, not ceil(0.9 * 14)
    (11, 14, False),
    (13, 15, True),      # 15 - 2
    (12, 15, False),
    (8, 10, True),       # absolute 10 percentage points, not relative 10%
    (7, 10, False),
    (0, 2, True),        # allowance is still two; required count floors at zero
    (18, 20, True),
    (17, 20, False),
    (20, 15, True),      # better than the reference is fine
])
def test_the_candidate_may_answer_two_fewer_correctly(candidate, reference, holds):
    got = pc.judge_answers(_probe(candidate, 20), _probe(reference, 20)["replies"],
                           ANSWERS, 0.1)
    assert got["evaluable"] is True and got["holds"] is holds
    assert got["candidate_correct"] == candidate
    assert got["reference_correct"] == reference
    assert got["allowed_fewer_correct"] == 2
    assert got["min_candidate_correct"] == max(0, reference - 2)
    assert got["max_absolute_accuracy_drop"] == .1
    assert "min_relative_accuracy" not in got


def test_an_unanswered_question_counts_as_wrong():
    got = pc.judge_answers(_probe(20, 20, errors=5), _probe(20, 20)["replies"],
                           ANSWERS, 0.1)
    assert got["candidate_correct"] == 15 and got["candidate_errors"] == 5
    assert got["holds"] is False and "unanswered" in got["why"]


def test_a_broken_reference_is_not_evaluable_rather_than_a_pass():
    got = pc.judge_answers(_probe(20, 20), _probe(0, 20)["replies"], ANSWERS, 0.1)
    assert got["evaluable"] is False and got["holds"] is False


def test_no_question_asked_fails_closed():
    got = pc.judge_answers({"questions": [], "error": "pool missing"}, [], {}, 0.1)
    assert got["evaluable"] is False and got["holds"] is False
    assert "pool missing" in got["why"]


@pytest.mark.parametrize("total,allowed", [(10, 1), (19, 1), (20, 2), (25, 2), (30, 3)])
def test_allowance_uses_total_questions_not_reference_correct(total, allowed):
    answers = {f"q{i}": 1.0 for i in range(total)}
    reference = _probe(total, total)["replies"]
    got = pc.judge_answers(_probe(total - allowed, total), reference, answers, .1)
    assert got["holds"] and got["allowed_fewer_correct"] == allowed
    missed = pc.judge_answers(_probe(total - allowed - 1, total), reference, answers, .1)
    assert not missed["holds"]


@pytest.mark.parametrize("drop", [-.1, 1.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_absolute_allowance_cannot_pass(drop):
    got = pc.judge_answers(_probe(20, 20), _probe(20, 20)["replies"], ANSWERS, drop)
    assert not got["evaluable"] and not got["holds"]


def test_zero_allowance_requires_reference_accuracy():
    got = pc.judge_answers(_probe(13, 20), _probe(14, 20)["replies"], ANSWERS, 0)
    assert not got["holds"]


def test_the_draw_is_seeded_and_carries_answers(tmp_path, monkeypatch):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"items": [
        {"id": f"p{i}", "question": f"q{i}", "reference_answer": f"x\n#### {i}"}
        for i in range(50)]}))
    monkeypatch.setattr(pc, "ANSWER_POOL", pool)
    first = pc.select_questions(20, 7)
    assert first == pc.select_questions(20, 7)
    assert first != pc.select_questions(20, 8)
    assert len({q["id"] for q in first}) == 20
    assert all(q["answer"] == float(q["id"][1:]) for q in first)
    with pytest.raises(ValueError):
        pc.select_questions(51, 7)


def test_a_modal_candidate_phase_reads_the_shipped_questions(tmp_path, monkeypatch):
    shipped = tmp_path / "questions.json"
    shipped.write_text(json.dumps([{"id": "a", "question": "1+1?"}]))
    monkeypatch.setenv(pc.QUESTIONS_FILE_ENV, str(shipped))
    monkeypatch.setattr(pc, "ANSWER_POOL", tmp_path / "absent.json")
    asked = []
    monkeypatch.setattr(pc, "ask", lambda url, questions, **kw: asked.append(questions) or [])
    got = pc.answer_probe("http://x", contract=None, destination=tmp_path / "o.json")
    assert asked == [[{"id": "a", "question": "1+1?"}]]
    assert "answer" not in json.dumps(got["questions"])


def test_the_request_is_greedy_and_stops_naturally():
    body = pc.answer_body("What is 2+2?")
    assert body["temperature"] == 0 and body["ignore_eos"] is False
    assert body["stream"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "What is 2+2?" in body["messages"][0]["content"]


def test_the_gate_is_never_packaged():
    assert "private_correctness.py" in package_grader.NEVER_PACKAGED
