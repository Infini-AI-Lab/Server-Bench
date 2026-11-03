"""No graded question, held-out answer or correctness material travels.

The verdict grades a candidate on items from the held-out `final_validation`
half of an answer pool. None of those items -- id, question or answer -- may
reach a workspace. Beyond the held-out guarantee, nothing about correctness
travels at all: not the public pool, not the `reference.yaml` correctness
block, not the timed-reply selector. The agent is told output correctness is
verified privately and builds its own checks. `--check-protocol-context`
checks endpoint protocol, native ids and context; the binding answer gate
samples already-committed timed AgentX replies after the candidate stops, on
the host.

Built through `workspace.build`, because what matters is what is on disk in
the workspace, not what a delivery function intended.
"""

from __future__ import annotations

import json
import re
import sys
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import pytest

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "serverbench" / "harness"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASKS = REPO / "serving_tasks"
SECRET_ID = "gsm8k-held-out-0001"
SECRET_QUESTION = "How many held-out apples are there?"
SECRET_ANSWER = "424242"


def _tasks() -> list[str]:
    return sorted(p.name for p in TASKS.glob("*") if (p / "contract.yaml").is_file())


@pytest.fixture()
def pools(tmp_path) -> Path:
    """A stand-in eval_samples root: one public half, one held-out half."""
    root = tmp_path / "eval_samples"
    root.mkdir()
    (root / "gsm8k-final_validation.json").write_text(json.dumps({
        "items": [{"id": SECRET_ID, "dataset": "gsm8k",
                   "question": SECRET_QUESTION,
                   "reference_answer": SECRET_ANSWER}]}), "utf-8")
    return root


def _public_pool(root: Path, count: int) -> None:
    (root / "gsm8k-in_harbor.json").write_text(json.dumps({
        "dataset": "gsm8k",
        "items": [{"id": f"gsm8k-public-{n:04d}", "dataset": "gsm8k",
                   "question": f"How many public apples are there, take {n}?",
                   "reference_answer": str(n)} for n in range(count)]}), "utf-8")


def _built(task: str, tmp_path: Path, pools: Path, public_items: int = 16):
    from lifecycle import workspace as ws_mod
    from lifecycle.record import Journal, RunIdentity

    identity = RunIdentity(
        run_id="r", task_id=task, lineage_id="l", control_epoch="e",
        workspace_id="w", namespace="evolve-scratch",
        created_at="2026-09-15T00:00:00Z",
        deadline_utc="2026-09-15T08:00:00Z", root=tmp_path / "run")
    identity.root.mkdir(parents=True)
    _public_pool(pools, public_items)
    task_dir = TASKS / task
    ws_mod.build(identity=identity, task_dir=task_dir, repo=REPO,
                 journal=Journal(identity.root), seed=task_dir / "demo",
                 brief="brief")
    return identity.root / "workspace"


def _leaked(body: str) -> list[str]:
    """Which pieces of held-out material this text actually carries.

    The ID and the QUESTION are distinctive strings: a substring match is the
    right test, and anything containing them is a leak.

    THE ANSWER IS A BARE NUMBER, and that needs a boundary: the canary answer
    `424242` can appear inside a delivered latency value such as
    `50.88881742424242`. Six digits inside a float are not a leak -- nothing
    can be recovered from them, and no question travels with them -- whereas
    `"reference_answer": "424242"`, `answer: 424242`, or the number in prose
    all still match, because a leak that matters puts the answer where it can
    be read as a value.

    `test_the_boundary_still_catches_a_real_leak` holds that line.
    """
    found = [needle for needle in (SECRET_ID, SECRET_QUESTION)
             if needle in body]
    # Not part of a LONGER NUMBER, which is the only thing excluded: a digit
    # before it, a digit after it, or a decimal point on either side with a
    # digit beyond. A trailing sentence period still matches.
    if re.search(r"(?<!\d)(?<!\d\.)" + re.escape(SECRET_ANSWER)
                 + r"(?!\d)(?!\.\d)", body):
        found.append(SECRET_ANSWER)
    return found


def test_the_boundary_still_catches_a_real_leak() -> None:
    """The positive control for the rule above."""
    for leak in (f'{{"reference_answer": "{SECRET_ANSWER}"}}',
                 f"the answer is {SECRET_ANSWER}.",
                 f"answer={SECRET_ANSWER}",
                 f'["{SECRET_ANSWER}"]',
                 f"{SECRET_ID} something",
                 SECRET_QUESTION):
        assert _leaked(leak), f"a real leak went unnoticed: {leak!r}"
    for innocent in ('{"value": 50.88881742424242}',
                     '{"latency_ns": 1789651651424242999}',
                     '{"value": 424242.5}'):
        assert not _leaked(innocent), (
            f"a digit run inside a number was called a leak: {innocent!r}")


@pytest.mark.parametrize("task", _tasks())
def test_no_held_out_question_or_answer_is_materialised(task, tmp_path, pools):
    workspace = _built(task, tmp_path, pools)

    hits = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        try:
            body = path.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            continue                      # a binary asset carries no prompt
        hits += [f"{path.relative_to(workspace)}: {needle!r}"
                 for needle in _leaked(body)]
    assert not hits, (
        f"held-out material reached the workspace: {hits[:5]}")

    # The evaluator's own source names the partition -- it has to, it is the
    # code that draws from the other half -- so the check above is about
    # CONTENT. This is about the file itself.
    carried = [str(p.relative_to(workspace)) for p in workspace.rglob("*")
               if p.is_file() and "final_validation" in p.name]
    assert not carried, f"a held-out partition file travelled: {carried}"


@pytest.mark.parametrize("task", _tasks())
def test_the_questions_the_verdict_grades_on_do_not_travel(
        task, tmp_path, pools):
    """The held-out pool itself is the source of truth for what must not appear.

    Reading the held-out half directly is stronger than checking against a
    copy of part of it.
    """
    held_out = json.loads(
        (pools / "gsm8k-final_validation.json").read_text("utf-8"))
    items = [row for row in (held_out.get("items") or held_out)
             if isinstance(row, dict) and row.get("id")][:40]
    assert items, "the held-out pool records no items"

    workspace = _built(task, tmp_path, pools)
    hits = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        try:
            body = path.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for row in items:
            for needle in (row["id"], str(row.get("question") or "")):
                if needle and needle in body:
                    hits.append(f"{path.relative_to(workspace)}: {needle[:40]!r}")
    assert not hits, (
        f"questions the verdict grades on reached the workspace: {hits[:5]}")


@pytest.mark.parametrize("task", _tasks())
def test_the_frozen_reference_travels_as_a_score_not_as_questions(
        task, tmp_path, pools):
    import yaml

    workspace = _built(task, tmp_path, pools)
    delivered = workspace / "task_context" / task / "reference.yaml"
    assert delivered.is_file()
    assert "correctness" not in (
        yaml.safe_load(delivered.read_text("utf-8")) or {}), (
        "how correctness is judged is private; the delivered copy strips it")
    # THE SCORE, AND NOT ONE QUESTION. `reference.yaml` records the accuracy,
    # the sample size, the floor and the tolerated delta; the probes it was
    # measured from are not in the repository, so a delivery that carries this
    # whole file carries no graded item.
    body = delivered.read_text("utf-8")
    assert "probes" not in body and "gsm8k_seed" not in body
    assert "response_text" not in body


@pytest.mark.parametrize("task", _tasks())
def test_no_pool_and_no_correctness_block_reaches_the_workspace(
        task, tmp_path, pools):
    """Neither half of any answer pool, public or held out, is delivered."""
    import yaml

    workspace = _built(task, tmp_path, pools, public_items=240)
    assert not (workspace / "tests" / "final_verdict" / "pool").exists()
    assert not list(workspace.rglob("gsm8k-*.json"))
    delivered = workspace / "task_context" / task / "reference.yaml"
    assert "correctness" not in (
        yaml.safe_load(delivered.read_text("utf-8")) or {})


@pytest.mark.parametrize("task", _tasks())
def test_the_correctness_sampler_stays_on_the_host(task, tmp_path, pools):
    """The timed-reply selector is not in the workspace, and still works here."""
    import subprocess

    from final_validate import agentx_reference

    workspace = _built(task, tmp_path, pools, public_items=240)
    got = subprocess.run(
        [sys.executable, "-c", "from final_verdict import agentx_reference"],
        cwd=str(workspace / "tests"), capture_output=True, text=True,
        timeout=180, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    assert got.returncode != 0, "the packaged copy carries the selector"

    artifact = tmp_path / "timed-agentx"
    artifact.mkdir()
    request_ids = [f"timed-request-{index:02d}" for index in range(11)]
    (artifact / "outputs.json").write_text(json.dumps({"data": [{
        "x_request_id": request_id, "conversation_id": f"c-{index:02d}",
        "turn_index": index, "response_text": f"committed reply {index}",
    } for index, request_id in enumerate(request_ids)]}), "utf-8")
    (artifact / "profile_export_raw.jsonl").write_text("".join(
        json.dumps({
            "metadata": {"x_request_id": request_id,
                         "benchmark_phase": "profiling"},
            "payload": {"messages": [{"role": "user", "content": f"t{index}"}],
                        "temperature": 0, "ignore_eos": True},
            "responses": [{"packets": [{"value": json.dumps({
                "choices": [{"index": 0, "delta": {
                    "content": "x", "token_ids": [index + 1]}}]})}]}],
        }) + "\n" for index, request_id in enumerate(request_ids)), "utf-8")
    body = agentx_reference.select_timed_requests(artifact, seed=17,
                                                  sample_size=8)
    assert body["evaluable"] is True
    drawn = {str(r["x_request_id"]) for r in body["items"]}
    assert len(drawn) == 8 and drawn <= set(request_ids)
    assert all(len(r["candidate_tokens"]) == 1 for r in body["items"])


@pytest.mark.parametrize("task", _tasks())
def test_the_delivered_contract_states_the_current_bar_only(
        task, tmp_path, pools):
    """The agent reads the current bar, not its history."""
    import yaml

    workspace = _built(task, tmp_path, pools)
    delivered = yaml.safe_load(
        (workspace / "task_context" / task / "contract.yaml").read_text("utf-8"))
    assert delivered["service_levels"], "the current bar must travel"
    # `contract.yaml` carries no history to strip; this asserts that no
    # history-shaped key appears in it.
    for gone in ("superseded_service_levels", "superseded", "notes",
                 "history"):
        assert gone not in delivered, (
            f"{gone!r} is in the contract; the bar's history belongs in "
            f"git, not in front of the agent")
    # `model_config` is the model's own config.json, data rather than prose,
    # and its size is the model's.
    assert delivered["model_config"]["config"]["model_type"]
    rest = {key: value for key, value in delivered.items() if key != "model_config"}
    assert len(yaml.safe_dump(rest)) < 1500, (
        "the delivered contract has grown a prose block")


@pytest.mark.parametrize("task", _tasks())
def test_the_workspace_carries_the_two_data_files_and_nothing_else(
        task, tmp_path, pools):
    """What `task_context/` delivers, exactly.

    Two files, both generated, neither carrying a date or a digest.
    """
    workspace = _built(task, tmp_path, pools)
    context = workspace / "task_context" / task
    delivered = sorted(p.name for p in context.iterdir() if p.is_file())
    assert delivered == ["contract.yaml", "reference.yaml"], delivered
    for name in delivered:
        body = (context / name).read_text("utf-8")
        assert "measured_at" not in body
        assert "sha256" not in body
        assert not re.search(r"\b20\d\d-\d\d-\d\d\b", body), (
            f"{name} carries a date; a task's data states what is true now")
