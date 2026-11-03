"""Every task has one verdict inside and outside Harbor, with two hard gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import tomllib

REPO = Path(__file__).resolve().parents[3]
TASKS_ROOT = REPO / "serving_tasks"
for path in (str(REPO / "serverbench" / "harness"),):
    if path not in sys.path:
        sys.path.insert(0, path)

from final_validate import (  # noqa: E402
    package_grader,
    private_correctness,  # noqa: E402
    scenario,
    spine,
    task_spec,
)


def _tasks() -> list[Path]:
    return sorted(path for path in TASKS_ROOT.glob("*")
                  if (path / "contract.yaml").is_file())


TASKS = _tasks()


def _row(contract: spine.Contract, artifact_dir: Path, *,
         replay: bool, correctness: bool) -> dict:
    levels = list(contract.slo["service_levels"])
    aggregate = {
        "benchmark_duration_s": 1800,
        "total_input_tokens": 1_000_000,
        "total_output_tokens": 100_000,
        "theoretical_prefix_cache_hit": {"count": 1, "sum": 0},
    }
    for level in levels:
        percentile = int(level["percentile"])
        aggregate[f"ttft_p{percentile}_ms"] = float(level["ttft_ms"]) * 0.5
        aggregate[f"tpot_p{percentile}_ms"] = float(level["tpot_ms"]) * 0.5
    replay_gate = {"holds": replay,
                   "why": "replay gate failed" if not replay else ""}
    answer_gate = {"holds": correctness, "evaluable": True, "available": True,
                   "why": "correctness gate failed" if not correctness else ""}
    combined = private_correctness.combined_correctness(replay_gate, answer_gate)
    row = {"arm": "candidate", "aggregate": aggregate,
           "correctness": combined, "stream_timing": {"buffered": False}}
    if contract.slo.get("request_scoring"):
        aggregate.update(request_count=20, error_request_count=0)
        row["artifact_dir"] = str(artifact_dir)
        records = [{
            "metadata": {"benchmark_phase": "profiling", "x_request_id": str(i)},
            "metrics": {
                "time_to_first_token": {"value": aggregate["ttft_p90_ms"], "unit": "ms"},
                "inter_token_latency": {"value": aggregate["tpot_p90_ms"], "unit": "ms"},
                "input_sequence_length": {"value": 50000, "unit": "tokens"},
                "output_sequence_length": {"value": 5000, "unit": "tokens"},
            },
        } for i in range(20)]
        (artifact_dir / "profile_export.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))
    return row


@pytest.mark.parametrize("task_dir", TASKS, ids=lambda path: path.name)
def test_harbor_copy_is_the_host_verdict_source_byte_for_byte(task_dir: Path) -> None:
    assert package_grader.check(task_dir) == [], (
        f"{task_dir.name}: the Harbor verdict differs from the host verdict")


@pytest.mark.parametrize("task_dir", TASKS, ids=lambda path: path.name)
def test_replay_and_correctness_are_independent_hard_gates(task_dir: Path, tmp_path) -> None:
    contract = spine.load_contract(task_dir.name)
    spec = task_spec.read(task_dir.name)
    usd_per_hour = spec.devices * spec.usd_per_gpu_hour

    passing = scenario.decide(
        contract, [_row(contract, tmp_path, replay=True, correctness=True)],
        usd_per_hour)["arms"][0]
    assert passing["outcome"] == "HOLDS"
    assert passing["replay_gate"]["holds"] is True
    assert passing["correctness_gate"]["holds"] is True
    assert passing["eligibility_gates_hold"] is True
    assert passing[contract.objective] is not None

    # Corrupt the cached combined boolean deliberately. The verdict must read
    # the two children and refuse either failure rather than trust this field.
    replay_failure = _row(contract, tmp_path, replay=False, correctness=True)
    replay_failure["correctness"]["holds"] = True
    replay_decision = scenario.decide(
        contract, [replay_failure], usd_per_hour)["arms"][0]
    assert replay_decision["outcome"] == "MISSES"
    assert replay_decision["eligibility_gates_hold"] is False

    correctness_failure = _row(contract, tmp_path, replay=True, correctness=False)
    correctness_failure["correctness"]["holds"] = True
    correctness_decision = scenario.decide(
        contract, [correctness_failure], usd_per_hour)["arms"][0]
    assert correctness_decision["outcome"] == "MISSES"
    assert correctness_decision["eligibility_gates_hold"] is False


@pytest.mark.parametrize("task_dir", TASKS, ids=lambda path: path.name)
def test_task_metadata_declares_both_non_tradeable_gates(task_dir: Path) -> None:
    metadata = tomllib.loads((task_dir / "task.toml").read_text("utf-8"))[
        "metadata"]
    assert metadata["objective"] == "dollars_per_dollar"
    assert metadata["gate_replay_fidelity"] is True
    if metadata.get("request_scoring_kind") == "request_p90_billable_tokens_v4":
        # Request-scored tasks name no correctness method: it is private.
        assert metadata["hard_correctness_and_replay_gates"] is True
        assert "gate_answer_correctness" not in metadata
        assert "gate_correctness" not in metadata
    else:
        assert metadata["gate_answer_correctness"] == "timed_agentx_vs_plain_checkpoint"


def test_modal_backend_is_visible_in_each_modal_task_identity() -> None:
    modal = {path.name for path in TASKS if (path / "MODE.md").is_file()
             and (path / "MODE.md").read_text("utf-8").startswith("# Modal mode")}
    assert modal == {
        "qwen3_5_9b_standard_modal_2xL40S",
        "gemma4_e4b_it_standard_modal_1xL40S",
    }
    assert all("_modal_" in task for task in modal)
