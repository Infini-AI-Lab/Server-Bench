"""Rapid iteration is cheap without weakening final correctness or replay."""
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]


def test_all_live_tasks_share_the_private_correctness_settings():
    from final_validate import private_correctness, spine, task_spec
    for task in task_spec.live_tasks():
        assert not hasattr(task_spec.read(task), "correctness"), (
            "the packaged task_spec must not carry the private settings")
        contract = spine.load_contract(task)
        assert private_correctness.answer_settings(contract) == (20, 0.2)
        assert private_correctness.suspect_regret(contract) == 4.0


def test_csv_copies_the_verdicts_accuracy_decision():
    recorder = _recorder()
    def label(settings):
        return recorder._correctness({"output_correctness": {
            "candidate_correct": 13, "reference_correct": 15, "questions": 20,
            "max_absolute_accuracy_drop": .1, **settings}})[0]
    assert label({"evaluable": True, "holds": True}).endswith(" pass")
    assert label({"evaluable": True, "holds": False}).endswith(" FAIL")
    assert label({"evaluable": False, "holds": False}).endswith(" unevaluable")


def test_technique_menu_is_present_in_tasks_skill_and_reminders():
    import sys
    sys.path.insert(0, str(REPO / "serving_tasks/_manage"))
    import instruction_render
    from final_validate import task_spec
    from lifecycle import instructions
    bundle = REPO / "serverbench/skills_instructed/serving-system-evolution-instructed"
    detailed = [(bundle / "SKILL.md").read_text(),
                (bundle / "references/17-optimization-techniques.md").read_text()]
    compact = []
    for task in task_spec.live_tasks():
        if "_instructed_" not in task:
            continue
        compact += [instruction_render.render(task)]
        for kind in instructions.SESSION_HEADINGS:
            compact += [instructions.session_prompt(REPO / "serving_tasks" / task,
                kind=kind, budget_seconds=3600, deadline_utc="2099-01-01T00:00:00Z")]
    # Technique-specific names belong in the optional menu, not in every prompt.
    for term in ("PagedAttention", "Radix Cache", "HiCache", "batching",
                 "scheduling", "overlap", "kernel", "framework"):
        assert term in detailed[1], term
    for text in (detailed[0],):
        for term in ('deployment', 'semantics', 'state_memory', 'scheduling',
                     'host_io', 'device_compute', 'topology_transfer', './full-test'):
            assert term in text
    for text in compact:
        for term in ("replay", "protocol", "full-test", "performance"):
            assert term in text, term
        assert 'tests/reference-fidelity' not in text
        assert "candidate SHA" in text or "candidate_tree_sha" in text
    for path in (bundle / "references").glob("[0-9]*.md"):
        number = int(path.name.split("-", 1)[0])
        assert number <= 19, path
    assert not (bundle / "references/17-speculative-decoding.md").exists()


def test_new_technique_references_are_conceptual_and_links_resolve():
    import re
    bundle = REPO / "serverbench/skills_instructed/serving-system-evolution-instructed"
    for filename in ("13-vllm-v1.md", "14-sglang-runtime.md", "17-optimization-techniques.md"):
        text = (bundle / "references" / filename).read_text()
        assert "```" not in text, "no copied code or pseudocode recipes"
        assert "docs." in text
    for path in [bundle / "SKILL.md", *(bundle / "references").glob("*.md")]:
        for link in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", path.read_text()):
            if "://" not in link:
                assert (path.parent / link).exists(), f"{path}: {link}"


def test_evidence_and_trial_snapshots_are_not_published_source(tmp_path):
    from lifecycle.workspace import candidate_manifest
    for name in ("serve.sh", "engine/cache.py", "agentx/trial/candidate/serve.sh",
                 "agentx/quick.py", "agentx/run/report.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    assert candidate_manifest(tmp_path) == ["engine/cache.py", "serve.sh"]


def test_explicit_promotion_accepts_version_or_candidate_id():
    from lifecycle.run import Lifecycle
    life = Lifecycle.__new__(Lifecycle)
    version = SimpleNamespace(version_id="v1", run_id="run", candidate_id="c1")
    life.identity = SimpleNamespace(run_id="run")
    life.registry = SimpleNamespace(
        all=lambda: [version],
        get=lambda target: version if target == "v1" else None)
    life.publisher = SimpleNamespace(head_version=lambda: "v1")
    life.journal = SimpleNamespace(append=lambda *a, **kw: None)
    calls = []
    life._publish = lambda *a, **kw: calls.append((a, kw))
    assert life.promote_checkpoint("v1") == "v1"
    assert life.promote_checkpoint("c1") == "v1"
    assert calls[0][1]["explicit_selection"] is True
    with pytest.raises(ValueError, match="not a version id or candidate id"):
        life.promote_checkpoint("does-not-exist")
    version.run_id = "another-run"
    with pytest.raises(ValueError, match="belongs to"):
        life.promote_checkpoint("v1")


def test_promote_latest_withdraws_an_earlier_concrete_selection():
    """`./promote v1` then `./promote latest` must not leave v1 judged."""
    from lifecycle.run import Lifecycle
    life = Lifecycle.__new__(Lifecycle)
    versions = [SimpleNamespace(version_id=v, run_id="run", candidate_id=c, round_index=i)
                for i, (v, c) in enumerate((("v1", "c1"), ("v2", "c2")), 1)]
    life.identity = SimpleNamespace(run_id="run")
    life.registry = SimpleNamespace(
        all=lambda: versions,
        get=lambda target: next((v for v in versions if v.version_id == target), None))
    head = {"v": "v2"}
    life.publisher = SimpleNamespace(head_version=lambda: head["v"])
    life.journal = SimpleNamespace(append=lambda *a, **kw: None)
    life._threads = []

    def publish(cid, version, **kw):
        head["v"] = version.version_id
    life._publish = publish
    life.promote_checkpoint("v1")
    assert life._explicit_head_override == "v1"
    life.promote_checkpoint("latest")
    assert life._explicit_head_override == "", (
        "promote latest left the concrete v1 selection in force; the run end "
        "would judge v1 instead of the newest checkpoint")


def test_run_end_defaults_to_newest_saved_checkpoint_without_capturing():
    from lifecycle.run import Lifecycle
    life = Lifecycle.__new__(Lifecycle)
    versions = [
        SimpleNamespace(version_id="v1", run_id="run", candidate_id="c1",
                        round_index=1),
        SimpleNamespace(version_id="v2", run_id="run", candidate_id="c2",
                        round_index=2),
    ]
    head = [""]
    events = []
    life.identity = SimpleNamespace(run_id="run")
    life.registry = SimpleNamespace(
        all=lambda: versions,
        get=lambda target: next((v for v in versions
                                 if v.version_id == target), None))
    life.publisher = SimpleNamespace(head_version=lambda: head[0])
    life.journal = SimpleNamespace(
        append=lambda event, **fields: events.append((event, fields)))
    life._publish = lambda _cid, version, **_kw: head.__setitem__(
        0, version.version_id)

    assert life.default_instructed_head_at_run_end() == "v2"
    assert head[0] == "v2"
    assert any(event == "head_defaulted_at_run_end" for event, _ in events)


def _headless_instructed_run(task_id):
    from lifecycle.run import Lifecycle
    life = Lifecycle.__new__(Lifecycle)
    versions = [SimpleNamespace(version_id=f"v{i}", run_id="run", candidate_id=f"c{i}",
                                round_index=i) for i in (1, 2, 3)]
    head, events = [""], []
    life.identity = SimpleNamespace(run_id="run", task_id=task_id)
    life.registry = SimpleNamespace(
        all=lambda: versions,
        get=lambda target: next((v for v in versions if v.version_id == target), None))
    life.publisher = SimpleNamespace(head_version=lambda: head[0])
    life.journal = SimpleNamespace(append=lambda event, **fields: events.append((event, fields)))
    life._publish = lambda _cid, version, **_kw: head.__setitem__(0, version.version_id)

    def container_gone(*_a, **_kw):
        raise RuntimeError("container is not running")
    life.take_requested_promotions = container_gone
    return life, head, events


def test_a_host_failure_mid_run_still_gives_an_instructed_run_its_head():
    # A run that saved three checkpoints and then lost its container skips
    # finish(); without a settled head the verdict would REFUSE a headless run.
    life, head, events = _headless_instructed_run("qwen3_5_2b_instructed_1xL40")
    assert life.settle_head_after_harness_failure() == "v3"
    assert head[0] == "v3"
    assert any(event == "head_defaulted_at_run_end" for event, _ in events)


def test_head_settlement_leaves_standard_runs_alone():
    life, head, _ = _headless_instructed_run("qwen3_5_2b_standard_1xL40")
    assert life.settle_head_after_harness_failure() == ""
    assert head[0] == ""


def test_full_replay_also_triggers_first_replay_checkpoint(tmp_path):
    from lifecycle.record import Journal
    from lifecycle.run import Lifecycle
    task = "qwen3_5_9b_instructed_2xL40"
    root = tmp_path / "workspace/agentx/full"
    root.mkdir(parents=True)
    (root / "development-evidence.json").write_text(json.dumps({
        "schema": "final_validate.development_evidence/1", "task_id": task,
        "mode": "full", "full_window_complete": True, "replay_fidelity_holds": True}))
    life = Lifecycle.__new__(Lifecycle)
    life.identity = SimpleNamespace(task_id=task)
    life.built = SimpleNamespace(path=tmp_path / "workspace")
    life.journal = Journal(tmp_path / "run")
    life._automatic_checkpoints = set()
    life.boundary = lambda **kw: "candidate"
    life.take_evidence_boundaries()
    assert life._automatic_checkpoints == {"replay_passed", "full_window_completed"}
    assert life.take_evidence_boundaries() == []


def _recorder():
    spec = importlib.util.spec_from_file_location(
        "record_run", REPO / "evaluation_progress/record_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _monitor():
    spec = importlib.util.spec_from_file_location(
        "watch_evolve", REPO / "evaluation_progress/watch_evolve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scenario(tmp_path, task, arm, objective_name, outcome="MISSES"):
    scenario = tmp_path / "scenario-candidate.json"
    scenario.write_text(json.dumps({
        "row": {"arm": "candidate", "aggregate": {"ttft_p90_ms": 20000},
                "correctness": {}},
        "verdict": {"arms": [{"arm": "candidate", "outcome": outcome, **arm}],
                    "task_id": task, "objective": objective_name}}))
    return scenario


def test_final_csv_keeps_the_score_the_gross_figure_and_the_verdict_apart(
        tmp_path):
    recorder = _recorder()
    recorder._stream_timing = lambda _: {}
    task = "qwen3_5_9b_instructed_2xL40"
    scenario = _scenario(tmp_path, task, {
        "gross_dollars_per_dollar": 2, "dollars_per_dollar": 0.25,
        "request_multiplier": 0.125}, "dollars_per_dollar", outcome="HOLDS")
    result = recorder.row_from_evidence(
        scenario, final={"status": "HOLDS", "dollars_per_dollar": 0.25,
                         "gross_dollars_per_dollar": 2,
                         "measured_utc": "2026-01-01T00:00:00Z"},
        context={"task": task, "run_id": "run-1"})
    assert result["gross_dollars_per_dollar"] == "2.00"
    assert result["dollars_per_dollar"] == "0.25"
    assert result["request_multiplier"] == "0.1250"
    assert "objective" not in recorder.FIELDS            # one score column
    assert "objective_value" not in recorder.FIELDS
    assert result["verdict"] == "HOLDS"
    assert result["tpot_p90_ms"] == ""
    assert not any("soft" in field or "p95" in field
                   for field in recorder.FIELDS)


def test_task_csv_keeps_one_atomic_row_per_evolve(tmp_path):
    recorder = _recorder()
    path = tmp_path / "result.csv"
    recorder.write_final_result(path, {"task": "some_task", "run_id": "first",
                                       "verdict": "MISSES"})
    recorder.write_final_result(path, {"task": "some_task", "run_id": "second",
                                       "verdict": "HOLDS"})
    recorder.write_final_result(path, {"task": "some_task", "run_id": "first",
                                       "version": "v9", "verdict": "HOLDS"})
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    by_run = {row["run_id"]: row for row in rows}
    assert by_run["first"]["version"] == "v9"
    assert by_run["first"]["verdict"] == "HOLDS"
    assert by_run["second"]["verdict"] == "HOLDS"


def test_lifecycle_events_are_rejected_as_csv_results(tmp_path):
    recorder = _recorder()
    for status in ("RUN_CREATED", "FINAL_VALIDATION_STARTED"):
        with pytest.raises(ValueError, match="lifecycle telemetry"):
            recorder.write_final_result(
                tmp_path / f"{status}.csv", {"verdict": status})


def test_failed_evolve_records_one_terminal_result(tmp_path):
    recorder = _recorder()
    task = "gemma4_e4b_it_instructed_1xL40"
    run = tmp_path / "run-failed"
    run.mkdir()
    (run / "journal.jsonl").write_text("".join(json.dumps(event) + "\n" for event in [
        {"event": "run_created", "at_utc": "2026-01-01T00:00:00Z",
         "task_id": task, "agent_model": "vendor/model", "hours": 8},
        {"event": "final_validation", "at_utc": "2026-01-01T00:00:01Z",
         "outcome": "REFUSED"},
    ]))
    (run / "SUMMARY.json").write_text(json.dumps({
        "task_id": task, "state": "FAILED", "accepted_head": ""}))
    output = tmp_path / "result.csv"
    assert recorder.record(run, output=output) == 0
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["verdict"] == "FAILED"
    assert rows[0]["source"] == "run_summary"


def test_stopped_early_evolve_records_final_refusal(tmp_path):
    recorder = _recorder()
    task = "gemma4_e4b_it_instructed_1xL40"
    run = tmp_path / "run-stopped"
    run.mkdir()
    events = [
        {"event": "run_created", "at_utc": "2026-01-01T00:00:00Z",
         "task_id": task, "agent_model": "vendor/model", "hours": 8},
        {"event": "final_validation", "at_utc": "2026-01-01T08:00:00Z",
         "outcome": "REFUSED"},
    ]
    (run / "journal.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events))
    (run / "SUMMARY.json").write_text(json.dumps({
        "task_id": task, "state": "STOPPED_EARLY", "accepted_head": ""}))
    output = tmp_path / "result.csv"
    assert recorder.record(run, output=output) == 0
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["verdict"] == "REFUSED"
    assert rows[0]["measured_utc"] == "2026-01-01T08:00:00Z"


def test_unfinished_evolve_cannot_create_a_csv(tmp_path):
    recorder = _recorder()
    run = tmp_path / "still-running"
    run.mkdir()
    (run / "SUMMARY.json").write_text(json.dumps({"state": "RUNNING"}))
    output = tmp_path / "result.csv"
    assert recorder.record(run, output=output) == 1
    assert not output.exists()


def test_monitor_does_not_create_a_progress_csv(tmp_path):
    monitor = _monitor()
    result_dir = tmp_path / "results"
    state_dir = tmp_path / "operations"
    result_dir.mkdir()
    state_dir.mkdir()
    args = SimpleNamespace(output_dir=result_dir, state_dir=state_dir,
                           campaign="campaign", model="vendor/model", hours=8)
    job = monitor.Job("Some_Task_1xL40", args)
    assert job.state == "WAITING_GPU"
    assert job.csv_path.name == "Some_Task_1xL40.csv"
    assert not job.csv_path.exists()
    assert not list(result_dir.glob("*.lock"))
    assert (state_dir / "some_task_1xl40.launch.lock").is_file()


def test_recorder_canonicalises_a_timestamped_output_to_the_task_file():
    recorder = _recorder()
    task = "qwen3_5_9b_instructed_2xL40"
    stale = recorder.HERE / f"{task}_rapid-20990101T000000Z.csv"
    assert recorder.result_path(stale, task) == recorder.HERE / f"{task}.csv"


def test_a_misses_row_carries_no_score_and_no_rank(tmp_path):
    """A verdict with a failed hard gate scores nothing, even when its
    evidence carries a measured number."""
    recorder = _recorder()
    recorder._stream_timing = lambda _: {}
    task = "qwen3_5_9b_instructed_2xL40"
    scenario = _scenario(tmp_path, task, {
        "gross_dollars_per_dollar": 0.52, "dollars_per_dollar": 0.0376,
        "request_multiplier": 0.0722}, "dollars_per_dollar")
    missed = recorder.row_from_evidence(
        scenario, final={"status": "MISSES", "dollars_per_dollar": 0.0376,
                         "gross_dollars_per_dollar": 0.52},
        context={"task": task, "run_id": "run-missed"})
    assert missed["verdict"] == "MISSES"
    assert missed["dollars_per_dollar"] == ""
    assert missed["gross_dollars_per_dollar"] == "0.52"     # diagnostic
    held = {"task": task, "run_id": "run-held", "system": "b",
            "verdict": "HOLDS", "dollars_per_dollar": "0.0044"}
    # A hand-carried MISSES row with a number is blanked on write too.
    stale = {"task": task, "run_id": "run-stale", "system": "c",
             "verdict": "MISSES", "dollars_per_dollar": "0.0125"}
    path = tmp_path / "result.csv"
    recorder.write_rows(path, [dict(missed, system="a"), held, stale])
    rows = {row["run_id"]: row for row in recorder.read_rows(path)}
    assert rows["run-held"]["rank"] == "1"
    for run_id in ("run-missed", "run-stale"):
        assert rows[run_id]["rank"] == ""
        assert rows[run_id]["dollars_per_dollar"] == ""
    assert list(recorder.read_rows(path))[0]["run_id"] == "run-held"


def test_measure_published_reports_no_score_for_a_misses_arm(tmp_path):
    import sys
    sys.path.insert(0, str(REPO / "serverbench" / "harness"))
    from final_validate.tools import measure_published

    (tmp_path / "scenario-candidate.json").write_text(json.dumps({
        "row": {}, "verdict": {"arms": [{
            "arm": "candidate", "outcome": "MISSES",
            "dollars_per_dollar": 0.04, "gross_dollars_per_dollar": 0.5}]}}))
    got = measure_published._decision_from(tmp_path, ["candidate"])
    assert got["status"] == "MISSES"
    assert got["dollars_per_dollar"] is None
    assert got["gross_dollars_per_dollar"] == 0.5



def test_the_row_names_where_the_agent_ran_and_what_was_checked(tmp_path):
    """`backend` is container or modal; `correctness` carries the output
    check's numbers; `versions_published` counts the run's versions."""
    recorder = _recorder()
    run = tmp_path / "run"
    run.mkdir()
    (run / "container.json").write_text(json.dumps({"container": "sb-x-main-1",
                                                    "backend": "harbor"}))
    (run / "journal.jsonl").write_text("".join(
        json.dumps({"event": e}) + "\n" for e in
        ("run_created", "version_registered", "version_registered")))
    context = recorder.context_of(run)
    assert context["backend"] == "container"
    assert context["versions_published"] == "2"
    (run / "container.json").write_text(json.dumps({"container": "modal:sb-1"}))
    assert recorder.context_of(run)["backend"] == "modal"
    accuracy, eos, replay = recorder._correctness({"correctness": {
        "replay_fidelity": {"holds": True},
        "output_correctness": {"candidate_correct": 14, "reference_correct": 13,
                               "questions": 20, "max_absolute_accuracy_drop": 0.2,
                               "evaluable": True, "holds": True,
                               "eos_gate": {"holds": False}}}})
    assert (accuracy, eos, replay) == ("gsm8k 14/20 vs ref 13/20 pass", "FAIL", "ok")
    assert recorder._requests({"request_attainment": {"request_count": 111}}) == "111"
    assert recorder._requests({"aggregate": {"request_count": 55.0,
                                             "error_request_count": 2.0}}) == "57"
    for gone in ("request_multiplier", "objective", "objective_value"):
        assert gone not in recorder.FIELDS
    for added in ("requests", "eos_gate", "gross_dollars_per_dollar",
                  "dollars_per_dollar"):
        assert added in recorder.FIELDS


def test_a_gate_missed_in_several_places_is_named_once(tmp_path):
    recorder = _recorder()
    recorder._stream_timing = lambda _: {}
    task = "qwen3_5_2b_standard_1xL40"
    scenario = _scenario(tmp_path, task, {"dollars_per_dollar": 1.0},
                         "dollars_per_dollar", outcome="MISSES")
    body = json.loads(scenario.read_text())
    why = "answer accuracy 13/20 is below 0.9 x the plain checkpoint's 15/20"
    body["row"]["output_correctness"] = {"holds": False, "why": why}
    body["row"]["correctness"] = {"holds": False, "why": why}
    body["verdict"]["arms"][0]["why_misses"] = why
    scenario.write_text(json.dumps(body))
    row = recorder.row_from_evidence(scenario, final={"status": "MISSES"},
                                     context={"task": task, "run_id": "r"})
    assert row["gates_missed"].count(why) == 1
