"""The load a task states is the load SGLang's search adopted, and no other.

Every live task lets the candidate declare its own offered concurrency, and
states SGLang's measured operating point on the task's hardware as a reference
(`REFERENCE_CONCURRENCY`). That number is a property of the model, the hardware
and the p90 bars together, so it must be the adopted point of the recorded
search in `reference.yaml`, and each recorded point must carry the p90 numbers
it was judged on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "serving_tasks"

yaml = pytest.importorskip("yaml")


def _tasks() -> list[str]:
    if not TASKS.is_dir():
        return []
    return sorted(p.name for p in TASKS.glob("*")
                  if (p / "contract.yaml").is_file())


def _reference(task: str) -> dict:
    import yaml

    path = TASKS / task / "reference.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text("utf-8")) or {}


def test_there_are_tasks_to_check() -> None:
    assert _tasks(), "no task carries a contract, so nothing here checks anything"


@pytest.mark.parametrize("task", _tasks())
def test_the_offered_load_is_the_one_the_search_adopted(task: str) -> None:
    """The load a task is measured at, stated once and backed by evidence.

    Every live task lets the candidate declare its own load in
    `concurrency.json`, so the module fixes none (`OFFERED_CONCURRENCY` is
    None) and states the recorded SGLang search's adopted point as
    `REFERENCE_CONCURRENCY` -- a reference the instruction states, never the
    load it offers. The two cannot drift apart without this failing.
    """
    import paths

    module = paths.task_module(task)
    assert getattr(module, "CONCURRENCY_POLICY", None) == "candidate", (
        f"{task} does not let the candidate declare its load")
    assert getattr(module, "OFFERED_CONCURRENCY", None) is None, (
        f"{task} lets the candidate choose its load and also fixes one")
    search = _reference(task).get("concurrency_search") or {}
    reference = getattr(module, "REFERENCE_CONCURRENCY", None)
    if not search:
        assert reference is None, (
            f"{task} states reference load {reference} with no recorded "
            f"search behind it. The number is a property of this model on "
            f"this hardware and is never inherited")
        return
    assert reference == search.get("adopted"), (
        f"{task}: states reference load {reference} but the recorded "
        f"search adopted {search.get('adopted')}")
    rule = str(search.get("selection_rule", ""))
    if rule.startswith("operator_confirmed_concurrency_"):
        # A confirmed operating point replaces the search: the rule names the
        # point, and the SLO seeds in `slo_derivation` are the measurements
        # taken at it.
        assert rule.startswith(f"operator_confirmed_concurrency_{reference}_"), (
            f"{task}: the operator confirmed a point other than {reference}: {rule}")
        assert (search.get("slo_derivation") or {}).get("runs"), (
            f"{task}: an operator-confirmed point with no seeds measured at it")
        return
    assert any(int(p["offered"]) == int(reference)
               for p in search.get("points") or []), (
        f"{task}: the adopted point {reference} is not one the search measured")


@pytest.mark.parametrize("task", _tasks())
def test_the_search_only_adopts_a_point_it_measured(task: str) -> None:
    """Every recorded point carries the p90 numbers it was judged on."""
    search = _reference(task).get("concurrency_search") or {}
    if not search:
        pytest.skip(f"{task} has no concurrency search yet")
    import yaml
    contract = yaml.safe_load(
        (TASKS / task / "contract.yaml").read_text("utf-8")) or {}
    percentiles = sorted(int(r["percentile"])
                         for r in contract.get("service_levels") or [])
    fields = [f"{m}_p{pc}_ms" for pc in percentiles for m in ("ttft", "tpot")]
    for point in search.get("points") or []:
        for field in (*fields, "output_tokens_per_s"):
            assert point.get(field) is not None, (
                f"{task}: point at offered {point.get('offered')} records no "
                f"{field}, so nothing can say whether it held the gates")
