"""What every analysis script shares: where runs live, how a task scores, how a run reads.

Everything here reads evidence the harness already wrote; nothing measures.
The defaults are this checkout's own trees, and every one is overridable on
the command line of the script that uses it.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVOLVE = REPO / "serverbench" / "evolve_space" / "evolve"
SERVER_OUTPUT = REPO / "server_output"
LEDGER = REPO / "evaluation_progress"
TASKS = REPO / "serving_tasks"
FINAL_VALIDATION = REPO / "serverbench" / "artifacts" / "final_validation"
#: Where the retrospective-budget experiment puts its re-measured verdicts.
REMEASURED = FINAL_VALIDATION / "budget"
OUT = REPO / "experiments" / "analysis" / "out"

#: Development exports that are not the agent's own measurement of its server:
#: mock and stub servers, and the harness's review and full-test copies.
NOT_A_MEASUREMENT = ("mock", "stub", ".review/", "/full-tests/")


def add_paths(ap: argparse.ArgumentParser, *names: str) -> None:
    """The input and output roots a script reads, each with its repo default."""
    defaults = {
        "evolve": (EVOLVE, "run roots, <task>/<run-id>/"),
        "server_output": (SERVER_OUTPUT, "published lineages and their verdicts"),
        "tasks": (TASKS, "task directories (contract.yaml)"),
        "out": (OUT, "where results are written and read back"),
    }
    for name in names:
        default, what = defaults[name]
        ap.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default,
                        help=f"{what} (default: {display(default)})")


def display(path: Path) -> str:
    path = Path(path)
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        lines = Path(path).read_text("utf-8", errors="ignore").splitlines()
    except OSError:
        return out
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def ts(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {display(path)}")


def read_csv(path: Path) -> list[dict]:
    if not Path(path).is_file():
        raise SystemExit(f"{display(path)} does not exist; run the script that writes it first")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- a task's scoring terms ----------------------------------------------------

def contract(task: str, tasks: Path = TASKS) -> dict:
    import yaml

    return yaml.safe_load((Path(tasks) / task / "contract.yaml").read_text("utf-8")) or {}


def task_terms(task: str, tasks: Path = TASKS) -> dict:
    """Token prices, the p90 bars and the GPU cost rate, from the task's contract."""
    spec = contract(task, tasks)
    prices = spec["token_prices"]
    level = next(lv for lv in spec["service_levels"] if int(lv["percentile"]) == 90)
    return {"p_in": float(prices["usd_per_million_input"]),
            "p_out": float(prices["usd_per_million_output"]),
            "tau_first": float(level["ttft_ms"]), "tau_token": float(level["tpot_ms"]),
            "usd_per_hour": float(spec["usd_per_gpu_hour"]) * int(spec["devices"])}


# --- a run ---------------------------------------------------------------------

def run_roots(evolve: Path = EVOLVE) -> list[Path]:
    return sorted(p for p in Path(evolve).glob("*/*") if (p / "journal.jsonl").is_file())


def run_info(root: Path) -> dict:
    """What a run says about itself: configuration, start, and each published version."""
    root = Path(root)
    identity = load_json(root / "identity.json")
    events = read_jsonl(root / "journal.jsonl")
    created = next((e for e in events if e.get("event") == "run_created"), {})
    started = identity.get("created_at") or created.get("at_utc")
    t0 = ts(started) if started else None
    versions = []
    for e in events:
        if e.get("event") == "version_registered":
            versions.append({"version": e.get("version_id", ""),
                             "hours": round((ts(e["at_utc"]) - t0) / 3600, 3) if t0 else None,
                             "at_unix": ts(e["at_utc"]),
                             "directory": e.get("directory", "")})
    notes = [{"hours": round((ts(e["at_utc"]) - t0) / 3600, 2) if t0 else None,
              "note": e.get("note") or ""}
             for e in events if e.get("event") == "candidate_noted"]
    lineage = Path(versions[-1]["directory"]).parent.name if versions else ""
    return {"root": root, "run_id": root.name,
            "task": identity.get("task_id") or created.get("task_id", ""),
            "agent_cli": created.get("agent_cli", ""), "agent_model": created.get("agent_model", ""),
            "effort": created.get("agent_effort", ""), "evolve_hours": created.get("hours", ""),
            "started_utc": started or "", "t0": t0, "lineage": lineage,
            "versions": versions, "notes": notes}


def runs_by_lineage(evolve: Path = EVOLVE) -> dict[str, dict]:
    """Every run under `evolve`, keyed by the lineage it published into."""
    out = {}
    for root in run_roots(evolve):
        info = run_info(root)
        if info["lineage"]:
            out[info["lineage"]] = info
    return out


def model_label(info_or_row: dict) -> str:
    model = str(info_or_row.get("agent_model") or "").rsplit("/", 1)[-1]
    return " ".join(p for p in (model, str(info_or_row.get("effort") or "")) if p)


def is_measurement(path: str) -> bool:
    return not any(marker in path for marker in NOT_A_MEASUREMENT)
