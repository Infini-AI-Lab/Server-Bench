"""What OFFERED LOAD should a task carry on THIS machine? Measure it.

    $SERVERBENCH_PY -m final_validate.concurrency \
        --task <task> --arm sglang --model-dir <snapshot> --gpu <ids> \
        --points 1,2,4,8

**Run this on a new machine BEFORE the first evolve, once per (task, model,
hardware).** The answer is a property of all three and transfers between none
of them: a 9B model on two L40s and a 12B model on two H200s saturate at
different loads, and a task whose gates are tighter saturates earlier than one
whose gates are loose. A calibration copied from another host is a number
about that host.

## Why offered load is set explicitly

aiperf's default `--concurrency` is ONE in-flight request, which measures a
server single-stream. `AGENTIC_REPLAY` does not derive concurrency from the
trace: it governs timing WITHIN a session (inter-turn delays, think time),
while how many sessions are in flight is the load generator's business, so
every measurement passes `--concurrency` itself.

`--concurrency` is not a scenario lock violation. The validator rejects only
rate-shaped scheduling -- `--request-rate`, `--user-centric-rate`,
`--fixed-schedule`, `--adaptive-scale` -- and says so in its own words: *"the
default concurrency shape is the only one the lock may stamp."*

## The shape of this package

One responsibility per module, so a caller who wants only one point does not
have to go through the CLI:

    arm.py        standing the reference framework up on the right cards
    point.py      measuring ONE offered load and reading the task's gates
    strategy.py   which loads to measure -- frontier search, or a flat sweep
    report.py     what gets written down, and what it says it is about
    configs/      server launch lines, one file per (model, arm, variant)

`Plan` below is the input to all of them. It is a dataclass rather than an
argparse namespace so a task, a test or a deploy script can drive a search
without building a command line:

    from final_validate.concurrency import Plan, sweep, frontier

    plan = Plan(task="qwen3_5_9b_standard_2xL40", arm="sglang",
                model_dir=Path(snapshot), gpu="4,5",
                out_dir=Path("/tmp/cal"), duration=1800)

    report = frontier(plan)                  # where does it stop holding?
    report = sweep(plan, points=[4, 5, 6])   # which of these is best?

## Two questions, two strategies, and they are not the same question

`frontier()` answers **"where does this arm stop holding the gates"** --
doubling until a gate misses, walking down if the start point itself missed,
then bisecting. That is the right search when a frontier exists.

`sweep()` answers **"which of these loads is best"** -- every point measured,
none skipped, no bisection. Use it when the frontier is known to be absent, or
when the question is an operating point rather than a limit.

The task build workflow uses `sweep()` because it needs a comparable record of
every candidate load before it freezes the operating point.

## What a search can and cannot establish inside a time box

A fixed time box plus a heavy-tailed corpus means **different offered loads do
not complete the same work**: request counts and mean output lengths can
differ several-fold between adjacent points.

Fixing the seed does not fix this: `--concurrency N` opens N lanes that
consume the seeded stream in parallel, so adjacent points diverge immediately
and share few conversations. A fixed seed makes REPEATS of one point
reproducible; it does not make different points comparable. By default each
point draws a fresh development seed.

So a fine-grained ranking across loads is not something this measurement can
deliver, and a search that claims one is claiming the draw. What IS stable
across repeats -- where throughput peaks, where eviction starts, whether the
client ever reaches the offered number -- is what an operating point should be
chosen on. `report.py` records both, and says which is which.

## What it writes

    serverbench/artifacts/concurrency/<host>/<task>-<arm>/calibration.json

Gitignored, because a calibration is about ONE host. What ships with the
repository is the reusable half: `configs/*.txt`, read with
`--serve-extra-file`.

`serving_tasks/_concurrency_search_archive/<task>/` (gitignored) is where an
operator records the SGLang search exports behind a task's reference operating
point (its `REFERENCE_CONCURRENCY`) and its p90 bars, which are the worst p90
of five random-seed SGLang runs at that point.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve()
PACKAGE = HERE.parent
SERVERBENCH = HERE.parents[3]
REPO = SERVERBENCH.parent
if str(SERVERBENCH / "harness") not in sys.path:
    sys.path.insert(0, str(SERVERBENCH / "harness"))

SCHEMA = "final_validate.concurrency_calibration/1"

#: Where a calibration lands. Per host, because that is what it is about.
ARTIFACTS = SERVERBENCH / "artifacts" / "concurrency"

#: The committed launch lines. One file per (model, arm, variant), read with
#: `--serve-extra-file` so no shell touches the JSON inside them.
CONFIGS = PACKAGE / "configs"


@dataclass
class Plan:
    """Everything one search needs to know, and nothing about how it searches.

    Deliberately a plain dataclass with the same attribute names the CLI uses,
    so `point.measure()` accepts either this or an argparse namespace and the
    CLI needs no adapter.
    """

    task: str
    model_dir: Path
    gpu: str
    arm: str = "sglang"
    out_dir: Path = None            # type: ignore[assignment]
    duration: int = 1800
    port: int = 8931
    ready_timeout: float = 1800.0
    #: Override the task's frozen launch line for this arm. Recorded in the
    #: report so a reader can tell a study from a calibration.
    serve_extra: str = ""
    #: None draws a fresh development seed per point; an integer fixes one,
    #: which makes repeats of a point replay the same sessions.
    random_seed: int | None = None
    #: Frontier-search bounds. Ignored by `sweep()`.
    start: int = 4
    ceiling: int = 512
    resolution: int = 2
    points: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        if int(self.duration) != 1800:
            raise ValueError(
                f"concurrency calibration requires the complete 1800-second "
                f"verdict workload at every point; got {self.duration}")
        if self.out_dir is None:
            import socket
            self.out_dir = (ARTIFACTS / socket.gethostname()
                            / f"{self.task}-{self.arm}")
        # ABSOLUTE, ALWAYS. The run directory is written relative to the
        # caller's cwd, but `point.measure` runs aiperf with `cwd=REPO` and
        # hands it the artifact directory as a string, so a relative path
        # would name two places: aiperf's export would land in one and
        # `aggregate()` would glob the other, find nothing, and record a full
        # 1,800 s point as `INVALID (nothing measured)`. Resolving here makes
        # both halves name the same place whatever the caller's cwd.
        self.out_dir = Path(self.out_dir).expanduser().resolve()

    @property
    def task_dir(self) -> Path:
        return REPO / "serving_tasks" / self.task


def config(name: str) -> Path:
    """A committed launch line by name, e.g. `config("sglang-qwen3.5-cookbook")`.

    Resolves a name to a file under `configs/`, so callers never hardcode the
    path.
    """
    for candidate in (CONFIGS / name, CONFIGS / f"{name}.txt"):
        if candidate.is_file():
            return candidate
    available = sorted(p.stem for p in CONFIGS.glob("*.txt"))
    raise SystemExit(f"no launch-line config named {name!r}. "
                     f"available: {', '.join(available)}")


from final_validate.concurrency.arm import (  # noqa: E402
    docker_device_spec,
    launch_line_from_file,
    serve_extra_for,
    wait_for_quiet_cards,
)
from final_validate.concurrency.point import aiperf_command, measure  # noqa: E402
from final_validate.concurrency.report import save  # noqa: E402
from final_validate.concurrency.strategy import frontier, sweep  # noqa: E402

__all__ = [
    "ARTIFACTS", "CONFIGS", "Plan", "SCHEMA", "aiperf_command", "config",
    "docker_device_spec", "frontier", "launch_line_from_file", "measure",
    "save", "serve_extra_for", "sweep", "wait_for_quiet_cards",
]
