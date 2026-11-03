"""The CLI. One entry point, two strategies, same Plan.

    # where does SGLang stop holding this task's gates on this machine?
    $SERVERBENCH_PY -m final_validate.concurrency \
        --task qwen3_5_9b_standard_2xL40 --arm sglang \
        --model-dir <snapshot> --gpu 4,5

    # which of these operating points is best? (no frontier to find)
    $SERVERBENCH_PY -m final_validate.concurrency \
        --task qwen3_5_9b_standard_2xL40 --arm sglang \
        --model-dir <snapshot> --gpu 4,5 --points 4,5,6,7,8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from final_validate.concurrency import ARTIFACTS, Plan
from final_validate.concurrency.arm import launch_line_from_file
from final_validate.concurrency.report import summarise
from final_validate.concurrency.strategy import frontier, sweep


def main(argv: list[str] | None = None) -> int:
    import final_validate.concurrency as pkg

    ap = argparse.ArgumentParser(
        prog="final_validate.concurrency",
        description=pkg.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True,
                    help="task directory name, e.g. qwen3_5_9b_standard_2xL40")
    ap.add_argument("--arm", default="sglang",
                    help="reference framework to calibrate against")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gpu", required=True, help="e.g. 3,4")
    ap.add_argument("--out-dir", default="",
                    help="default: serverbench/artifacts/concurrency/<host>/"
                         "<task>-<arm>")
    ap.add_argument("--serve-extra", default="",
                    help="override the task's frozen launch line for this arm")
    ap.add_argument("--serve-extra-file", default="",
                    help="read the override from a FILE, or name a committed "
                         "one from configs/ without its .txt. Prefer this "
                         "whenever the line contains JSON: a shell mangles "
                         "--json-model-override-args and the server dies on "
                         "JSONDecodeError rather than on anything about "
                         "serving")
    ap.add_argument("--points", default="",
                    help="measure exactly these offered loads, e.g. 4,5,6,7,8, "
                         "and RANK them. Skips the frontier search entirely: "
                         "no doubling, no walk-down, no bisect, and a miss "
                         "does not stop the sweep. Use it when the frontier is "
                         "known to be absent and the question is which "
                         "operating point is best")
    ap.add_argument("--start", type=int, default=4)
    ap.add_argument("--ceiling", type=int, default=512)
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--duration", type=int, default=1800,
                    help="the scored window per point; must be exactly 1800 "
                         "seconds, matching evolve and final verdict")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--ready-timeout", type=float, default=1800.0)
    ap.add_argument("--random-seed", type=int, default=None,
                    help="measure on this workload seed; default draws a fresh "
                         "development seed per point, like every other "
                         "development measurement")
    args = ap.parse_args(argv)

    if args.serve_extra_file:
        if args.serve_extra:
            raise SystemExit("pass --serve-extra or --serve-extra-file, "
                             "not both")
        path = Path(args.serve_extra_file)
        if not path.is_file():
            path = pkg.config(args.serve_extra_file)
        args.serve_extra = launch_line_from_file(path)

    points = [int(x) for x in str(args.points).split(",") if x.strip()]
    if any(p < 1 for p in points):
        raise SystemExit("--points takes offered loads >= 1")

    plan = Plan(
        task=args.task, arm=args.arm, model_dir=Path(args.model_dir),
        gpu=args.gpu,
        out_dir=(Path(args.out_dir) if args.out_dir
                 else ARTIFACTS / __import__("socket").gethostname()
                 / f"{args.task}-{args.arm}"),
        duration=args.duration, port=args.port,
        ready_timeout=args.ready_timeout, serve_extra=args.serve_extra,
        random_seed=args.random_seed, start=args.start,
        ceiling=args.ceiling, resolution=args.resolution, points=points)

    if not (plan.task_dir / "contract.yaml").is_file():
        raise SystemExit(f"{args.task} is not a task on disk")

    report = sweep(plan) if points else frontier(plan)
    print(summarise(report))
    print(f"  written to {Path(plan.out_dir) / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
