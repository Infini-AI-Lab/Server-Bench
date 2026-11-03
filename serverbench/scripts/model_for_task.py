#!/usr/bin/env python3
"""Print the model a task serves, read from the task's own module.

    model_for_task.py serving_tasks/<task>              -> the local snapshot directory
    model_for_task.py serving_tasks/<task> --id         -> the Hugging Face id
    model_for_task.py serving_tasks/<task> --revision   -> the pinned revision

The snapshot must be the pinned revision: different weights under one name is
exactly what a pin exists to prevent. Exits non-zero with a message naming what
is missing, so a caller refuses rather than falling back to another model.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3) or (len(argv) == 3 and argv[2] not in ("--id", "--revision")):
        print(__doc__, file=sys.stderr)
        return 2
    task_dir = Path(argv[1])
    if not task_dir.is_dir():
        print(f"no task directory at {task_dir}", file=sys.stderr)
        return 2

    from final_validate import model_identity

    model_id, revision = model_identity.declared_for(task_dir)
    if not model_id:
        print(f"{task_dir.name}'s module declares no MODEL_ID, so the weights "
              f"to serve cannot be derived from the task", file=sys.stderr)
        return 3
    if argv[2:] == ["--id"]:
        print(model_id)
        return 0
    if argv[2:] == ["--revision"]:
        print(revision)
        return 0

    snapshot = model_identity.local_snapshot_of(model_id)
    if snapshot is None:
        print(f"{task_dir.name} serves {model_id!r} and no local snapshot of it "
              f"was found in any Hugging Face cache on this host; stage it "
              f"before running the task", file=sys.stderr)
        return 4
    if revision and snapshot.name != revision:
        print(f"{task_dir.name} pins {model_id}@{revision} and the local "
              f"snapshot is {snapshot.name}", file=sys.stderr)
        return 5
    print(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
