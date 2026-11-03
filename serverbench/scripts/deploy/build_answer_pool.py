#!/usr/bin/env python3
"""Write the held-out answer pool the verdict's private correctness stage draws from.

The pool is the GSM8K test split (openai/grade-school-math, 1,319 problems),
written to `serverbench/artifacts/eval_samples/
gsm8k-final_validation.json` (gitignored, host-only). Each verdict draws a
seeded sample of 20 questions from it and compares the candidate's answers
with the plain checkpoint's (`final_validate/private_correctness.py`).

Stdlib only. Needs network once; an existing pool is left untouched.

    "$SERVERBENCH_PY" serverbench/scripts/deploy/build_answer_pool.py [--source test.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
POOL = REPO / "serverbench" / "artifacts" / "eval_samples" / "gsm8k-final_validation.json"
SOURCE_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
              "grade_school_math/data/test.jsonl")
PROBLEMS = 1319


def build(raw: bytes) -> dict:
    items = []
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        items.append({"id": f"gsm8k-test-{index:04d}", "question": row["question"],
                      "reference_answer": row["answer"]})
    if len(items) != PROBLEMS:
        raise SystemExit(f"the GSM8K test split has {len(items)} problems, not {PROBLEMS}")
    return {"schema": "serverbench.answer_pool/1", "source": SOURCE_URL, "items": items}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, help="a local copy of the GSM8K test.jsonl")
    ap.add_argument("--out", type=Path, default=POOL)
    args = ap.parse_args(argv)
    if args.out.is_file():
        print(f"answer pool exists: {args.out}")
        return 0
    if args.source:
        raw = args.source.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:  # noqa: S310
            raw = response.read()
    body = build(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1) + "\n", "utf-8")
    print(f"wrote {len(body['items'])} items to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
