#!/usr/bin/env python3
"""A handful of requests shaped like qwen3_5_9b_standard_2xL40's own workload.

Lengths come from the task's measured corpus shape -- prompts 64-989824 tokens, outputs
1-40339 -- so a smoke test exercises the regime the task is scored in
rather than a generic short prompt. It is NOT the frozen trace: the trace is
held out, and this is here so you can see the server answer something.
"""

import json
import os
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")

#: Roughly 64-989824 prompt tokens once tokenized, which is this task's range.
_FILLER = ("scheduler batch kv cache prefill decode throughput latency "
           "tokenizer attention checkpoint quantization ") * 6

REQUESTS = [
    {"prompt": _FILLER + "\n\nSummarise the terms above in one sentence.",
      "max_tokens": 128},
    {"prompt": _FILLER + "\n\nName the two that concern memory.",
      "max_tokens": 128},
]


def main() -> int:
    for i, body in enumerate(REQUESTS):
        body = {**body, "model": "target", "temperature": 0.0, "stream": False}
        req = urllib.request.Request(
            f"{BASE}/v1/completions", method="POST",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        text = (out.get("choices") or [{}])[0].get("text", "")
        print(f"[{i}] {len(text)} chars: {text[:120]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
