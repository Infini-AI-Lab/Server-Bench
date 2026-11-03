"""What each final verdict's published tree is built from (static scan, no GPU).

Scans the agent-authored source of the version the final verdict measured
(everything except the files and directories the harness provides) and counts
identifiers that mark a technique. A technique counts as present at a minimum
number of hits, so a comment alone does not count.

    $SERVERBENCH_PY experiments/analysis/design_scan.py [--out DIR]

Reads <out>/final_rescored.csv; writes <out>/design_features.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import common

FEATURES = {  # name: (regex, minimum hits)
    "triton_kernels": (r"@triton\.jit", 1),
    "cuda_source": (r"load_inline|cpp_extension|__global__", 1),
    "cuda_graphs": (r"CUDAGraph|torch\.cuda\.graph\(|cuda\.graphs|make_graphed_callables", 2),
    # draft/accept logic, not a checkpoint's MTP weight names (which a Qwen3.5 loader mentions anyway)
    "speculative_decoding": (r"draft\w*|accept(ed|ance)\w*", 5),
    "prefix_reuse": (r"prefix[_ ]?(cache|hash|match|reuse|tree)|radix|longest_common|lcp", 5),
    "state_snapshots": (r"snapshot|checkpoint_state|state_cache|ring_snapshot|gdn_state", 5),
    "paged_kv": (r"paged|page_table|block_table|num_blocks", 5),
    "tensor_parallel": (r"all_reduce|init_process_group|nccl|tensor_parallel|tp_size|\bTP\b|world_size", 3),
    "host_kv_tier": (r"pin_memory|host[_ ]tier|host_cache|cpu_offload|offload", 3),
}
SOURCE_SUFFIXES = (".py", ".sh", ".cu", ".cpp", ".h", ".cuh")
HARNESS_FILES = {"start_server.sh", "stop_server.sh", "smoke.sh", "example_requests.py", "native_tokens.py",
                 "gpu_env.sh"}
HARNESS_DIRS = {"tests", "wait", "full-test", "gpu-clock", "clock", "measurements", "task_context"}


def scan(root: Path) -> dict:
    files = [f for f in root.rglob("*") if f.is_file() and f.suffix in SOURCE_SUFFIXES
             and not (HARNESS_DIRS & set(f.relative_to(root).parts)) and f.name not in HARNESS_FILES]
    texts = [f.read_text(errors="ignore") for f in files]
    text = "\n".join(texts)
    out = {k: len(re.findall(rx, text)) >= n for k, (rx, n) in FEATURES.items()}
    out["agent_source_lines"] = sum(len(t.splitlines()) for t in texts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths(ap, "server_output", "out")
    args = ap.parse_args()
    rows = [r for r in common.read_csv(args.out / "final_rescored.csv")
            if r["kind"] == "final" and not r["undecided"]]
    out = []
    for r in rows:
        tree = args.server_output / r["lineage"] / r["version_dir"]
        if not tree.is_dir():
            print(f"skip {r['version_dir']}: no published tree under {common.display(args.server_output)}")
            continue
        out.append({"task": r["task"], "run_id": r["run_id"], "agent_model": r["agent_model"],
                    "effort": r["effort"], "eligible": r["eligible"], "dpd": r["dpd"],
                    "gross_dpd": r["gross_dpd"], "concurrency": r["concurrency"],
                    **scan(tree), "version_dir": str(tree)})
    if not out:
        print("no published trees found")
        return 1
    common.write_csv(args.out / "design_features.csv", out)
    keys = list(FEATURES)
    print(f"{'run':40}" + " ".join(k[:6] for k in keys) + "  lines  $/$")
    for o in sorted(out, key=lambda o: (o["task"], -float(o["dpd"]))):
        marks = " ".join(("  X   " if o[k] else "  .   ")[:6] for k in keys)
        label = f"{o['task'][:12]} {common.model_label(o)}"[:40]
        print(f"{label:40}{marks} {o['agent_source_lines']:6} {float(o['dpd']):.2f} "
              f"{'' if o['eligible'] == 'True' else 'INEL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
