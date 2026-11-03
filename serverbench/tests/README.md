# Test-suite policy

The default suite protects boundaries whose failure wastes a benchmark run or
can publish a false result:

- task, model, hardware, workload, seed, concurrency and pricing identity;
- the 1,800-second final-verdict window, and the 300-1,800-second bounds on a
  development window;
- the p90 TTFT/TPOT bars and per-request scoring, streamed-response
  correctness, and declared long-context input fidelity;
- SGLang bar evidence and replay reproducibility;
- the run lifecycle: resumption, lineages and the final verdict;
- agent isolation: no internet, no held-out data, no other run, one model at
  its full context window.

Tests that enumerate repository text, mirror implementation structure, inspect
run workspaces, or repeat the same generated-task assertion do not belong here:
they are expensive to maintain and fail after valid refactors without
protecting benchmark behavior. Prefer one behavior-level test at the narrowest
public boundary that can fail, and consolidate parametrized repository-wide
scans before adding new ones.

How to run it, and how the packages are organised, is
[`docs/harness/testing.md`](../docs/harness/testing.md).
