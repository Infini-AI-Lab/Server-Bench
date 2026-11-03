# Quantified design review for a specialized serving engine

Read this once, together with 13-vllm-v1.md and 14-sglang-runtime.md.
Then update only the affected measurements each iteration.
The obligation is small: one hypothesis, seven impact rows, source-bound evidence.
This is executable checking of bounded experiments, not a proof for every input.

## Set the boundary before selecting an architecture

Record the pinned model/revision, model and cache dtype, allocated GPU count and
interconnect, offered concurrency, prompt/output/context limits, conversation
reuse, exact protocol and objective from the task contract. Keep all declared
requests usable. Specialize shapes and execution, never answers or trace IDs.
Do not generalize to arbitrary models, tenants, training, autoscaling, cross-node
placement, LoRA or modalities unless this task actually requires them. A one-GPU
task has no inter-device communication, but still has host/device transfer.
Model layer types, recurrent state and shared KV remain required even when an
upstream generic engine takes a different path.

## Seven responsibility axes

Each cost or invariant gets one owner below. A design change can affect several
owners; record those dependency edges without counting the same duration twice.
This partition covers the serving request lifecycle, not every possible hardware
mechanism. Add model-specific invariants within the appropriate row.

| Owner | Required invariant / boundary | Useful measurable quantities |
|---|---|---|
| deployment | clean source boots with only staged assets and declared devices; readiness and shutdown work | cold-start seconds, startup failures, health success, rebuild time |
| semantics | weights/tokenizer/dtype, layer equations, positions/masks, sampling and native IDs preserve the declared model | your own output-correctness evidence, token-ID/text mismatches, protocol/context failures |
| state_memory | request/slot identity is injective; no stale prefix or use-after-free; state fits per GPU | live KV/recurrent bytes, peak reserved bytes, margin, prefix reuse tokens, evictions, invalid transitions |
| scheduling | admit/advance/retire every offered request fairly; batching respects ownership and capacity | queue delay, decode batch occupancy, prompt-token budget, preemptions, starvation/unfinished count |
| host_io | parsing/tokenizing, metadata/IPC, H2D preparation, detokenization and SSE advance without losing tokens | host preparation us/step, API/IPC time, H2D bytes, launches/step, D2H sync count, backpressure |
| device_compute | selected attention/GEMM/recurrent/sampler/compiler/graph route is supported and semantically correct | CUDA kernel time by prefill/decode shape, GPU gaps, graph hit rate, temporary bytes, achieved bandwidth/FLOPS |
| topology_transfer | every rank owns the right shard; collectives agree and complete; communication actually overlaps as claimed | collective bytes/time, uncovered communication, rank skew, device utilization imbalance |

`topology_transfer` may be inapplicable with a reason on one GPU. Other rows
remain applicable even when unchanged. Unknown is unknown, not zero. The script
requires numeric before/after values, units and existing evidence for each row;
copy unchanged measurements only after verifying the source and dependency
assumption. A reference/file is supporting evidence, not automated proof that
an agent-entered measurement was interpreted correctly.

## Small proof obligations and bounds

- Per GPU: weights + KV + recurrent state + graphs + peak temporary + runtime
  reserve <= usable memory. Use measured peaks; allocator reserved and allocated
  overlap, so do not add them. Count shared KV storage once. For ordinary KV,
  bytes/token = sum_over_stored_layers(2 * local_kv_heads * head_dim * dtype_bytes).
  Sliding windows, recurrent layers and shared KV require their actual storage law.
- State: allocated = free + live + retained, with disjoint sets; every active
  request owns exactly one valid slot/generation. Test cancel, error, slot reuse,
  divergent suffix, repeated prefix and mixed-length interleavings.
- Time: TTFT = queue + first-token critical path. TPOT comes from visible token
  intervals. Build a dependency timeline; overlapped CPU/GPU/communication times
  cannot simply be added. A sum of kernel durations is not elapsed request time.
- Potential speedup <= 1 / ((1-f) + f/s), where f is measured critical-path share
  and s is the proposed component speedup. Peak FLOPS/bandwidth are lower-bound
  estimates of duration, not observed throughput. Rank changes by removable time.
- Correctness: finite protocol/replay tests exercise state. Microbenchmarks must
  cover short requests and extreme input lengths (at least 90% of the declared
  context limit, with room for decode). Record actual input/output token counts,
  passing observations and evidence paths in `microbenchmarks.short/extreme`.
  Set `scope.context_limit` to the task's numeric limit, not a narrative string.
  Complete `./full-test --wait` at least once during the evolve. It measures
  performance and replay fidelity, not output correctness; build and run your
  own correctness checks.

## Fast iteration and runnable commands

Use the supplied scripts from `.claude/skills/serving-system-evolution-instructed/scripts`.
All drivers, reports and traces live below `/workspace/agentx`; deployed source
contains only runtime/build files. Freeze a clean candidate and use that same
tree for every check. Keep an already verified fallback.

1. `run_trial.py --task TASK --candidate-dir SOURCE --model-dir "$MODEL_PATH"
   --run-dir /workspace/agentx/quick-N --through correctness` cold-starts and
   runs short multi-turn replay plus protocol/context gates. It never runs 1800s.
2. Write a small Python driver with `setup(candidate_dir)` loading the actual
   candidate and `run(state)` exercising representative prefill/decode shapes.
   Imports must resolve into the frozen source; do not profile only a client.
   `profile_candidate.py --candidate-dir /workspace/agentx/quick-N/candidate
   --driver /workspace/agentx/probe.py --run-dir /workspace/agentx/profile-N
   --backend torch --workload 'prefill/decode; ISL=...; OSL=...; batch=...; BF16; TP=...'`.
   `torch` uses `torch.profiler` (not `torch.profile`). `nsys`/`ncu` are alternatives
   for worker-process or kernel questions. Use the GPU-owning thread/process;
   empty/CPU-only traces, stale source and failed captures fail the check.
3. `design_review.py init --output /workspace/agentx/design-N.json` emits a small
   schema. Fill scope, reading lessons, hypothesis, impacts, invariants and numeric
   metrics. Optional `min`/`max` are checked as inequalities. Then run
   `design_review.py check --task TASK --candidate-dir SOURCE --design DESIGN.json
   --profile PROFILE/profile.json --trial QUICK/trial.json
   --output /workspace/agentx/iteration-N.json`.
4. Repair or revert a failed iteration before building further changes on it.
   Save exploratory checkpoints whenever useful; a saved experiment is not a
   verified runnable iteration. The harness reports missing/stale profiling at
   checkpoint boundaries. Source changes invalidate the receipt.
5. At an initial working baseline, a meaningful bottleneck shift, and before
   the deadline, run an uninstrumented 1800s AgentX window with
   `run_trial.py --stage full --design-evidence /workspace/agentx/iteration-N.json ...`.
   This rejects unverified instructed candidates before consuming the window.
   It does not rerun the short replay. Also complete `./full-test --wait` early
   in this evolve, stopping other GPU work first; inspect its full verdict and
   cold-start logs, then repair and retest failures. Compare full-window score only here;
   short captures provide diagnosis, not surrogate TTFT/TPOT or dollars/dollar.

Profile one representative region with explicit warmup and a few iterations.
Use a baseline capture and a changed capture with the same driver/shapes; note
graph/eager differences. Collect instrumentation only in short diagnostic runs.
Scored serving has no profiler, extra tracing or analysis process on its hot path.
The checker writes bounded summaries, keeping raw captures separately.

Sources: [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html),
[vLLM V1](https://vllm.ai/blog/2025-01-27-v1-alpha-release),
[SGLang overlap scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/).
