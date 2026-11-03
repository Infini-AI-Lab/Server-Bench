# Serving optimization techniques: an open menu

Use this reference when choosing a small experiment, not as a curriculum to
complete. It covers the serving stack, not only KV caching. None of the named
techniques is required; invent alternatives, combine mechanisms, question an
upstream design, or keep a simpler implementation if feedback favors it.
No framework source code or implementation recipes are supplied here.

Official vLLM/SGLang docs and engineering blogs are linked beside each
mechanism. Historical posts explain design decisions, not current
feature support or expected speedups on these tasks. Verify the installed
version, model/state family, dtype, hardware and feature interactions. The
small experiments and tradeoff analysis below are our engineering guidance,
not upstream performance claims or additional evidence obligations.

## Navigate by the current question

- Host overhead: framework boundaries, persistent input preparation, output path.
- Utilization and tail latency: batching, admission and scheduling.
- Device execution: kernels, compilation and CUDA graphs.
- Capacity and state: memory budgeting, paging, prefix reuse and offload.
- Concurrency and scale: async overlap, parallelism, communication and placement.
- Fewer serial steps: speculative execution and exact verification.
- End-to-end balance: routing, prefill/decode placement and streaming.

| Symptom | Example direction, not a prescription | One lightweight probe |
|---|---|---|
| GPU waits between small forwards | persistent metadata, IPC reduction, CPU/GPU overlap | short mixed-length stream and step timers |
| decode slows during long arrivals | chunked prefill, token budgets, admission | one long prompt alongside ongoing decode |
| batch is large but useful work is small | ragged batching, bucketing, packing | compare active versus padded tokens |
| one operator dominates | backend dispatch, tiling, fusion, layout | representative-shape timing and numerical check |
| many tiny launches or recompilations | graphs, compiler fusion, shape specialization | compare common shapes and one outlier |
| memory pressure or cache churn | capacity budgeting, paging, reuse/eviction | bounded grow/retire/reuse sequence |
| adding GPUs makes steps slower | topology, TP/PP/DP choice, collective overhead | same short workload on allowed placements |
| MoE ranks wait unevenly | expert placement, dispatch/combine, DBO/TBO | per-rank times on a small representative batch |
| draft work grows but tokens do not | speculative depth or fallback | accepted tokens per total elapsed step |
| engine is fast but client sees bursts | sampling, detokenization, output queues | client-side token arrival timestamps |

Pick the row that helps, or a completely different hypothesis. Supplied gates
and AgentX reports remain the acceptance/score evidence; these probes are
diagnostics. No matrix-wide test pass is required before checkpointing.

## Framework, host work and execution boundaries

vLLM's V1 engineering account separates the API-facing work from the engine
loop, caches worker request state, and sends incremental updates rather than
rebuilding all inputs each step. These are framework optimizations even when
the model kernels remain unchanged.
[Official V1 architecture blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release).

Possible experiments: move invariant processing out of decode iterations,
reduce Python object churn, batch metadata updates, reuse host buffers, avoid
redundant serialization/copies, or separate tokenizer/output work from the
scheduler. A generic framework fast path, a specialized runner, or a different
process/thread decomposition are all possibilities within task restrictions.
A framework reference does not require using that framework.

Tradeoffs: IPC and extra processes add overhead; more CPU threads can contend
with the GPU submission thread; cached metadata can go stale. A short request
stream with coarse timestamps at API, scheduler, forward and output boundaries
often gives enough feedback for the next edit. A full profiler is optional.

## Batching: membership, representation and shape

Continuous batching replaces finished requests between iterations. Persistent
batching retains input/metadata buffers and updates changed entries, reducing
host preparation. They solve different problems and can be combined.
The V1 blog documents persistent input preparation.
[Official V1 batching discussion](https://vllm.ai/blog/2025-01-27-v1-alpha-release).

Other candidates include ragged/packed tokens instead of rectangular padding,
length-aware buckets, microbatches, persistent slots, and combining prefill with
decode work. Batch size by sequence count alone hides token-count differences.
Use the actual length/head/state shapes rather than a convenient tiny prompt.

Tradeoffs: a larger batch can improve arithmetic efficiency while lengthening
every decode step; many buckets increase complexity and graph memory; stable
slots complicate retirement. Try mixed lengths with one early completion and a
new arrival. Check request identity, positions, sampler state and slot reuse.
These suggestions are not a required batching architecture or test suite.

## Scheduling and admission

vLLM documents decode-prioritized chunked prefill under a shared token budget.
SGLang tuning relates chunk size, running-request limits, memory fraction and
admission conservativeness to queueing, token use and retractions.
[vLLM tuning](https://docs.vllm.ai/en/stable/configuration/optimization/),
[SGLang tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning).

Explore token-budget allocation, prefill chunk size, decode reservation,
FCFS versus age/priority/prefix-aware ordering, and memory-aware admission.
Scheduling policy chooses the next work; admission decides how much state can
safely be live. Preemption/recompute can reclaim capacity but may amplify churn.

Tradeoffs: small chunks protect decode yet add launches and slow prompt progress;
locality can starve cold requests; over-admission can repeatedly evict useful
work. Try one budget/policy adjustment with mixed long and short requests, inspect
queue delay and completed work, then checkpoint. Preserve offered arrivals,
concurrency and requests: the benchmark load is not a tuning knob.

## Kernels and backend dispatch

Optimize the operator that costs time, not merely the one with an interesting
implementation. Attention, GEMM/GEMV, normalization, activation, RoPE, cache writes,
sampling and MoE dispatch can each matter at different shapes.

SGLang's gpt-oss engineering post describes fusing normalization with all-reduce,
combining RoPE with KV writes, and removing unnecessary kernels. Its v0.3 post
describes compiler optimization of linear/norm/activation work alongside
specialized attention and sampling kernels. These are design examples, not
permission to apply those exact paths to every model.
[gpt-oss optimization blog](https://www.lmsys.org/blog/2025-08-27-gpt-oss/),
[SGLang compiler/kernel blog](https://www.lmsys.org/blog/2024-09-04-sglang-v0-3/).

Possible experiments: choose a better-supported attention/GEMM backend; specialize
prefill versus decode; improve tiling, vectorization, memory coalescing or layout;
fuse producer/consumer operations; remove an intermediate tensor or redundant
launch. Fusion can increase register pressure or lose a better library path.
Shape-specific dispatch needs a visible fallback outside its supported domain.

Use a representative-shape microbenchmark and a small independent numerical
comparison when useful; then obtain service feedback. Check the actual GPU,
head layout, dtype and recurrent/attention layer support. Hopper/Blackwell-only
instructions are not L40 optimizations.
[vLLM backend matrix](https://docs.vllm.ai/en/stable/design/attention_backends/),
[SGLang backend guide](https://docs.sglang.io/docs/advanced_features/attention_backend).

## Compilation and CUDA graphs

Compilation can fuse operators, eliminate redundant copies and specialize
execution. CUDA graph replay reduces repeated submission overhead for compatible
execution; it is not the same optimization. vLLM describes compiler passes
including no-op/copy elimination and collective/norm fusion, while its graph
design supports full and piecewise execution depending on batch/backend support.
[Compiler engineering blog](https://vllm.ai/blog/2025-08-20-torch-compile),
[CUDA graph design](https://docs.vllm.ai/en/latest/design/cuda_graphs/).

Try dynamic versus specialized shapes, a small set of common buckets, eager
versus compiled execution, or a smaller captured region around dynamic work.
Tradeoffs include compile/capture time, graph-pool memory, padded work, fallback
frequency and recompilation. Persistent addresses alone do not ensure that the
right request metadata reaches a replay.

Compare a few common shapes and one outlier; include cold-start cost when launch
paths change. Keep an eager fallback while iterating. Quantized fusion examples
in upstream posts do not authorize changing a task's fixed precision.

## Memory capacity and lifetime, beyond prefix hits

Weights, request state, activations, graph pools, temporary kernel workspaces
and allocator slack all consume capacity. SGLang's tuning guide distinguishes
prefill, decode and static-allocation pressure.
[Official memory tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning).

Consider workspace/buffer reuse, bounded temporary allocations, graph-pool size,
chunking, selective recomputation or offload when supported. Look for peak
rather than only steady-state usage. Shrinking one pool may merely move pressure
to another, and enlarging KV can leave too little space for a long prefill.
A short long/short request mix with retire-and-reuse feedback may expose the
problem without a large tracing project.

Quantization and approximate compression are general serving techniques, but
not available experiments where the task freezes weights, dtype or semantics.
Do not treat their appearance in an upstream menu as authorization.

## Paging, prefix reuse and hierarchical storage

These are separate, composable layers rather than three names for the same cache.

- **PagedAttention:** logical KV blocks can occupy non-contiguous physical
  memory. Investigate allocation waste, block-table cost, page size and safe
  reuse after device readers retire.
  [Paged-attention design](https://docs.vllm.ai/en/v0.8.1/design/kernel/paged_attention.html).
- **Automatic prefix caching:** vLLM identifies reusable full blocks using
  parent hashes, token content and relevant extra identity. Observe actual
  prefill avoided, not only cache-hit counts.
  [APC design](https://docs.vllm.ai/en/stable/design/prefix_caching/).
- **Radix Cache:** SGLang organizes reusable prefixes as a tree and evicts
  eligible unlocked leaves. Recency/frequency/priority retention can favor
  different workloads, with host lookup and eviction costs.
  [Radix eviction](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy).
- **HiCache:** GPU, host and optional shared storage tiers expand reuse capacity,
  adding prefetch, restore and write-back decisions. A hit helps only if useful
  restored work outweighs transfer and contention.
  [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design).

A cheap experiment can repeat a prefix with changed suffixes, cross a block
boundary, then evict/reload while unrelated requests run. Cache identity and
state must match; hybrid models can need recurrent/convolution state as well
as KV at the reuse boundary. Never cache benchmark answers. A simpler resident
cache is a valid alternative to a hierarchy.

## Async overlap: identify what is being overlapped

**Host/device overlap:** SGLang describes preparing a future batch while the
current GPU batch executes; vLLM exposes async scheduling controls.
[SGLang scheduler blog](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/),
[vLLM scheduler config](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/).

Possible extensions include overlap of input staging, result processing,
tokenization or output handling. Stable/versioned metadata and completion events
matter: a new plan must not overwrite data still used by the preceding batch.
Try on/off comparison over a short request stream; extra streams alone are not
evidence of speedup.

**Communication/compute overlap:** vLLM Dual Batch Overlap and SGLang Two-Batch
Overlap interleave microbatch compute with MoE dispatch/combine communication.
SGLang also describes Single-Batch Overlap around shared-expert work.
[vLLM DBO design](https://docs.vllm.ai/en/stable/design/dbo/),
[SGLang EP and overlap](https://docs.sglang.io/docs/advanced_features/expert_parallelism).

These mechanisms are not a generic replacement for host overlap. They require
relevant communication, model structure and backend support. Smaller microbatches
can reduce kernel efficiency, while competing transfers can erase the gain.
A one-GPU dense model may have no corresponding overlap opportunity.

## Parallelism and topology

Choose among supported forms only within the assigned GPU count and any frozen
topology requirements. More parallelism is not automatically more throughput.

| Technique | What it divides | Main question before trying it |
|---|---|---|
| Tensor parallelism (TP) | tensors/layer computation across ranks | does memory/compute relief outweigh frequent collectives? |
| Pipeline parallelism (PP) | model layers across stages | do reduced intra-layer transfers outweigh bubbles and imbalance? |
| Data parallelism (DP) | requests across model replicas | do weights fit per replica, and does splitting batches hurt efficiency? |
| Context parallelism (CP/DCP) | context/attention work or KV across ranks, depending on implementation | is long-context memory the bottleneck, and is the model/backend supported? |
| Expert parallelism (EP) | MoE experts across ranks | do dispatch/combine and uneven expert demand dominate? |

vLLM's scaling guide discusses TP/PP placement and interconnect effects; its
context-parallel guide explains KV sharding and communication costs. SGLang's EP
guide describes distributed expert execution and associated communication.
[vLLM scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/),
[vLLM context parallel](https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/),
[SGLang expert parallel](https://docs.sglang.io/docs/advanced_features/expert_parallelism).

Our application: compare viable layouts at the SAME total allocated hardware
and offered workload. Include duplicated weights/state, KV head divisibility,
per-rank memory and link topology. PCIe-only placements should not inherit NVLink
assumptions. EP is irrelevant without routed experts; one allocated GPU cannot
become a multi-GPU experiment. These are applicability checks, not a required
topology search before publication.

## Communication and distributed work balance

Potential targets include fewer/smaller metadata broadcasts, collective choice,
packing/layout, avoiding unnecessary synchronization, topology-aware placement,
and balancing expert or replica work. The vLLM compiler blog gives sequence
parallelism/async TP and collective fusion as examples of optimizing across
communication and compute boundaries.
[Official compiler/communication discussion](https://vllm.ai/blog/2025-08-20-torch-compile).

Compare collective and useful compute time on the available ranks, or run a
small end-to-end test on two permitted layouts. Do not optimize network bandwidth
in isolation if the real cost is many small synchronization points. Faster
average ranks cannot compensate for one persistent straggler. Count packing,
extra launches and shared-link contention when judging an overlap experiment.

## Speculative execution and verification

vLLM and SGLang support speculative mechanisms; SGLang documents EAGLE-family
drafting and controls for draft depth, branching and verification size. The
concept is to propose tokens cheaply and verify with the target model, reducing
serial target steps when enough proposals are accepted.
[vLLM speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/),
[SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding).

Possible experiments include draft length, proposal source, verification batch
shape and fallback decisions. Try only if required weights/capabilities are
available and allowed. Measure accepted tokens per total draft-plus-verify time,
not acceptance rate alone. Extra draft memory, low acceptance, large batches or
host synchronization can erase a gain. Preserve the task's target distribution,
token counts and streaming semantics; speculation is not permission to emit
unverified tokens, shorten output or change sampling.

## Routing, stage placement and output delivery

**Routing:** SGLang's cache-aware balancing design motivates balancing reuse
against worker load. Locality can save prefill but create a hot replica.
[Official cache-aware routing blog](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/).

**Stage placement:** separating prefill and decode can isolate latency and allow
different execution choices, but requires state transfer and capacity on both
sides. vLLM explicitly warns that disaggregated prefill is not a throughput
improvement by itself.
[Official P/D design](https://docs.vllm.ai/en/stable/features/disagg_prefill/).

Our application: consider queue-aware placement, locality hints, stage fusion
or separation only where resources and the task allow it. Compare colocated
execution against transfer plus duplicated state and imbalance; tiny allocations
may favor colocation. Task-specific recipes are deliberately not part of this
reference.

**Output delivery:** sampling, stop handling, detokenization and SSE flushing
can dominate client-visible latency even with a fast forward pass. Reuse buffers,
avoid repeated host/device synchronization, or move CPU output work off the
submission path when dependencies permit. Check a short streaming request at
the client. Buffering all tokens until completion is not a TPOT optimization.

## Combine selectively, explore beyond the menu

Interactions often matter more than isolated wins: fusion changes graph
compatibility; capture consumes memory otherwise available for requests; DP
changes batch sizes and cache locality; chunking changes kernel shapes; overlap
changes buffer lifetime; speculation changes verification and scheduling work.

Use these mechanisms as vocabulary for forming a hypothesis, not as an allowed
strategy list. A novel scheduler, memory layout, kernel, runtime organization or
combination is welcome. Start with a small useful test, keep a runnable fallback,
save versions early, and decide the next experiment from feedback. There is no
requirement to exhaust these topics, prove a complete bottleneck theory, or wait
for a full 1800-second result before publishing. Final correctness, replay and
the task-defined objective remain unchanged.
