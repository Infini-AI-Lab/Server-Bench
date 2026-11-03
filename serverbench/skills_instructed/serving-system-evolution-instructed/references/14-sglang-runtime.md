# SGLang: mechanisms, not a task catalogue

Distilled from the official documentation. Learn design ideas without requiring
SGLang itself, any particular strategy, or upstream source code. Features depend
on installed version, model and hardware. See [the optimization menu](17-optimization-techniques.md).

## Reuse and memory

Radix Cache finds reusable token prefixes; eviction policies rank eligible
unlocked leaves. A retention policy is not a guarantee that state stays resident.
[Radix eviction](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy).

HiCache adds GPU, host and optional shared storage tiers, with configurable
write-back and data movement. It extends reuse capacity but adds transfer cost.
[HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design).

Session-aware radix caching can favor state referenced by an active session.
The session label does not reconstruct the prompt, and soft protection does
not prevent eviction under sufficient pressure.
[Session radix cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache).

## Work and execution

Chunk size, static memory allocation, running-request limits and conservative
admission affect throughput, latency and retractions together.
[Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning).

The overlap scheduler prepares the next batch while the GPU runs the current
one. This targets host-side bubbles, not a change to request semantics.
[Scheduler explanation](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/).

Attention and recurrent-layer backend coverage differs by hardware and phase.
Do not assume that a fast conventional-attention route supports every layer.
[Backend guide](https://docs.sglang.io/docs/advanced_features/attention_backend).

## Beyond cache: cross-stack choices

The [kernel optimization blog](https://www.lmsys.org/blog/2025-08-27-gpt-oss/)
illustrates fusion and removal of unnecessary execution work.
The [expert-parallel guide](https://docs.sglang.io/docs/advanced_features/expert_parallelism)
describes TBO/SBO communication overlap, distinct from CPU/GPU scheduling overlap.
These are model/hardware-dependent examples, not universal recommendations.

[The full-stack menu](17-optimization-techniques.md) connects kernels, compiler
work, batching, admission, parallelism, communication, exact speculative
verification and routing to small useful experiments. Other designs remain
welcome; the reference is not an allowed-strategy list.

Our application: use a repeated-prefix, mixed-length, slot-reuse or small
backend comparison when it answers a question. Record artifacts under
/workspace/agentx, checkpoint asynchronously, and continue. These are optional
experiments. Each design change requires short real GPU profiling plus
correctness/replay checks. Exploratory checkpoint saving stays unconditional.
