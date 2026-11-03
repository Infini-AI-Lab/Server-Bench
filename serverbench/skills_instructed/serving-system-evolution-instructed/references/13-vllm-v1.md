# vLLM V1: mechanisms, not a task catalogue

Distilled from the official documentation. This is conceptual guidance, not
source code, a required architecture, or a promise about the installed version.
See [the optimization menu](17-optimization-techniques.md) for experiments.

## Runtime organization

The API server handles request-facing work; the engine core coordinates
scheduling and state; GPU workers execute the model. This separation makes
host processing, queues and IPC optimization targets as well as kernels.
[Architecture](https://docs.vllm.ai/en/stable/design/arch_overview/).

## Cache and work management

PagedAttention uses block-addressed KV storage. APC adds prefix reuse using
parent-linked block hashes and relevant extra identity; only complete blocks
are cached. Paging and prefix matching solve different problems.
[Paged attention](https://docs.vllm.ai/en/v0.8.1/design/kernel/paged_attention.html),
[APC](https://docs.vllm.ai/en/stable/design/prefix_caching/).

Token-budget scheduling and chunked prefill balance prompt work with decode.
Async scheduling targets CPU overhead. These controls interact with residency,
preemption and feature compatibility; do not import example parameter values
as optimal settings for this model or hardware.
[Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/),
[scheduler controls](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/).

## Execution acceleration

Full/piecewise graph modes and attention backend support are separate choices.
Inspect the supported route for the actual model, batch composition and GPU.
[Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/),
[backend matrix](https://docs.vllm.ai/en/stable/design/attention_backends/).

## Beyond cache: cross-stack choices

The [V1 engineering blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)
explains persistent input preparation and incremental worker-state updates.
The [compiler blog](https://vllm.ai/blog/2025-08-20-torch-compile) describes
fusion, redundant-copy elimination and communication-related passes.
These motivate experiments in host work, batching and device execution together,
not a cache-only optimization plan.

For topology, collectives, DBO, speculative verification and stage placement,
use the relevant section of [the full-stack menu](17-optimization-techniques.md).
Feature support, fixed precision and allocated hardware limit applicability;
the menu does not limit which novel strategies may be explored.

Our application: make one change, capture a short real GPU profile with
torch.profiler, nsys or ncu, pass the relevant correctness/replay checks, then
keep or revert. Use periodic 1800s AgentX windows to measure the task-defined
score. Save exploratory checkpoints freely, but mark missing evidence unverified.
