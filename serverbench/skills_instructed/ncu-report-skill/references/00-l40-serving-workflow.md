# L40 serving profiler workflow

This is the architecture-specific front door for the instructed tasks. The
allocated cards report compute capability 8.9. Do not apply B200 / `sm_100`
metric tables or Hopper/Blackwell instruction advice to them.

## Nsight Systems answers “where?”

Collect a bounded serving interval with CUDA, NVTX, and OS runtime tracing and
CPU sampling disabled. Use NVTX phase ranges, then inspect GPU idle gaps,
launch-to-execution delay, CUDA synchronizations, memory copies, allocation
bursts, stream serialization, and the cumulative time of kernel families.
Correlate the timeline with request state: offered, admitted, first output,
decode steps, finish, and release. Otherwise queueing or a hung warmup can be
misread as model compute.

The useful summaries are `cuda_api_sum`, `cuda_gpu_kern_sum`,
`cuda_kern_exec_sum`, and `nvtx_sum`. A timeline capture is deliberately short
and instrumented. It diagnoses the system but cannot establish a hard gate.

## Nsight Compute answers “why this kernel?”

Filter by kernel name and warmed invocation. Begin with the basic set; collect
the full set only after verifying the matched launch is the intended one. NCU's
kernel replay can change cache state and latency, so isolate a deterministic
launch whenever possible.

Read in this order:

1. Launch dimensions, duration, registers/thread, static/dynamic shared memory,
   waves, and achieved occupancy.
2. Compute and memory speed-of-light sections to classify the broad ceiling.
3. Memory workload to distinguish useful bytes from transactions, cache
   behavior, and spills.
4. Scheduler and warp-state sections to see eligible warps and dominant stalls.
5. Source/SASS correlation, enabled by line information, to locate the
   instruction responsible for the observed signal.

Low occupancy is a clue, not automatically a cause. High occupancy is not proof
of efficiency. A memory-bound label requires memory evidence; a stall label
requires the corresponding sampled/counter evidence. Record actual metric names
and values from the report because availability differs by architecture and
tool version.

## From diagnosis to candidate

Estimate the kernel's share of the real critical path and the maximum removable
fraction. State the predicted end-to-end ceiling before implementing. After a
change, keep input signature and launch conditions identical, verify independent
numerical parity and boundary shapes, compare reports, and check wall-clock
kernel timing with CUDA events. Re-run the system timeline after integration.

The public correctness check can reject a semantically wrong server. Only the
exact 1,800-second AgentX replay can accept the integrated system with respect
to all four hard service-level gates and dollars per dollar.
