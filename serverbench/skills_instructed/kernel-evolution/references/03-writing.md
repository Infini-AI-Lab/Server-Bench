# Writing CUDA and PTX kernels for the instructed tasks

No implementation body is provided here. Derive the kernel from the exact
profile-selected operation, freeze its capability signature, and verify it
against an independent reference.

## Hardware and instruction set

The cards are Ada L40 and report compute capability 8.9. Confirm this at
runtime, then compile the deployment path for `sm_89`. Hopper/Blackwell
features such as WGMMA and TMA are unavailable. Ada supports `mma.sync`,
`cp.async`, and FP8 tensor-core paths, but availability never proves that an
instruction is appropriate for the model dtype or operator semantics.

Write down before building: operation and accumulation semantics; input/output
dtypes; shapes and their observed frequency; strides/contiguity; alignment;
cache layout and valid extent; stream/context; supported boundaries; scratch
space; and explicit fallback behavior.

## Choose the level from the missing control

- Triton is often the quickest route for bandwidth-bound/fused tensor work and
  offers autotuning and masking.
- CUDA C++ gives explicit launch, shared-memory, instruction, and pipeline
  control and integrates through PyTorch's C++ extension machinery.
- NVRTC compiles CUDA C++ at runtime when specialisation depends on a shape
  known only in the running server.
- Hand-written PTX exposes virtual instructions, registers, and scheduling when
  the profile identifies a limitation the higher-level path cannot control.

Lower level is not automatically faster. The expected end-to-end ceiling and
verification cost decide whether the additional control is worthwhile.

## Ahead-of-time CUDA C++ extension

1. Separate the host binding from device compilation. Match PyTorch's ABI,
   device, dtype, current CUDA context, and current stream.
2. Compile an optimised timing build with line information and an `sm_89` cubin.
   Include `compute_89` PTX only as a deliberate forward-JIT choice. Do not
   profile a debug build.
3. Use the offline toolkit under `/usr/local/cuda`, the installed PyTorch C++
   extension support, and `ninja`. Keep extension/build caches under the
   writable workspace and preserve the complete compiler/linker log.
4. Inspect compiler resource reporting for registers, spills, stack, constant
   memory, and shared memory. Excess resource use can reduce resident blocks
   and erase a local instruction win.
5. Load and warm the extension before timed traffic. Fail loudly on build,
   symbol, or dispatch errors. Count every fallback by reason.
6. Verify representative and boundary shapes, including batch one, lengths
   around tile edges, long history, numeric extremes, and supported
   non-contiguous layouts. Refuse unsupported inputs explicitly.
7. Time with CUDA events on the same stream, after warmup, with synchronization
   at measurement boundaries. Retain raw samples and interleave reference and
   candidate order.

Useful compiler phases include CUDA-to-PTX, CUDA/PTX-to-cubin, object/shared
library creation, and keeping intermediates for inspection. Use `nvcc`'s
documented `--ptx`, `--cubin`, `--generate-code`, `--keep`, `--lineinfo`, and
resource-usage options rather than depending on an implicit architecture.

## NVRTC runtime compilation

NVRTC accepts CUDA C++ source and returns PTX or cubin data without spawning
`nvcc`. Compile for `compute_89`, pass required include directories explicitly,
and always capture the compile log. Cache successful output by source digest,
compiler/toolkit identity, architecture, flags, and capability signature.

The CUDA Driver API loads the output. Ensure PyTorch's primary context is
current before module loading, retrieve the exact entry symbol, and launch on
the serving stream. Retain module and argument storage for the required
lifetime, check each driver result, and surface asynchronous launch failures.
Compilation and first-JIT latency are startup work and must never occur on the
first scored request.

## Hand-written PTX and cubins

PTX is a versioned virtual ISA. It is not the L40's final SASS. A module needs a
toolchain-supported PTX version, a compatible target, 64-bit addressing,
visible entries, and an exact parameter ABI. The CUDA driver JITs PTX for the
device; `ptxas` assembles it offline into an `sm_89` cubin.

1. Derive pointer/scalar parameter order, width, and alignment from the caller.
   Record tensor strides and launch geometry. ABI mistakes can launch
   successfully while reading the wrong memory.
2. Choose the lowest PTX version and target that express the needed
   Ada-supported instructions. A newer manual does not make a newer
   architecture instruction legal on `sm_89`.
3. Run `ptxas` for `sm_89` before integration. Preserve verbose register,
   spill, stack, and shared-memory reporting. Assembly failure or surprising
   resource use is evidence, not a packaging nuisance.
4. Load PTX for driver-JIT portability or the exact cubin for fixed deployment.
   Use the current primary context and serving stream, exact symbol lookup,
   correctly packed arguments, and checked synchronous/asynchronous errors.
5. Warm the driver JIT and launch path. Cache by the full identity so a stale
   binary cannot be attributed to current source.
6. Verify against an independent implementation over the entire declared
   envelope. Include out-of-bounds guards, partial tiles, long cache history,
   NaN/Inf behavior where relevant, and negative tests for unsupported inputs.
7. Profile a warmed filtered launch with Nsight Compute. Use observed L40
   metrics for occupancy, memory transactions, register/spill pressure,
   scheduler state, and warp stalls; do not reuse B200 counter names.

## Integration and acceptance

Dispatch only when the full capability predicate holds, and expose path/fallback
counters. Run model-level parity, long generation, heterogeneous batching, and
join/leave tests after operator parity. Re-run Nsight Systems after integration:
metadata construction, launch overhead, or a neighboring kernel can consume the
isolated win.

A microbenchmark or profiler can reject this candidate but cannot accept it.
`--check-protocol-context` establishes only protocol/native-token/context integrity,
not semantic answer fidelity or performance.
Only the exact 1,800-second AgentX replay can confirm all four hard gates and
measure dollars per dollar.

## Primary sources

- CUDA Programming Guide and NVCC workflow:
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html
- NVCC compiler driver manual:
  https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
- PTX ISA:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- NVRTC guide:
  https://docs.nvidia.com/cuda/nvrtc/
- CUDA Best Practices Guide:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
