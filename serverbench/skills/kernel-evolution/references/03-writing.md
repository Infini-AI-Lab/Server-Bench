# Writing the kernel

## The hardware you are on

**Ada L40, `sm_89`.** Check rather than assume — `torch.cuda.get_device_capability()`
— and compile for what it returns.

This matters because most published kernel craft describes Hopper or Blackwell.
`wgmma`, TMA, warp-specialised producer/consumer pipelines and `mbarrier` rings
are **H100 and later**. None of them exist on `sm_89`. A design built around
them will not compile, and a metric table keyed to B200 counters names hardware
counters this device does not have.

What Ada does give you: `mma.sync` at the PTX level, `cp.async` for global to
shared, 128 KB of shared memory per SM, and fourth-generation tensor cores with
FP8 support. Occupancy and register budget are the usual limiters, as always.

## Four levels, no default

All four are installed. The right level is the one that gives you the control
the profile says you are missing — nothing here is a badge, and nothing is the
expected answer.

**Triton.** Fastest to a competitive bandwidth-bound kernel: autotuning over
block sizes, boundary masking without hand-written predication. For norms,
activations, elementwise fusions and many attention variants it is also where
you finish. What you give up is the PTX — it is generated for you, and the
control you cannot reach is exactly the control a hand-written kernel is
usually written to get.

**CUDA C++.** `torch.utils.cpp_extension` builds and imports against torch's
own tensors. `nvcc` is CUDA 12.8, the release torch is built against; `ninja` and the CUDA headers are present. Reach
for it when you want explicit shared-memory layout, an instruction Triton does
not expose, or pipeline control.

```python
from torch.utils import cpp_extension
module = cpp_extension.load_inline(name="...", cpp_sources="",
                                   cuda_sources=SOURCE, with_cuda=True)
```

**PTX from NVRTC, compiled in process.** Useful when you want to generate or
specialise a kernel at run time from shapes you only know then.

Three details cost time if you find them yourself:

```python
import ctypes, glob
# NVRTC is NOT in the CUDA tree. `CDLL("libnvrtc.so")` fails on every plain
# name; it ships with torch, so load it by path.
lib = sorted(glob.glob("/usr/local/lib/python3.12/site-packages/nvidia/"
                       "cuda_nvrtc/lib/libnvrtc.so*"))[-1]
nvrtc = ctypes.CDLL(lib)
```

* NVRTC resolves no include directory of its own, so a kernel including
  `<cuda_bf16.h>` or `<cuda_fp16.h>` needs `-I/usr/local/cuda/include` as a
  compile option. Without it the error is `catastrophic error: could not open
  source file "cuda_bf16.h" (no directories in search list)`, which reads like
  a missing package and is a missing flag.
* Compile for `--gpu-architecture=compute_{major}{minor}` from
  `get_device_capability()`, not a hardcoded arch.
* Allocate one CUDA tensor before the first driver call. Torch's primary
  context becomes current only once something is on the device, and
  `cuModuleLoadData` fails without a current context.

Then load and launch through the driver API: `cuModuleLoadData`,
`cuModuleGetFunction`, `cuLaunchKernel` via `ctypes`, on torch tensors through
`data_ptr()`.

**PTX by hand.** `cuModuleLoadData` takes PTX text directly, so a
`.visible .entry` you wrote yourself loads and runs exactly like a compiled
one. This is the level at which register budget, instruction scheduling,
`ld.global.nc` and a chosen `mma` shape are yours to set. It is also where you
end up when you are reading the PTX of a kernel that is slower than its
arithmetic says it should be and want to fix one specific thing.

## What decides whether the kernel was worth writing

Only the full scored scenario. A microbenchmark can **reject** a kernel — if it
is not faster in isolation it will not be faster in the server — but it cannot
accept one. Between the two sit every effect a microbenchmark does not have:
launch overhead at the real batch shape, cache state after a real prefill,
contention with everything else in the step, and the scheduler that decides
when your kernel runs at all.

So: profile, write, check parity, microbenchmark to reject, then measure the
serving system end to end and read the gates.

## Parity before speed

A scheduler bug announces itself. A kernel bug produces fluent, plausible,
subtly wrong output at a very good latency, and the correctness rule is
checked before any latency number is read.

Check your kernel against the reference implementation it replaces on the
shapes the workload actually serves — including the awkward ones: sequence
lengths that are not multiples of the block, a batch of one, the longest
context you support, and the empty-cache first call. Exact bitwise equality is
the right check for a change that is supposed to preserve arithmetic; it is a
diagnostic you choose, not a shipping gate the task imposes.

## Measuring a kernel change

`ncu --set full --kernel-name <kernel>` before and after.
`.claude/skills/ncu-report-skill/` explains how to read the report — **skip its
Blackwell material**, which names counters `sm_89` does not have.

Compare like with like: the same shapes, the same dtype, the same occupancy
regime. A kernel that is 3x faster on a shape the workload never serves has not
made the server faster, and the profile of the real scenario is what says which
shapes those are.
