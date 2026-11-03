## GPU clock state

The workspace supplies `./gpu-clock`, which reads the profiler clock state of
the cards this run is allocated and, under guard, resets a profiler clock lock
on one of them.

Clock state can change latency measurements. Clock behaviour has no
hardware-specific frequency threshold: low MHz alone is not proof of a fault,
an idle card reads low, and power or thermal limits under load are different
from a profiler clock lock. A profiling process group that is interrupted can
leave a lock behind. Neither clock state nor a successful reset identifies who
set a lock.

Useful times for a read-only check are before a baseline measurement, after
Nsight Compute exits or is interrupted, and after an unexplained slowdown.
Keep the result under `/workspace/agentx`. If clock state changes, remeasure
the affected comparison; restored clocks are not an algorithmic speedup.

`./gpu-clock check` is read-only: it queries the CUDA-visible UUID/PCI
identities and the explicit PerfWorks clock state, without starting a profiling
session or changing clocks. `LOCKED` / `LOCKED_TO_RATED_TDP` is a lock;
`BOOST_ENABLED` / `UNLOCKED` is not. `BOOST_DISABLED`, an unknown state,
missing libraries and denied access are reported as what they are, never as a
pass.

```bash
./gpu-clock check
nvidia-smi --query-gpu=uuid,pci.bus_id,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv
```

`nvidia-smi` may list more devices than the task owns, and host and container
ordinals differ; devices correspond by UUID/PCI.

```bash
./gpu-clock recover --uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --exclusive
```

`recover` acts on ONE full UUID that `./gpu-clock check` reported. `--exclusive`
is the caller's attestation that the card is allocated to this run; it does not
acquire a lease, and device visibility or an empty process list does not prove
exclusivity. It refuses a missing `--exclusive`, a device that is not uniquely
CUDA-visible, an active or unreadable process list, a state that is not
explicitly locked, and a container without `ncu`. Otherwise it invokes
`ncu --clock-control reset --devices 0` once, with `CUDA_VISIBLE_DEVICES`
restricted to the chosen UUID, and reports the state before and after. A
refusal or an unknown result is infrastructure evidence; this container has no
channel to an operator, so that evidence exists only where it is recorded in
the workspace.

Escalating privileges, widening device visibility, changing power limits,
forcing boost clocks and resetting the GPU are outside this task. Clock control
is not part of a candidate: `serve.sh` and the server do not change clock
state. Profiling with `ncu --clock-control none` requests no new clock lock; it
does not release an existing one.

The scoped Nsight cleanup mechanism is documented in NVIDIA's
[clock-control CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options).
