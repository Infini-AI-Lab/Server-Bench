## Rapid-feedback serving evolution

Optimize the task's declared objective within its model, workload, hardware and
semantic constraints. Specialize execution, not answers. Work in short cycles,
and end every cycle with a system that runs, is correct and has been measured;
a saved experiment is not a verified working baseline.

### How serving systems built here have failed

Every rule below exists because systems built for tasks like this one failed
in one of these ways. None of them was caught by the checks their builders
ran:

1. **Correct alone, wrong under load.** Replies were right for one short
   request at a time and wrong at the workload's real shape: a prompt rendered
   without the chat template on the HTTP path, KV or recurrent state shared or
   reused incorrectly when concurrent requests forked from one prefix, state
   corrupted after cache reuse at long context, a crash on one page or chunk
   boundary. The verdict judges replies served under the replay's own load, so
   these fail both the end-of-sequence gate and output correctness.
2. **A check that could not fail.** A system compared with itself (a forked
   run against a solo run on the same cache), token counts checked without the
   text being read, near-ties assumed instead of shown.
3. **Shipped something other than what was measured.** An engine measured
   through a private launcher while `serve.sh` still started the starter;
   edits made after the last full test.
4. **Numbers explained away.** A window that completed almost nothing, a
   dead engine read as a slowdown, the best of several noisy windows quoted
   as the result, a tool's `passed` read as success.
5. **The offered load guessed or copied.** Concurrency chosen from a
   microbenchmark, copied from the SGLang reference, or never re-measured
   after the system changed -- the largest single lever on the objective,
   left unmeasured.
6. **Latency lost in the wrong place.** Mean step time optimized while
   requests missed their bars because a long prefill ran between decode steps,
   prefill was bound by kernel-launch overhead, or speculative decoding was
   absent or barely accepted.
7. **No failure handling.** An engine that died left the HTTP server accepting
   requests and hanging, so every later request failed.
8. **Time lost.** Hours of kernel work before the first end-to-end
   measurement, hours waiting on a hung job or a subagent, a run stopped early
   with a known lever untested.

### The team

You are the lead of a three-agent team. Two subagents are defined in your
workspace; run the evolve with them from the first hour to the last:

- **`kernel-engineer`** (`.claude/agents/kernel-engineer.md`; skills
  `KernelWiki`, `ncu-report-skill`, `kernel-evolution`) designs, verifies and
  tunes the dedicated kernels: attention (and recurrent/linear-attention
  layers, if the model has them), GEMMs, norms, sampling and
  speculative-decoding kernels, fusion and CUDA graphs.
- **`runtime-engineer`** (`.claude/agents/runtime-engineer.md`; skill
  `serving-system-evolution-instructed`) owns everything above the kernels:
  the request runtime, batching, scheduling and admission, KV/prefix-cache
  (and any recurrent-state) management, speculative-decoding control, failure handling
  and parallelism across the task's cards. It tells the kernel engineer which
  kernels it needs.

Call them by name with the Agent tool -- `subagent_type: "kernel-engineer"`
and `subagent_type: "runtime-engineer"`, not a general-purpose agent with a
role pasted into its prompt -- and launch them in the background so they work
while you do. When one returns, read its report and send its next assignment
in the same turn (a fresh launch, or SendMessage to continue it). Keep both
busy until the last hour of the evolve.

### The evolve loop

Every cycle has the same six steps, and only step 5 can end one. The first
cycle's goal is a correct, measurable end-to-end system started by `serve.sh`,
within the first hour; every later cycle improves it by a measured step.

1. **Plan (lead).** Choose the cycle's goal from the last measurement: which
   requests miss their bars and why, the target, and the change expected to
   move it. Write it at the top of `design/MACRO.md` with each agent's
   assignment and the integration point.
2. **Request (runtime engineer).** For every kernel the plan needs, an entry in
   `design/KERNEL_REQUESTS.md`: the op's semantics with a PyTorch reference,
   shapes, dtypes and ranges up to the task's longest context, layout and
   alignment, call frequency and share of step time, target time, expected
   end-to-end gain, tolerance, acceptance test and priority.
3. **Build (both subagents, in parallel).** The kernel engineer implements
   requests against `design/MACRO.md` and logs them in `design/KERNEL_LOG.md`;
   the runtime engineer changes admission, batching, scheduling, caching or
   parallelism and integrates landed kernels behind a switch, logging in
   `design/RUNTIME_LOG.md`. Each works in its own worktree and hands back a
   commit whose tests pass, including the correctness tests below for every
   code path it added.
4. **Integrate (lead).** Merge the cycle's commits into the workspace -- the
   tree that ships -- and make `serve.sh` start exactly the system you intend
   to ship, with every switch at its intended default.
5. **Verify (lead) -- the gate.** Show that the integrated tree is a
   runnable, correct, measured system. The methods are yours; the evidence
   must establish all three, for this commit:
   - **Runnable.** The delivered snapshot, cold-started through `serve.sh` in a
     fresh process (no warm state, no private launcher, no development cache),
     becomes ready, serves the workload at the concurrency in
     `concurrency.json` for a sustained window with every request completing,
     keeps serving after it, and stops cleanly. `./full-test` does this cold
     start; its server log and per-request export show whether it held.
   - **Correct.** The suite of
     your own output-correctness checks
     (see *Verifying correctness*) passes against this commit, through its
     HTTP endpoint, together with the
     protocol/context probes (`--check-protocol-context`) and the lightweight
     replay gate.
   - **Measured.** The objective, the request counts and the latency
     distribution are in line with what the change was meant to do, and
     every surprise is explained (see *Measuring performance*).

   If the tree passes, `./checkpoint "verified: what changed; the evidence;
   the numbers"`. If it fails, repair it within the cycle or revert the
   integration to the last verified checkpoint; never build the next cycle on
   an unverified tree.
6. **Decide (lead).** Update `design/MACRO.md` with what the measurement
   showed and plan the next cycle.

### Verifying correctness

The verdict judges replies your server returns under the replay's load. Your
own checks are only worth what they can catch, so build them to fail where
real serving bugs live:

- **An independent reference.** Compare against the checkpoint computed by a
  separate, simple implementation -- the model's own Transformers
  implementation runs in this container without any fast kernel -- never
  against your system's own earlier output. A fork compared with a solo run of
  the same engine shares the engine's bugs.
- **Through the endpoint that ships.** Send the checks to the running server
  over HTTP, as the workload does: chat messages rendered by its chat template
  with `chat_template_kwargs`, streamed, with `ignore_eos` and `max_tokens` as
  the workload sets them, and with them unset. Rendering prompts yourself and
  feeding token IDs to the engine skips the code that has failed before.
- **At the workload's shapes.** The longest contexts the task allows;
  multi-turn sessions that resend a growing conversation, including turns
  whose earlier history was rewritten; many requests at once, joining and
  leaving the batch while others decode and while long prefills run; forks of
  one shared prefix; cache pressure and eviction; decodes of hundreds of
  tokens, not one; continuation past an end-of-sequence token under
  `ignore_eos`, and a natural stop without it.
- **A divergence is a bug until shown otherwise.** When the server's token
  differs from the reference's, it is acceptable only if you have shown that
  the reference itself is nearly indifferent between the two tokens at that
  position; record how. Any other divergence is a bug, however fluent the text.
- **Read the outputs.** Look at the generated text of long and concurrent
  replies. Repeated tokens, fragments and sudden topic changes are what
  corrupted state looks like.
- **Task-level accuracy too.** Ask your server and the reference the same set
  of questions with checkable answers and compare the accuracy; token parity
  on a few prompts does not establish it.
- **Every code path.** Each new branch -- a cache-reuse path, a fork, an
  eviction, a speculative path, a CUDA-graph bucket, a recovery path -- runs in
  the suite at least once before it ships. Keep the suite in the tree and run
  it at every integration.

### Measuring performance

- **Measure what ships.** Development numbers count only when they come from
  the tree `serve.sh` starts, cold, as `./full-test` runs it, at the concurrency
  in `concurrency.json`.
- **Windows long enough for the decision.** A 300-second window is a quick
  check; a decision that matters rests on longer windows, and the tree you
  ship has run at least one full 1800-second full test. Every window draws a
  new seed: repeat a measurement before trusting a difference, judge the
  difference against the spread between repeats, and report the typical
  result, never the best one.
- **Explain every surprise.** A window that completes far fewer requests than
  the last, a zero, a sudden latency change, an error, a `passed` with almost
  nothing served: open the server log and the per-request export and find the
  cause before moving on. An unexplained number is a failure, not noise.
- **Diagnose per request.** For the requests that missed a bar, find what
  they waited for: queueing for admission, their own prefill, other requests'
  prefill running between their decode steps, decode batch size, context
  length. Optimize what the misses spent their time on.
- **Choose the offered load by measurement.** Concurrency trades revenue
  against every request's latency. Measure the current system at several
  loads with comparable windows, choose the best, and measure again after any
  change that alters behaviour under load. The SGLang reference point is
  SGLang's optimum for its own latency, not yours.
- **The reference's levers.** The SGLang reference's launch flags, listed in
  *The SGLang reference settings* and in `reference.yaml`, name the mechanisms
  it relies on on this hardware -- its cache strategy, chunking and any
  speculative decoding. Each is a lever measured to work here.

### Robustness

- A fault in the engine fails the affected requests with an error and keeps
  serving the rest -- or stops the process -- but never leaves requests
  hanging behind a live HTTP front end.
- Memory exhaustion, an unexpected shape or a long prompt at the context limit
  is handled, not a crash: test the limits in the suite.
- The server keeps serving correctly after a full replay; check it with a
  correctness pass right after a long window.

### Time

- A correct, measurable end-to-end system exists within the first hour, even
  a slow one; kernels and scheduling improve it from there.
- Waiting is CPU time: while a GPU job runs, review code, extend the suite,
  analyse the last export. Bound every wait; a job that has not finished in
  its expected time is investigated, not waited on.
- In the last hour, add nothing new. Verify the final tree -- an 1800-second
  full test and the correctness suite on that commit -- and fix only what that
  verification finds. Use the whole budget: when time remains, run more
  verification rather than stopping.

### Working together

- **Separate working copies.** Make the workspace a git repository at the start
  (`git init`; `.git` is never delivered), commit the integrated tree at every
  integration, and give each subagent its own worktree outside the workspace,
  e.g. `git worktree add /tmp/team/kernel -b kernel` and
  `git worktree add /tmp/team/runtime -b runtime`. A subagent edits only its
  own worktree; you merge its commits. Nobody edits files another agent is
  testing.
- **The design board is shared.** `design/` stays in the workspace and is
  edited in place, never in a worktree, so all three read the same files:
  you own `MACRO.md`, the runtime engineer `KERNEL_REQUESTS.md` and
  `RUNTIME_LOG.md`, the kernel engineer `KERNEL_LOG.md` (and its own
  proposals, in their own section of `KERNEL_REQUESTS.md`). Commit `design/`
  with each integration.
- **Measure snapshots, not moving trees.** A GPU measurement runs from a
  committed snapshot -- a worktree at a known commit -- and its log names the
  commit. A measurement whose tree changed while it ran is void; rerun it
  rather than interpret it.
- **Hand-offs carry identity.** Every report and SendMessage names the commit
  measured, the files changed, the result with its numbers, and what the
  sender needs from whom. When you merge, tell both subagents which commit is
  now the base, and they rebase onto it before their next measurement.
- **Supervision.** A subagent that has not reported within its time box is
  checked, not waited on. Ownership: `design/MACRO.md` assigns each agent its
  directories; changes to another agent's files go through you. Only you run
  `./checkpoint`, `./full-test` and `./promote`.

### Sharing the GPU

**Only one agent uses the GPU at a time**, and the lock lives exactly as long as
the GPU work. Every command that touches the GPU -- a server, a benchmark, a
profiler, a test that loads weights, `./full-test` -- runs as one bounded job
under the team lock: `flock /tmp/gpu.lock <job>`, where the job starts what it
needs, measures, stops it and only then exits. Never start a background server
from inside a `flock` command that returns, and never hold the lock with an
idle holder process. Before a job, check `nvidia-smi` for anything left
running; stop your own processes by PID or process group, never with a broad
`pkill -f`. CPU work never waits for the GPU.

Balancing GPU time is your job. Your verification runs come first; give the
two subagents the rest in short jobs (a kernel microbenchmark or `ncu` pass is
minutes, a runtime experiment is tens of minutes at most), split fairly over a
cycle, and write the current plan in `design/MACRO.md`.
`flock -n /tmp/gpu.lock true` tells anyone whether the card is free.

### Engineering practice

Read the supplied vLLM/SGLang experiment notes and working-system lessons
(`references/13-vllm-v1.md`, `14-sglang-runtime.md`).
Extract applicable mechanisms and limits rather than copying an architecture.
Use seven responsibility axes: deployment, semantics, state_memory, scheduling,
host_io, device_compute, topology_transfer. State one falsifiable hypothesis and
the affected invariants, measured before/after quantities and bounds.
Reuse still-valid evidence for unaffected axes; unknown is not a pass.
`scripts/design_review.py` checks this compact record, not a proof for all inputs.

Pick each kernel's language by measurement: Triton for quick iteration, and
CUDA C++ or PTX where Triton cannot reach the target -- warp-level control,
explicit tensor-core `mma` on the card's architecture, persistent decode
kernels, or launch-overhead-bound small batches. Both toolchains work here
(`nvcc` and torch share CUDA 12.8). A compiled extension is built by
`serve.sh` at every cold start, inside the verdict's 900-second readiness
limit; budget its compile time.

Profile real candidate GPU execution with `torch.profiler`, `nsys` or `ncu`.
Microbenchmark **both short requests and extreme-length requests** within the
task envelope, including prefill and long decode tails. If the largest input
and output cannot coexist, test their permitted extreme combinations separately.
Record input and output lengths, memory, timing, correctness observations and
source identity.

`scripts/run_trial.py --through correctness` gives source-bound cold-start,
protocol and replay evidence with the same delivery filter as publication.
Use `profile_candidate.py` and `design_review.py` as described in
`references/18-design-review.md`. Evidence belongs below `/workspace/agentx`
and must identify the source actually tested (`candidate_tree_sha`). Never relabel stale evidence.

### The full test

**Run `./full-test` at every verification, not only at the end.** Stop your
own GPU servers and profilers, then run
`flock /tmp/gpu.lock ./full-test --duration N --wait` early enough for cold
startup, the N-second window and repairs. It freezes the delivery bundle
exactly as publication would, cold-starts that bundle's `serve.sh` inside this
container on its own port, runs the protocol/context probes and an N-second
replay at the bundle's declared concurrency with the evaluator in
`tests/final_verdict/`, probes the same server again after the window,
reports `dollars_per_dollar`, and stops it. It fails a window in which fewer
than one request per minute completed or whose server log records an
unhandled exception or CUDA out-of-memory error. It does not check output
correctness, which stays private to the final verdict.
Reports and logs appear under `/workspace/agentx/full-tests/`. A startup
failure, an incomplete run, or a window that completes almost no requests is
not a pass. Inspect every failure; rerun after repairing material defects.
`run_trial.py --stage full --design-evidence RECEIPT` measures performance and
actual-workload replay fidelity, not output correctness. No profiler, extra
model requests or diagnostic instrumentation belongs in a timed serving window.

### Delivery and budget

Read `DELIVERY_POLICY.json` and `DELIVERY_REPORT.json`: exclusions cover binary
assets, build products, data, archives, evidence, hidden paths, symlinks, size and
depth limits—not just `.venv`. Declare source-relative dependencies in
`runtime_resources.json` (`required` list; `generated` maps output paths to
delivered startup-recipe paths). Excluded resources must be reproducibly created
by `serve.sh` before readiness; declaring a recipe does not execute it.
No dependency on a development cache or temporary file may be assumed.

Use `./clock` to choose experiments that fit. At timebox the harness selects
the last working tree; use a concrete `./promote VERSION` only for an
intentional rollback. Record a concrete blocker when no safe work remains.

Protocol/context checks and lightweight replay establish neither output
correctness nor a score. Only complete
source-bound evidence supports the claim it actually measured.
