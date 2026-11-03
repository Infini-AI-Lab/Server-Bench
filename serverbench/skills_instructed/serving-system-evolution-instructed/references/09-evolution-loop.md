## Rapid-feedback serving evolution

Optimize the task's declared objective within its model, workload, hardware and
semantic constraints. Specialize execution, not answers. Keep a runnable
fallback; a saved experiment is not a verified working baseline.

### Scope and reasoning

Read the supplied vLLM/SGLang experiment notes and working-system lessons
(`references/13-vllm-v1.md`, `14-sglang-runtime.md`).
Extract applicable mechanisms and limits rather than copying an architecture.
Use seven responsibility axes: deployment, semantics, state_memory, scheduling,
host_io, device_compute, topology_transfer. State one falsifiable hypothesis and
the affected invariants, measured before/after quantities and bounds.
Reuse still-valid evidence for unaffected axes; unknown is not a pass.
`scripts/design_review.py` checks this compact record, not a proof for all inputs.

### Fast loop

1. Make one small reversible change. Cold-start the **delivered snapshot**, not
   a previously running workspace service. Account for every runtime dependency.
2. Use `torch.profiler`, `nsys` or `ncu` on real candidate GPU execution.
   Run the protocol probes (`--check-protocol-context`), your own correctness checks and lightweight
   replay gate in the routine loop. Repair or revert failures before building
   on that iteration. No separate reference test/canary is required.
3. Microbenchmark **both short requests and extreme-length requests** within the
   task envelope, including prefill and long decode tails. If the largest input
   and output cannot coexist, test their permitted extreme combinations separately.
   Cover changed shape/state
   boundaries and mixed-request interference where applicable. Record input and
   output lengths, memory, timing, correctness observations and source identity.
   Short replay alone does not cover long-context capacity or numerical fidelity.
4. Save useful states asynchronously with `./checkpoint "change; evidence; next"`
   and continue. Publication neither runs GPU tests nor certifies correctness.

Use the supplied `scripts/run_trial.py --through correctness` for source-bound
cold-start and two-gate evidence; it applies the same delivery filter as
publication. Use `profile_candidate.py` and `design_review.py` as described in
`references/18-design-review.md`. Evidence belongs below `/workspace/agentx`
and must identify the source actually tested (`candidate_tree_sha`). Never relabel stale evidence.

### Periodic measurement and mandatory full test

Run uninstrumented 1800-second AgentX performance windows at useful
milestones, not every edit. `run_trial.py --stage full --design-evidence RECEIPT`
measures performance and actual-workload replay fidelity, not complete correctness.
A full window includes replay; do not prepend its lightweight proxy.

**During every evolve cycle, run at least one complete final-verdict full test.**
Stop your own GPU servers/profilers, then run `./full-test --wait` early enough
for cold startup, the 1800-second window, correctness audit and repairs.
The harness freezes and deploys the actual delivery bundle using the real
final-verdict implementation and pinned reference runtime, on the declared
GPU type/count (the run's devices on Docker; fresh allocations on Modal).
Only public development traffic/results are exposed; the independent private
final verdict remains separate. Full-test reports and logs appear under
`/workspace/agentx/full-tests/`. A startup failure or incomplete run is not
completion. Inspect every failed gate; rerun after repairing material defects.
A backend that cannot execute the full verifier reports an infrastructure
blocker, never an endpoint-only substitute.

No profiler, extra model requests or diagnostic instrumentation belongs in a
timed serving window. The model-fidelity audit is internal to the full verifier
and runs after timing; it is not a third per-edit test.

### Delivery and budget

Read `DELIVERY_POLICY.json` and `DELIVERY_REPORT.json`: exclusions cover binary
assets, build products, data, archives, evidence, hidden paths, symlinks, size and
depth limits—not just `.venv`. Declare source-relative dependencies in
`runtime_resources.json` (`required` list; `generated` maps output paths to
delivered startup-recipe paths). Excluded resources must be reproducibly created
by `serve.sh` before readiness; declaring a recipe does not execute it.
No dependency on a development cache or temporary file may be assumed.

Use `./clock` to choose experiments that fit. A returned answer does not use the
remaining budget: continue bounded profiling, correctness coverage, dependency
audits or bottleneck experiments. If a new performance change no longer fits,
verify the existing fallback and analyze remaining uncertainty instead of waiting
or making an untested last-minute change. Record a concrete blocker when no safe
work remains. At timebox the harness selects the last working tree; use a
concrete `./promote VERSION` only for an intentional rollback.

Protocol/context checks do not establish model probabilities; lightweight
replay does not establish full workload correctness or score. Only complete
source-bound evidence supports the claim it actually measured.
