# Contract reference

Two different things are called a "contract" in this repository.

* **A task's contract** is data: `serving_tasks/<task>/contract.yaml` (bars,
  prices, devices, window, request scoring) and `reference.yaml` (the SGLang
  operating point and reference arms). `final_validate/spine.py` reads them for
  the verdict and the driver; `taskctl` is the only writer. See
  [`../../../serving_tasks/README.md`](../../../serving_tasks/README.md).
* **A typed contract** is a frozen, versioned dataclass with a canonical digest,
  in `harness/serving_evolve/contracts/`. These back the gate ladder
  (`serving_evolve/evaluator/`), the kernel-evolution subsystem
  (`harness/kernel_evolve/`) and the `serverbench evolve` CLI. The evolve
  driver (`harness/lifecycle/`) records its run as JSON and does not use them.

The rest of this page is about the second kind.

## Shared behaviour (`contracts/base.py::Contract`)

| | |
|---|---|
| `SCHEMA_VERSION` | class attribute; reading a document written by a different version raises |
| `KIND` | stable discriminator written into every serialized form |
| `to_dict()` / `from_dict()` | round-trip; `from_dict` rebuilds nested contracts and **refuses unknown fields** |
| `evolve(**changes)` | returns a copy; the original is untouched |
| `digest()` | SHA-256 over normalized semantic content |
| `validate()` | invariants types cannot express; runs on construction |

**Digests bind meaning, not the clock.** Keys that record *when* or *where*
(`generated_at`, `timestamp`, `hostname`, `artifact_root`, …) are stripped before
hashing, so the same plan hashes identically on any host on any day.

**Unknown fields are refused, not ignored.** Silently dropping a field means a
document written by something you do not understand is read as though you did.

## What exists

| module | contracts | what they pin |
|---|---|---|
| `workload.py` | `ModelIdentity`, `ShapeEnvelope`, `SemanticInvariant`, `WorkloadSpec` | model and **pinned revision**; prompt/output/batch/concurrency ranges; invariants whose `hard` form fails a candidate outright; the whole workload including SLO and arrival process |
| `gates.py` | `GateSpec`, `GateResult`, `AcceptancePolicy`, `AcceptanceDecision` | what is checked, PASS / FAIL / INCONCLUSIVE / INVALID with denominators, and the rule fixed before results |
| `run.py` | `ArtifactRef`, `CommandReceipt` | a pointer into the content-addressed store, and proof that a command ran, with what it saw and produced |
| `environment.py` | `ExecutionEnvironment` | the declared interpreter a gate may run under; a bare `python3` is refused |

A revision is required: "latest" is a moving target. `AcceptancePolicy`
refuses to be constructed with `inconclusive_is_acceptance`, and a `PASS` with
recorded failures raises.

## Adding a contract

1. Subclass `Contract`, set `KIND` and `SCHEMA_VERSION`.
2. Frozen dataclass, defaults for every field.
3. Put the invariants types cannot express in `validate()`.
4. Export it from `contracts/__init__.py` (and `serving_evolve/__init__.py` if it is public).
5. Give it a `KIND` no other contract uses, and add a round-trip test and at
   least one refused-state test.
