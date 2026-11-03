# `server_output/` — what agents built, and the verdict on it

One directory per **lineage**: one `(task, model, agent)` ascent, opened by an
evolve run and extended by its resumes. Everything here is written by the evolve
driver (`serverbench/harness/lifecycle/`) and the final verdict
(`serverbench/harness/final_validate/`); nothing is written by hand, and recorded
evidence is never edited or renamed afterwards.

The summary of what each lineage achieved is recorded in
[`../evaluation_progress/`](../evaluation_progress/README.md); this directory
holds the source and the evidence behind it.

## Layout

```
server_output/
  <UTC start>-<task id, lower case>-<agent model>-<effort>/        a lineage
    <lineage>_v1/  <lineage>_v2/  ...                              published versions
    <lineage>_v<N>.status.json                                     each version's status record
    versions.jsonl                                                 the version index, append-only
    ACCEPTED_HEAD.json, head_history.jsonl                         the one head, and every move of it
    MEMORY/                                                        the agent's notes, when the lineage keeps any
    FINAL.json                                                     the verdict's summary for the final head
    FINAL_VALIDATION/<version>/                                    the verdict's evidence
  .objects/                                                        the captured git trees
```

The timestamp leads the name so lineages sort by start time.

### A version directory

A frozen, read-only snapshot of the agent's workspace at the moment it ran
`./version "why"`: its `serve.sh`, `server.py`, `concurrency.json` (the offered
load the verdict will use) and whatever else it built, plus files the harness
writes beside the source:

| file | written by | what it says |
|---|---|---|
| `VERSION.json` | `lifecycle/boundary.py` | identity: run, lineage, commit and tree digest, predecessor, status, and the approximate API cost of getting there; `judged: false`, because publishing is not judging |
| `DATASHEET.json`, `DATASHEET.md`, `MEASUREMENTS.jsonl` | `boundary.py` | the development measurements the agent took (not a verdict) |
| `REVIEW.md` | `lifecycle/boundary_checks.py` | deterministic checks over the frozen tree (entry point, portable paths, the mounted checkpoint, no held-out surface); recorded, not gating |
| `DECISIONS.md`, `TRAJECTORY.md` | `lifecycle/run.py` at run end | the agent's recorded decisions, and a readable digest of its trajectory |

Publishing is not judging: every version is kept, and the head only moves
forward along its own ancestry (`lifecycle/publish.py`).

### The verdict

After the deadline the driver freezes the final working tree and
`final_validate/tools/validate_evolve.py` measures the head once: the task's
1,800 s AgentX replay at the declared load on a fresh odd seed, then the
private correctness stage. `FINAL.json` carries the outcome (`HOLDS`, `MISSES`
or `NOT_RUN`), the scored `dollars_per_dollar` and a pointer to
`FINAL_VALIDATION/<version>/`, which holds `MEASURED.json`,
`scenario-candidate.json` (replay fidelity, correctness, per-request scores,
the objective), the AgentX exports and the server logs. The authoritative copy
of every verdict is under `serverbench/artifacts/final_validation/` on the host
that measured it.

## What is in git

Only this README. Everything else here is written by runs on the operator's
machine and is gitignored, as is each task's results file
`evaluation_progress/<task>.csv`.

## Rules for what an agent builds

1. **A prototype never imports the harness.** `serverbench` launches and probes
   it; it must not know `serverbench` exists.
2. **No absolute paths, campaign evidence or run identifiers** in prototype
   source.
3. **Superseded versions stay.** A rejected or outdated version is evidence of
   what was tried; it is never deleted or renamed to look better.
4. **An agent sees its own lineage's `MEMORY/` and nothing else** from this
   directory: never another model's or agent's results.
