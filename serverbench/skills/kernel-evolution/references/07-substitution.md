# Putting the kernel into the model

An operator that is correct in isolation and wrong in the model is the common
case, not the exotic one. This is the step where that is caught.

## Check the semantics before the shapes

Shapes agreeing is not evidence of anything. These are the switches that make a
substitution safe, and every one of them can mismatch while every shape agrees:

| switch | what a mismatch does |
|---|---|
| **QK normalisation** | silently drops or adds a normalisation the model was trained with. Output stays fluent |
| **GQA group size** | wrong K/V head serves a query head. Well-formed, wrong values |
| **mask convention** | causal vs sliding-window vs none; a leak of future tokens looks like a better model |
| **position convention** | NeoX vs GPT-J pairing; also "position zero for every token", which is right for the first token of every sequence |
| **KV cache layout** | contiguous vs paged; reading the right addresses for the wrong sequence |
| **head dimension** | derived rather than read from the config. Qwen3-4B has hidden 2560 and head dim 128 — the derivation gives 80 |

These are the comparisons to write yourself, one per axis. A mismatch is a
refusal, not a warning.

## Integration checks that are worth their time

1. **Exact-output equivalence on a fixed prompt set, greedy.** Token for token,
   against the model before the substitution. This is the check that catches
   everything above.
2. **A long generation**, not a short one. Position and cache-offset bugs need
   history to appear.
3. **A batch with heterogeneous lengths.** Padding and masking bugs need ragged
   input.
4. **Requests joining and leaving mid-batch**, if the server does that. A kernel
   holding per-request state across a step is a state leak.
5. **The unsupported input.** Confirm the fallback is taken *and* that it says so.

## When integration passes and end-to-end does not move

That is a result, not a failure of the experiment. Common causes, in the order
worth checking:

- the operator was overlapped, so its cost was never on the critical path;
- the win is at a shape the workload does not produce;
- new per-step host work — metadata, index tensors — ate the saving;
- occupancy changed and a neighbouring kernel got slower.

Each of those is worth recording in the lineage's memory. The next attempt should
not rediscover it.
