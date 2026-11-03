# Negative controls — the check on the checks

A check battery quietly stops checking, and that is the failure this rung exists
for.

It happens like this. A tolerance is widened slightly to admit a candidate that
was nearly right. A boundary case is dropped during a refactor. A fixture stops
being regenerated and starts describing a program that no longer exists. From
that moment every candidate passes, including the broken ones, and the battery
looks exactly like a healthy one — green, fast, and blind.

So the battery has to be shown to **kill** things.

## The classes, and which rung must catch each

Every one is a defect that has shipped in a real kernel
(`kernel_evolve/contracts/verification.py::MutantClass`). Seven are required of
every plan (`REQUIRED_MUTANTS`); `dropped_gqa_group`, `truncated_vocabulary` and
`wrong_position` are the three that are not.

| class | what it does | must be caught by |
|---|---|---|
| `wrong_mask` | attends past the current position | parity |
| `missing_normalization` | skips the RMS norm on Q and K | parity |
| `wrong_reduction` | sums where the reference takes a max | parity |
| `dropped_gqa_group` | broadcasts one KV head across the group | parity |
| `stale_kv_offset` | reads the cache at the previous step's offset | boundaries |
| `truncated_vocabulary` | reduces over the first 2ᵏ logits | boundaries |
| `wrong_position` | applies RoPE at position zero for every token | boundaries |
| `partial_write` | leaves the last output elements unwritten | numerics |
| `silent_fallback` | dispatches to the library without saying so | dispatch |
| `inaccurate_timing` | measures without synchronising | microbenchmark |

The **pairing** is the interesting part. A dropped normalisation is numerical and
belongs to parity. A stale KV offset only appears once the cache has history, so
it belongs to the boundary sweep. A silent fallback is not a numerical defect at
all — the answer is right — and only the dispatch rung can see it.

## Four ways a matrix fails, and they mean different things

**Survived.** The checks cannot see that class. The hole is exactly the shape of
a defect that has already shipped.

**Caught by the wrong rung.** Usually means the rung that should have caught it is
not checking what its name says.

**A legitimate variant rejected.** The battery is over-sensitive and will now
reject real work for the rest of the campaign. This fails the matrix just as hard
as a survivor, because a battery that kills everything is not sensitive — it is
broken.

**Never tested.** A matrix that checks only "nothing survived" lets a battery
covering three of seven classes report itself qualified, so an untested required
class counts against it too.

## Scope, so that narrowing is honest

An RMS norm has no mask and no KV cache. Requiring it to kill `wrong_mask` would
be theatre, and worse: it would train people to add empty controls to make a
check go green.

So `kernel_evolve.verification.applicable_to(operator)` names the classes that can actually occur,
and a `VerificationPlan` may narrow its required set — **with a stated reason**.
An unknown operator gets the full set, because testing less is never the safe
default.
