# Qwen3 — the semantics that will trip a borrowed kernel

*Applies to: every operator, when the model is any Qwen3 checkpoint.*

## Shape

| | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| layers | 28 | 28 | 36 | 36 |
| hidden | 1024 | 2048 | 2560 | 4096 |
| q heads | 16 | 16 | 32 | 32 |
| kv heads | 8 | 8 | 8 | 8 |
| head dim | 128 | 128 | 128 | 128 |
| intermediate | 3072 | 6144 | 9728 | 12288 |
| vocabulary | 151,936 | 151,936 | 151,936 | 151,936 |

Read the exact values from the checkpoint's `config.json` rather than this table.
The table is for orientation; the checkpoint is the truth, and a kernel tuned to
the wrong head dimension is a compile-time constant away from wrong.

Note `head_dim = 128` with `hidden = 2560` for the 4B: the head dimension is
**not** `hidden / num_heads`. A kernel that derives it is wrong on this model.

## The four things that bite

**1. Q and K are RMS-normalised inside the attention block.** Before RoPE, per
head. Qwen2 does not do this. A fused attention kernel written for the older shape
runs, is faster, and omits an operation the model needs — output stays fluent and
is simply wrong. This is the single most important line in this file.

**2. The vocabulary is 151,936, which is not a power of two.** 593.5 tiles of 256.
Any reduction over the vocabulary that assumes a whole number of tiles drops the
last 128 logits and does it silently, losing rare tokens.

**3. GQA with 8 KV heads.** Group size is 2 for the small models and 4 for the
larger ones. K/V traffic divides by the group size if — and only if — the query
rows in a group share a K/V tile load. Getting the group indexing wrong keeps the
shapes valid and changes the values.

**4. RoPE is NeoX-convention**, pairing `(i, i + d/2)`. Applying the GPT-J pairing
produces well-formed, wrong output.

## What decode looks like

At batch 1 for the 4B model, one decode step reads roughly 8 GB of weights in
BF16. On an H200 at ~4.8 TB/s that is about 1.7 ms of pure weight streaming — a
floor no kernel work goes below. If your measured step is 3 ms, half of it is
already accounted for, and the remaining half is where the optimisation lives.

Per step the model issues on the order of a few hundred kernel launches. At a few
microseconds of launch overhead each, launch cost is a visible fraction of a
decode step, which is why fusion and graph capture are worth measuring before
tiling is.

## Cross-checking a substitution

Before substituting any kernel for a Qwen3 operator, confirm all of:

- `qk_norm` matches (this one first, every time);
- GQA group size matches;
- RoPE convention matches;
- mask convention matches;
- KV cache layout matches;
- head dimension is what the checkpoint says, not what you derived.

`KernelCapability.disagreements_with` reports these in words. A mismatch is a
refusal, not a warning: the failure mode is fluent, plausible, wrong output that
no timing measurement can see.
