# Qwen/Qwen3.5-9B: exact serving notes

Applies to revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` used by the
two-L40 instructed task. Facts below were checked against that checkpoint's
local config and tensor shapes; the official model card supplies the family
description and supported native context.

## Architecture inventory

| Part | Pinned value | Serving consequence |
|---|---:|---|
| text layers | 32 | state and kernels follow each layer's declared type |
| layout | 8 repeats of 3 Gated DeltaNet layers + 1 full-attention layer | 24 layers use recurrent/conv state; only 8 grow full KV with sequence length |
| hidden / FFN | 4,096 / 12,288, SiLU | dense 9B model; no MoE routing in this checkpoint |
| full attention | 16 Q heads, 4 KV heads, head dim 256 | GQA; Q projection has 4× KV groups |
| Q/K normalization | learned `q_norm` and `k_norm` weights | omitting/folding them incorrectly changes logits |
| attention output | configured output gate | a generic attention-output kernel is not automatically equivalent |
| rotary | 64 rotary dims (0.25 of 256), theta 10,000,000, interleaved mRoPE sections 11/11/10 | position construction and rotary layout are part of correctness |
| linear attention | 16 key heads, 32 value heads, key/value dim 128, conv width 4 | persistent state is not a conventional KV block table |
| recurrent state dtype | float32 | storing the long-lived state in BF16 to save memory changes the model |
| embeddings / output | vocab 248,320, untied | embedding and LM head are distinct weights |
| checkpoint dtype | BF16 | task semantics freeze precision unless the brief says otherwise |
| MTP | one dedicated hidden layer, shared embeddings | an available draft mechanism, not proof that speculation wins |
| vision tower | 27 layers, hidden 1,152, patch 16 | text-only traffic should not execute vision work; loading policy still must match checkpoint APIs |

The full-attention projection tensor shapes confirm Q `8192 × 4096` and K
`1024 × 4096`; a DeltaNet layer's combined input projection is `8192 × 4096`
and depthwise convolution has width four. Do not transplant Qwen3, Qwen3-Next,
or Qwen3-0.6B kernels by family name.

## Two different cache/state regimes

For each token, conventional BF16 KV in the eight full-attention layers costs:

```text
8 layers × 4 KV heads × 256 dims × 2 (K,V) × 2 bytes = 32 KiB/token
```

One million tokens are therefore about 32 GiB of full-attention KV before
allocator metadata and fragmentation. TP2 can shard compatible attention/KV
work across two L40s, but collectives and head partitioning must be measured.

DeltaNet layers instead need recurrent matrix state plus short convolution
history. They avoid KV growth with sequence length but are not stateless. A
request snapshot, prefix entry, cancellation, preemption, chunk boundary, or
speculative rollback must preserve:

- all eight full-attention KV histories and committed lengths;
- all 24 DeltaNet recurrent states in their required accumulation dtype;
- convolution history for every linear-attention layer;
- absolute/logical positions and YaRN/mRoPE policy;
- sampling state and output accounting.

```text
# Pseudocode state transaction.
mark = state.committed_position
candidate = run_tokens(tokens, kv, recurrent_fp32, conv_history)
if accepted:
    state.commit(candidate)
else:
    state.restore(mark)  # every state family, not only KV
```

Prefix caching by token ids alone is insufficient. Cache identity includes the
model revision, exact post-template tokens, YaRN/mRoPE configuration, dtype,
and any modality/adapter identity. Publish only at a commit point.

## Required long-context policy

The checkpoint is native to 262,144 tokens; the official card describes
extension up to 1,010,000. This task deliberately requires whole requests up
to 1,048,576 and pins a factor-4 YaRN override in its own reference file:

```text
# Configuration sketch copied from task facts, not implementation source.
max_positions = 1_048_576
rope = yarn(factor=4, original=262_144, theta=10_000_000,
            partial_rotary=0.25, interleaved_sections=[11, 11, 10])
```

The task's override is authoritative for the benchmark even though it is beyond
the model card's rounded extension statement. Accepting a long request without
applying the exact positions/rotary semantics is a correctness failure. Chunked
prefill must carry both recurrent and convolution boundary state exactly.

## Parallelism and scheduling hypotheses

- TP2 is the natural capacity path for the long full-attention cache, but it
  introduces per-layer communication. Partition 16 Q, 4 KV, 16 linear-key, and
  32 linear-value heads only in ways supported by the exact operators.
- DP2 duplicates roughly 19.3 GB of checkpoint shards on each card and divides
  the prefix cache, which may be unattractive for million-token sessions.
- Chunk long prefills to protect decode TPOT, but sweep chunk sizes: DeltaNet
  scan efficiency, GEMMs, metadata, and launch count respond differently.
- Prefix reuse can save enormous prefill work, but cached recurrent snapshots
  are large. Measure bytes retained, lookup overhead, hit distance, and actual
  prefill tokens skipped—not hit count alone.
- The MTP head makes draft/verify plausible. Qualify token parity and rollback,
  then measure acceptance length versus verifier expansion at offered
  concurrency four. A draft head that increases verifier batch can worsen TPOT.

## Fast-path qualification

Exercise batch one and four, each batch position, prompt chunks around every
tile/block boundary, native and extended positions, prefix hit/miss/eviction,
request cancellation, output limits, EOS/ignore-EOS behavior, and long decode.
Stateful detokenization must stream exactly the requested text; do not count or
drop special tokens differently merely to satisfy an output-length check.

Profile before selecting the target: possible costs include DeltaNet scan,
full-attention prefill, MLPs, TP collectives, host metadata, cache movement, and
re-prefill. The exact 1,800-second replay is the only hard-gate and
dollars-per-dollar evidence.

Primary sources: pinned `config.json` and safetensor shapes, plus
https://huggingface.co/Qwen/Qwen3.5-9B and
https://huggingface.co/docs/transformers/model_doc/qwen3_5.
