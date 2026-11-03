# google/gemma-4-E4B-it: exact serving notes

Applies to revision `ee0ef6023621cff504d758262d4e04895a5af4a2` on one L40.
Facts below were checked against the pinned local config and tensor shapes, with
the official Google Gemma 4 overview used for family terminology.

## Architecture inventory

| Part | Pinned value | Serving consequence |
|---|---:|---|
| model | dense E4B, 4.5B effective / about 8B including embeddings | PLE tables dominate parameters but are lookup-heavy rather than dense compute |
| text layers | 42 | per-layer inputs and attention geometry must follow the layer index |
| layout | 5 sliding layers then 1 global, repeated 7 times | 36 local-window layers and 6 full-context layers; final layer is global |
| hidden / FFN | 2,560 / 10,240, GeGLU-tanh | dense decoder, not the larger Gemma 4 MoE family |
| PLE | 256 dims per layer, vocab 262,144 | one weight tensor is `[262144, 42×256]`; each layer projects/gates its slice into hidden state |
| sliding attention | window 512, 8 Q / 2 KV heads, head dim 256 | bounded local cache and 2,048-dim Q projection |
| global attention | 8 Q / 2 KV heads, head dim 512 | distinct 4,096-dim Q and 1,024-dim K/V geometry |
| global KV | unified/shared as described by Google; config declares 18 KV-shared layers | do not assume ordinary independent KV at every layer; verify the runtime's exact sharing map |
| Q/K normalization | learned q/k norm weights | required on both attention forms |
| RoPE | local theta 10,000; global proportional RoPE theta 1,000,000 with 0.25 partial rotary | backend dispatch must preserve per-layer position policy |
| maximum context | 131,072 | admit only complete traces that fit; never truncate an admitted request |
| logits | final soft cap 30 | omitting soft-capping changes sampling and greedy tokens |
| embeddings | tied token/output embedding plus PLE | tied LM head and per-layer lookup have different reuse patterns |
| modalities | text, image, audio; separate vision/audio towers | text-only benchmark should not run encoder work, but loader/API compatibility remains exact |

The PLE tensor shape is `[262144, 10752]`, or 5.25 GiB in BF16 by itself.
Each decoder layer has a `2560 × 256` PLE projection and a `256 × 2560` gate.
Treating E4B as a conventional 4.5B checkpoint gives the wrong weight-memory
and bandwidth model.

## Hybrid cache sizing

Ignoring declared cross-layer KV sharing, the upper-bound BF16 storage is:

```text
global:  6 × 2 KV heads × 512 dims × 2 (K,V) × 2 bytes = 24 KiB/token
sliding: 36 × 2 × 256 × 2 × 2 bytes × at most 512 resident positions
         = 36 MiB/request once every local window is full
```

At 131,072 tokens, the unshared global upper bound is about 3 GiB per request,
plus bounded local state, allocator metadata, and workspaces. Actual storage can
be lower where the supported runtime honors unified/shared KV. Do not implement
sharing from the integer `18` alone: inspect the pinned model implementation or
prove the layer mapping against the reference. Incorrect sharing produces
plausible text with wrong logits.

```text
# Pseudocode layer dispatch.
if layer.kind == SLIDING:
    attend(head_dim=256, window=512, rope=local_rope)
else:
    attend(head_dim=512, history=full, rope=proportional_rope,
           kv_owner=declared_shared_group(layer))
```

A single rectangular cache wastes memory and invites wrong head dimensions.
Use typed per-layer metadata even when the physical allocator is shared.

## PLE execution opportunities

PLE is a token-indexed lookup, then a small per-layer projection/gate. Avoid
materializing redundant expanded tensors across the whole vocabulary or copying
all 42 slices per step. Useful hypotheses include gathering only current token
rows, retaining stable indices, fusing small elementwise work where the profile
shows launch overhead, and arranging weights for coalesced reads. Every change
must preserve the exact layer slice, projection, gate, normalization order, and
accumulation dtype.

## Prompt and generation semantics

The checkpoint has native system-role and configurable thinking behavior. This
task passes `chat_template_kwargs` including `enable_thinking=false`; use the
pinned Jinja template and preserve that field. EOS is a list at the top-level
config, and the instruction-tuned template can emit control tokens. Validate
finish reasons, ignore-EOS behavior, visible text, usage counts, and stateful
streaming rather than relying on one decoded final string.

The loader filters out complete traces beyond 131,072 tokens. It does not grant
permission to truncate an admitted trace. At offered concurrency two, profile
whether the tight TTFT gate is limited by prefix miss/re-prefill, long prefill
blocking, PLE/weight bandwidth, attention, host gaps, or queueing.

## Speculative path

The reference SGLang arm uses the separately pinned
`google/gemma-4-E4B-it-assistant` with NEXTN-style settings. That is a separate
asset and state machine, not an architectural property of the target checkpoint.
The target remains authoritative. Measure accepted tokens per verification,
draft time, target expansion, and rollback. Compare request tokens and final
answers to the non-speculative path; at small live batches, added draft work can
help or hurt depending on acceptance.

## Fast-path qualification

Test all six layer-transition boundaries, both head dimensions, positions 511/
512/513, full-attention long positions, PLE layer zero and 41, batch positions,
prefix reuse/eviction, cancellation, and graph/eager fallback. Verify logits or
tokens against an independent reference on small and boundary shapes before
full request correctness. The exact 1,800-second replay alone decides the hard
gates and dollars per dollar.

Primary sources: pinned `config.json`, safetensor shapes, and chat template,
plus https://ai.google.dev/gemma/docs/core.
