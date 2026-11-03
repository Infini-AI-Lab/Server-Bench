# Questions worth asking before you call a kernel change done

Not a checklist to tick, and nothing reads it. A kernel bug produces
fluent, plausible, subtly wrong output at a very good latency, so most of these
are about correctness rather than speed.

**Does it run your kernel?** On the workload being measured, on the
primary compute path -- not a wrapper that dispatches to the library
the baseline already used. That passes every numerical check, because
the answer is right, and reports the library's speed as yours.

**Is it correct on the awkward shapes?** Sequence lengths that are not
a multiple of the block, a batch of one, the longest context you
support, the first call against an empty cache.

**Model semantics.** RoPE base and scaling, the norm's epsilon and
whether it is `w` or `1+w`, head dimension and grouping, the KV layout
you assumed, the dtype of every accumulator. A kernel that is right for
a Llama-shaped block and wrong for this one is the classic way to get
plausible wrong output.

**Does a fallback announce itself?** A silent fallback for shapes you
do not support reports the fallback's numbers under your kernel's name.

**Did you compare like with like?** Same shapes, same dtype, same
occupancy regime, before and after.

**Did the serving system get faster?** A microbenchmark can reject a
kernel and cannot accept one. Only the full scored scenario says
whether the gates still hold.
