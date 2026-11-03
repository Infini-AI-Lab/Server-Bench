# Which kernels you may write

Some tasks freeze a set of kernels and ask you to use them; others ask you to
write your own. The task contract says which, under `kernel_strategy`.

The serving tasks in this suite are `evolve`: the kernels are yours to write,
no catalog of frozen implementations is supplied, and a kernel you wrote is the
only kind that counts as your work.

What does not change either way: the model semantics, the four latency gates,
the correctness rule, the measurement window and the clock. A frozen kernel you
were handed and one you wrote are judged by exactly the same numbers.
