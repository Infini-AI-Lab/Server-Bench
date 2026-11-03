# Kernel evolution

Evolving one composable GPU kernel, safely. A separate subsystem from the
serving verdict: no serving run uses it, and every serving task lets the agent
write kernels anyway.

| page | what it answers |
|---|---|
| [ladder.md](ladder.md) | what gets checked, in what order, and why that order |
| [controls.md](controls.md) | the check on the checks |
| [scope.md](scope.md) | what is deliberately out of scope, and how that is enforced |
| [knowledge.md](knowledge.md) | what a kernel writer is given, and what is withheld |

## The shape of it

`kernel_evolve` is a **sibling** of `serving_evolve`, not a fork. Everything a
campaign must have exactly one of (the candidate controller, the ledger, the
accepted-head compare-and-swap, the budget, the evidence index) stays single; a
second copy of any of those is how a campaign spends twice its ceiling, or
accepts two different heads. So this package adds only what is kernel-shaped:

```
serverbench/harness/kernel_evolve/
  contracts/       what a kernel claims: capability, envelope, model semantics,
                   candidates, budgets, verification plans and mutant classes
  verification.py  the eleven-rung ladder, the reference operators, the
                   negative controls and kill matrix, the out-of-scope scan
                   and the import check
  shapes.py        the strategy catalog: what a candidate may dispatch to
  sandbox.py       a build that cannot see the harness
  campaign.py      a deterministic scripted campaign, no GPU required
  smoke.py         one real kernel, all the way up, on a real device
  commands.py      the command runner that records receipts
```

Nothing in it is copied from another kernel repository. A kernel pasted from a
production codebase is not a kernel the agent designed, and the benchmark asks
what an agent can do; the design principles are re-expressed as original code
and prose.

## Two rules that are enforced rather than documented

**A microbenchmark may reject and may never accept.** The rung carries
`can_accept=False`, and the acceptance policy will not take it. A kernel three
times faster on an operator holding four per cent of the critical path cannot
move the served system by more than about three per cent; if the operator was
not on the critical path, it cannot move it at all.

**The serving runtime never imports this package.** Evolution control code in a
serving hot path makes the thing being measured depend on the thing measuring
it. `kernel_evolve.verification.check_imports` checks this rather than
asserting it.

## Running it

```bash
serverbench evolve kernels list       # the ladder and the required controls
serverbench evolve kernels doctor     # can this host build and run a candidate
serverbench evolve kernels knowledge  # does the knowledge catalog resolve
```

`doctor` takes `--interpreter`, an absolute path to the environment a build
would actually use; the path is the identity, because resolving a virtualenv's
`bin/python` symlink yields the base interpreter and loses the venv's packages.
