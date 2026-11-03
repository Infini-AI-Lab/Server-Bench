"""The evolve lifecycle: one run, one agent, one final head.

Everything a run does is in this package; the production entrypoint is
:mod:`lifecycle.run`, driven by `serverbench/scripts/evolve`. The modules, in
roughly the order a run uses them:

    record          who this run is, what happened, what state it is in
    workspace       build the workspace, prove it boots, own what that took
    delivery        copy the seed, evaluator, skills and waiters into it
    instructions    the brief the agent receives
    agent           the processes started inside the agent's container
    boundary        freeze a tree, then register it as a version
    boundary_checks deterministic checks over a frozen tree
    publish         move the final head by compare-and-swap
    full_test       the development full test an instructed agent may request
    decisions       the agent's stated reasoning, for each version's DECISIONS.md
    report          the run's ATIF trajectory
    verdict         whose fault a failure is
    utils           helpers with no phase of their own

Nothing in a run measures the candidate: the agent measures itself with the
pinned `aiperf`, and the final verdict (`serverbench/harness/final_validate`) runs
after the evolve, outside the container.
"""
