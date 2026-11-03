# `wait/`

Blocking waiters. Each returns the moment its condition holds; they are
written in `bash` and `python3`.

    bash wait/wait_ready.sh [port] [deadline_s] [route]   # the server answers
    bash wait/wait_for.sh --file /tmp/done  [--deadline 900]
    bash wait/wait_for.sh --pid 12345       [--deadline 900]
    bash wait/wait_for.sh --match 'DONE' --in /tmp/sweep.log
    bash wait/run_long.sh --tag sweep -- <command>        # start it waitably
    bash wait/gpu_free.sh [min_free_MiB] [deadline_s]     # or name the holder
    bash wait/await_agent.sh [agentId] [--deadline 7200]  # a subagent finishes

`run_long.sh` returns the output inline and the real exit code when the command
finishes quickly, and otherwise prints a log path, a done-file and the line
that blocks on it.

`await_agent.sh` reads the subagent's output file, which ends with
`[exited with code N]` when it is finished, and prints that code and a bounded
tail.
