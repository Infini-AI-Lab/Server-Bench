# Questions worth asking before you call a serving change done

Not a checklist to tick, and nothing reads it. These are the questions
that have most often turned out to matter, offered so you do not have to
rediscover which ones those are.

**Is it still correct?** The correctness rule is checked before any
latency number is read, so a change that is faster and answers
differently scores nothing.

**Did you measure the thing you changed?** One hypothesis, one change.
Two at once and a moved number attributes to neither.

**Is the margin real?** Run-to-run spread on this corpus is wide. A gate
met by a few percent is noise; if it matters, measure again.

**Does it hold at the offered concurrency you are scored at?** A gain at
another load is not a gain on this task.

**Does the published tree still start?** The verdict re-serves your
published version in a fresh container by running `serve.sh`, and
`.venv` is not published -- so `serve.sh` has to rebuild it when the
interpreter is absent.

**Is anything hard-coded to the workspace?** The verdict mounts your
tree somewhere else and exports `APP_DIR`.
