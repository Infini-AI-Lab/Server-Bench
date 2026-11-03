# When you are stuck

Being stuck is a measurement. It says the model you have of the system is wrong
somewhere, and the fastest way forward is to find out *where* rather than to try
another change.

The single most useful prior:

> **A failure message names a subject, and the subject is often wrong.**
> "X failed" usually means the thing that *measured* X failed, or the thing that
> *set X up* failed. Before you debug X, check the two things standing between
> you and X.

In every example below the named subject is innocent. They are worth reading
once because the shape repeats, not because you will meet these particular bugs.

| the message says | what can actually be happening |
|---|---|
| `the prefill role exited before it answered` | a *previous* process still holds most of the device's memory |
| `SLO MISS` at a value well under the bar | two modules disagree about what `slo_pass` means |
| `the router did not answer within 10 minutes` | the router is healthy; the probe asks for an endpoint it does not implement |
| `exit_code: 1` after hours of work | the credentials expired; every call returned 401 |
| a process killed with no output | its own `pkill -f server.py` matched its own command line |
| a published result with no verdict | a `None` percentile formatted with `:.1f`, swallowed by a "never fatal" step |
| `ImportError: libcuda.so.1` | a container started without `--gpus` |

## The order to check things in

Cheapest and most often wrong first. Do not skip ahead because a step feels
beneath the problem — the reason these are ordered this way is that the early
ones are the ones that actually fire.

1. **Is the input there?** Not "is the path mounted" — is the *content* readable.
   A directory of symlinks whose targets are absent lists perfectly and opens
   nothing. Read one file end to end and check its size.
2. **Does the artefact you are reading belong to this run?** Check its
   timestamp against the run's start. A log from a previous attempt reads
   exactly like a diagnosis of this one and is not. If a failure path does not
   *write* evidence, an old file is what you will find.
3. **Is anything else on the device?** Your own previous process counts. So does
   a neighbour's. `nvidia-smi --query-compute-apps` names them.
4. **Does the failing subject match the failing thing?** Read the message
   literally, then ask what it would take for that message to appear when the
   named subject is fine.
5. **Is the check itself the problem?** A probe that demands a 200 from an
   endpoint the server never implements will wait forever against a healthy
   server. A predicate that folds two different questions together will fail on
   the one you were not asking.
6. **Only now: is the idea wrong?**

## Reproduce it smaller before you explain it

An explanation you cannot reproduce is a story. Shrink until the failure is
either obvious or gone:

- **One request, two tokens.** A context server that cannot answer
  `{"prompt": "ok", "max_tokens": 1}` in two minutes is not slow, it is stuck,
  and that is a different investigation from a slow one.
- **One device, one process.** If it only fails with two, the interaction is the
  bug.
- **One component, started by hand.** Launching a single role outside the
  harness separates "this component is broken" from "the harness starts it
  wrong". The latter is common.
- **The same input twice.** A failure that moves between blocks is not
  contention if it also moves on an idle machine — it is a property of the
  system at that operating point.

## Silence is not evidence

Some of the most expensive mistakes are silence read as a result:

- A point where every request fails records **no percentiles at all**, and a
  check that skips missing values concludes the arm met its SLO everywhere.
  *Absent is not passing.*
- A "never fatal" publish step swallows an exception **after** it has written
  half its output, leaving a directory that looks published and carries no
  verdict. *Partially done looks identical to done unless something checks.*
- A timeout path returns a message and captures no log, so the only file on
  disk is an earlier attempt's. *A failure that writes nothing teaches nothing.*

When you write a failure path, make it record what it saw. When you read one,
ask what it would have looked like had it recorded nothing.

## When the instrument and the subject disagree

If two things that should agree do not — a datasheet and a brief, a percentile
and a bar, a log and an exit code — **stop and resolve the disagreement before
acting on either.** Do not pick the one that supports your plan.

This is not pedantry. When a datasheet and a brief state different bars,
choosing the wrong one spends hours optimising against a bar that does not
exist. Resolve which source is authoritative, and say so.

## Record the bug, not just the fix

Keep `MEMORY/bugs.md` — [templates/bugs.md](../templates/bugs.md) is the
starting point, and a starting point for one is in this skill's `templates/`. One entry per
bug that cost you more than a few minutes, written **when you understand
it**, not when you first hit it:

```markdown
## <short name>  (<date>)

**Symptom, verbatim.** The exact message or number, quoted. Not paraphrased —
the next person greps for it.

**What it named.** The subject the message pointed at.

**What it actually was.** The real cause, and how you proved it rather than
concluded it.

**Why it looked like the other thing.** This is the part that generalises.

**The check that would have caught it sooner.** A guard, a test, an assertion,
or an ordering. If you can add it now, add it and link it here.
```

That last field is the point of the file. A bug you fixed is worth a few minutes;
a bug you turned into a check is worth every future run that never hits it. An
entry that ends in a test keeps catching the same class of mistake whenever it
is reintroduced.

Read `MEMORY/bugs.md` **before** proposing a candidate that touches an area you
have been stuck in before. The ledger is not a diary; it is the list of things
you already know are traps.

## What not to do when stuck

- **Do not bundle changes to escape.** Three changes at once turns one unknown
  into three, and the result tells you nothing either way.
- **Do not raise a timeout to make a symptom go away.** If a queue grows linearly
  with concurrency, a longer deadline moves the cliff and leaves the serialisation
  in place. Ask what the timeout was protecting you from noticing.
- **Do not conclude from one run** that a failure is transient. Run it again on
  a quiet machine. "It was contention" is the most attractive wrong answer
  available, because it requires no further work.
- **Do not fix what you have not reproduced.** A fix for a cause you inferred is
  a change with no evidence behind it, and it will be in the tree forever.
- **Do not stop recording because you are frustrated.** The entries written at
  the worst moment are the ones that save the most time later.
