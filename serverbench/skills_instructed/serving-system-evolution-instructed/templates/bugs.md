# Bug ledger

Bugs that cost real time, and the checks that now prevent them.

Write an entry when you **understand** a bug, not when you first hit it. Read
this file before proposing a candidate in an area you have been stuck in before —
it is the list of traps you already know about, and it is carried forward into
every later run of this lineage.

The last field of each entry is the point of the file. A bug you fixed is worth a
few minutes; a bug you turned into a check is worth every future run that never
hits it.

---

## <short name>  (<date>)

**Symptom, verbatim.** The exact message, number or behaviour, quoted. Not
paraphrased — the next person greps for the string.

**What it named.** The subject the message pointed at.

**What it actually was.** The real cause, and how you *proved* it — the smaller
reproduction, the isolated component, the second run on a quiet machine — rather
than what you concluded.

**Why it looked like the other thing.** This is the part that generalises to the
next bug, which will not be this one.

**The check that would have caught it sooner.** A test, an assertion, an
ordering, a guard. Add it now and link it here. If you could not add one, say
why — that is a finding too.

---

<!--
Worked example. Delete it once you have entries of your own.

## router-readiness-demanded-a-missing-endpoint

**Symptom, verbatim.** `the router did not answer within 10 minutes`, repeated
across two full measurement passes, while the router's own log read
`INFO: Application startup complete.` followed by
`INFO: "GET /v1/models HTTP/1.1" 404 Not Found` every five seconds.

**What it named.** The router — as failed, unhealthy or slow to start.

**What it actually was.** The router was healthy the entire time. The readiness
probe required HTTP 200 from `/v1/models` before it would attempt any real work,
and that server does not implement `/v1/models`. Proved by reading the captured
log: startup completed, and every probe was answered — with a 404.

**Why it looked like the other thing.** "Did not answer" is true of the probe's
question and false of the server. A 404 *is* an answer, and a probe that only
recognises 200 cannot tell "not implemented" from "not up".

**The check that would have caught it sooner.** Make readiness the completions
trial — the actual work — with `/v1/models` used only for what it can honestly
report: whether anything is answering at all. Any status code counts as up; only
a connection failure counts as down.
-->
