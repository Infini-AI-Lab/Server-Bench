# Authoring and validating a knowledge bundle

A skill bundle is what an agent reads before it designs anything. It ships in
`skills/<name>/` and is installed into a candidate workspace at
`.claude/skills/<name>/`, where the Claude Code CLI discovers it.

```
serverbench/skills/<name>/
  SKILL.md            the router — small on purpose
  provenance.yaml     what was read, from where, under which licence (never delivered)
  references/         focused files, loaded only when routed to (optional)
  templates/          original starting points, meant to be rewritten (optional)
```

`serving-system-evolution` is `SKILL.md` and `provenance.yaml` only;
`kernel-evolution` adds `references/`, `templates/` and a `knowledge/` corpus
indexed by `knowledge/catalog.yaml`, whose `root` is skill-relative so it is
true inside a workspace too.

## The four rules

**Every reference resolves.** A skill that routes to a file which does not exist
sends the agent nowhere at exactly the moment it needed help.

**The entry point stays a router.** Progressive disclosure only works if the top
level is small. A `SKILL.md` that grew into an encyclopedia is loaded in full,
every time, relevant or not. The limit is 24 KB and it is enforced.

**No upstream source ships.** The bundle is *derived from* reading pinned
framework source and must contain none of it — no copied files, no distinctive
excerpts, no translated implementation. Naming a system is fine and often
necessary; reproducing it is not. The templates deliberately avoid upstream type
names so nothing can be mistaken for a copy.

**Provenance is recorded.** Repository, commit, licence, retrieval date, the
files inspected, and what each one taught. Without it, "distilled from upstream"
is an unverifiable claim.

## Validating

```bash
serverbench evolve skill serverbench/skills/serving-system-evolution
```

It also checks that nothing under `references/` or `templates/` is unreachable
from the router (material the agent is never routed to is material it never
sees) and that no local checkout path is referenced.

The source scan is deliberately conservative: it can flag prose that quotes an
identifier, which a reviewer resolves by looking, and it cannot miss a copied
source file, which is the direction that matters.

## Researching upstream without vendoring it

1. Check the bundle out **outside this repository**, read-only, pinned to a
   commit. `third_party/` is gitignored for exactly this.
2. Prefer shallow, blobless, sparse checkouts of the runtime paths you need. Do
   not build it, run it, benchmark it, or download weights or engines.
3. Read enough to understand the *boundaries and failure modes*, not just the
   marketing architecture diagram.
4. Write the lesson in your own words, with the *reason* it exists. "SGLang keeps
   a result queue" is trivia; "overlap is a one-batch lag, and there are steps
   where it must be disabled and drained first" is knowledge.
5. Record every commit and file in `provenance.yaml`.
6. Run the validator, and read every distributed file yourself.

## Delivery

`harness/lifecycle/delivery.py` copies each bundle into the workspace at
`.claude/skills/<name>/` when the run builds it. `run.py` picks the tree by
task variant: `serverbench/skills/` for `_standard_` tasks,
`serverbench/skills_instructed/` for `_instructed_` ones (which also get the
pinned `KernelWiki` submodule plus the overlay in `_overlays/KernelWiki/`).

The copy is filtered, not verbatim: a file that carries upstream serving-framework
source is held back, and a model note about another task's model is removed.
Both are recorded in the run's delivery manifest (`withheld`), next to the digest
of every file that was delivered, so a run can show which bytes the agent
received. A bundle's `provenance.yaml` is a repository record, not reference
material: it stays in the checkout for the validator and is never copied
(`delivery.SKILL_RECORDS_NOT_DELIVERED`).

The instruction carries a **pointer, not a summary**: a table naming each
delivered bundle's path and subject (`instruction_render.SKILL_SUBJECTS`). Summarising the skill in the
brief would load all of it into context every iteration, which is what routing
exists to avoid.

## Updating

A skill is a dependency: changing it changes what future agents know. So —

- re-read the pinned source, and move the pin deliberately;
- update `provenance.yaml` in the same change;
- re-run the validator;
- say in the commit what an agent will now do differently.
