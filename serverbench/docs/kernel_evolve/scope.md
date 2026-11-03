# What is out of scope, and how that is held

A **megakernel** is a single persistent GPU kernel that owns the whole model
iteration: one launch, an internal instruction schedule, no per-operator kernel
boundaries. It is a legitimate and interesting technique.

It is deliberately out of scope here, for three reasons:

- it asks a different research question from "can an agent evolve a serving
  system";
- it needs a schedule compiler and a virtual machine before it computes anything,
  which is a large amount of machinery to qualify before the first measurement;
- it does not compose with an ordinary composable-kernel candidate loop, so the
  candidate/budget/ladder design here would have to be rebuilt around it.

## Out of scope is easy to say and hard to hold

Quarantine that is not *checked* decays: a reference survives in a skill, an
import survives in a package, a prompt still routes an agent there, and the
exclusion becomes a claim in a document.

So `kernel_evolve.verification.scan` scans, and the scan is a gate that fails
rather than warns.

## What it checks

Scope is a list of `(surface, path)` pairs, and the surface is part of the
result: "three hits" means very different things in the package, the knowledge
tree and a skill. The default surfaces are the kernel harness, the serving
harness, both knowledge bundles, the skill bundles, the tasks, and the agent
environment images.

It matches `megakernel`, `mega_kernel`, `megair`, `three_layer_mk`, and the
phrase for a persistent single kernel owning the iteration, case-insensitively.

`check_imports` complements it: a static text scan cannot see a module that
arrives through a registry lookup or a dynamic import, and importing then reading
`sys.modules` can.

## What is exempt, and why that is not a loophole

**Provenance files.** Naming the excluded material is their content. A reader
learns from `provenance.yaml` that the megakernel subsystem was left out *on
purpose* rather than lost, and that is worth more than a clean grep.

**Files carrying an explicit marker.** A module whose entire subject is the
quarantine has to be able to name what it quarantines. The marker string is long
and specific precisely so it cannot arrive in a file by accident, and an exempt
file is *listed* in the report rather than silently skipped.

What is not a solution: exempting a file because its name looks like part of the
machinery. A file that trips the scan on an innocent label changes the label; it
is not exempted wholesale, which would blind the scan to a real routing decision
added there later.

Run the scan with `serverbench evolve kernels doctor`, which reports the files
and surfaces scanned and every exempt provenance file.
