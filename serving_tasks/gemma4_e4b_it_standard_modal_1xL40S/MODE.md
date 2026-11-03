# Modal mode

**Final validation uses the shared evaluator on Modal.** The verdict requires
a pinned, isolated SGLang reference with `/tokenize` and `/generate` logprob
APIs, in a separate fresh sandbox after the candidate sandbox has stopped.
The image pin, reference flags, checks, workload, 1800-second window and scoring
come from the same code as Docker. Performance requests stay on loopback;
only post-measurement evidence is downloaded. Both public `./full-test --wait`
and private final validation use this path. Startup/reference/infrastructure
failures remain explicit and cannot become a correctness pass.

Instructed and standard guidance are the same as on Docker; the profile is
independent of the mode.

The run happens in a **Modal sandbox**. `serving_tasks/_modal/modalctl.py` is
the only place a run's sandbox is created, and it composes the image from the
same two Dockerfiles this repository builds locally -- our image, adapted, not
a second one.

```bash
harbor/.venv/bin/python serving_tasks/_modal/modalctl.py setup --task <TASK>
bash serverbench/scripts/server_evolve/modal-evolve.sh <TASK> --hours 8
```

The interpreter is `harbor/.venv/bin/python`, which can `import modal`. The
harness interpreter deliberately cannot: it is shared with live runs, and
pulling modal's dependency tree into it could move grpcio or protobuf
underneath a driver that is mid-run.

| what | where it comes from |
|---|---|
| the cards | Modal's, requested from the task id's hardware suffix |
| the image | composed by `_modal/docker_to_modal.py` from this repository's Dockerfiles |
| the checkpoint | fetched into a Modal Volume at the task's pinned `MODEL_REVISION` |
| the corpus | fetched into a Modal Volume at the task's `CORPUS` |
| the way out | Modal's sandbox has no route to this repository, and the task's own files are uploaded rather than mounted |

**The hardware suffix must be a card Modal offers.** `modalctl.gpu_spec`
refuses anything else by name, and its list is the authority: `L40S`, `A100`,
`A100-40GB`, `A100-80GB`, `H100`, `H200`, `B200`, `L4`, `A10`, `T4`. A task
named `_2xL40` cannot run in this mode -- Modal has no L40 -- which is why the
docker-mode tasks on this fleet carry `L40` and the modal-mode ones carry
`L40S`.

The suffix also has to agree with `contract.yaml`'s `devices`: the id decides
WHICH card because everything else in the suite is named after it, and the
contract decides HOW MANY because that is what the objective bills.

**Only this run's sandboxes are stopped when it finishes.** Model and wheelhouse
volumes are shared and mounted read-only by running tasks. They are kept by
default so one run cannot delete another's inputs. After the campaign, an
operator may explicitly purge unused volumes; purge refuses while sandboxes
remain live. Storage continues to incur Modal charges until explicitly removed.

Before a task's first run on these cards, measure SGLang's operating point on
them and record it with `taskctl set <task> --reference-concurrency N`. It does
not transfer from the docker-mode sibling, even for the same model -- different
cards, different KV pool, different operating point.
