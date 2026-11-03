"""The agent's document states facts. It does not teach, advise or instruct.

The standard treatment measures what an agent does with the facts alone, so
guidance in its brief would confound the experiment. The document carries: the
goal, the two gates, the workload, an inventory of the workspace and the
container, how the verdict runs, the entrypoint it requires, and how versions
and the budget work. It carries no directory layout to copy, no order of work,
no profiler or compiler tutorial, and no sentence telling the agent what to do
next.

Rendered for EVERY task under `serving_tasks/`, so a task added later is held
to the same line with nothing to remember.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "serverbench" / "harness"),
           str(REPO / "serving_tasks" / "_manage")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASKS = sorted(p for p in (REPO / "serving_tasks").glob("*")
               if (p / "contract.yaml").is_file())

#: Advisory shapes, each labelled with the kind of material it belongs to. A
#: match is a sentence telling the agent how to work rather than what is true.
FORBIDDEN = (
    (r"\byou should\b", "advice"),
    (r"\bwe recommend\b", "advice"),
    (r"\bit is worth\b", "advice"),
    (r"\bworth reaching for\b", "the CUDA/PTX tutorial"),
    (r"\bbefore you optimise\b", "the prescribed order of work"),
    (r"\bdoes it start\?", "the prescribed order of work"),
    (r"\bthe way you would lay out\b", "the directory layout to copy"),
    (r"engine/scheduler\.py", "the directory layout to copy"),
    (r"kernels/attention\.py", "the directory layout to copy"),
    (r"\bspend that headroom\b", "the throughput advice"),
    (r"\badmit more, batch wider\b", "the throughput advice"),
    (r"\bis usually the fastest way\b", "the profiler tutorial"),
    (r"\buse a profiler before\b", "the profiler tutorial"),
    (r"\bnsys is usually\b", "the profiler tutorial"),
    (r"cuModuleLoadData", "the CUDA driver-API tutorial"),
    # NAMING A PACKAGE IS AN INVENTORY; TEACHING ITS USE IS NOT.
    #
    # The document says the container can build and profile CUDA and PTX, so
    # the toolchain row names `nvcc`, `ptxas`, NVRTC, `ninja` and
    # `torch.utils.cpp_extension`, and says nothing about when or whether to
    # reach for any of them. What is banned is the imperative around them.
    (r"NVRTC is not in the CUDA tree", "the NVRTC tutorial"),
    (r"\bload_inline\b", "the CUDA extension tutorial"),
    (r"\bbuild (?:a|your own) (?:CUDA|custom) (?:extension|kernel)",
     "the CUDA extension tutorial"),
    (r"register budget", "the PTX tutorial"),
    (r"can load your checkpoint", "the imperative checklist"),
    (r"\bthe first thing to do\b", "the prescribed order of work"),
    (r"\bIterate fast\b", "the working-style advice"),
    (r"\bAsk often\b", "the working-style advice"),
    (r"\bkeep these open as you work\b", "the reading list framing"),
    (r"\bRead this first\b", "the reading list framing"),
    (r"\bWalk the whole workspace\b", "the inventory instruction"),
    (r"\bAn hour spent here\b", "the working-style advice"),
    (r"\bit is not optional\b", "the required loop"),
    (r"\bRun it after every change\b", "the required loop"),
    (r"\byour job\b", "advice"),
    (r"\bmake sure you\b", "advice"),
    (r"\bdo not spend\b", "advice"),
    (r"\bSpecialise\b", "the design advice"),
    (r"\bkeep working\b", "advice"),
    (r"\bREAD IT\b", "the imperative"),
    (r"\byou are done\b", "advice"),
    (r"\bnext thing to do\b", "advice"),
    # Each tells the agent how to work.
    (r"Make as many requests as you can", "the objective coaching"),
    (r"straightforward reference path", "the accuracy-method coaching"),
    (r"A version whose outputs you have not", "the accuracy-method coaching"),
    (r"compare systems over repeated runs", "the measurement coaching"),
    (r"expected to pass this replay gate", "the replay-gate coaching"),
    (r"full window is\s+not required merely", "the replay-gate coaching"),
    (r"extra per-token GPU\s+synchronisation", "the native-token coaching"),
    (r"Use a proper incremental detokenizer", "the detokenizer coaching"),
    (r"check the adapter and then", "the native-token checklist"),
    (r"\bRun it at the start\b", "the GPU-clock order of work"),
    (r"\bRerun any measurement\b", "the GPU-clock order of work"),
    (r"\bDiagnose unexpected clock\b", "the GPU-clock order of work"),
    # The container has no channel to an operator.
    (r"\bask the operator\b", "a channel the agent does not have"),
    (r"\boperator diagnosis\b", "a channel the agent does not have"),
    # Not what is enforced: the CLI's settings are pinned, routes in general
    # are logged and audited.
    (r"Every model route in this\s+container is pinned", "an overclaim"),
)

#: Facts the document must carry.
REQUIRED = (
    ("dollars_per_dollar", "the goal"),
    ("## The hard gates", "the hard gates"),
    ("## The three hard gates", "the three separate hard gates"),
    ("**1. The replay gate.**", "the replay gate"),
    ("**2. The EOS (end-of-sequence) gate.**", "the EOS gate"),
    ("**3. The correctness gate.**", "the correctness gate: what correct output means"),
    ("Output correctness is verified privately",
     "that correctness is checked privately, without saying how"),
    ("Design and run your own scripts", "that the agent verifies its own"),
    ("You measure your server's accuracy yourself, inside this container",
     "that accuracy is the agent's own measurement, in the container"),
    ("scores **0 `dollars_per_dollar`**", "that a hacked system scores zero"),
    ("Latency is not a hard gate", "that latency is scored, not gated"),
    ("## How AgentX measures your server", "how the workload drives the server"),
    ("## Measure dollars_per_dollar the way the verdict does",
     "the in-container full test, with dollars_per_dollar"),
    ("./full-test --duration N --wait", "the full-test command"),
    ("**You choose N.**", "that the development window is the agent's"),
    ("Use only that model", "one model for every route"),
    ("A more capable model will inspect your work", "the reviewer notice"),
    ("replay fidelity", "what correctness means"),
    ("## What is in your workspace", "the workspace inventory"),
    ("## What is in the container", "the container inventory"),
    ("## How the final verdict runs", "how the verdict runs"),
    ("## The entrypoint the verdict requires", "the verdict's entrypoint"),
    ("serve.sh", "the entry point itself"),
    ("GET /health", "the readiness endpoint"),
    ("POST /v1/chat/completions", "the served endpoint"),
    ("max_completion_tokens", "the request fields the server must honour"),
    ("chat_template_kwargs", "the chat-template field"),
    ("enable_thinking=false", "what the correctness requests set"),
    ("tests/final_verdict/", "the evaluator copy in the workspace"),
    ("--check-protocol-context", "the evaluator's protocol/context mode"),
    ("tests/replay-verify", "the lightweight AgentX replay verifier"),
    ("can be used for lightweight replay verification",
     "the standard treatment's replay-verification availability"),
    ("final published head", "which version is judged"),
    ("## The image, and the packages in it", "the image and its packages"),
    ("## The offline build inputs", "the wheelhouse"),
    ("torch==", "the installed torch version"),
    ("transformers==", "the installed transformers version"),
    ("Nsight Systems", "the profilers' versions"),
    ("nvcc", "the CUDA toolchain"),
    ("## Versions", "what gets judged"),
    ("## Budget", "the budget"),
)


def _document(task_dir: Path) -> str:
    from lifecycle import instructions

    return instructions.build_brief(task_dir=task_dir, model_id="", hours=8.0,
                                    carried=False)


def test_there_are_tasks_to_check() -> None:
    assert TASKS and STANDARD


#: The instructed treatment deliberately carries coaching; it is held to its
#: declared curriculum below instead.
STANDARD = [t for t in TASKS if "_instructed_" not in t.name.lower()]


@pytest.mark.parametrize("task_dir", STANDARD, ids=[t.name for t in STANDARD])
def test_the_document_advises_nothing(task_dir: Path) -> None:
    body = _document(task_dir)
    found = [f"{pattern!r} ({what})" for pattern, what in FORBIDDEN
             if re.search(pattern, body, re.I)]
    assert not found, (
        f"{task_dir.name}'s instruction.md carries advisory material:\n  "
        + "\n  ".join(found)
        + "\n\nThe document states what is true -- the goal, the gates, the "
          "inventory, the verdict and its entrypoint. Guidance on how to "
          "work belongs nowhere.")


@pytest.mark.parametrize(
    "task_dir",
    [t for t in TASKS if "_instructed_" in t.name.lower()],
    ids=lambda t: t.name,
)
def test_the_instructed_document_carries_the_declared_curriculum(
        task_dir: Path) -> None:
    body = _document(task_dir)
    for required in (
            "## Rapid-feedback serving evolution",
            "### The evolve loop",
            "**Verify (lead) -- the gate.**",
            "your own output-correctness checks",
            "Run `./full-test` at every verification",
            "candidate_tree_sha",
            "--check-protocol-context",
            "./full-test --duration N --wait",
            "extreme-length requests",
            "DELIVERY_POLICY.json",
            ".claude/skills/serving-system-evolution-instructed/",
            "last working tree"):
        assert required in body, f"{task_dir.name} is missing {required!r}"


@pytest.mark.parametrize("task_dir", TASKS, ids=[t.name for t in TASKS])
def test_the_document_states_every_required_fact(task_dir: Path) -> None:
    import yaml

    body = _document(task_dir)
    contract = yaml.safe_load((task_dir / "contract.yaml").read_text("utf-8"))
    required = list(REQUIRED)
    if (contract.get("request_scoring") or {}).get("kind") \
            == "request_p90_billable_tokens_v4":
        required.remove(("## The hard gates", "the hard gates"))
        required.append(("## The p90 bars", "the p90 bars"))
        required.append(("request_p90_billable_tokens_v4", "the scoring policy"))
        required.append(("gross_dollars_per_dollar", "the gross diagnostic"))
    missing = [f"{needle!r} ({what})" for needle, what in required
               if needle not in body]
    assert not missing, (
        f"{task_dir.name}'s instruction.md does not state:\n  "
        + "\n  ".join(missing))


@pytest.mark.parametrize("task_dir", TASKS, ids=[t.name for t in TASKS])
def test_a_context_probe_is_named_only_where_one_runs(task_dir: Path) -> None:
    """The verdict sends a near-maximum context probe only for a task whose
    module declares `REQUIRED_CONTEXT_LENGTH`
    (`scenario._context_probe_tokens`); elsewhere naming one is false."""
    from final_validate import tasks

    body = _document(task_dir)
    module = tasks.python_module(task_dir.name)
    if getattr(module, "REQUIRED_CONTEXT_LENGTH", None):
        assert "near-maximum context probe" in body
    else:
        assert "near-maximum context probe" not in body
        if "_instructed_" not in task_dir.name.lower():
            # The instructed appendix is the curriculum's own text.
            assert "context probe" not in body


@pytest.mark.parametrize("task_dir", TASKS, ids=[t.name for t in TASKS])
def test_the_workspace_record_is_described_as_it_is(task_dir: Path) -> None:
    """The deadline the brief places in `WORKSPACE.json` is written there,
    under the key it names."""
    body = _document(task_dir)
    assert "`deadline_utc` in `WORKSPACE.json`" in body
    assert "identity, paths and deadline" not in body


@pytest.mark.parametrize("task_dir", TASKS, ids=[t.name for t in TASKS])
def test_the_verdicts_own_command_is_stated(task_dir: Path) -> None:
    """The verdict's command, with the load the candidate declares.

    Every live task lets the candidate declare its load, so the command names
    where the load comes from, and the SGLang reference point is stated, not
    imposed.
    """
    from final_validate import tasks

    body = _document(task_dir)
    module = tasks.python_module(task_dir.name)
    assert getattr(module, "CONCURRENCY_POLICY", None) == "candidate"
    assert "aiperf profile" in body
    assert "--concurrency <N from concurrency.json>" in body
    assert "## Offered concurrency: yours to choose" in body
    reference = getattr(module, "REFERENCE_CONCURRENCY", None)
    if reference is not None:
        assert f"offered session concurrency **{reference}**" in body
    else:
        assert "No reference operating point is stated" in body
    assert getattr(module, "RANDOM_SEED", 0) is None
    assert "--random-seed <fresh random seed>" in body


@pytest.mark.parametrize("task_dir", TASKS, ids=[t.name for t in TASKS])
def test_the_gates_come_from_the_contract(task_dir: Path) -> None:
    import yaml
    from final_validate import service_levels

    spec = yaml.safe_load((task_dir / "contract.yaml").read_text("utf-8")) or {}
    # `contract.yaml` states the p90 row at its top level.
    levels = service_levels.read(
        {"service_levels": spec.get("service_levels")}, task_id=task_dir.name)
    body = _document(task_dir)
    for level in levels:
        for ceiling in (level["ttft_ms"], level["tpot_ms"]):
            assert f"**{float(ceiling):g} ms**" in body, (
                f"{task_dir.name}: the p{int(level['percentile'])} ceiling "
                f"{ceiling} ms is not in the document")


@pytest.mark.parametrize("task_dir", TASKS, ids=[t.name for t in TASKS])
def test_the_session_prompts_are_the_tasks_own(task_dir: Path) -> None:
    """And they are not delivered as part of the specification."""
    from lifecycle import instructions

    body = _document(task_dir)
    for kind, heading in instructions.SESSION_HEADINGS.items():
        assert heading not in body, (
            f"{heading} is delivered in the document the agent reads at "
            f"launch; it is a later prompt, not part of the specification")
        text = instructions.session_prompt(
            task_dir, kind=kind, budget_seconds=3600.0,
            deadline_utc="2026-09-18T00:00:00Z")
        assert text and "{budget_hours}" not in text
        assert "1.00 h" in text and "2026-09-18T00:00:00Z" in text
        if "_instructed_" in task_dir.name.lower():
            for expected in (
                    "### Instructed loop reminder",
                    "torch.profiler",
                    "design_review.py",
                    "--check-protocol-context",
                    "your own output-correctness checks",
                    "immutable candidate SHA",
                    "./full-test --duration N --wait",
                    "last working tree"):
                assert expected in text, (
                    f"{task_dir.name}/{kind} is missing {expected!r}")
        else:
            assert "### Instructed loop reminder" not in text


def test_the_harness_composes_no_agent_facing_prose() -> None:
    """The harness files that load and assemble the brief carry no prompt of their own."""
    banned = {
        "serverbench/harness/lifecycle/instructions.py": 200,
        "serverbench/harness/lifecycle/agent.py": None,
        "serverbench/harness/lifecycle/workspace.py": None,
        "serverbench/harness/final_validate/task_spec.py": None,
    }
    for path, line_cap in banned.items():
        body = (REPO / path).read_text("utf-8")
        if line_cap is not None:
            assert len(body.splitlines()) < line_cap, (
                f"{path} has grown to {len(body.splitlines())} lines; it "
                f"is a loader, not a template")
        # STRING LITERALS ONLY, and never a docstring: those are notes to
        # whoever reads the code, and this test is about what an agent reads.
        # Parsed rather than grepped, because a docstring's continuation lines
        # look like ordinary prose to a line scanner.
        import ast

        tree = ast.parse(body)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                first = (node.body or [None])[0]
                if (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    docstrings.add(id(first.value))
        literals = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings]
        for text in literals:
            for pattern, what in FORBIDDEN:
                assert not re.search(pattern, text, re.I), (
                    f"{path} carries {what} in a string it can emit: "
                    f"{text[:90]!r}")
