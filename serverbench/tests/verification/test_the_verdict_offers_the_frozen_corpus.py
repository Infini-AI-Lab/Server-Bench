"""The verdict must offer the corpus the task froze, not a sibling of it.

The task freezes `semianalysisai/cc-traces-weka-062126` at a pinned sha256.
The scenario's allowed-loader list also carries a 256k-capped sibling, which
is NOT used: capping the corpus drops every turn whose input exceeds the
model's native 262,144 context -- roughly half the turns, and exactly the
population that forces eviction and re-prefill, which is what makes this a
hard serving problem. A verdict that offered the cap would report an easier,
better-looking workload under this task's name.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serverbench" / "harness"))

TASK = REPO / "serving_tasks" / "qwen3_5_9b_standard_2xL40"


#: The corpus this task's p90 bars were measured on, and its digest.
FROZEN = {
    "corpus": "semianalysisai/cc-traces-weka-062126",
    "corpus_sha256": "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276",
}


def _frozen() -> dict:
    return dict(FROZEN)


def test_the_module_offers_exactly_the_frozen_corpus() -> None:
    from final_validate.tasks import qwen3_5_9b_standard_2xl40 as module

    frozen = _frozen()
    assert frozen["corpus"] == module.CORPUS, (
        f"the verdict offers {module.CORPUS!r} and the task freezes "
        f"{frozen['corpus']!r} -- a different workload under this task's name")
    assert frozen["corpus_sha256"] == module.CORPUS_SHA256, (
        "the digest the verdict pins is not the digest the task froze")


def test_the_capped_sibling_is_not_what_gets_offered() -> None:
    from final_validate.tasks import qwen3_5_9b_standard_2xl40 as module

    argv = module.invocation(url="http://h:8000", tokenizer="/tok",
                             artifact_dir="/art", concurrency=3)
    joined = " ".join(argv)
    assert "256k" not in joined, (
        f"the verdict offers the 256k cap the task rejects:\n"
        f"  {joined}")
    assert module.CORPUS in joined


def test_the_invocation_carries_the_calibrated_load_and_no_lock_violation() -> None:
    """AgentX owns request semantics; the task owns trajectory concurrency."""
    from final_validate.tasks import qwen3_5_9b_standard_2xl40 as module

    # The candidate declares its load; whatever it declares is exactly what is
    # offered, once.
    joined = " ".join(module.invocation(url="http://h:8000", tokenizer="/tok",
                                        artifact_dir="/art", concurrency=5))
    at = joined.split().index("--concurrency")
    assert module.CONCURRENCY_POLICY == "candidate"
    assert joined.split()[at + 1] == "5" and joined.count("--concurrency") == 1
    # The seed is part of the offer too: the task's fixed public seed, or on
    # a task that fixes none, a fresh draw that is still passed explicitly.
    seed_at = joined.split().index("--random-seed")
    if module.RANDOM_SEED is not None:
        assert joined.split()[seed_at + 1] == str(module.RANDOM_SEED)
    else:
        assert joined.split()[seed_at + 1].isdigit()
    words = joined.split()
    extra_at = words.index("--extra-inputs")
    assert words[extra_at + 1] == (
        '{"temperature":0,"return_token_ids":true,"skip_special_tokens":false,'
        '"chat_template_kwargs":{"enable_thinking":false}}'), (
        "the timed workload samples by default, so plain-reference output "
        "cannot be compared reproducibly")
    for flag in ("--request-count", "--custom-dataset-type",
                 "--unsafe-override"):
        assert flag not in joined, (
            f"{flag} is a scenario lock violation; "
            f"--unsafe-override additionally stamps submission_valid=false")


def test_the_readme_quotes_the_frozen_corpus_too() -> None:
    """The agent reads the README; it must not be sent at the wrong workload."""
    body = (TASK / "README.md").read_text("utf-8")
    from final_validate.tasks import qwen3_5_9b_standard_2xl40 as module

    frozen = {"corpus": module.CORPUS}
    # The README is a facts card; the command lives in the task's
    # `instruction.md`, rendered from the verdict's own invocation. What the
    # README must not do is name a DIFFERENT corpus.
    assert frozen["corpus"] in body, (
        "the README does not name the frozen corpus, "
        "so an agent following it measures something else")
    assert "--public-dataset semianalysis_cc_traces_weka_062126_256k" not in body
