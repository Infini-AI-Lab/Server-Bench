"""A delivered skill bundle carries no upstream serving-framework source.

A kernel wiki reasonably stores the key files from the upstream pull requests
it documents, and several of those are against SGLang, so the tree contains
SGLang's own kernel implementations. Handing an agent those is handing it
somebody else's answer, which is the one thing the offline task exists to
prevent. So every delivered skill bundle goes through `_copy_skill`, which
drops upstream source and records what it withheld.

The skill bundles are vendored submodules; on a host where they are not
initialised they are empty directories and the copy has nothing to carry, so
the filter's call site is asserted directly rather than observed.
"""

from __future__ import annotations

import sys

import pytest
from paths import SERVERBENCH

sys.path.insert(0, str(SERVERBENCH / "harness"))

from lifecycle.delivery import UPSTREAM_MARKERS, _copy_skill  # noqa: E402

#: One real line from each framework the offline task must not hand over.
UPSTREAM_SOURCE = {
    "sglang_attention.py": "from sglang.srt.layers.attention import AttentionBackend\n",
    "vllm_paged.py": "import vllm\nfrom vllm.attention import PagedAttention\n",
}


def _bundle(tmp_path):
    """A vendored skill checkout: some curated notes, some upstream source."""
    src = tmp_path / "KernelWiki"
    (src / "references").mkdir(parents=True)
    (src / "SKILL.md").write_text("# how to think about attention\n", "utf-8")
    (src / "references" / "notes.md").write_text(
        "Persistent batching keeps the GPU busy between steps.\n", "utf-8")
    for name, body in UPSTREAM_SOURCE.items():
        (src / "references" / name).write_text(body, "utf-8")
    return src


def test_the_markers_name_the_frameworks_this_benchmark_competes_with():
    """The filter is only as good as its list; the list must not be empty."""
    joined = " ".join(UPSTREAM_MARKERS).lower()
    for framework in ("sglang", "vllm"):
        assert framework in joined, (
            f"{framework} is a comparison target and its source must never "
            f"reach a workspace; the marker list does not mention it"
        )


def test_upstream_source_is_dropped_and_the_rest_is_delivered(tmp_path):
    src = _bundle(tmp_path)
    delivered, dropped = _copy_skill(src, tmp_path / "out", tmp_path)

    landed = {p.rsplit("/", 1)[-1] for p in delivered}
    assert "SKILL.md" in landed and "notes.md" in landed, (
        "the curated half of the bundle is the point of delivering it at all"
    )
    for name in UPSTREAM_SOURCE:
        assert name not in landed, f"{name} carries upstream source and was delivered"
        assert not (tmp_path / "out" / "references" / name).exists()
    assert len(dropped) == len(UPSTREAM_SOURCE), (
        f"dropped {dropped}, expected one entry per upstream file so the "
        f"manifest can say what was withheld and why"
    )


@pytest.mark.parametrize("key", ["serving_skill", "kernel_skill"])
def test_both_delivered_bundles_go_through_the_filter(key):
    """A source grep is the wrong check; this asserts the call site by name.

    The skills go through `_copy_skill` in one loop; the unabridged
    `_copy_tree` is for KernelWiki alone. The call site is what is pinned,
    because a host whose submodules are empty cannot observe the difference.
    """
    source = (SERVERBENCH / "harness" / "lifecycle" /
              "delivery.py").read_text("utf-8")
    loop = source[source.index("for source in (sources.serving_skill"):]
    loop = loop[:loop.index("sources.kernelwiki_skill is not None")]
    assert f"sources.{key}" in loop and "_copy_skill(" in loop, (
        f"sources.{key} is delivered with the unfiltered copy. Every skill "
        f"bundle is a vendored checkout that nobody curated for this "
        f"benchmark, and one of them stores SGLang's own kernels."
    )
    assert "_copy_tree(" not in loop


def test_an_empty_bundle_delivers_nothing_and_withholds_nothing(tmp_path):
    """An uninitialised submodule: present as an empty directory."""
    src = tmp_path / "ncu-report-skill"
    src.mkdir()
    delivered, dropped = _copy_skill(src, tmp_path / "out", tmp_path)
    assert not delivered and not dropped, (
        "an uninitialised submodule is absent, not withheld, and must not "
        "appear in the manifest as an ablation that never happened"
    )


def test_standard_ncu_overlay_is_delivered_and_hashed_without_editing_source(tmp_path, monkeypatch):
    from lifecycle import delivery
    from serving_evolve.hashing import digest_file

    src = tmp_path / "source" / "ncu-report-skill"
    (src / "reference").mkdir(parents=True)
    (src / "SKILL.md").write_text("# NCU report skill\n")
    reference = src / "reference/09-common-issues.md"
    reference.write_text("upstream clock advice\n")
    monkeypatch.setattr(delivery, "NCU_SKILL", src)
    workspace = tmp_path / "workspace"
    result = delivery.deliver(workspace, delivery.Sources(ncu_report_skill=src))
    relative = ".claude/skills/ncu-report-skill/reference/09-common-issues.md"
    landed = workspace / relative
    assert "./gpu-clock check" in landed.read_text()
    assert "sudo nvidia-smi" not in landed.read_text()
    assert result.mounted[relative] == digest_file(landed)
    assert reference.read_text() == "upstream clock advice\n"


def test_custom_ncu_bundle_is_not_overlaid(tmp_path):
    from lifecycle import delivery

    src = tmp_path / "ncu-report-skill"
    (src / "reference").mkdir(parents=True)
    (src / "SKILL.md").write_text("# Custom NCU skill\n")
    (src / "reference/09-common-issues.md").write_text("custom guidance\n")
    workspace = tmp_path / "workspace"
    delivery.deliver(workspace, delivery.Sources(ncu_report_skill=src))
    delivered = workspace / ".claude/skills/ncu-report-skill/reference/09-common-issues.md"
    assert delivered.read_text() == "custom guidance\n"


def test_uninitialized_ncu_does_not_receive_an_orphan_overlay(tmp_path, monkeypatch):
    from lifecycle import delivery

    src = tmp_path / "ncu-report-skill"
    src.mkdir()
    monkeypatch.setattr(delivery, "NCU_SKILL", src)
    workspace = tmp_path / "workspace"
    result = delivery.deliver(workspace, delivery.Sources(ncu_report_skill=src))
    assert not any("ncu-report-skill" in name for name in result.mounted)
