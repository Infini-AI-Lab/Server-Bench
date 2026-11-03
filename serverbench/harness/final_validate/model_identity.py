"""Is the model on the card the model the task froze?

LIVES IN `final_validate/`, NOT IN `lifecycle/`, ON PURPOSE.
------------------------------------------------------------
Both the driver and the verdict have to ask this question, and
`tests/integrity/test_the_trajectory_never_reaches_the_verdict.py` forbids any
`final_validate` module from importing `lifecycle` -- the rule that keeps the
verdict from ever reading the agent's own measurements. So the shared check
lives on the verdict's side of that line and the driver imports it, rather than
the other way round or the logic existing twice.

WHY THIS CHECK REFUSES
----------------------
Every task serves its own model, and a model directory staged for one task is
easy to mount into another. A wrong-model run is the most expensive failure
this harness has, because it does not look like a failure: the agent builds
against the brief, the server runs, and it produces a full curve of results, a
plausible tokens-per-dollar curve, and a verdict -- against gates calibrated
for a model that never loaded. So this check compares the mounted weights with
the task's declared `MODEL_ID` and `MODEL_REVISION` BEFORE the container
starts, and it REFUSES rather than warns.

HOW IDENTITY IS ESTABLISHED
---------------------------
Two independent routes, strongest first:

1. **The cache layout.** A Hugging Face cache encodes the repository in the
   directory name (`models--Qwen--Qwen3.5-9B`) and the revision in the
   snapshot directory. When the mounted path is a snapshot, identity is read
   rather than inferred, and a pinned `revision` can be checked exactly.

2. **The architecture fingerprint.** A staged flat copy has no repo id, so the
   declared model is resolved to a local snapshot and the two `config.json`s
   are compared on the fields that decide what a serving system must build:
   `model_type`, `architectures`, depth, width, and the attention layer plan.

If neither route can establish identity the run is refused too. "I could not
tell" is not "they match".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The config fields that decide what a serving system has to build. Two models
#: agreeing on all of these are interchangeable for this harness's purposes;
#: two differing on any are not.
FINGERPRINT_FIELDS = (
    "model_type", "architectures", "num_hidden_layers", "hidden_size",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "vocab_size", "layer_types",
)

#: Where to look for a local snapshot of a declared repo id, beyond the caller's
#: own hint. Derived from the environment, never a path copied from a document.
def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ("HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        roots += [path, path / "hub"]
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    out: list[Path] = []
    for root in roots:
        if root.is_dir() and root not in out:
            out.append(root)
    return out


class ModelMismatch(RuntimeError):
    """The mounted weights are not the model the task froze."""


@dataclass
class Identity:
    """What the bytes at a path actually are."""

    path: Path
    repo_id: str | None = None
    revision: str | None = None
    fingerprint: dict[str, Any] = field(default_factory=dict)
    problem: str | None = None

    def describe(self) -> str:
        who = self.repo_id or "an unidentified model"
        if self.revision:
            who += f"@{self.revision[:12]}"
        kind = self.fingerprint.get("model_type")
        layers = self.fingerprint.get("num_hidden_layers")
        extra = f" (model_type={kind!r}, {layers} layers)" if kind else ""
        return f"{who}{extra}"


def _read_config(model_dir: Path) -> dict[str, Any] | None:
    body = model_dir / "config.json"
    if not body.is_file():
        return None
    try:
        return json.loads(body.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def _fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    """The architectural fields, flattened across a possible `text_config`.

    Multimodal configs (Qwen3-VL, Qwen3.5) put the language fields one level
    down. Reading only the top level makes two different models look equally
    blank, which is the same failure as not checking at all.
    """
    merged: dict[str, Any] = {}
    for source in (config.get("text_config") or {}, config):
        if not isinstance(source, dict):
            continue
        for key in FINGERPRINT_FIELDS:
            if key in source and key not in merged:
                merged[key] = source[key]
    layers = merged.get("layer_types")
    if isinstance(layers, list):
        # The ORDER of a layer plan is an implementation detail; the census is
        # not -- 24 linear + 8 full is a different machine from 36 dense.
        counts: dict[str, int] = {}
        for item in layers:
            counts[str(item)] = counts.get(str(item), 0) + 1
        merged["layer_types"] = counts
    return merged


def _from_cache_layout(model_dir: Path) -> tuple[str | None, str | None]:
    """`(repo_id, revision)` when the path is inside a Hugging Face cache."""
    resolved = model_dir.resolve()
    parts = resolved.parts
    for index, part in enumerate(parts):
        if not part.startswith("models--"):
            continue
        repo = part[len("models--"):].replace("--", "/")
        revision = None
        rest = parts[index + 1:]
        if len(rest) >= 2 and rest[0] == "snapshots":
            revision = rest[1]
        return repo, revision
    return None, None


def identify(model_dir: Path) -> Identity:
    """What is actually staged at `model_dir`."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return Identity(path=model_dir,
                        problem=f"{model_dir} is not a directory")
    config = _read_config(model_dir)
    if config is None:
        return Identity(
            path=model_dir,
            problem=f"{model_dir}/config.json is missing or unreadable, so "
                    f"what is staged there cannot be established")
    repo_id, revision = _from_cache_layout(model_dir)
    return Identity(path=model_dir, repo_id=repo_id, revision=revision,
                    fingerprint=_fingerprint(config))


def local_snapshot_of(repo_id: str) -> Path | None:
    """A local Hugging Face snapshot of `repo_id`, or None.

    MATCHED CASE-INSENSITIVELY, because a cache directory records the casing
    the DOWNLOAD used and a task records the casing the Hub publishes, and the
    two can differ (`google/gemma-4-E4B-it` against
    `models--google--gemma-4-e4b-it`) while naming the same model.

    Hugging Face resolves repo ids case-insensitively, so this does too. An
    EXACT match is still preferred when both exist, so a deliberate
    distinction -- if the Hub ever published two ids differing only in case --
    is never silently collapsed.
    """
    wanted = "models--" + repo_id.replace("/", "--")
    fallback: Path | None = None
    for root in _hf_cache_roots():
        if not root.is_dir():
            continue
        for repo in sorted(root.iterdir()):
            if not repo.is_dir() or not repo.name.startswith("models--"):
                continue
            exact = repo.name == wanted
            if not exact and repo.name.lower() != wanted.lower():
                continue
            snapshots = repo / "snapshots"
            if not snapshots.is_dir():
                continue
            for entry in sorted(snapshots.iterdir()):
                if (entry / "config.json").exists():
                    if exact:
                        return entry
                    fallback = fallback or entry
                    break
    return fallback


#: WHAT THIS CHECKS: identity, not bytes. Route 1 below refuses a snapshot
#: whose repository or REVISION is not the pinned one, and route 2 refuses one
#: whose architecture fingerprint differs from a local copy of the declared
#: model. A re-tagged snapshot fails route 1, because a revision is a commit.
#:
#: What it cannot detect: a snapshot that carries the right repository and
#: revision in its path while its FILES have been edited. A task carries no
#: per-file digests.

def check(*, model_dir: Path, declared_model_id: str,
          declared_revision: str = "", task_dir=None) -> None:
    """Raise `ModelMismatch` unless `model_dir` holds the declared model.

    Silence means the weights on the card are the weights the task froze.
    """
    declared_model_id = (declared_model_id or "").strip()
    if not declared_model_id:
        # A task that declares no model cannot be contradicted by one. Tasks
        # under development legitimately reach this; a released one should not,
        # and `tests/` asserts that separately.
        return

    found = identify(Path(model_dir))
    if found.problem:
        raise ModelMismatch(
            f"the task declares {declared_model_id!r} and {found.problem}. "
            f"Refusing rather than guessing: a run against unidentified "
            f"weights produces a full set of results, and a verdict, that "
            f"answer a question nobody asked")

    # -- route 1: the cache layout names the repository outright -------------
    if found.repo_id:
        if found.repo_id.lower() != declared_model_id.lower():
            raise ModelMismatch(
                f"the task declares {declared_model_id!r} and {model_dir} "
                f"holds {found.describe()}. Pass --model-dir for this task, or "
                f"stage the declared model; do NOT run and re-label")
        if declared_revision and found.revision \
                and found.revision != declared_revision:
            raise ModelMismatch(
                f"the task pins {declared_model_id}@{declared_revision} and "
                f"{model_dir} is revision {found.revision}. A revision is part "
                f"of a frozen identity: different weights under one name are "
                f"exactly what a pin exists to prevent")
        return

    # -- route 2: compare architecture against a local copy of the declared --
    reference = local_snapshot_of(declared_model_id)
    if reference is None:
        raise ModelMismatch(
            f"{model_dir} carries no repository identity (it is a flat copy, "
            f"not a Hugging Face snapshot) and no local snapshot of the "
            f"declared {declared_model_id!r} was found to compare it against, "
            f"so the two cannot be shown to be the same model. Stage the "
            f"declared model, or point --model-dir at a cache snapshot whose "
            f"path names the repository")
    want = _fingerprint(_read_config(reference) or {})
    got = found.fingerprint
    differing = {key: (want.get(key), got.get(key))
                 for key in FINGERPRINT_FIELDS
                 if key in want and want.get(key) != got.get(key)}
    if differing:
        detail = "; ".join(
            f"{key}: declared {declared!r} vs staged {staged!r}"
            for key, (declared, staged) in sorted(differing.items()))
        raise ModelMismatch(
            f"the task declares {declared_model_id!r} and {model_dir} holds a "
            f"different architecture -- {detail}. A serving system built for "
            f"one of these is not measurable on the other")


def declared_for(task_dir) -> tuple[str, str]:
    """The task's frozen `(model_id, revision)`, or `("", "")`.

    Read from the task's own Python module -- `MODEL_ID` and
    `MODEL_REVISION` -- the one place a task's identity lives, so the pin
    cannot disagree with itself.
    """
    try:
        from final_validate import tasks as task_modules
        module = task_modules.python_module(Path(task_dir).name)
        return (str(getattr(module, "MODEL_ID", "") or ""),
                str(getattr(module, "MODEL_REVISION", "") or ""))
    except Exception:                                          # noqa: BLE001
        return ("", "")
