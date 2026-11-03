"""Make the sibling helper modules importable by their bare names.

One rule follows from this: **a test directory must never be named after the
package it tests.** A `tests/kernel_evolve/` directory would shadow `kernel_evolve`
on `sys.path`, so every `from kernel_evolve.contracts import ...` would resolve to
the test package, and its modules would be reported as collection errors rather
than as failures. That is why the kernel tests live in `tests/kernels/`.

`tests/` is a package (it has `__init__.py`), so pytest puts the *repository*
directory on `sys.path`, not `tests/` itself. The test modules import their
helpers as `paths` / `qualification`, which only resolves if `tests/` is on
the path. Adding it here keeps those imports working without rewriting every
module, and without depending on the caller's cwd or `PYTHONPATH`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

# `harness/` is the one package root: `lifecycle`, `serving_evolve`,
# `kernel_evolve` and `final_validate` all import from it. `pyproject.toml`'s
# `pythonpath` puts it on the path for a pytest run from `serverbench/`; this
# keeps the same true for a run started anywhere else.
_HARNESS = _TESTS.parent / "harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, resolved from this file rather than from the cwd."""
    return _TESTS.parent.parent


@pytest.fixture(autouse=True)
def _never_touch_the_fleets_claims(tmp_path_factory, monkeypatch):
    """No test writes to the real card-claim directory.

    `/tmp/serverbench-card-claims` is what every scheduler on the host reads --
    `scripts/deploy/with-gpu`, `evolve-common.sh::gpu_is_held`,
    `lifecycle.workspace.Ownership` and `final_validate.lease`. A claim written
    there by a test is an action on the live machine: `pick_gpus` consults
    claims, so a stray one makes a card that belongs to someone else read as
    held by this harness.

    Autouse and suite-wide on purpose: naming the tests that need it would
    protect only the ones already written. Redirecting the directory protects
    the ones nobody has written yet.
    """
    claims = tmp_path_factory.mktemp("card-claims")
    monkeypatch.setenv("SERVERBENCH_CARD_CLAIMS", str(claims))
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path_factory.mktemp("gpu-locks")))
    for module, attribute in (("lifecycle.workspace", "Ownership"),):
        try:
            import importlib

            owner = getattr(importlib.import_module(module), attribute)
        except Exception:                                       # noqa: BLE001
            continue
        monkeypatch.setattr(owner, "CLAIMS", claims, raising=False)
    try:
        from final_validate import lease as lease_mod

        monkeypatch.setattr(lease_mod, "CLAIMS", claims)
        monkeypatch.setattr(lease_mod, "LOCKS",
                            Path(os.environ["GPU_LOCK_DIR"]))
    except Exception:                                           # noqa: BLE001
        pass
