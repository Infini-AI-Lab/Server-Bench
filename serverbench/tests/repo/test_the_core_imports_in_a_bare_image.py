"""The harness core must import where torch, transformers and yaml are not.

The harness core is stdlib-only by design: it must verify a candidate inside a
bare image. The reason is operational rather than tidy. The driver and the
verdict run inside images built for the CANDIDATE, and a candidate may pin,
break or omit any of those three. A module-scope `import yaml` in `lifecycle/`
turns that into a run that cannot start, hours after the cards were claimed.

This asks the question the way production asks it -- import the real modules
with those three names made unimportable -- rather than parsing ASTs, which
mirrors implementation structure and fails on valid refactors. A lazily
imported `yaml` inside a function still passes, which is exactly the
convention the rule asks for.

No GPU, no network. One subprocess, ~1 s.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SERVERBENCH = REPO / "serverbench"

#: What a bare image may not have. `yaml` is here for the same reason as the
#: other two: it is a third-party package, and the core reads YAML through
#: function-level imports precisely so that its absence is survivable.
FORBIDDEN = ("torch", "transformers", "yaml")

#: The modules a run actually imports before it can refuse anything: the
#: driver, the workspace builder it calls, and the contracts package both read.
CORE = ("lifecycle.run", "lifecycle.workspace", "lifecycle.instructions",
        "serving_evolve")

_PROBE = """
import importlib.abc, importlib.machinery, sys

FORBIDDEN = {forbidden!r}


class Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in FORBIDDEN:
            raise ImportError(f"{{name}} is not installed in this image")
        return None


sys.meta_path.insert(0, Absent())
for name in FORBIDDEN:
    sys.modules.pop(name, None)

for module in {core!r}:
    __import__(module)
print("bare-import-ok")
"""


@pytest.mark.parametrize("module", CORE)
def test_the_module_imports_with_no_third_party_package(module: str) -> None:
    got = subprocess.run(
        [sys.executable, "-c", _PROBE.format(forbidden=FORBIDDEN,
                                             core=(module,))],
        capture_output=True, text=True, timeout=120, cwd=str(REPO),
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPATH": f"{SERVERBENCH / 'harness'}:{SERVERBENCH}"})
    assert got.returncode == 0 and "bare-import-ok" in got.stdout, (
        f"`import {module}` needs a package a bare image may not carry, so a "
        f"run inside a candidate's image cannot start:\n"
        f"{(got.stderr or got.stdout)[-800:]}")


def test_the_probe_would_notice_a_module_scope_import() -> None:
    """The guard has to fail on the thing it is guarding against."""
    probe = _PROBE.replace('for module in', 'import yaml  # noqa: F401\nfor module in')
    got = subprocess.run(
        [sys.executable, "-c", probe.format(forbidden=FORBIDDEN, core=())],
        capture_output=True, text=True, timeout=120, cwd=str(REPO),
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPATH": f"{SERVERBENCH / 'harness'}:{SERVERBENCH}"})
    assert got.returncode != 0 and "not installed" in got.stderr, (
        "the blocker does not block, so the tests above prove nothing")
