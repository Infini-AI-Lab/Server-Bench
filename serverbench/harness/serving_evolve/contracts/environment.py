"""The interpreter and environment a gate is allowed to run in.

A gate that shells out to ``python3`` runs whatever ``PATH`` resolves, which
inside a container or a cron job can be a bare system interpreter with none of
the dependencies; the gate then fails and the failure is charged to the
candidate for the evaluator's misconfiguration.

So the interpreter is *declared*, not discovered. A bare name is refused. The
resolved absolute path, its version, the working directory, the argument vector
and the exit code all land in the receipt, and a missing or broken environment is
``INVALID`` — an evaluator failure with its own denominator — rather than a
candidate ``FAIL``.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path
from typing import ClassVar

from ..errors import ExternalBlocker
from .base import Contract

#: Names that must never be executed as written. Each resolves through ``PATH``,
#: which is precisely the thing this contract exists to stop depending on.
BARE_NAMES = frozenset({"python", "python3", "python3.10", "python3.11", "python3.12",
                        "python3.13", "pytest", "py.test", "pip", "pip3", "uv",
                        "nvcc", "gcc", "g++", "cc", "make", "ninja", "cmake"})


@dataclasses.dataclass(frozen=True)
class ExecutionEnvironment(Contract):
    """Where a command runs: which interpreter, which paths, which variables."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "execution_environment"

    environment_id: str = ""
    #: Absolute path to the interpreter. Never a bare name.
    interpreter: str = ""
    #: Directories prepended to ``PYTHONPATH`` for commands run here.
    python_paths: list[str] = dataclasses.field(default_factory=list)
    #: Variables set for every command.
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    #: Tools other than the interpreter this environment promises, by absolute
    #: path. ``resolve()`` refuses when one is missing.
    tools: dict[str, str] = dataclasses.field(default_factory=dict)
    description: str = ""

    def validate(self) -> None:
        self.require(bool(self.environment_id), "an execution environment needs an id")
        self.require(bool(self.interpreter),
                     "an execution environment must declare its interpreter; a gate "
                     "that resolves one from PATH charges the evaluator's "
                     "misconfiguration to the candidate")
        self.require(self.interpreter not in BARE_NAMES,
                     f"interpreter {self.interpreter!r} is a bare name that resolves "
                     f"through PATH; declare the absolute path")
        self.require(os.path.isabs(self.interpreter),
                     f"interpreter {self.interpreter!r} is not an absolute path")
        for name, path in self.tools.items():
            self.require(os.path.isabs(path),
                         f"tool {name!r} is declared as {path!r}, which is not absolute")

    # -- resolution ----------------------------------------------------------

    def resolve(self) -> dict[str, str]:
        """Prove the environment exists, and describe it.

        Raises :class:`ExternalBlocker` when it does not. That exception is what a
        caller turns into ``INVALID`` — the environment is missing, which is not
        something a candidate did.
        """
        interp = Path(self.interpreter)
        if not interp.exists():
            raise ExternalBlocker(
                f"declared interpreter {self.interpreter} does not exist",
                remedy=f"create the environment, or point {self.environment_id} at one "
                       f"that exists")
        if not os.access(interp, os.X_OK):
            raise ExternalBlocker(
                f"declared interpreter {self.interpreter} is not executable",
                remedy="fix its mode, or declare a different interpreter")
        try:
            out = subprocess.run([str(interp), "-V"], capture_output=True, timeout=30,
                                 text=True)
            version = (out.stdout or out.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - rare
            raise ExternalBlocker(
                f"declared interpreter {self.interpreter} could not be run: {exc}",
                remedy="repair or replace the environment") from exc
        missing = {n: p for n, p in self.tools.items() if not Path(p).exists()}
        if missing:
            raise ExternalBlocker(
                f"declared tool(s) missing: {sorted(missing)}",
                remedy="install them, or stop declaring them")
        return {"environment_id": self.environment_id,
                "interpreter": str(interp),
                # Both, because they differ and both matter: the path is what runs
                # and carries the virtualenv's identity, while the real path says
                # which base interpreter is underneath it.
                "real_path": str(interp.resolve()),
                "version": version,
                "python_paths": os.pathsep.join(self.python_paths)}

    def has_module(self, module: str) -> bool:
        """Whether the declared interpreter can import ``module``."""
        try:
            proc = subprocess.run([self.interpreter, "-c", f"import {module}"],
                                  capture_output=True, timeout=60,
                                  env=self.process_env())
        except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - rare
            return False
        return proc.returncode == 0

    # -- use -----------------------------------------------------------------

    def process_env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """The environment a command run here should be given."""
        environ = dict(os.environ if base is None else base)
        environ.update(self.env)
        if self.python_paths:
            existing = environ.get("PYTHONPATH", "")
            joined = os.pathsep.join([*self.python_paths, existing] if existing
                                     else self.python_paths)
            environ["PYTHONPATH"] = joined
        return environ

    def argv(self, *args: str) -> list[str]:
        """An argument vector that starts with the declared interpreter."""
        return [self.interpreter, *args]

    # -- construction --------------------------------------------------------

    @classmethod
    def from_interpreter(cls, interpreter: str | os.PathLike, *, environment_id: str,
                         python_paths: list[str] | None = None,
                         description: str = "") -> ExecutionEnvironment:
        """Pin an interpreter by absolute path, **without** resolving symlinks.

        Resolving is wrong here and the mistake is easy to make. A virtualenv's
        ``bin/python`` is a symlink to the base interpreter; following it produces
        a path that runs but has none of the virtualenv's packages, so a perfectly
        good environment reports its dependencies as missing. The absolute path is
        the identity; ``resolve()`` records the target separately.
        """
        return cls(environment_id=environment_id,
                   interpreter=os.path.abspath(os.fspath(interpreter)),
                   python_paths=list(python_paths or []),
                   description=description)

    @classmethod
    def current(cls, *, environment_id: str = "harness-current",
                python_paths: list[str] | None = None) -> ExecutionEnvironment:
        """The interpreter running this process.

        Correct for in-process harness use and for a container where the harness
        and the candidate share one environment. It is still *declared*: the
        resolved path is written down, so a later reader can see which
        interpreter ran rather than inferring it from the machine.
        """
        import sys

        return cls.from_interpreter(
            sys.executable, environment_id=environment_id,
            python_paths=list(python_paths or []),
            description="the interpreter running the harness process")
