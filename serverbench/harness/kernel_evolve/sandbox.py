"""Building a kernel candidate, in a process that is not the harness.

Three things make this its own module rather than a subprocess call inline.

**A build failure is ambiguous.** "nvcc exited 1" is a candidate failure when the
candidate's CUDA does not compile, and an evaluator failure when nvcc is not
installed, the disk is full, or the build cache is corrupt. The two are different
denominators and only one of them is the candidate's fault, so the classifier
here is the point of the module.

**A build must not be able to see the harness.** A candidate that imports the
evaluator can read the answer, and a candidate that mutates it can rewrite the
question. The build runs with a declared environment and a scrubbed
``PYTHONPATH``, not with whatever the parent process happened to have.

**Compilation is where a candidate escapes the sandbox if it is going to.** Build
scripts run arbitrary code by design. The scope check refuses a manifest whose
sources point outside the candidate root before anything is executed.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
from pathlib import Path
from typing import ClassVar

from serving_evolve.contracts.base import Contract
from serving_evolve.contracts.environment import ExecutionEnvironment
from serving_evolve.contracts.gates import GateResult, ReasonCode
from serving_evolve.contracts.run import CommandReceipt
from serving_evolve.hashing import digest_file

from .commands import CommandRunner
from .contracts.candidate import BuildManifest
from .verification import BUILD, MANIFEST

#: Patterns in build output that mean the *toolchain* failed, not the code. Each
#: was chosen because it is produced by the environment and never by a candidate's
#: own source being wrong.
INFRASTRUCTURE_PATTERNS = (
    r"No such file or directory:\s*'?nvcc",
    r"command not found",
    r"No space left on device",
    r"Cannot allocate memory",
    r"unable to execute .*: No such file",
    r"CUDA driver version is insufficient",
    r"failed to open cache",
    r"Permission denied",
    r"Read-only file system",
)


@dataclasses.dataclass(frozen=True)
class BuildSandboxSpec(Contract):
    """Where and how a candidate is built."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "kernel_build_sandbox"

    sandbox_id: str = ""
    environment: ExecutionEnvironment | None = None
    #: Absolute paths the build may read outside its own root. Kept explicit and
    #: small: a build that can read the evaluator can read the answer.
    readable_roots: list[str] = dataclasses.field(default_factory=list)
    #: Whether the parent's PYTHONPATH is scrubbed. Almost always yes — inheriting
    #: it is how a candidate silently imports the harness.
    scrub_python_path: bool = True
    #: Cache directory for compiled extensions. Isolated per candidate, because a
    #: shared cache is how candidate N is measured running candidate N-1's binary.
    build_cache_dir: str = ""

    def validate(self) -> None:
        self.require(bool(self.sandbox_id), "a build sandbox needs an id")
        self.require(self.environment is not None,
                     "a build sandbox needs a declared environment; resolving a "
                     "compiler from PATH at build time is how a candidate gets "
                     "measured against a toolchain nobody recorded")

    def process_env(self) -> dict[str, str]:
        assert self.environment is not None
        base = dict(os.environ)
        if self.scrub_python_path:
            base.pop("PYTHONPATH", None)
        env = self.environment.process_env(base)
        if self.build_cache_dir:
            env["TORCH_EXTENSIONS_DIR"] = self.build_cache_dir
        return env


def classify_output(text: str) -> ReasonCode:
    """Decide whether build output describes a broken toolchain or broken code."""
    for pattern in INFRASTRUCTURE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return ReasonCode.INFRASTRUCTURE_ERROR
    return ReasonCode.BUILD_FAILED


def check_scope(manifest: BuildManifest, candidate_root: Path,
                permitted_extra: list[str] | None = None) -> list[str]:
    """Sources that resolve outside the candidate root. Empty is the good answer."""
    root = Path(candidate_root).resolve()
    allowed = [root, *(Path(p).resolve() for p in (permitted_extra or []))]
    outside: list[str] = []
    for source in manifest.sources:
        resolved = (root / source.path).resolve()
        if not any(resolved == a or a in resolved.parents for a in allowed):
            outside.append(source.path)
    return outside


def verify_manifest(manifest: BuildManifest, candidate_root: Path,
                    permitted_extra: list[str] | None = None) -> GateResult:
    """Rung 1: the declared source exists, hashes as declared, and stays in scope."""
    root = Path(candidate_root)
    outside = check_scope(manifest, root, permitted_extra)
    if outside:
        return GateResult.failed(
            MANIFEST, ReasonCode.PROTECTED_PATH_TOUCHED,
            f"source path(s) {outside} resolve outside the candidate root; a build that "
            f"reaches outside its own tree is not the thing being evaluated",
            observed=len(manifest.sources), failures=len(outside))

    missing: list[str] = []
    mismatched: list[str] = []
    for source in manifest.sources:
        path = root / source.path
        if not path.exists():
            missing.append(source.path)
            continue
        if digest_file(path) != source.digest:
            mismatched.append(source.path)

    if missing:
        return GateResult.invalid(
            MANIFEST, ReasonCode.FIXTURE_STALE,
            f"declared source file(s) {missing} are not present; the manifest "
            f"describes a candidate that is not here")
    if mismatched:
        return GateResult.failed(
            MANIFEST, ReasonCode.EVALUATOR_TAMPERED,
            f"source file(s) {mismatched} do not hash to their declared digest; what "
            f"would be built is not what was frozen",
            observed=len(manifest.sources), failures=len(mismatched))
    if not manifest.has_device_source:
        return GateResult.failed(
            MANIFEST, ReasonCode.HIDDEN_FALLBACK,
            "the manifest declares no device-language source, so whatever this "
            "candidate does it is not evolving a kernel",
            observed=len(manifest.sources), failures=1)
    return GateResult.passed(
        MANIFEST, observed=len(manifest.sources),
        message=f"{len(manifest.sources)} source file(s) present and matching")


def build(manifest: BuildManifest, spec: BuildSandboxSpec, candidate_root: Path,
          runner: CommandRunner | None = None) -> tuple[GateResult, CommandReceipt | None]:
    """Rung 2: compile and import the candidate in a separate process.

    Import happens in a child process rather than here for a plain reason: a CUDA
    extension that fails to load can take the process with it, and losing the
    harness to a candidate's build is an evaluator failure that also loses the
    evidence explaining it.
    """
    assert spec.environment is not None
    runner = runner or CommandRunner()

    try:
        identity = spec.environment.resolve()
    except Exception as exc:  # noqa: BLE001 - a missing environment is INVALID
        return GateResult.invalid(
            BUILD, ReasonCode.INFRASTRUCTURE_ERROR,
            f"the declared build environment is unusable ({exc}); this is an evaluator "
            f"failure and is not charged to the candidate"), None

    if spec.build_cache_dir:
        Path(spec.build_cache_dir).mkdir(parents=True, exist_ok=True)

    probe = (f"import importlib,sys;"
             f"m=importlib.import_module({manifest.entry_module!r});"
             f"fn=getattr(m,{manifest.entry_callable!r},None);"
             f"sys.exit(0 if callable(fn) else 3)")
    receipt = runner.run(spec.environment.argv("-c", probe),
                         cwd=candidate_root, env=spec.process_env(),
                         timeout_s=manifest.timeout_seconds, keep_output=True,
                         label="kernel_build")

    detail = {"interpreter": identity["interpreter"], "version": identity["version"],
              "entry_module": manifest.entry_module,
              "entry_callable": manifest.entry_callable,
              "exit_code": receipt.exit_code}

    if receipt.timed_out:
        return GateResult.invalid(
            BUILD, ReasonCode.TIMEOUT,
            f"the build did not finish within {manifest.timeout_seconds:.0f}s",
            detail=detail), receipt
    if receipt.exit_code == 3:
        return GateResult.failed(
            BUILD, ReasonCode.BUILD_FAILED,
            f"{manifest.entry_module} imported but exposes no callable "
            f"{manifest.entry_callable!r}", detail=detail, failures=1), receipt
    if receipt.exit_code != 0:
        text = _output_text(receipt, runner)
        reason = classify_output(text)
        message = (f"the build environment failed (exit {receipt.exit_code}); this is "
                   f"an evaluator failure, not a candidate failure"
                   if reason is ReasonCode.INFRASTRUCTURE_ERROR
                   else f"the candidate does not build (exit {receipt.exit_code})")
        maker = (GateResult.invalid if reason is ReasonCode.INFRASTRUCTURE_ERROR
                 else GateResult.failed)
        return maker(BUILD, reason, message, detail=detail), receipt

    return GateResult.passed(BUILD, observed=1,
                             message=f"built and imported {manifest.entry_module}",
                             detail=detail), receipt


def _output_text(receipt: CommandReceipt, runner: CommandRunner) -> str:
    """Best-effort text of a receipt's output, for classification only.

    Output is stored by digest rather than inline, so this resolves it back out of
    the content-addressed store. If the store is not available the classifier
    simply has less to go on and errs toward calling the failure the candidate's —
    which is the safe direction: a candidate wrongly charged is visible in its
    result, an evaluator failure wrongly excused is not.
    """
    if runner.ledger is None:
        return ""
    parts: list[str] = []
    for ref in (receipt.stderr_ref, receipt.stdout_ref):
        if ref is None or not ref.sha256:
            continue
        try:
            parts.append(runner.ledger.get_artifact(ref.sha256).read_text(
                "utf-8", errors="replace"))
        except (OSError, ValueError):  # pragma: no cover - store missing or corrupt
            continue
    return "\n".join(parts)


def isolated_cache(root: Path, candidate_id: str) -> str:
    """A per-candidate build cache path, created fresh.

    Fresh because a stale extension in a shared cache is loaded in preference to
    rebuilding, and the result is candidate N measured running candidate N-1's
    binary — a correct-looking, entirely fictitious result.
    """
    path = Path(root) / "build-cache" / candidate_id
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return str(path)
