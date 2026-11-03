"""Command receipts: proof that something ran, and what it saw.

A receipt records the exact argv, the working directory, a digest of the
environment the process was given, the exit code, the wall time, and digests of
both output streams. Output is hashed rather than quoted, so a receipt cannot
leak a token that happened to be printed; the environment digest redacts the
values of secret-looking variables.

A `CommandRunner` may be given a content-addressed store (`put_bytes`,
`get_artifact`, `append`, `events`) to keep full output; without one it keeps
digests only.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from serving_evolve.contracts.run import ArtifactRef, CommandReceipt
from serving_evolve.hashing import digest

COMMAND_RAN = "COMMAND_RAN"

#: Environment variable names that mark a secret. The names stay in the
#: digest, so a changed set is detectable; the values never do.
SECRET_HINTS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE", "OAUTH")


def environment_digest(env: Mapping[str, str]) -> str:
    """Digest names and non-secret values. Never a secret value."""
    safe = {k: ("<redacted>" if any(h in k.upper() for h in SECRET_HINTS) else v)
            for k, v in env.items()}
    return digest(safe, semantic=False)


class CommandRunner:
    """Runs commands and records receipts."""

    def __init__(self, ledger: Any = None) -> None:
        self.ledger = ledger

    def run(self, argv: Sequence[str], *, cwd: os.PathLike | str | None = None,
            env: dict[str, str] | None = None, timeout_s: float | None = None,
            keep_output: bool = False, label: str = "") -> CommandReceipt:
        environ = dict(os.environ if env is None else env)
        started_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(list(argv), cwd=str(cwd) if cwd else None, env=environ,
                                  capture_output=True, timeout=timeout_s)
            out, err, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            out = exc.stdout or b""
            err = exc.stderr or b""
            code = -1

        receipt = CommandReceipt(
            argv=list(argv), cwd=str(cwd or Path.cwd()), exit_code=code,
            wall_seconds=round(time.monotonic() - t0, 6),
            stdout_sha256=hashlib.sha256(out).hexdigest(),
            stderr_sha256=hashlib.sha256(err).hexdigest(),
            stdout_bytes=len(out), stderr_bytes=len(err),
            env_digest=environment_digest(environ),
            started_utc=started_wall, timed_out=timed_out,
        )
        if keep_output and self.ledger is not None:
            receipt = receipt.evolve(
                stdout_ref=ArtifactRef(sha256=self.ledger.put_bytes(out), bytes=len(out),
                                       media_type="text/plain", label=f"{label}:stdout"),
                stderr_ref=ArtifactRef(sha256=self.ledger.put_bytes(err), bytes=len(err),
                                       media_type="text/plain", label=f"{label}:stderr"))
        if self.ledger is not None:
            self.ledger.append(COMMAND_RAN, {"label": label, "receipt": receipt.to_dict()})
        return receipt

    def receipts(self) -> list:
        if self.ledger is None:
            return []
        return [CommandReceipt.from_dict(e["payload"]["receipt"])
                for e in self.ledger.events(COMMAND_RAN)]
