"""What a command produced: content-addressed artifact references and receipts.

An :class:`ArtifactRef` names bytes in the content-addressed store by digest; a
:class:`CommandReceipt` records what a command ran, where, how it exited and
the digests of what it printed.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from .base import Contract


@dataclasses.dataclass(frozen=True)
class ArtifactRef(Contract):
    """A pointer into the content-addressed store.

    Path is a convenience; ``sha256`` is the identity. A reference that resolves
    to different bytes than its digest is a corruption, not a newer version.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "artifact_ref"

    sha256: str = ""
    bytes: int = 0
    media_type: str = "application/json"
    original_path: str = ""
    label: str = ""

    def validate(self) -> None:
        self.require(len(self.sha256) == 64,
                     "an artifact reference is a full sha256, never a prefix",
                     sha256=self.sha256)


@dataclasses.dataclass(frozen=True)
class CommandReceipt(Contract):
    """Proof that a command ran, with what it saw and what it produced."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "command_receipt"

    argv: list[str] = dataclasses.field(default_factory=list)
    cwd: str = ""
    exit_code: int = 0
    wall_seconds: float = 0.0
    stdout_sha256: str = ""
    stderr_sha256: str = ""
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    env_digest: str = ""
    environment_digest: str = ""
    started_utc: str = ""
    timed_out: bool = False
    #: Present only when the caller asked to keep the output.
    stdout_ref: ArtifactRef | None = None
    stderr_ref: ArtifactRef | None = None

    def validate(self) -> None:
        self.require(bool(self.argv), "a receipt must record what was run")

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out
