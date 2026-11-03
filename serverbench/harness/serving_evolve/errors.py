"""Structured domain errors for the evolution harness.

Every failure a caller might reasonably branch on gets its own type and a stable
``reason_code``. Callers match on the code, never on message text, so wording can
improve without breaking automation.

Nothing here catches broadly. An error that reaches the CLI is reported with its
code and a non-zero exit status chosen by :mod:`serving_evolve.cli`; an error
nobody anticipated is allowed to propagate rather than be flattened into a
misleading "failed" result.
"""

from __future__ import annotations

from typing import Any


class EvolveError(Exception):
    """Base class. Carries a stable machine-readable code and free-form detail."""

    reason_code = "evolve_error"

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail

    def as_dict(self) -> dict[str, Any]:
        return {"reason_code": self.reason_code, "message": self.message, "detail": self.detail}


# --- contract / schema -------------------------------------------------------

class ContractError(EvolveError):
    """A contract object is malformed, or its schema version is unsupported."""

    reason_code = "contract_invalid"


class SchemaVersionError(ContractError):
    """The serialized form was written by an incompatible schema version."""

    reason_code = "schema_version_unsupported"

    def __init__(self, kind: str, found: int, supported: int) -> None:
        super().__init__(
            f"{kind} schema version {found} is not supported (this build understands {supported})",
            kind=kind, found=found, supported=supported,
        )


# --- lifecycle ---------------------------------------------------------------

class IllegalTransition(EvolveError):
    """A candidate was pushed into a state it cannot reach from where it is."""

    reason_code = "illegal_transition"

    def __init__(self, frm: str, to: str, allowed: Any) -> None:
        super().__init__(f"{frm} -> {to} is not a legal transition "
                         f"(legal: {', '.join(sorted(allowed)) or 'none, terminal'})",
                         **{"from": frm, "to": to, "allowed": sorted(allowed)})


# --- external ----------------------------------------------------------------

class ExternalBlocker(EvolveError):
    """Progress needs something outside the harness: a credential, a service, a decision."""

    reason_code = "external_blocker"

    def __init__(self, message: str, resume_command: str | None = None, **detail: Any) -> None:
        super().__init__(message, resume_command=resume_command, **detail)
        self.resume_command = resume_command

