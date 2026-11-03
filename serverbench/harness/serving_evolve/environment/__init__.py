"""The environment a coding agent builds for itself, and the proof it is alone.

The agent is handed a Python runtime, a `uv` binary and a read-only wheelhouse —
not a working virtualenv. Building the environment is the first construction
milestone and part of the evolved artifact, because an environment the agent
inherited is an environment nobody can reconstruct.

The proof that matters is negative: no host `site-packages` on the path, no
inherited interpreter prefix, and no built general-purpose serving framework
reachable at any of eight layers. See :mod:`.denylist` for why eight, and for
the line between a *framework* (denied) and a *generic dependency or composable
kernel* (allowed, and needed).
"""

from .denylist import (
    ALLOWED_GENERIC,
    DENIED,
    AbsenceReport,
    DeniedFramework,
    LayerFinding,
    audit,
    audit_script,
)

__all__ = [
    "ALLOWED_GENERIC",
    "DENIED",
    "AbsenceReport",
    "DeniedFramework",
    "LayerFinding",
    "audit",
    "audit_script",
]
