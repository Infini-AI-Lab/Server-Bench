"""``serving_evolve`` -- the harness library the evolve driver and tools share.

Standard library only, side-effect free, and safe to import in a bare
container. What is here:

``contracts``    frozen, versioned, canonically hashable values (the gate
                 contracts that `kernel_evolve` decides with, command receipts,
                 execution environments, workload identity)
``evaluator``    the gate suite and its golden cases, and the static integrity
                 scan the driver runs over every frozen tree
``environment``  the serving-framework denylist the offline wheelhouse is
                 audited against
``kernels``      kernel catalogs: research, validation and provisioning
``runtime``      the pinned baseline images (`image_for`)
``curve``        what a run did, per version, in one table
``cli``          `serverbench evolve <verb>`

The driver itself is `lifecycle` (sibling package); the final verdict is
`serverbench/harness/final_validate`.
"""

from __future__ import annotations

from .contracts.base import Contract
from .contracts.gates import (
    AcceptanceDecision,
    AcceptancePolicy,
    GateResult,
    GateSpec,
    GateStatus,
    ReasonCode,
    decide,
)
from .errors import ContractError, EvolveError

#: This package versions with the harness, not with whatever happens to import it.
__version__ = "0.1.0"

__all__ = [
    "AcceptanceDecision", "AcceptancePolicy", "Contract", "ContractError",
    "EvolveError", "GateResult", "GateSpec", "GateStatus", "ReasonCode",
    "__version__", "decide",
]
