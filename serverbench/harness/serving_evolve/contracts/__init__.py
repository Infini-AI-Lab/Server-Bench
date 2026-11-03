"""Versioned, frozen, canonically hashable contracts.

Importing this package must stay free of side effects: no model load, no CUDA
init, no thread, no filesystem write. It is types and validation only.
"""

from __future__ import annotations

from .base import Contract
from .environment import ExecutionEnvironment

__all__ = ["Contract", "ExecutionEnvironment"]
