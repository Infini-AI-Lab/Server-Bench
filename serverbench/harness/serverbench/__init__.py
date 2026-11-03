"""``serverbench`` -- the console script, and nothing else.

`pip install -e serverbench` installs a `serverbench` command whose one verb,
``serverbench evolve <verb>``, is `serving_evolve.cli` (``doctor``, ``curve``,
``golden``, ``skill``, ``kernels``). An evolve run itself is started by
``serverbench/scripts/evolve``, not from here.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
