"""The `serverbench` console script. One verb, delegated.

`serverbench evolve ...` hands the rest of the argument list to
`serving_evolve.cli` (`doctor`, `curve`, `golden`, `skill`, `kernels`).
Anything else prints usage and exits 2.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "evolve":
        # Delegated before any parser sees it. `argparse.REMAINDER` mis-handles
        # a leading option -- `serverbench evolve --help` would be reported as
        # an unrecognised argument rather than reaching the harness parser.
        from serving_evolve.cli import main as evolve_main

        return evolve_main(tokens[1:])

    print("usage: serverbench evolve <command> [...]\n"
          "\n"
          "`evolve` is the only command. Everything a run does is in\n"
          "`serverbench/harness/lifecycle/run.py`, and `serverbench/scripts/evolve`\n"
          "is the entrypoint that drives one.\n"
          "\n"
          "  serverbench evolve --help        every evolve verb\n"
          "  serverbench evolve doctor        python, imports, device inventory\n",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
