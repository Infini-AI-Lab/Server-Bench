"""Whose fault a failure is.

Every failure the driver journals carries a `classification`, because a
harness fault charged to a candidate -- or a candidate fault excused as the
harness's -- makes a run's record say something that did not happen.
"""

from __future__ import annotations

import enum


class FailureClass(str, enum.Enum):
    HARNESS = "HARNESS"
    CANDIDATE = "CANDIDATE"
    EXTERNAL = "EXTERNAL"
