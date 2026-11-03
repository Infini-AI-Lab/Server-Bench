"""A task's guidance profile (standard or instructed), read from its id."""
import re


def profile_of(task: str, requested: str | None = None) -> str:
    # A scoring variant may sit between the guidance profile and hardware.
    # It still receives exactly the profile named before it.
    found = re.search(
        r"_(standard|instructed)(?:_modal)?_"
        r"\d+x[A-Za-z0-9-]+$", task)
    if not found:
        raise ValueError("task id must carry _standard_ or _instructed_ before hardware")
    profile = found.group(1)
    if requested is not None and requested != profile:
        raise ValueError(f"profile {requested!r} disagrees with task id {task!r}")
    return profile


