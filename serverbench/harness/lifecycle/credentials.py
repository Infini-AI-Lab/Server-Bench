"""The operator's token, at an inode this run owns.

## The failure this module exists for

The agent's credentials reach its container as a bind of ONE FILE. A docker
bind of a file resolves to that file's **inode** when the container starts and
holds it for the container's life.

Claude Code refreshes its OAuth token by ATOMIC REPLACE -- write a new file,
rename it over the old one -- which produces a NEW INODE. And the refresh
REVOKES the token it replaced. So after any refresh by anything else logged in
as the same operator:

  * the host's file holds a fresh, working token;
  * the container still sees the old inode, holding the previous token;
  * and that previous token no longer works.

The agent then dies with `401 OAuth access token has been revoked` and
`"apiKeySource":"none"` -- which reads like a misconfigured container and is in
fact a race with a login the run does not control.

## The fix

`stage()` copies the credentials to `<run>/secrets/.credentials.json`, and that
copy is what gets bound. Its inode belongs to the run, so `refresh()` can
rewrite it **in place** -- open `r+`, write, truncate -- and the container,
which is holding exactly that inode, sees the new bytes on its next read. No
remount, no restart, and no window in which the agent holds a revoked token.

In place is the whole point. A `shutil.copy2` or a rename here would give the
file a new inode and put the container back on the stale one.

## Why this is a module and not two functions in `harbor_env.py`

`harbor_env.py` is a SCRIPT -- it composes the container and exits -- and the
refresh has to keep happening for the life of the run, which is the driver's
job. `tests/repo/test_scripts_hold_only_entrypoints.py` forbids a package
importing a script, and it is right to: a script's contract is its command
line, and importing one couples the driver to something free to change shape.
So the logic lives here, stdlib-only like the rest of `lifecycle`, and both the
script and the driver import it.

The VALUE is never read for content, never logged, and never written into
evidence. What is recorded is that a refresh happened.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Inside the RUN root, never the workspace: the agent must not be able to list
#: the directory its own token came from. Mode 700.
SECRETS_DIR = "secrets"

#: The operator's own file. Read every refresh, never held open.
SOURCE = "~/.claude/.credentials.json"

#: Where the token sits INSIDE the container, on both backends. One definition,
#: because the two backends get it there by different means and only one of them
#: gets it there again:
#:
#:   docker  a bind of this run's staged inode, so `refresh()` writing in place
#:           is immediately visible and nothing further is needed;
#:   modal   a sandbox has no bind mounts, so `modal_env.py` COPIES the file in
#:           once at setup. A later host refresh reaches nothing, and the
#:           refresh revokes what the sandbox is still holding -- which is the
#:           401 this module exists to prevent, arriving by the other path. The
#:           driver therefore pushes the new bytes through `backend.put_file`.
IN_CONTAINER = "/home/agent/.claude/.credentials.json"


def source_path() -> Path:
    return Path(SOURCE).expanduser()


def staged_path(run_root: Path | str) -> Path:
    return Path(run_root) / SECRETS_DIR / source_path().name


def stage(credentials: Path, run_root: Path | str) -> Path | None:
    """Copy the credentials into the run and return the copy, or None.

    None when there is nothing to copy -- a host with no login. The caller
    reports that; it is not this function's job to decide whether a run without
    credentials should start.
    """
    credentials = Path(credentials)
    if not credentials.is_file():
        return None
    secrets = Path(run_root) / SECRETS_DIR
    secrets.mkdir(parents=True, exist_ok=True)
    secrets.chmod(0o700)
    staged = secrets / credentials.name
    staged.write_bytes(credentials.read_bytes())
    staged.chmod(0o600)
    return staged


def refresh(staged: Path, source: Path | None = None) -> bool:
    """Rewrite the staged copy in place when the host's token has moved on.

    Returns True when the bytes changed. Every failure is False rather than an
    exception: this runs on a timer inside a live run, and a transient unlink
    while the host is mid-rename must not end it. The next tick tries again.
    """
    staged = Path(staged)
    source = Path(source) if source is not None else source_path()
    try:
        fresh = source.read_bytes()
    except OSError:
        return False
    try:
        if staged.read_bytes() == fresh:
            return False
        with open(staged, "r+b") as handle:
            handle.seek(0)
            handle.write(fresh)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        return False
    return True
