"""The round boundary: freeze the tree, then make it a visible version.

Capture is a git tree object written over the candidate manifest, in seconds,
never a directory walk. Registration turns that captured tree into an immutable
version directory whose bytes come from `git archive` -- read-only afterwards,
and kept forever whether it was accepted or rejected.

## Capture
Round-boundary capture: freeze the exact candidate tree, in seconds.

The identity is a **git tree object**, in a bare repository owned by the run.
That is deliberate and it is the whole design:

* Immutability and content addressing come from git, so there is no bespoke
  hashing system here and nothing for the coding agent to compute or embed.
* Capture is ``add`` + ``write-tree`` over an explicit file list -- an index
  write, not a copy -- so the agent is stopped for a moment, not for minutes.
* The boundary checks are handed ``git archive`` of that exact tree while the
  agent goes straight back to editing the live workspace. The two cannot
  interfere: one is a directory, the other is an object.

The file list comes from the candidate manifest and never from a directory
walk, which would treat a multi-gigabyte ``.venv`` as candidate source.

## Versions
Materialise a captured tree as an immutable, visible version.

A version is a directory of exactly the bytes that were captured, plus the
records that say where it came from and what happened to it. Two properties
carry the design:

* **Immutable.** The content comes from ``git archive`` of the captured tree,
  not from the workspace, so edits the agent made after the boundary cannot
  reach it. After materialisation the tree is made read-only.
* **Visible whatever the verdict.** A rejected candidate is a version with
  ``REJECTED`` in its status, kept forever. Rejections are most of what a
  lineage learns, and a harness that hides them reports a straight ascent that
  never happened.

Registration is separate from acceptance. Being in the registry means "this
existed and here is its provenance". Only :mod:`publish` moves the head.
"""
from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

from .record import Journal, RunIdentity, utc_now
from .utils import write_atomically

# --- capture -----------------------------------------------------------------------

class CaptureError(RuntimeError):
    pass


def _git(*args: str, git_dir: Path, work_tree: Path | None = None,
         index: Path | None = None, check: bool = True) -> str:
    cmd = ["git", f"--git-dir={git_dir}"]
    if work_tree is not None:
        cmd.append(f"--work-tree={work_tree}")
    cmd += list(args)
    env = dict(os.environ)
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    # A capture must not read the operator's identity or hooks.
    env.setdefault("GIT_CONFIG_GLOBAL", "/dev/null")
    env.setdefault("GIT_CONFIG_SYSTEM", "/dev/null")
    # The harness's own identity, so a capture never carries the operator's.
    for who in ("AUTHOR", "COMMITTER"):
        env.setdefault(f"GIT_{who}_NAME", "serverbench")
        env.setdefault(f"GIT_{who}_EMAIL", "serverbench@localhost")
    got = subprocess.run(cmd, capture_output=True, text=True, env=env,
                         check=False)
    if check and got.returncode != 0:
        raise CaptureError(
            f"git {' '.join(args)} failed rc={got.returncode}: "
            f"{got.stderr.strip()[:400]}")
    return got.stdout.strip()


@dataclasses.dataclass(frozen=True)
class Captured:
    candidate_id: str
    tree_sha: str
    #: The commit wrapping that tree, whose parent is the capture it was edited
    #: from. A tree says WHAT the code was; only the commit says what it came
    #: after, and "does this candidate descend from the accepted head" is not
    #: answerable without it.
    commit_sha: str
    round_index: int
    files: tuple[str, ...]
    run_id: str
    workspace_id: str
    accepted_head_at_capture: str
    captured_utc: str
    capture_seconds: float

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["files"] = list(self.files)
        d["kind"] = "captured_candidate"
        d["schema_version"] = 1
        return d


class CaptureStore:
    """One bare repository per run. Holds every candidate tree the run made."""

    def __init__(self, root: Path) -> None:
        self.git_dir = Path(root) / "candidates.git"
        self._last_commit = ""
        if not (self.git_dir / "HEAD").exists():
            self.git_dir.mkdir(parents=True, exist_ok=True)
            _git("init", "--bare", "--quiet", str(self.git_dir),
                 git_dir=self.git_dir)

    def capture(self, *, workspace: Path, files: list[str], round_index: int,
                run_id: str, workspace_id: str, accepted_head: str,
                journal: Journal | None = None,
                parent: str = "") -> Captured:
        """Freeze `files` (workspace-relative) as one immutable tree."""
        import time

        started = time.monotonic()
        workspace = Path(workspace)
        present = [f for f in files if (workspace / f).exists()]
        if not present:
            raise CaptureError(
                f"nothing to capture: none of the {len(files)} manifested "
                f"candidate files exist under {workspace}")
        index = self.git_dir / f"index-r{round_index:03d}"
        if index.exists():
            index.unlink()
        # `--` then explicit paths: no globbing, no walk, no ignore rules.
        _git("add", "--force", "--", *present, git_dir=self.git_dir,
             work_tree=workspace, index=index)
        tree = _git("write-tree", git_dir=self.git_dir, work_tree=workspace,
                    index=index)
        if not tree:
            raise CaptureError("git write-tree produced no object")
        # The commit: parented on whatever this candidate was edited from --
        # the previous capture in this run, or the accepted head this run was
        # seeded with. That parent chain is what makes descent checkable
        # instead of assumed, and it works across runs because the object
        # store belongs to the LINEAGE rather than to one run.
        parents: list[str] = []
        for candidate_parent in (parent, self._last_commit):
            if candidate_parent and self.exists(candidate_parent, "commit"):
                parents = ["-p", candidate_parent]
                break
        commit = _git("commit-tree", tree, *parents, "-m",
                      f"round {round_index} of {run_id}",
                      git_dir=self.git_dir)
        self._last_commit = commit
        elapsed = round(time.monotonic() - started, 3)
        candidate_id = f"r{round_index:03d}-{tree[:12]}"
        captured = Captured(
            candidate_id=candidate_id, tree_sha=tree, commit_sha=commit,
            round_index=round_index,
            files=tuple(sorted(present)), run_id=run_id,
            workspace_id=workspace_id, accepted_head_at_capture=accepted_head,
            captured_utc=utc_now(), capture_seconds=elapsed)
        # Journalled by the caller, not here. A boundary that finds nothing
        # changed still has to write a tree to discover that, and recording
        # THAT as a captured candidate makes the run look like it produced
        # candidates it then lost -- which is the shape of a real defect.
        if journal is not None:
            journal.append("candidate_captured", **captured.to_dict())
        return captured

    def extract(self, tree_sha: str, dest: Path) -> Path:
        """Materialise a captured tree into `dest`, exactly as captured.

        Used for the boundary checks and for the version directory. The live
        workspace is never the thing either of them reads.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(
            ["git", f"--git-dir={self.git_dir}", "archive", tree_sha],
            capture_output=True, check=False)
        if archive.returncode != 0:
            raise CaptureError(
                f"no captured tree {tree_sha}: "
                f"{archive.stderr.decode()[:300]}")
        tar = subprocess.run(["tar", "-x", "-C", str(dest)],
                             input=archive.stdout, capture_output=True,
                             check=False)
        if tar.returncode != 0:
            raise CaptureError(f"extracting {tree_sha}: "
                               f"{tar.stderr.decode()[:300]}")
        return dest

    def exists(self, sha: str, kind: str = "tree") -> bool:
        if not sha:
            return False
        got = _git("cat-file", "-t", sha, git_dir=self.git_dir, check=False)
        return got == kind

    def descends_from(self, ancestor_commit: str, commit: str) -> bool | None:
        """Is `ancestor_commit` genuinely an ancestor of `commit`?

        The general form of "may this candidate become the accepted head". It
        holds within a run (each capture is parented on the last) and across
        runs (a run seeded from a head parents its first capture on that head's
        commit), because the object store belongs to the lineage.

        Returns None when it cannot be established -- a missing object, or a
        version with no commit recorded. None is not False: the
        caller decides what to do about not knowing, and must not be able to
        mistake it for a yes.
        """
        if not ancestor_commit or not commit:
            return None
        if not (self.exists(ancestor_commit, "commit")
                and self.exists(commit, "commit")):
            return None
        if ancestor_commit == commit:
            return True
        got = subprocess.run(
            ["git", f"--git-dir={self.git_dir}", "merge-base",
             "--is-ancestor", ancestor_commit, commit],
            capture_output=True, check=False)
        return got.returncode == 0


# --- versions ----------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Version:
    version_id: str
    candidate_id: str
    tree_sha: str
    run_id: str
    lineage_id: str
    workspace_id: str
    control_epoch: str
    predecessor: str
    directory: str
    registered_utc: str
    status: str
    review: dict
    qualification: dict
    #: The boundary this version's tree was frozen at. Within one run every
    #: candidate comes from the same workspace edited forward, so a higher
    #: round index IS descent -- and it is the only thing that makes "this
    #: candidate descends from the accepted head" checkable rather than assumed.
    round_index: int = 0
    captured_utc: str = ""
    #: The commit this tree was captured as. Ancestry -- "does this descend
    #: from the accepted head" -- is asked of git, not inferred from ordering,
    #: because ordering only holds inside a single run.
    commit_sha: str = ""
    #: What producing this version cost in API spend, from
    #: `lifecycle.api_cost`. The TOTAL is the agent's own `total_cost_usd`
    #: where it recorded one; the per-version SPLIT is proportional and says so
    #: in its own `split_caveat`. Empty on any run whose agent stream is
    #: missing -- absent, never zero, because a run with no stream did not cost
    #: nothing. The per-publication figure is carried in `qualification`.
    api_cost: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["kind"] = "version"
        d["schema_version"] = 1
        return d


#: Names the VERSION DIRECTORY generates. A candidate file carrying one of
#: these is a collision, not source, and both halves of the collision are real:
#: the agent's file and the harness's record of the version.
#:
#: An agent's own `DATASHEET.md` would be captured as candidate source,
#: overwritten by `_write_datasheet` at materialisation, frozen by `_freeze` as
#: source, and then refuse the later outcome write: the agent's notes gone and
#: the datasheet stuck at REGISTERED.
GENERATED_NAMES = ("VERSION.json", "REVIEW.md", "DATASHEET.json",
                   "DATASHEET.md", "MEASUREMENTS.jsonl")


def _write_generated(path: Path, text: str) -> None:
    """Write one of the harness's OWN files, over a read-only copy if need be.

    `_move_colliding_source` stops collisions. This covers the case where the
    rename itself failed, so a generated name was frozen as source and no
    later outcome could otherwise update it.

    Narrow on purpose: only the names in `GENERATED_NAMES`, only inside a
    version directory, and it restores nothing it did not change. Anything the
    CANDIDATE owns stays frozen -- the point of the freeze is that an
    accidental edit to a candidate's source fails loudly, and this must not
    become a way around it.
    """
    if path.name not in GENERATED_NAMES:
        raise ValueError(f"{path.name} is not a generated name; "
                         f"{GENERATED_NAMES} are")
    try:
        path.write_text(text)
        return
    except PermissionError:
        if not path.exists():
            raise
    mode = path.stat().st_mode
    os.chmod(path, mode | stat.S_IWUSR)
    try:
        path.write_text(text)
    finally:
        os.chmod(path, mode)


def _move_colliding_source(directory: Path, source: list[Path],
                           journal: Journal) -> list[Path]:
    """Rename candidate files that collide with a generated name.

    RENAMED, never deleted and never left in place. Deleting would destroy work
    the agent chose to do; leaving it would have the harness overwrite that work
    and then freeze its own record against itself. `candidate_<name>` keeps the
    bytes, keeps them beside the version they belong to, and says in the
    filename which of the two writers put them there.
    """
    kept: list[Path] = []
    for path in source:
        if path.name not in GENERATED_NAMES or path.parent != directory:
            kept.append(path)
            continue
        moved = directory / f"candidate_{path.name}"
        try:
            path.replace(moved)
        except OSError as exc:
            journal.append(
                "candidate_file_collision_unresolved", name=path.name,
                why=(f"{exc}. The harness will write its own {path.name} over "
                     f"it, and the candidate's copy is lost"))
            continue
        kept.append(moved)
        journal.append(
            "candidate_file_renamed", name=path.name, to=moved.name,
            why=(f"the candidate carried a file named {path.name}, which is a "
                 f"name the version directory generates. Renamed so the "
                 f"harness's own record and the candidate's file can both "
                 f"exist; nothing was discarded"))
    return kept


def _freeze(paths: list[Path]) -> None:
    """Drop the write bit on the candidate's SOURCE files.

    Only the source. The directory stays writable because a version's outcome
    is recorded after materialisation, into `VERSION.json` and the datasheet.
    Freezing the whole directory would mean unfreezing to write them, which is
    how "immutable" quietly stops being true.

    Not a security boundary: the same uid can chmod it back. It turns an
    accidental edit into an error at the moment it happens.
    """
    for path in paths:
        if path.is_symlink() or not path.exists():
            continue
        mode = path.stat().st_mode
        os.chmod(path, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


class VersionRegistry:
    """Every version this lineage has, accepted or not, in creation order."""

    def __init__(self, lineage_dir: Path, task_dir: Path | None = None,
                 agent: dict | None = None) -> None:
        #: Where the measured baselines and the scored objective are read from.
        #: Without it a datasheet reports a number; with it, a position.
        self.task_dir = Path(task_dir) if task_dir else None
        #: WHO BUILT THIS, and what it was serving. `{"model", "effort"}` for
        #: the coding agent; the served model is read from the task.
        #:
        #: A datasheet without it names a `lineage_id` and a `run_id` and
        #: nothing a reader recognises. A lineage is one (task, model, agent)
        #: triple and comparing agents on one task is the point of the
        #: benchmark, so the sheet that describes a version has to say which
        #: agent produced it -- rather than sending the reader to a journal in
        #: a run directory that may have been pruned.
        self.agent = dict(agent or {})
        # NOT created here. server_output/ is the published-artifact tree,
        # and a directory in it says a lineage exists. Creating one at run
        # startup would leave an empty lineage behind for every dry run and
        # every run that published nothing.
        self.root = Path(lineage_dir)
        self.index = self.root / "versions.jsonl"

    def _open(self) -> None:
        """Called on the first write, and only then."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.index.touch(exist_ok=True)

    @property
    def lock(self) -> Path:
        """A SEPARATE file from the index, and it has to be.

        `record_outcome` rewrites the index through `write_atomically`, which is
        `os.replace` and swaps the inode, so a lock taken on the index itself
        would be silently dropped halfway through the update. `publish.py` keeps
        `.head.lock` separate from `ACCEPTED_HEAD.json` for the same reason.
        """
        return self.root / ".versions.lock"

    @contextlib.contextmanager
    def _locked(self):
        """Serialise a read-modify-write of the index across threads and runs.

        Three paths write `versions.jsonl` and all three genuinely run at once
        -- `_after_capture` is a daemon thread per boundary, publication runs on
        the agent tick, and `finish()` takes the final capture. Without the lock:

        * `next_version_id()` counts the index, but `materialize` appends to it
          only after extracting the tree, copying MEMORY and writing the
          datasheet, so concurrent boundaries would all be handed `v1` and all
          but one would hit `FileExistsError`.
        * `record_outcome` reads every version and rewrites the whole file, so
          an append landing in between would be erased -- leaving a version
          directory on disk that the index denies, whose id `next_version_id`
          would hand out again, wedging the lineage on `FileExistsError`.

        `flock` is per open file description, so this serialises threads of one
        process as well as separate runs sharing a lineage.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock.touch(exist_ok=True)
        with open(self.lock, "r+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def all(self) -> list[Version]:
        if not self.index.exists():
            return []
        out = []
        for line in self.index.read_text().splitlines():
            if line.strip():
                out.append(Version(**{k: v for k, v in json.loads(line).items()
                                      if k not in ("kind", "schema_version")}))
        return out

    def get(self, version_id: str) -> Version | None:
        for v in self.all():
            if v.version_id == version_id:
                return v
        return None

    def next_version_id(self) -> str:
        return f"v{len(self.all()) + 1}"

    def materialize(self, *, captured: Captured, store: CaptureStore,
                    identity: RunIdentity, journal: Journal, review: dict,
                    predecessor: str,
                    memory_dir: Path | None = None,
                    workspace: Path | None = None) -> Version:
        """Write the version directory from the captured tree, and register it.

        Called after the boundary checks complete. The bytes come from the
        captured tree, so what the final verdict measures and what a reader can
        look at are the same.
        """
        # The id is chosen from the index and the index is only appended to at
        # the very end, so the whole body runs under the lock: releasing it
        # after `next_version_id` would leave the window that hands concurrent
        # boundaries the same `v1`.
        with self._locked():
            return self._materialize_locked(
                captured=captured, store=store, identity=identity,
                journal=journal, review=review, predecessor=predecessor,
                memory_dir=memory_dir, workspace=workspace)

    def _materialize_locked(self, *, captured: Captured, store: CaptureStore,
                            identity: RunIdentity, journal: Journal,
                            review: dict, predecessor: str,
                            memory_dir: Path | None = None,
                            workspace: Path | None = None) -> Version:
        self._open()
        version_id = self.next_version_id()
        directory = self.root / f"{identity.lineage_id}_{version_id}"
        if directory.exists():
            raise FileExistsError(
                f"{directory} already exists; a version directory is written "
                f"once and never rewritten")
        store.extract(captured.tree_sha, directory)
        source = sorted(directory / f for f in captured.files)
        # BEFORE the first generated write. Afterwards the candidate's file has
        # already been overwritten and there is nothing left to preserve.
        source = _move_colliding_source(directory, source, journal)
        version = Version(
            version_id=version_id, candidate_id=captured.candidate_id,
            tree_sha=captured.tree_sha, run_id=identity.run_id,
            lineage_id=identity.lineage_id,
            workspace_id=captured.workspace_id,
            control_epoch=identity.control_epoch, predecessor=predecessor,
            directory=str(directory), registered_utc=utc_now(),
            status="REGISTERED", review=dict(review), qualification={},
            round_index=captured.round_index,
            captured_utc=captured.captured_utc,
            commit_sha=captured.commit_sha)
        _write_generated(
            directory / "VERSION.json",
            json.dumps(version.to_dict(), indent=1, sort_keys=True) + "\n")
        _write_generated(directory / "REVIEW.md", _review_markdown(review))
        if memory_dir and Path(memory_dir).is_dir():
            shutil.copytree(Path(memory_dir), directory / "MEMORY",
                            dirs_exist_ok=True)
        _deliver_reproduction_kit(directory, workspace)
        # The datasheet exists from the moment the version does, and says
        # plainly that it has no measurement yet. A version directory that
        # simply lacks one reads like a version nobody got round to measuring,
        # which is the same shape as a version that failed.
        _write_datasheet(directory, version, qualification=None,
                         task_dir=self.task_dir, agent=self.agent)
        _freeze(source)
        with open(self.index, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(version.to_dict(), sort_keys=True) + "\n")
        journal.append("version_registered", version_id=version_id,
                       candidate_id=captured.candidate_id,
                       tree_sha=captured.tree_sha, directory=str(directory),
                       predecessor=predecessor)
        return version

    def record_outcome(self, version_id: str, *, status: str,
                       qualification: dict, journal: Journal) -> Version:
        """Attach a verdict. Rewrites the index line, never the tree."""
        # Read and rewrite under one lock: this replaces the whole file from a
        # list read at the top, so an append landing in between would be erased.
        with self._locked():
            return self._record_outcome_locked(
                version_id, status=status, qualification=qualification,
                journal=journal)

    def _record_outcome_locked(self, version_id: str, *, status: str,
                               qualification: dict, journal: Journal) -> Version:
        versions = self.all()
        updated = None
        lines = []
        for v in versions:
            if v.version_id == version_id:
                v = dataclasses.replace(v, status=status,
                                        qualification=dict(qualification))
                updated = v
            lines.append(json.dumps(v.to_dict(), sort_keys=True))
        if updated is None:
            raise KeyError(f"no version {version_id} in {self.index}")
        write_atomically(self.index, "\n".join(lines) + "\n")
        # The status file sits beside the frozen tree, not inside it.
        (Path(updated.directory).parent /
         f"{Path(updated.directory).name}.status.json").write_text(
            json.dumps({"kind": "version_status", "schema_version": 1,
                        "version_id": version_id, "status": status,
                        "qualification": qualification,
                        "at_utc": utc_now()}, indent=1, sort_keys=True) + "\n")
        # The version's OWN file, inside the directory, beside the datasheet.
        #
        # Materialisation writes it as REGISTERED with an empty qualification.
        # The outcome must reach this file as well as the index, the status
        # file and the datasheet, because this is the file a reader picks up
        # first: it is the one named after the version.
        #
        # Safe against `_freeze`, which drops the write bit on the candidate's
        # SOURCE only and leaves the directory writable for exactly this reason;
        # `_write_datasheet` below relies on the same thing.
        _write_generated(
            Path(updated.directory) / "VERSION.json",
            json.dumps(updated.to_dict(), indent=1, sort_keys=True) + "\n")
        _write_datasheet(Path(updated.directory), updated,
                         qualification=qualification, task_dir=self.task_dir,
                         agent=self.agent)
        journal.append("version_outcome", version_id=version_id, status=status,
                       qualification=qualification)
        return updated


#: What a published version needs to be MEASURED again, not merely read.
#:
#: Without these a version would be source plus numbers and no way to
#: reproduce either. Names absent from the workspace are skipped.
#:
#: Small on purpose: the client, the public tests, the baselines it was
#: compared against, and the waiters its scripts call. Not the toolkit -- the
#: measurement implementation stays on the harness side, which is the whole
#: point of handing out a client.
REPRODUCTION_KIT = ("measure", "tests", "baselines", "wait", "recall")


def _deliver_reproduction_kit(directory: Path, workspace: Path | None) -> None:
    """Copy the delivered client and comparison set beside the frozen source."""
    if workspace is None:
        return
    for name in REPRODUCTION_KIT:
        source = Path(workspace) / name
        if not source.exists():
            continue
        target = directory / name
        try:
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
        except OSError:
            # A version missing part of its kit is still a version; it is not
            # worth failing a publication over a file that would not copy.
            continue


def _review_markdown(record: dict) -> str:
    """Render the deterministic boundary checks recorded for a candidate."""
    if not record:
        return ("# Boundary checks\n\nNothing was recorded for this "
                "candidate.\n")
    lines = ["# Boundary checks", "",
             "Deterministic checks over the frozen tree.", "",
             f"**Took:** {record.get('seconds')} s", ""]
    failed = list(record.get("failed") or [])
    lines.append(f"**Failed:** {', '.join(failed) if failed else 'none'}")
    lines += ["", "| check | outcome | detail |", "|---|---|---|"]
    for check in record.get("checks") or []:
        detail = str(check.get("detail", "")).replace("|", "\\|")
        lines.append(f"| `{check.get('name')}` | {check.get('outcome')} "
                     f"| {detail} |")
    lines += ["", str(record.get("note") or "")]
    return "\n".join(lines) + "\n"


def _served_model(task_dir: Path | None) -> dict:
    """The model this version SERVES, from the task's own module.

    Read from the task rather than named here: the tasks in this suite serve
    different checkpoints, and a datasheet that hardcoded one would describe
    the wrong model with total confidence.

    Identity lives in the task's Python module -- `MODEL_ID`, `MODEL_REVISION`
    -- which is what the verdict passes to the tokenizer, so this reads the
    same fact from the same place.
    """
    if not task_dir:
        return {}
    try:
        from .utils import ensure_harness_on_path
        ensure_harness_on_path()
        from final_validate import tasks as task_modules
        module = task_modules.python_module(Path(task_dir).name)
    except Exception:                                          # noqa: BLE001
        return {}
    found = {
        "model_id": getattr(module, "MODEL_ID", ""),
        "tokenizer_id": getattr(module, "MODEL_ID", ""),
        "revision": getattr(module, "MODEL_REVISION", ""),
        "dtype": getattr(module, "DTYPE", ""),
    }
    return {k: v for k, v in found.items() if v}


def _write_datasheet(directory: Path, version: Version,
                     qualification: dict | None,
                     task_dir: Path | None = None,
                     agent: dict | None = None) -> None:
    """What this version is: `DATASHEET.json` for machines, `.md` for people.

    Nothing in a run measures a version. Its numbers come from the final
    verdict, run outside Harbor after the evolve, so the sheet says that
    rather than reporting an absent measurement as a zero.
    """
    sheet = {
        "kind": "version_datasheet", "schema_version": 2,
        "version_id": version.version_id,
        "lineage_id": version.lineage_id,
        "candidate_id": version.candidate_id,
        "tree_sha": version.tree_sha,
        "run_id": version.run_id,
        "workspace_id": version.workspace_id,
        "control_epoch": version.control_epoch,
        # The coding agent that built this (model and effort are different
        # budgets of one model) and the checkpoint the version serves.
        "agent": {
            "model": (agent or {}).get("model") or (
                (version.api_cost or {}).get("model") or None),
            "effort": (agent or {}).get("effort") or None,
            "source": ("the run's plan" if (agent or {}).get("model")
                       else "the agent stream's own model field"),
        },
        "served_model": _served_model(task_dir),
        "predecessor": version.predecessor or None,
        "status": (qualification or {}).get("status") or version.status,
        "registered_utc": version.registered_utc,
        "measurement": {
            "status": "NOT_MEASURED_IN_RUN",
            "why": ("a version is published unjudged; the final verdict "
                    "measures the lineage's head after the evolve"),
        },
    }
    _write_generated(directory / "DATASHEET.json",
                     json.dumps(sheet, indent=1, sort_keys=True) + "\n")

    lines = [f"# {version.version_id}", "",
             f"- lineage `{version.lineage_id}`",
             f"- from run `{version.run_id}`, workspace `{version.workspace_id}`",
             f"- tree `{version.tree_sha}`",
             f"- predecessor `{version.predecessor or '(none -- first of the lineage)'}`",
             f"- control epoch `{version.control_epoch}`",
             f"- status **{sheet['status']}**"]
    ag, served = sheet["agent"], sheet["served_model"]
    if ag.get("model"):
        effort = f" at `{ag['effort']}` effort" if ag.get("effort") else ""
        lines.append(f"- built by **{ag['model']}**{effort}")
    if served.get("model_id"):
        rev = served.get("revision") or ""
        lines.append(
            f"- serves **{served['model_id']}**"
            + (f" at `{rev[:12]}`" if rev else "")
            + (f", {served['dtype']}" if served.get("dtype") else ""))
    lines += ["", "## Measurement", "", sheet["measurement"]["why"] + "."]
    record = version.review or {}
    if record.get("checks"):
        lines += ["", "## Boundary checks", "",
                  "| check | outcome | detail |", "|---|---|---|"]
        for check in record["checks"]:
            detail = str(check.get("detail", "")).replace("|", "\\|")
            lines.append(f"| `{check.get('name')}` | {check.get('outcome')} "
                         f"| {detail} |")
    _write_generated(directory / "DATASHEET.md", "\n".join(lines) + "\n")
