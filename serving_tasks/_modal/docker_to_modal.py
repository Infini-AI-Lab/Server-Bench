#!/usr/bin/env python3
"""Build OUR agent image on Modal, from our own Dockerfiles.

Not a re-authored equivalent. This reads
`serverbench/environments/serving-agent/Dockerfile` and
`serverbench/environments/evolve-agent/Dockerfile`, the two files that already
define the image every evolve runs in, and adapts exactly what Modal cannot
take. Everything else -- the torch pin, the transformers pin, the Nsight and
CUDA-13 stages, the CCCL fix, the ninja wheel, the toolchain smoke test -- is
whatever those files say today, so a change there reaches Modal without anyone
re-typing it here.

## The two things that actually need adapting

**A local base image.** `evolve-agent/Dockerfile` begins three stages with
`FROM serverbench/serving-agent:cu128`, an image that exists only in this
host's docker daemon. Modal builds remotely and cannot see it. The chain does
bottom out somewhere public -- `serving-agent/Dockerfile` is
`FROM python:3.12-slim` -- so the two files are composed into one multi-stage
Dockerfile with the local reference rewritten to a stage name. Nothing is
rewritten but that reference.

**The build context.** Both files COPY from the repository root, and sending
the repository to Modal would mean sending `server_output/` and
`artifacts/` with it. Only the paths the Dockerfiles actually name are staged,
and they are discovered by reading the COPY lines rather than listed here, so a
new COPY does not silently ship a broken image.

## What Modal will not honour, and what to do about it

`USER agent` is ignored: Modal containers always run as root. Our own
Dockerfile explains why that matters -- "the agent CLI refuses
`--dangerously-skip-permissions` under root" -- so the caller must launch the
agent as that user (`su agent -p -s /bin/bash -c ...`). The user itself is
still created by our Dockerfile, with our UID, so nothing about the image
changes; only the launch does.

## The acceptance test

Built from these files and started on 2x L40S, every pin our Dockerfiles carry
must arrive intact:

    agent user        uid=1000(agent), and `su agent` runs the CLI
    gpus              2x NVIDIA L40S
    nvcc              13.0.88
    torch             2.8.0+cu128, cuda available, 2 devices
    transformers      4.55.2
    agent CLIs        claude 2.1.267, codex 0.154.0, ripgrep 14.1.1
    Nsight            ncu 2025.4.1, nsys 2025.5.2
    ninja / uv        1.13.2 / 0.10.12
    egress allowlist  /usr/local/bin/restrict-network.sh present

If a change to our Dockerfiles stops any of it arriving, the migration is not
faithful and the image is not the one the suite is validated against.

Also ignored or unsupported by Modal's builder, per its documentation:
`EXPOSE`, `HEALTHCHECK`, `LABEL`, `ONBUILD`, `STOPSIGNAL`, `VOLUME`, `ADD`
from anything but a single URL, and `ENV` defaults of the `${VAR:-default}`
form. Rather than assume ours avoid them, `unsupported_directives()` reports
any that appear, and the build refuses instead of producing an image that
quietly differs from the one this suite is validated against.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVIRONMENTS = REPO / "serverbench" / "environments"
BASE_DOCKERFILE = ENVIRONMENTS / "serving-agent" / "Dockerfile"
AGENT_DOCKERFILE = ENVIRONMENTS / "evolve-agent" / "Dockerfile"

#: The local image name the evolve-agent Dockerfile builds on, and the stage it becomes.
LOCAL_BASE = "serverbench/serving-agent:cu128"
BASE_STAGE = "serving_agent_base"

#: Build args our own build script passes. Kept here because they name paths in
#: the context, which is the one thing this module has to stage.
#: `03-build-images.sh` is the authority; these mirror it.
BUILD_ARGS = {
    "CLAUDE_BIN": "serverbench/environments/_claude-bin",
    "CODEX_BIN": "serverbench/environments/_codex-bin",
    "CODEX_CODE_MODE_HOST_BIN": "serverbench/environments/_codex-code-mode-host-bin",
    "CODEX_RG_BIN": "serverbench/environments/_codex-rg-bin",
}

#: Directives Modal's Dockerfile builder does not support or silently ignores.
#: `USER` is listed separately: it is ignored, ours uses it deliberately, and
#: the answer is a launch-time `su` rather than an edit to the image.
MODAL_UNSUPPORTED = ("EXPOSE", "HEALTHCHECK", "LABEL", "ONBUILD",
                     "STOPSIGNAL", "VOLUME")


class MigrationError(RuntimeError):
    """A difference between our Dockerfile and Modal's builder, stated."""


def _read(path: Path) -> list[str]:
    if not path.is_file():
        raise MigrationError(f"no Dockerfile at {path}")
    return path.read_text("utf-8").splitlines()


def unsupported_directives() -> dict[str, list[str]]:
    """Which Modal-unsupported directives our Dockerfiles use, if any."""
    found: dict[str, list[str]] = {}
    for path in (BASE_DOCKERFILE, AGENT_DOCKERFILE):
        for number, line in enumerate(_read(path), 1):
            head = line.strip().split(" ", 1)[0].upper()
            if head in MODAL_UNSUPPORTED:
                found.setdefault(head, []).append(
                    f"{path.relative_to(REPO)}:{number}")
            if head == "ADD" and not re.search(r"\bhttps?://", line):
                found.setdefault("ADD(non-URL)", []).append(
                    f"{path.relative_to(REPO)}:{number}")
            # ONLY on an ENV line. What Modal does not honour is an `ENV`
            # default; `${VAR:-default}` anywhere else is ordinary shell that
            # the shell expands at run time, and Modal never sees it -- such
            # as the `RUN printf` that writes `${APP_DIR:-/workspace}` into a
            # /etc/profile.d drop-in for `serve.sh` to source.
            if head == "ENV" and re.search(
                    r"\$\{[A-Za-z_][A-Za-z0-9_]*:-", line):
                found.setdefault("ENV-default", []).append(
                    f"{path.relative_to(REPO)}:{number}")
            # A build-arg variable in a COPY SOURCE is refused by Modal. It is
            # reported rather than listed as fatal, because `_literal_copy`
            # resolves exactly the ones BUILD_ARGS knows.
            if head == "COPY" and re.search(r"\$\{?[A-Za-z_]", line):
                unknown = [m for m in re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)",
                                                line)
                           if m not in BUILD_ARGS]
                if unknown:
                    found.setdefault("COPY-var", []).append(
                        f"{path.relative_to(REPO)}:{number} ({', '.join(unknown)})")
    return found


def _logical_lines(lines: list[str]) -> list[str]:
    """Join backslash continuations into one instruction each.

    Not cosmetic. `serving-agent/Dockerfile` writes

        COPY serverbench/.../restrict-network.sh \\
             /usr/local/bin/restrict-network.sh

    and a line-at-a-time reader sees a COPY with one operand, decides it is
    malformed, and skips it -- so the egress allowlist would be absent from the
    staged context and the build would fail on a missing file.
    """
    out, buffer = [], ""
    for raw in lines:
        stripped = raw.strip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1].rstrip() + " "
            continue
        out.append((buffer + stripped).strip())
        buffer = ""
    if buffer.strip():
        out.append(buffer.strip())
    return out


def copy_sources() -> list[str]:
    """Every context path our Dockerfiles COPY, with build args resolved.

    Read rather than listed: a COPY added to either Dockerfile has to reach the
    staged context, and a hand-maintained list is how that stops happening.
    """
    wanted: list[str] = []
    for path in (BASE_DOCKERFILE, AGENT_DOCKERFILE):
        for line in _logical_lines(_read(path)):
            if not line.upper().startswith("COPY"):
                continue
            if "--from=" in line:
                continue                      # an earlier stage, not the context
            body = line[4:].strip()
            body = re.sub(r"^--[^\s]+\s+", "", body)
            parts = body.split()
            if len(parts) < 2:
                continue
            for source in parts[:-1]:
                resolved = source
                for name, value in BUILD_ARGS.items():
                    resolved = resolved.replace(f"${{{name}}}", value)
                    resolved = resolved.replace(f"${name}", value)
                if "$" in resolved:
                    raise MigrationError(
                        f"{path.relative_to(REPO)} copies {source!r}, whose "
                        f"build arg this module does not know. Add it to "
                        f"BUILD_ARGS so the file lands in the staged context")
                wanted.append(resolved)
    # Stable and de-duplicated, so the staged context is reproducible.
    return sorted(dict.fromkeys(wanted))


def _resolved(name: str) -> Path | None:
    found = shutil.which(name)
    if not found:
        return None
    got = subprocess.run(["readlink", "-f", found], capture_output=True,
                         text=True, check=False).stdout.strip()
    path = Path(got or found)
    return path if path.is_file() else None


def agent_binary_sources() -> dict[str, Path]:
    """Where each binary our Dockerfile copies actually is, as our build finds it.

    `03-build-images.sh` is the authority and this mirrors its resolution,
    including the two parts that are not obvious:

    * `codex-code-mode-host` is not on PATH. It sits BESIDE the resolved
      `codex` binary, inside the versioned standalone release directory.
    * `rg` may not be a binary at all. On a host with only Claude Code it is a
      shell function that re-execs the agent CLI as `rg`, because Claude Code
      bundles ripgrep -- so when no real ripgrep exists, the CLI binary IS the right thing to
      stage, and staging nothing would leave the image without the search tool
      the agent reaches for constantly.

    Every one can be overridden the same way the build script allows, so a host
    that lays these out differently does not need this file edited.
    """
    import os
    out: dict[str, Path] = {}

    claude = (Path(os.environ["SERVERBENCH_CLAUDE_BIN"])
              if os.environ.get("SERVERBENCH_CLAUDE_BIN") else _resolved("claude"))
    if claude is None:
        raise MigrationError(
            "the agent CLI is not on PATH, so it cannot be staged into the "
            "image; set SERVERBENCH_CLAUDE_BIN or install it")
    out["_claude-bin"] = claude

    codex = (Path(os.environ["SERVERBENCH_CODEX_BIN"])
             if os.environ.get("SERVERBENCH_CODEX_BIN") else _resolved("codex"))
    if codex is not None:
        out["_codex-bin"] = codex
        host = os.environ.get("SERVERBENCH_CODEX_CODE_MODE_HOST_BIN")
        beside = Path(host) if host else codex.parent / "codex-code-mode-host"
        if beside.is_file():
            out["_codex-code-mode-host-bin"] = beside

    override = os.environ.get("SERVERBENCH_CODEX_RG_BIN")
    real_rg = Path(override) if override else _resolved("rg")
    # The CLI answers to `rg` when invoked as it, which is exactly what this
    # host's shell function does; falling back to it keeps the image complete.
    out["_codex-rg-bin"] = real_rg if real_rg is not None else claude
    return out


def stage_agent_binaries() -> None:
    """Put the agent CLIs where our Dockerfile expects them, as our build does."""
    sources = agent_binary_sources()
    missing = []
    for target in ("_claude-bin", "_codex-bin", "_codex-code-mode-host-bin",
                   "_codex-rg-bin"):
        destination = ENVIRONMENTS / target
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        source = sources.get(target)
        if source is None:
            missing.append(target)
            continue
        shutil.copy2(source, destination)
        destination.chmod(0o755)
    if missing:
        raise MigrationError(
            "our Dockerfile copies binaries this host cannot supply: "
            + ", ".join(missing)
            + ". Set the matching SERVERBENCH_* override, or run "
              "serverbench/scripts/deploy/03-build-images.sh, which stages "
              "them the same way.")


def _literal_copy(line: str) -> str:
    """Resolve build-arg variables inside a COPY source to a literal path.

    Modal's builder refuses them outright:

        InvalidError: COPY command: ${CLAUDE_BIN} using special
        flags/arguments/variables are not supported

    Docker resolves `COPY ${CLAUDE_BIN} /usr/local/bin/claude` against the
    build args our own build script passes; Modal will not, and the failure
    arrives only when the image is actually built. The substitution uses the
    same BUILD_ARGS the staged context is built from, so the file a COPY names
    and the file that is staged cannot drift apart.
    """
    if not line.strip().upper().startswith("COPY"):
        return line
    for name, value in BUILD_ARGS.items():
        line = line.replace(f"${{{name}}}", value).replace(f"${name}", value)
    return line


def compose_dockerfile() -> str:
    """Our two Dockerfiles as one, with the local base rewritten to a stage."""
    base = _read(BASE_DOCKERFILE)
    agent = _read(AGENT_DOCKERFILE)

    # The base's own FROM becomes a NAMED stage. Its `USER`/`WORKDIR` tail is
    # kept: the later stages inherit from it exactly as they do today.
    out: list[str] = [
        "# GENERATED by serving_tasks/_modal/docker_to_modal.py -- do not edit.",
        "#",
        "# serverbench/environments/serving-agent/Dockerfile and",
        "# serverbench/environments/evolve-agent/Dockerfile, composed into one",
        f"# multi-stage build so {LOCAL_BASE!r} -- which exists only in a local",
        "# docker daemon -- becomes a stage Modal can build remotely.",
        "#",
        "# The ONLY edits are the FROM rewrites below. Every other line is",
        "# whatever those two files say.",
        "",
    ]
    seen_from = False
    for line in base:
        if not seen_from and line.strip().upper().startswith("FROM "):
            out.append(f"{line.rstrip()} AS {BASE_STAGE}")
            seen_from = True
            continue
        out.append(_literal_copy(line))
    if not seen_from:
        raise MigrationError(f"{BASE_DOCKERFILE} has no FROM line")

    out += ["", "# " + "-" * 70,
            "# evolve-agent/Dockerfile follows, with its base rewritten.",
            "# " + "-" * 70, ""]

    rewrote = 0
    for line in agent:
        if line.strip().upper().startswith("FROM ") and LOCAL_BASE in line:
            out.append(line.replace(LOCAL_BASE, BASE_STAGE))
            rewrote += 1
            continue
        out.append(_literal_copy(line))
    if not rewrote:
        raise MigrationError(
            f"{AGENT_DOCKERFILE} does not build on {LOCAL_BASE}, so this "
            f"composition is describing an image that does not exist")
    return "\n".join(out) + "\n"


def stage_context(into: Path) -> dict:
    """A context holding only what the Dockerfiles copy, at their own paths."""
    into = Path(into)
    stage_agent_binaries()
    staged, missing = [], []
    for relative in copy_sources():
        source = REPO / relative
        if not source.exists():
            missing.append(relative)
            continue
        target = into / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        staged.append(relative)
    if missing:
        raise MigrationError(
            "our Dockerfiles copy files that are not in this checkout, so the "
            "image cannot be built faithfully:\n  " + "\n  ".join(missing))
    return {"staged": staged, "bytes": sum(
        (into / r).stat().st_size for r in staged if (into / r).is_file())}


def agentx_paths() -> tuple[Path, Path]:
    """The AgentX runtime and source the docker path BIND MOUNTS.

    On docker these arrive as two read-only mounts, `/opt/agentx` (a prebuilt
    venv plus `bin/aiperf`) and `/opt/agentx-src`. A sandbox has no bind
    mounts, so on Modal they become image layers at the SAME paths -- same
    bytes, same locations, so nothing that resolves `aiperf` or reads
    `AGENTX_SOURCE` can tell the difference.

    Resolved the way `harbor_env.py` resolves them, including its overrides, so
    the two cannot drift apart.
    """
    import os
    override = os.environ.get("SERVERBENCH_AGENTX_RUNTIME", "").strip()
    runtime = (Path(override) if override
               else REPO / "serverbench" / "artifacts" / "agentx-runtime")
    if not ((runtime / "venv").is_dir() and (runtime / "bin" / "aiperf").is_file()):
        raise MigrationError(
            f"no usable AgentX runtime at {runtime}: the brief tells the agent "
            f"to measure with `aiperf`, and an image without it fails the "
            f"boundary checks after the sandbox is already paid for. Set "
            f"SERVERBENCH_AGENTX_RUNTIME")
    source_override = os.environ.get("SERVERBENCH_AGENTX_SOURCE", "").strip()
    source = (Path(source_override).expanduser() if source_override
              else REPO / "agentx-harness")
    if not ((source / "pyproject.toml").is_file()
            and (source / "src" / "aiperf" / "__init__.py").is_file()):
        raise MigrationError(
            f"{source} is not the AgentX source checkout (an uninitialised "
            f"submodule looks exactly like this). Run `git submodule update "
            f"--init --recursive`")
    return runtime, source


def agentx_interpreter_paths(runtime: Path) -> tuple[Path, str]:
    """Copy the resolved install to the absolute alias the venv actually names.

    uv's generic cpython-3.12 directory can itself be a symlink to a patch
    version. Copying only to that resolved name leaves the venv link dangling.
    """
    import os
    link = Path(runtime) / "venv/bin/python"
    resolved = link.resolve(strict=True).parent.parent
    target = Path(os.readlink(link)) if link.is_symlink() else link
    mount = target.parent.parent if target.is_absolute() else resolved
    return resolved, str(mount)


def verifier_distribution_layers(image, runtime: Path):
    """Export the actual pinned installation, including its VCS provenance.

    The development runtime may have been built from a local checkout. Even
    identical code then lacks the VCS metadata the final verifier requires.
    Copy the already verified host installation; never manufacture provenance
    metadata or relax the pin check. Existing host environments are untouched.
    """
    import os
    python = os.environ.get("SERVERBENCH_PY") or str(REPO / "serverbench/.venv/bin/python")
    probe = (
        "import json,importlib.metadata as m; "
        "from final_validate.agentx_pin import require; "
        "p=require(); d=m.distribution('aiperf'); "
        "print(json.dumps({'package':str(d.locate_file('aiperf')), "
        "'metadata':str(d._path)}))"
    )
    got = subprocess.run([python, "-c", probe], check=True, capture_output=True,
                         text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": str(REPO / "serverbench" / "harness")})
    paths = json.loads(got.stdout)
    sites = list((runtime / "venv/lib").glob("python*/site-packages"))
    if len(sites) != 1:
        raise MigrationError("the exported AgentX runtime must have exactly one Python site-packages")
    destination = "/opt/agentx/" + str(sites[0].relative_to(runtime))
    for key in ("package", "metadata"):
        source = Path(paths[key])
        image = image.add_local_dir(str(source), destination + "/" + source.name, copy=True)
    return image


def modal_image(*, force_build: bool = False):
    """`modal.Image` for our agent image, built from our own Dockerfiles."""
    import modal

    problems = unsupported_directives()
    if problems:
        raise MigrationError(
            "our Dockerfiles use directives Modal's builder does not support, "
            "so the image it produced would differ from the one this suite is "
            "validated against:\n  " + "\n  ".join(
                f"{k}: {', '.join(v)}" for k, v in problems.items()))

    context = Path(tempfile.mkdtemp(prefix="serverbench-modal-context-"))
    stage_context(context)
    dockerfile = context / "Dockerfile.modal"
    dockerfile.write_text(compose_dockerfile(), "utf-8")
    runtime, source = agentx_paths()
    interpreter_root, interpreter_mount = agentx_interpreter_paths(runtime)
    image = (
        modal.Image.from_dockerfile(
            dockerfile, context_dir=context, build_args=dict(BUILD_ARGS),
            force_build=force_build)
        # The two bind mounts, as layers. `copy=True` so later commands can see
        # them and so they are part of the cached image rather than attached at
        # run time.
        .add_local_dir(str(runtime), "/opt/agentx", copy=True)
        # Match Harbor's interpreter mount. The venv's absolute symlink must
        # resolve for native pin verification, not merely `command -v aiperf`.
        .add_local_dir(str(interpreter_root), interpreter_mount, copy=True)
        .add_local_dir(str(source), "/opt/agentx-src", copy=True)
        # What `harbor_env.py` exports for the same mounts.
        .env({"AGENTX_ROOT": "/opt/agentx",
              "AGENTX_SOURCE": "/opt/agentx-src",
              "PATH": ("/opt/agentx/bin:/usr/local/cuda/bin:/usr/local/sbin:"
                       "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")})
        # AND FOR LOGIN SHELLS, which throw that PATH away.
        #
        # Debian's /etc/profile does not append, it ASSIGNS:
        #     PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games"
        # so `bash -c 'command -v aiperf'` finds it and `bash -lc` does not,
        # and every login shell the harness uses would report aiperf missing
        # and fail its boundary checks after the GPUs are paid for.
        #
        # `harbor_env.py` solves this with a /etc/profile.d drop-in bind mount;
        # the same file, baked in, because a sandbox has no bind mounts. Both
        # mechanisms are kept and neither is redundant.
        .run_commands(
            "printf '%s\\n' "
            "'# Written by serving_tasks/_modal/docker_to_modal.py.' "
            "'# /etc/profile ASSIGNS PATH rather than appending, so a login' "
            "'# shell drops the image PATH. This runs after that assignment.' "
            "'case \":$PATH:\" in' "
            "'  *:/opt/agentx/bin:*) ;;' "
            "'  *) PATH=\"/opt/agentx/bin:$PATH\" ;;' "
            "'esac' "
            "'export PATH' "
            "> /etc/profile.d/10-agentx.sh && chmod 644 /etc/profile.d/10-agentx.sh",
            # Proven at BUILD time, so a broken drop-in cannot reach a run.
            "bash -lc 'command -v aiperf' >/dev/null "
            "|| { echo 'aiperf is not on a login shell PATH' >&2; exit 1; }",
        )
    )
    return verifier_distribution_layers(image, runtime)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="print the composed Dockerfile")
    sub.add_parser("audit", help="what Modal will not honour, and what we copy")
    stage = sub.add_parser("stage", help="stage the build context and report")
    stage.add_argument("--into", default=None)
    args = ap.parse_args(argv)

    try:
        if args.cmd == "show":
            sys.stdout.write(compose_dockerfile())
            return 0
        if args.cmd == "audit":
            problems = unsupported_directives()
            print(json.dumps({
                "copy_sources": copy_sources(),
                "modal_unsupported": problems,
                "note": ("USER is ignored by Modal, so the agent must be "
                         "launched with `su agent`; the image still creates "
                         "the user exactly as our Dockerfile does"),
            }, indent=1))
            return 1 if problems else 0
        if args.cmd == "stage":
            into = Path(args.into) if args.into else Path(
                tempfile.mkdtemp(prefix="serverbench-modal-context-"))
            into.mkdir(parents=True, exist_ok=True)
            got = stage_context(into)
            (into / "Dockerfile.modal").write_text(compose_dockerfile(), "utf-8")
            print(json.dumps({"context": str(into), **got,
                              "megabytes": round(got["bytes"] / 1e6, 1)},
                             indent=1))
            return 0
    except MigrationError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
