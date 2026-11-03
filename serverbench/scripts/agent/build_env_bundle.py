#!/usr/bin/env python3
"""Freeze the offline inputs an agent needs to build its own environment.

The agent gets a Python runtime, a `uv` binary and a read-only wheelhouse. It
does not get a working virtualenv — building one is the first construction
milestone, and an environment the agent inherited is one nobody can reconstruct.

So every wheel it could need has to be here before the network goes away: a
missing wheel surfaces only inside a run, as a resolution failure that reads like
the agent's fault.

Three rules the builder enforces rather than documents:

**Wheels, never an installed tree.** Copying a live `.venv` or `site-packages`
carries absolute host paths in RECORD files, `.pth` injections, editable installs
and RPATHs — and produces an environment that works until it is moved. Wheels are
relocatable by construction.

**Every artefact is checked against the denylist before it is admitted**,
including transitively. A renamed wheel or a vendored copy does not get in by
being spelled differently, and an allowed package that *requires* a denied one is
refused with the dependency named.

**A frozen resolution is never replaced.** `--pins` replays a tracked
`resolved.requirements.txt` and copies it into the bundle; a bundle that already
holds a different resolution, or a fresh resolve over an existing one, is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness"))

from serving_evolve.environment.denylist import (  # noqa: E402
    ALLOWED_GENERIC,
    DENIED,
    scan_dependency_graph,
)

#: What a task needs to build, test, profile and serve. Deliberately not
#: "whatever was installed": every entry is here because something in the task
#: needs it, and the reason is recorded so a reader can challenge it.
REQUIRED: dict[str, str] = {
    "torch": "the model runtime and the only tensor library available",
    "transformers": "the tokenizer, the chat template and the reference forward",
    "tokenizers": "transformers' tokenizer backend",
    "safetensors": "the checkpoint format",
    "numpy": "array interchange, required by torch and transformers",
    "filelock": "transformers' cache locking",
    "huggingface-hub": "transformers' local snapshot resolution, offline",
    "packaging": "version handling, required across the stack",
    "pyyaml": "task and manifest files",
    "regex": "the tokenizer's pattern engine",
    "requests": "a transformers import-time dependency; no egress exists to use it",
    "tqdm": "transformers progress reporting",
    "fastapi": "an HTTP framework, if the agent chooses one",
    "starlette": "fastapi's ASGI layer",
    "uvicorn": "an ASGI server, if the agent chooses one",
    "pydantic": "fastapi's request validation",
    "pydantic-core": "pydantic's compiled core",
    "annotated-types": "pydantic dependency",
    "anyio": "starlette's async layer",
    "sniffio": "anyio dependency",
    "idna": "anyio and requests dependency",
    "click": "uvicorn's CLI",
    "h11": "uvicorn's HTTP/1.1 implementation",
    "httpx": "an HTTP client for the agent's own testing",
    "httpcore": "httpx's transport",
    "certifi": "TLS roots, required by requests at import",
    "charset-normalizer": "requests dependency",
    "urllib3": "requests dependency",
    "typing-extensions": "required across the stack",
    "pytest": "the agent's own test runner",
    "iniconfig": "pytest dependency",
    "pluggy": "pytest dependency",
    "triton": "the kernel language available under kernel_strategy=evolve",
    "jinja2": "torch's and transformers' template engine",
    "markupsafe": "jinja2 dependency",
    "fsspec": "torch and huggingface-hub dependency",
    "sympy": "torch dependency",
    "mpmath": "sympy dependency",
    "networkx": "torch dependency",
    "psutil": "process and memory introspection for the agent's own profiling",
    "hf-xet": "huggingface-hub storage backend",
    "accelerate": "transformers' device-placement path; several from_pretrained "
                  "options import it and fail unactionably offline without it",
    "protobuf": "transformers' slow-tokenizer fallback imports it, and the "
                "ImportError its absence raises cannot be fixed inside a "
                "container with no index",
    "sentencepiece": "the slow-tokenizer backend the same fallback needs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wheel_metadata(path: Path) -> dict:
    """Name, version, requirements and entry points, read from the wheel itself.

    From the artefact rather than from an index, because the artefact is what
    gets installed and an index answer describes a different file.
    """
    info: dict = {"requires": [], "entry_points": [], "name": "", "version": ""}
    try:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.namelist():
                # Only the wheel's *own* dist-info, which is always at the archive
                # root. A wheel can vendor other packages' METADATA (setuptools
                # ships many under `setuptools/_vendor/`); reading those would name
                # the wheel wrongly and scan the denylist against a vendored
                # package's requirements instead of the wheel's own.
                if entry.count("/") != 1:
                    continue
                if entry.endswith(".dist-info/METADATA"):
                    text = archive.read(entry).decode("utf-8", "replace")
                    for line in text.splitlines():
                        if line.startswith("Name: "):
                            info["name"] = line[6:].strip()
                        elif line.startswith("Version: "):
                            info["version"] = line[9:].strip()
                        elif line.startswith("Requires-Dist: "):
                            info["requires"].append(line[15:].strip())
                        elif line.startswith("License"):
                            info.setdefault("license", line.split(":", 1)[-1].strip())
                elif entry.endswith(".dist-info/entry_points.txt"):
                    info["entry_points"] = [
                        line.strip() for line in
                        archive.read(entry).decode("utf-8", "replace").splitlines()
                        if line.strip() and not line.startswith("[")]
    except zipfile.BadZipFile:
        info["error"] = "not a readable wheel"
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("serverbench/artifacts/env-input"))
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--torch-index",
                    default="https://download.pytorch.org/whl/cu128")
    ap.add_argument("--uv", default=shutil.which("uv") or "")
    ap.add_argument("--skip-download", action="store_true",
                    help="verify an existing wheelhouse without fetching")
    ap.add_argument("--pins", type=Path, default=None,
                    help="download exactly these pins instead of re-resolving. "
                         "The oracle's references were generated against a "
                         "specific transformers version, and the chat template "
                         "ships with the library: a fresh resolution that picks "
                         "up a newer one silently invalidates every reference. "
                         "Reuse the frozen resolution when one exists")
    args = ap.parse_args(argv)

    frozen = args.out / "resolved.requirements.txt"
    if args.pins:
        if (frozen.is_file() and frozen.resolve() != args.pins.resolve()
                and frozen.read_bytes() != args.pins.read_bytes()):
            raise SystemExit(f"{frozen} holds a different resolution from {args.pins}; "
                             "a replay never replaces a frozen resolution")
    elif not args.skip_download and frozen.is_file() and frozen.stat().st_size:
        raise SystemExit(f"{frozen} is a frozen resolution; replay it with "
                         f"--pins {frozen}, or remove it to resolve afresh")

    wheelhouse = args.out / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if not args.uv:
            raise SystemExit("uv is not on PATH and none was given; it is what "
                             "resolves and populates the wheelhouse")
        # uv has no `pip download`, and its cache stores wheels unpacked rather
        # than as `.whl` files, so the cache cannot be harvested. Resolve with
        # uv, which is fast and reports exactly what it picked, then fetch the
        # resolved artefacts with `pip download`, which keeps the wheels.
        staging = args.out / ".staging"
        if staging.exists():
            shutil.rmtree(staging)
        cache = args.out / ".uv-cache"

        print(f"resolving into a staging environment at {staging} ...", flush=True)
        # `--seed` because `uv venv` creates an environment without pip, and pip
        # is what keeps the downloaded wheels as files.
        subprocess.run([args.uv, "venv", str(staging), "--python", args.python,
                        "--seed"],
                       capture_output=True, text=True, timeout=600, check=True)
        staged_python = staging / "bin" / "python"

        if args.pins:
            # A resolution that already happened, replayed. The wheels still have
            # to be fetched -- only the *choice* of version is reused, and reusing
            # it is the point: these are the versions the oracle was generated
            # against.
            resolved = [line.strip() for line in
                        args.pins.read_text("utf-8").splitlines()
                        if "==" in line and not line.startswith("#")]
            print(f"  replaying a frozen resolution of {len(resolved)} package(s) "
                  f"from {args.pins}", flush=True)
        else:
            # Torch from its own index: the CUDA build is not on PyPI, and a
            # resolver that settles on the CPU wheel produces an environment that
            # imports and then cannot see the device — which surfaces much later
            # as a mysterious `cuda.is_available() == False`.
            print("resolving torch from the CUDA index ...", flush=True)
            result = subprocess.run(
                [args.uv, "pip", "install", "torch==2.8.0",
                 "--python", str(staged_python), "--index-url", args.torch_index,
                 "--cache-dir", str(cache)],
                capture_output=True, text=True, timeout=5400)
            if result.returncode != 0:
                print(result.stderr[-2000:], file=sys.stderr)
                raise SystemExit("torch resolution failed")

            rest = [name for name in REQUIRED if name != "torch"]
            print(f"resolving {len(rest)} further package(s) ...", flush=True)
            result = subprocess.run(
                [args.uv, "pip", "install", *rest, "--python", str(staged_python),
                 "--cache-dir", str(cache)],
                capture_output=True, text=True, timeout=5400)
            if result.returncode != 0:
                print(result.stderr[-2000:], file=sys.stderr)
                raise SystemExit("dependency resolution failed")

            freeze = subprocess.run(
                [args.uv, "pip", "freeze", "--python", str(staged_python)],
                capture_output=True, text=True, timeout=300)
            resolved = [line for line in freeze.stdout.splitlines() if "==" in line]
            frozen.write_text("\n".join(resolved) + "\n", "utf-8")
            print(f"  resolution pinned {len(resolved)} package(s)", flush=True)

        # Now fetch the exact pins as files. Torch's pins come from its own index;
        # everything else from PyPI. Split because one index does not serve both.
        torch_pins = [r for r in resolved
                      if r.split("==")[0].lower() in
                      {"torch", "triton", "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
                       "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12",
                       "nvidia-cufft-cu12", "nvidia-curand-cu12",
                       "nvidia-cusolver-cu12", "nvidia-cusparse-cu12",
                       "nvidia-nccl-cu12", "nvidia-nvtx-cu12",
                       "nvidia-cuda-cupti-cu12", "nvidia-cusparselt-cu12",
                       "nvidia-nvjitlink-cu12", "nvidia-cufile-cu12"}]
        other_pins = [r for r in resolved if r not in torch_pins]

        for label, pins, index in (("torch stack", torch_pins, args.torch_index),
                                   ("everything else", other_pins, "")):
            if not pins:
                continue
            print(f"downloading {len(pins)} {label} wheel(s) ...", flush=True)
            argv = [str(staged_python), "-m", "pip", "download", *pins,
                    "--dest", str(wheelhouse), "--only-binary", ":all:",
                    "--no-deps"]
            if index:
                argv += ["--index-url", index]
            result = subprocess.run(argv, capture_output=True, text=True,
                                    timeout=5400)
            if result.returncode != 0:
                print(result.stdout[-1500:], file=sys.stderr)
                print(result.stderr[-1500:], file=sys.stderr)
                raise SystemExit(f"{label} download failed")

        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(cache, ignore_errors=True)

    # The bundle carries the resolution it was built from: the run installs with
    # `-r /opt/env-input/resolved.requirements.txt`.
    if args.pins and not frozen.is_file():
        shutil.copyfile(args.pins, frozen)

    # -- inventory and audit --------------------------------------------------
    entries = []
    denied_names = {d.lower().replace("_", "-")
                    for f in DENIED for d in f.distributions}
    violations: list[str] = []

    for wheel in sorted(wheelhouse.glob("*.whl")):
        meta = wheel_metadata(wheel)
        name = (meta.get("name") or wheel.name.split("-")[0]).lower().replace("_", "-")
        if name in denied_names:
            violations.append(f"{wheel.name}: a denied serving framework")
            continue
        transitive = scan_dependency_graph(meta.get("requires", []))
        if transitive:
            violations.extend(
                f"{wheel.name}: requires {f.evidence}" for f in transitive)
        entries.append({
            "file": wheel.name,
            "name": meta.get("name") or name,
            "version": meta.get("version", ""),
            "sha256": sha256(wheel),
            "bytes": wheel.stat().st_size,
            "license": meta.get("license", "see the wheel's METADATA"),
            "requires": meta.get("requires", []),
            "entry_points": meta.get("entry_points", []),
            "reason": REQUIRED.get(name, "a transitive dependency of a required "
                                         "package"),
        })
    for sdist in sorted(wheelhouse.glob("*.tar.gz")):
        entries.append({
            "file": sdist.name, "name": sdist.name.split("-")[0], "version": "",
            "sha256": sha256(sdist), "bytes": sdist.stat().st_size,
            "license": "see the sdist", "requires": [], "entry_points": [],
            "reason": "no wheel was available; an offline build is intended",
        })

    manifest = {
        "schema_version": 1,
        "python_version": args.python,
        "torch_index": args.torch_index,
        "artifacts": entries,
        "artifact_count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "denylist_violations": violations,
        "clean": not violations,
        "allowlist_note": (
            "generic dependencies and composable kernels are allowed; a built "
            "general-purpose serving framework is not. torch, triton, "
            "transformers and CUDA user-space libraries are here because a "
            "serving system cannot be built without them, and because a "
            "general-purpose framework also using them is not a reason to deny "
            f"them. The full allowed-generic list is {list(ALLOWED_GENERIC)}"),
        "not_included_note": (
            "no live virtualenv, site-packages tree, interpreter prefix, editable "
            "install, .pth injection, build directory or package cache. Those "
            "carry absolute host paths in RECORD files and RPATHs and produce an "
            "environment that works until it is moved"),
    }
    (args.out / "wheelhouse.manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", "utf-8")

    # A checksum file over the whole bundle, so the container can verify what it
    # received rather than trusting the mount.
    lines = []
    for path in sorted(args.out.rglob("*")):
        if path.is_file() and path.name != "hashes.sha256":
            lines.append(f"{sha256(path)}  {path.relative_to(args.out)}")
    (args.out / "hashes.sha256").write_text("\n".join(lines) + "\n", "utf-8")

    print(f"wheelhouse   {len(entries)} artefact(s), "
          f"{manifest['total_bytes'] / 1e9:.2f} GB")
    print(f"denylist     {'clean' if not violations else violations[:3]}")
    print(f"manifest     {args.out / 'wheelhouse.manifest.json'}")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
