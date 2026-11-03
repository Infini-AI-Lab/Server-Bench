"""Proving a built serving framework is not reachable from the task.

The point of the benchmark is what an agent can *build*. An agent that can
`import vllm` is answering a different and much easier question, and the result
afterwards is indistinguishable from an honest one — so absence has to be
checked at every layer a package can arrive through, not just the obvious one.

Eight layers, because each catches what the others miss:

1. **distribution metadata** — installed under any name
2. **import discovery** — findable on the path, without importing it
3. **executables** — a console script on PATH is a framework whether or not it
   imports
4. **filesystem signatures** — a source tree or build directory that no package
   manager knows about
5. **loaded shared objects** — a `.so` already mapped into the process
6. **dependency graph** — something allowed that pulls a denied thing in
7. **subprocess receipts** — shelling out to a framework binary
8. **image layers** — present in the image even if nothing on the path finds it

A check that only confirms `import vllm` raises is layer 2 alone, and layer 2 is
the one a vendored copy walks straight past.

The hard part is not the denial, it is the **line**. CUDA, cuBLAS, NCCL, CUTLASS,
Triton, FlashAttention, PyTorch and Transformers are generic dependencies that
general-purpose frameworks also use, and denying them by association would leave
nothing to build with. What is denied is a *serving framework*: a scheduler, a
cache manager, an engine, a model runner, a server. What is allowed is a
composable kernel or a generic primitive — provided it is independently
installable and does not drag its parent framework in behind it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from ..contracts.base import Contract


@dataclasses.dataclass(frozen=True)
class DeniedFramework(Contract):
    """One framework, and every name it can arrive under."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "denied_framework"

    name: str = ""
    #: Canonical distribution names, as a package index would list them.
    distributions: list[str] = dataclasses.field(default_factory=list)
    #: Top-level import namespaces.
    modules: list[str] = dataclasses.field(default_factory=list)
    #: Console scripts it installs.
    executables: list[str] = dataclasses.field(default_factory=list)
    #: Shared-library filename patterns, as regexes.
    libraries: list[str] = dataclasses.field(default_factory=list)
    #: Directory names that mean a source or build tree is present.
    directories: list[str] = dataclasses.field(default_factory=list)
    why: str = ""

    def validate(self) -> None:
        self.require(bool(self.name), "a denied framework needs a name")
        self.require(bool(self.modules) or bool(self.distributions),
                     "a denial needs at least one name to match on")
        self.require(bool(self.why),
                     "a denial must say why; a list nobody can justify grows until "
                     "it denies something legitimate")


DENIED: tuple[DeniedFramework, ...] = (
    DeniedFramework(
        name="vllm",
        distributions=["vllm", "vllm-flash-attn", "vllm-nccl-cu12"],
        modules=["vllm", "vllm_flash_attn"],
        executables=["vllm"],
        # Anchored to a `vllm/` path component on purpose: a bare
        # `_C\.cpython.*\.so$` also matches PyTorch's own `torch/_C...so`, and
        # an anti-cheat layer that fires on every honest environment trains
        # whoever reads it to ignore the finding.
        libraries=[r"[/\\]vllm[/\\].*\.so$"],
        directories=["vllm", ".vllm"],
        why=("a complete general-purpose serving system: scheduler, paged KV cache, "
             "continuous batching, engine and server. It is the thing the agent is "
             "being asked to build")),
    DeniedFramework(
        name="sglang",
        distributions=["sglang", "sgl-kernel", "sglang-router"],
        modules=["sglang", "sgl_kernel", "sglang_router"],
        executables=["sglang", "sglang-router"],
        libraries=[r"sgl_kernel.*\.so$"],
        directories=["sglang", "sgl-kernel"],
        why=("a complete general-purpose serving system with its own runtime, "
             "radix cache and scheduler")),
    DeniedFramework(
        name="tensorrt_llm",
        distributions=["tensorrt-llm", "tensorrt_llm"],
        modules=["tensorrt_llm"],
        executables=["trtllm-build", "trtllm-serve", "trtllm-bench"],
        libraries=[r"libtensorrt_llm.*\.so", r"libnvinfer_plugin_tensorrt_llm.*\.so"],
        directories=["tensorrt_llm", "TensorRT-LLM"],
        why=("a complete general-purpose serving system plus an engine builder. Its "
             "compiled engines are also denied: a prebuilt engine is the framework's "
             "output and using one is using the framework")),
)

#: Generic dependencies a general-purpose framework also uses. Allowed, and named
#: here so that the denial scan does not flag them by association — which would
#: leave the agent nothing to build a serving system out of.
ALLOWED_GENERIC: tuple[str, ...] = (
    "torch", "triton", "transformers", "tokenizers", "safetensors", "numpy",
    "cuda", "cublas", "cudnn", "nccl", "cutlass", "flash_attn", "accelerate",
    "fastapi", "uvicorn", "httpx", "pydantic", "starlette", "pytest", "ruff",
)


@dataclasses.dataclass(frozen=True)
class LayerFinding(Contract):
    """What one layer of the scan found."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "denylist_layer_finding"

    layer: str = ""
    framework: str = ""
    evidence: str = ""

    def validate(self) -> None:
        self.require(bool(self.layer), "a finding names the layer that found it")


@dataclasses.dataclass(frozen=True)
class AbsenceReport(Contract):
    """Whether the denied frameworks are reachable, layer by layer."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "framework_absence_report"

    context: str = ""
    layers_run: list[str] = dataclasses.field(default_factory=list)
    findings: list[LayerFinding] = dataclasses.field(default_factory=list)
    #: Generic dependencies that *are* present. Expected, and recorded so a reader
    #: can see the scan distinguished them rather than missing them.
    allowed_present: list[str] = dataclasses.field(default_factory=list)

    def validate(self) -> None:
        self.require(bool(self.context), "an absence report names its context")
        self.require(bool(self.layers_run),
                     "a report that does not say which layers ran cannot be read as "
                     "evidence of absence")

    @property
    def clean(self) -> bool:
        return not self.findings

    def summary(self) -> str:
        if self.clean:
            return (f"{len(self.layers_run)} layer(s) checked in {self.context}: no "
                    f"denied framework reachable. "
                    f"{len(self.allowed_present)} allowed generic dependenc(ies) "
                    f"present, as expected")
        by_layer: dict[str, list[str]] = {}
        for finding in self.findings:
            by_layer.setdefault(finding.layer, []).append(finding.framework)
        return f"denied framework(s) reachable: {by_layer}"


# -- the eight layers ---------------------------------------------------------

def scan_distributions() -> list[LayerFinding]:
    """Layer 1: installed distribution metadata, under any name."""
    findings: list[LayerFinding] = []
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        return findings
    installed: set[str] = set()
    for distribution in metadata.distributions():
        meta = distribution.metadata
        name = meta["Name"] if meta is not None and "Name" in meta else None
        if name:
            installed.add(name.lower().replace("_", "-"))
    for framework in DENIED:
        for dist in framework.distributions:
            if dist.lower().replace("_", "-") in installed:
                findings.append(LayerFinding(
                    layer="distributions", framework=framework.name,
                    evidence=f"distribution {dist!r} is installed"))
    return findings


def scan_importable() -> list[LayerFinding]:
    """Layer 2: findable on the path — without importing it.

    Using the finder rather than a real import, because importing an untrusted
    package to discover it is present executes it, and a package that should not
    be here is exactly the one not to run.
    """
    import importlib.util

    findings: list[LayerFinding] = []
    for framework in DENIED:
        for module in framework.modules:
            try:
                spec = importlib.util.find_spec(module)
            except (ImportError, ValueError, ModuleNotFoundError):
                continue
            if spec is not None:
                findings.append(LayerFinding(
                    layer="importable", framework=framework.name,
                    evidence=f"module {module!r} resolves to "
                             f"{spec.origin or '<namespace>'}"))
    return findings


def scan_executables() -> list[LayerFinding]:
    """Layer 3: a console script on PATH is the framework, import or not."""
    findings: list[LayerFinding] = []
    for framework in DENIED:
        for name in framework.executables:
            found = shutil.which(name)
            if found:
                findings.append(LayerFinding(
                    layer="executables", framework=framework.name,
                    evidence=f"{name!r} on PATH at {found}"))
    return findings


def scan_filesystem(roots: Sequence[Path]) -> list[LayerFinding]:
    """Layer 4: a source or build tree no package manager knows about."""
    findings: list[LayerFinding] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for framework in DENIED:
            for directory in framework.directories:
                for hit in list(root.glob(f"**/{directory}"))[:3]:
                    if hit.is_dir():
                        findings.append(LayerFinding(
                            layer="filesystem", framework=framework.name,
                            evidence=f"directory {hit}"))
    return findings


def scan_loaded_libraries(pid: int | None = None) -> list[LayerFinding]:
    """Layer 5: shared objects already mapped into the process."""
    findings: list[LayerFinding] = []
    maps = Path(f"/proc/{pid or os.getpid()}/maps")
    if not maps.exists():
        return findings
    try:
        text = maps.read_text("utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return findings
    for framework in DENIED:
        for pattern in framework.libraries:
            if re.search(pattern, text, re.MULTILINE):
                findings.append(LayerFinding(
                    layer="loaded_libraries", framework=framework.name,
                    evidence=f"a mapped object matches {pattern!r}"))
    return findings


def scan_dependency_graph(requirements: Iterable[str]) -> list[LayerFinding]:
    """Layer 6: something allowed that pulls a denied thing in behind it."""
    findings: list[LayerFinding] = []
    denied_names = {d.lower().replace("_", "-")
                    for f in DENIED for d in f.distributions}
    for requirement in requirements:
        name = re.split(r"[<>=!~\[; ]", requirement.strip(), maxsplit=1)[0]
        if name.lower().replace("_", "-") in denied_names:
            framework = next(f.name for f in DENIED
                             if name.lower().replace("_", "-")
                             in {d.lower().replace("_", "-")
                                 for d in f.distributions})
            findings.append(LayerFinding(
                layer="dependency_graph", framework=framework,
                evidence=f"required by the dependency closure: {requirement!r}"))
    return findings


def scan_subprocess_receipts(receipts: Iterable[Sequence[str]]) -> list[LayerFinding]:
    """Layer 7: a command that shelled out to a framework binary."""
    findings: list[LayerFinding] = []
    for argv in receipts:
        joined = " ".join(argv)
        for framework in DENIED:
            for name in framework.executables:
                if re.search(rf"\b{re.escape(name)}\b", joined):
                    findings.append(LayerFinding(
                        layer="subprocess", framework=framework.name,
                        evidence=f"a recorded command invoked {name!r}: "
                                 f"{joined[:120]}"))
    return findings


def scan_image_inventory(inventory: Iterable[str]) -> list[LayerFinding]:
    """Layer 8: present in the image, even if nothing on the path finds it."""
    findings: list[LayerFinding] = []
    joined = "\n".join(inventory)
    for framework in DENIED:
        for name in {*framework.distributions, *framework.modules}:
            if re.search(rf"(^|[/\s]){re.escape(name)}([/\s=-]|$)", joined,
                         re.MULTILINE):
                findings.append(LayerFinding(
                    layer="image_inventory", framework=framework.name,
                    evidence=f"{name!r} appears in the image inventory"))
    return findings


def present_generics() -> list[str]:
    """Which allowed generic dependencies are here. Expected, and worth recording."""
    import importlib.util

    found: list[str] = []
    for name in ALLOWED_GENERIC:
        try:
            if importlib.util.find_spec(name) is not None:
                found.append(name)
        except (ImportError, ValueError, ModuleNotFoundError):
            continue
    return found


def audit(context: str, *, roots: Sequence[Path] = (),
          requirements: Iterable[str] = (),
          receipts: Iterable[Sequence[str]] = (),
          image_inventory: Iterable[str] = ()) -> AbsenceReport:
    """Run every layer that has an input, and say which ones ran."""
    layers: list[str] = []
    findings: list[LayerFinding] = []

    for name, result in (("distributions", scan_distributions()),
                         ("importable", scan_importable()),
                         ("executables", scan_executables()),
                         ("loaded_libraries", scan_loaded_libraries())):
        layers.append(name)
        findings.extend(result)

    if roots:
        layers.append("filesystem")
        findings.extend(scan_filesystem(roots))
    if requirements:
        layers.append("dependency_graph")
        findings.extend(scan_dependency_graph(requirements))
    if receipts:
        layers.append("subprocess")
        findings.extend(scan_subprocess_receipts(receipts))
    if image_inventory:
        layers.append("image_inventory")
        findings.extend(scan_image_inventory(image_inventory))

    return AbsenceReport(context=context, layers_run=layers, findings=findings,
                         allowed_present=present_generics())


def audit_script(context: str) -> str:
    """A self-contained script that runs the in-process layers where it lands.

    Written out as text rather than imported, because the place that needs
    auditing is inside a container in an environment the agent built — and
    shipping the harness there to check it would put the auditor on the path
    being audited.
    """
    payload = json.dumps({
        "context": context,
        "denied": [{"name": f.name, "distributions": list(f.distributions),
                    "modules": list(f.modules),
                    "executables": list(f.executables),
                    "libraries": list(f.libraries)} for f in DENIED],
        "allowed": list(ALLOWED_GENERIC),
    })
    return _AUDIT_TEMPLATE.replace("__PAYLOAD__", payload)


#: The audited-context script. A plain template with one substitution, so that
#: nothing here depends on quoting rules inside generated code.
_AUDIT_TEMPLATE = '''
import importlib.util, json, os, re, shutil, sys
from importlib import metadata

CONFIG = json.loads(r"""__PAYLOAD__""")
findings = []

installed = set()
for dist in metadata.distributions():
    meta = dist.metadata or {}
    name = meta["Name"] if "Name" in meta else ""
    if name:
        installed.add(name.lower().replace("_", "-"))


def findable(module):
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


for framework in CONFIG["denied"]:
    for name in framework["distributions"]:
        if name.lower().replace("_", "-") in installed:
            findings.append(["distributions", framework["name"], name])
    for module in framework["modules"]:
        if findable(module):
            findings.append(["importable", framework["name"], module])
    for executable in framework["executables"]:
        found = shutil.which(executable)
        if found:
            findings.append(["executables", framework["name"], found])

maps_path = "/proc/" + str(os.getpid()) + "/maps"
maps = open(maps_path).read() if os.path.exists(maps_path) else ""
for framework in CONFIG["denied"]:
    for pattern in framework["libraries"]:
        if re.search(pattern, maps, re.MULTILINE):
            findings.append(["loaded_libraries", framework["name"], pattern])

print(json.dumps({
    "context": CONFIG["context"],
    "layers_run": ["distributions", "importable", "executables",
                   "loaded_libraries"],
    "findings": findings,
    "allowed_present": [n for n in CONFIG["allowed"] if findable(n)],
    "sys_path": sys.path,
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "clean": not findings,
}))
'''
