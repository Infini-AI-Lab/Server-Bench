#!/usr/bin/env python3
"""Read CUDA-visible GPU clock state; explicitly guarded, single-GPU recovery.

No profiling session, counter collection, CUDA context, or clock-setting API is
used by check. The versioned PerfWorks/CUPTI ABI is shared by the CUDA 12.8 wheel
and CUDA 13 toolkit. Missing/unsupported APIs are UNKNOWN, never inferred from idle MHz.
Visibility is NOT an exclusive lease: recover requires the caller's --exclusive
attestation that the card is allocated to this run. The container has no
operator channel, so every refusal names itself as evidence to record.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import ctypes.util
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

#: What a refusal is: infrastructure evidence. Nobody reads the container
#: live, so it counts only when it is kept in the workspace.
EVIDENCE = "infrastructure evidence: record it in the workspace"


class ProbeError(RuntimeError):
    pass


class Init(ct.Structure):
    _fields_ = [("structSize", ct.c_size_t), ("pPriv", ct.c_void_p)]


class Count(ct.Structure):
    _fields_ = Init._fields_ + [("numDevices", ct.c_size_t)]


class Bus(ct.Structure):
    _fields_ = [("domain", ct.c_uint32), ("bus", ct.c_uint16), ("device", ct.c_uint16)]


class Buses(ct.Structure):
    _fields_ = Init._fields_ + [("pBusIds", ct.POINTER(Bus)), ("numDevices", ct.c_size_t)]


class Clock(ct.Structure):
    _fields_ = Init._fields_ + [("deviceIndex", ct.c_size_t),
                              ("clockStatus", ct.c_int), ("clockLevel", ct.c_int)]


STATES = {0: "UNKNOWN", 1: "LOCKED_TO_RATED_TDP", 2: "BOOST_ENABLED",
          3: "BOOST_DISABLED", 4: "LOCKED", 5: "UNLOCKED"}
LOCKED = {"LOCKED", "LOCKED_TO_RATED_TDP"}
NORMAL = {"BOOST_ENABLED", "UNLOCKED"}


def library(name):
    candidates = []
    for pattern in (f"/usr/local/cuda*/targets/*/lib/lib{name}.so*",
                    f"/usr/local/cuda*/extras/CUPTI/lib64/lib{name}.so*"):
        candidates += sorted(glob.glob(pattern), reverse=True)
    # The offline agent image carries CUPTI/PerfWorks in torch's NVIDIA wheel,
    # not in its intentionally minimal nvcc toolkit or the system loader path.
    for directory in sys.path:
        if os.path.isabs(directory):
            candidates += sorted(glob.glob(
                f"{directory}/nvidia/cuda_cupti/lib/lib{name}.so*"), reverse=True)
    candidates += [path for path in [ct.util.find_library(name)] if path]
    candidates += [f"lib{name}.so", f"lib{name}.so.1"]
    for path in dict.fromkeys(candidates):
        try:
            return ct.CDLL(path)
        except OSError:
            pass
    raise ProbeError(f"cannot load {name}: the CUDA/CUPTI library is not in this "
                     f"container ({EVIDENCE})")


def call(lib, name, *args):
    try:
        result = getattr(lib, name)(*args)
    except AttributeError as exc:
        raise ProbeError(f"unsupported SDK entry point: {name}") from exc
    if result != 0:
        raise ProbeError(f"{name} returned {result}")


def params(lib, name, value):
    value.structSize = ct.sizeof(value)
    call(lib, name, ct.byref(value))
    return value


def pci_key(text):
    match = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)(?:\.[0-7])?", text)
    if not match:
        raise ProbeError(f"unrecognized PCI identity: {text!r}")
    return tuple(int(part, 16) for part in match.groups())


def visible_devices():
    """CUDA ordinals honor CUDA_VISIBLE_DEVICES, unlike nvidia-smi indices."""
    cuda = library("cuda")
    call(cuda, "cuInit", 0)
    count = ct.c_int()
    call(cuda, "cuDeviceGetCount", ct.byref(count))
    devices = []
    for ordinal in range(count.value):
        device, raw_uuid, pci = ct.c_int(), (ct.c_ubyte * 16)(), ct.create_string_buffer(32)
        call(cuda, "cuDeviceGet", ct.byref(device), ordinal)
        get_uuid = "cuDeviceGetUuid_v2" if hasattr(cuda, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
        call(cuda, get_uuid, ct.byref(raw_uuid), device)
        call(cuda, "cuDeviceGetPCIBusId", pci, len(pci), device)
        devices.append({"cuda_ordinal": ordinal, "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid))),
                        "pci_bus_id": pci.value.decode()})
    return devices


def probe():
    devices = visible_devices()
    if not devices:
        raise ProbeError("no CUDA-visible GPUs; do not widen device visibility")
    cupti, nvperf = library("cupti"), library("nvperf_host")
    params(cupti, "cuptiProfilerInitialize", Init())
    params(nvperf, "NVPW_InitializeTarget", Init())
    # CUDA toolkit exports this standard size/pPriv initializer; older SDKs
    # without it fail closed. It discovers devices, not a profiling session.
    params(nvperf, "NVPW_CUDA_LoadDriver", Init())
    count = params(nvperf, "NVPW_GetDeviceCount", Count()).numDevices
    if not 0 < count <= 256:
        raise ProbeError(f"invalid PerfWorks device count: {count}")
    buses = (Bus * count)()
    params(nvperf, "NVPW_Device_GetPciBusIds", Buses(pBusIds=buses, numDevices=count))
    for device in devices:
        key = pci_key(device["pci_bus_id"])
        indices = [i for i, bus in enumerate(buses) if (bus.domain, bus.bus, bus.device) == key]
        if len(indices) != 1:
            raise ProbeError(f"ambiguous PerfWorks PCI mapping for {device['uuid']}")
        clock = params(nvperf, "NVPW_Device_GetClockStatus", Clock(deviceIndex=indices[0]))
        device.update(clock_status=STATES.get(clock.clockStatus, "UNKNOWN"),
                      clock_level=clock.clockLevel)
    return devices


def assert_idle(gpu_uuid):
    result = subprocess.run(["nvidia-smi", "-i", gpu_uuid, "-q", "-x"],
                            capture_output=True, text=True, timeout=15, check=True)
    root = ET.fromstring(result.stdout)
    devices = root.findall("gpu")
    if len(devices) != 1 or devices[0].findtext("uuid") != gpu_uuid:
        raise ProbeError("nvidia-smi device identity did not match; refusing recovery")
    processes = devices[0].find("processes")
    if processes is None or (processes.text or "").strip() not in {"", "None"}:
        raise ProbeError(f"process visibility unavailable; refusing recovery ({EVIDENCE})")
    if list(processes):
        raise ProbeError("GPU has active processes; stop only your own server/profiler first")


def recover(gpu_uuid, exclusive):
    if not exclusive:
        raise ProbeError("--exclusive is required: it attests that this card is "
                         "allocated to this run")
    if not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", gpu_uuid):
        raise ProbeError("use a full GPU UUID from check, never a host/container index")
    devices = probe()
    matches = [item for item in devices if item["uuid"] == gpu_uuid]
    if len(matches) != 1:
        raise ProbeError("target is not uniquely CUDA-visible; do not widen the allocation")
    before = matches[0]
    if before["clock_status"] in NORMAL:
        return {"action": "none", "reason": "already unlocked", "before": before}
    if before["clock_status"] not in LOCKED:
        raise ProbeError(f"no explicit recoverable lock; refusing recovery ({EVIDENCE})")
    ncu = shutil.which("ncu")
    if not ncu:
        raise ProbeError(f"ncu unavailable; refusing recovery ({EVIDENCE}); installing "
                         f"or escalating is outside the task")
    assert_idle(gpu_uuid)
    # --devices uses CUDA enumeration. Restrict that enumeration to this UUID,
    # then select only local ordinal 0. Never invoke an unscoped clock reset.
    command = [ncu, "--clock-control", "reset", "--devices", "0"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    report = {"action": "reset", "before": before, "command": command,
              "CUDA_VISIBLE_DEVICES": gpu_uuid, "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr, "recovered": False}
    try:
        report["after"] = next(item for item in probe() if item["uuid"] == gpu_uuid)
        report["recovered"] = result.returncode == 0 and report["after"]["clock_status"] in NORMAL
    except Exception as exc:
        report["after_error"] = str(exc)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="read-only; no clock modifications")
    recovery = commands.add_parser("recover", help="one idle, exclusively assigned GPU only")
    recovery.add_argument("--uuid", required=True)
    recovery.add_argument("--exclusive", action="store_true",
                          help="attest exclusive ownership; this does NOT acquire a GPU lease")
    args = parser.parse_args(argv)
    report = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        if args.command == "check":
            report.update(action="check", devices=probe())
            code = 0 if all(d["clock_status"] in NORMAL | LOCKED for d in report["devices"]) else 2
        else:
            report.update(recover(args.uuid, args.exclusive))
            code = 0 if report.get("recovered") or report["action"] == "none" else 2
    except Exception as exc:
        report.update(status="UNKNOWN_OR_REFUSED", error=str(exc))
        code = 2
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
