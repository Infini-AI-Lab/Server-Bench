"""Prove the kernel toolchain this image PROMISES, at image build time.

The agent brief and the evolution skill both tell a coding agent it may write
CUDA C++, compile PTX in process with NVRTC, or hand-write PTX. Each of those
has a failure that only shows up when it is tried:

* `torch.utils.cpp_extension` without `ninja` dies with
  "Ninja is required to load C++ extensions";
* every kernel that includes `<cuda_fp16.h>` or `<cuda_bf16.h>` dies on
  `fatal error: nv/target: No such file or directory` unless CCCL is
  installed -- and on this workload a kernel that cannot touch bfloat16 is not
  a kernel worth writing;
* NVRTC compiles nothing that includes a CUDA header unless it is given the
  include path, because NVRTC has no default one.

A brief that promises a toolchain the container does not have costs an agent an
hour to disprove and sends it back to Triton. So the promise is checked here,
in the layer that installs the toolchain, and a gap fails the image build.

No GPU is required, and that is deliberate: every one of the failures above is
a compile, link or import failure. Running the kernel needs a device and is
checked separately by the run's own boot contract, on the card the run holds.
"""

from __future__ import annotations

import ctypes
import glob
import shutil

CUDA_SOURCE = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <math.h>

__global__ void k(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = sinpif(x[i]) + tanhf(x[i]);
}

__global__ void scale_bf16(__nv_bfloat16* o, const __nv_bfloat16* a) {
    o[threadIdx.x] = __hmul(a[threadIdx.x], __float2bfloat16(2.f));
}

void bump(torch::Tensor t) {
    k<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<float>(), (int)t.numel());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("bump", &bump); }
"""

NVRTC_SOURCE = b"""
#include <cuda_bf16.h>
extern "C" __global__ void axpy_bf16(__nv_bfloat16* y, const __nv_bfloat16* x) {
    y[threadIdx.x] = __hadd(y[threadIdx.x], x[threadIdx.x]);
}
"""

#: NVRTC resolves no include directory of its own. Without this the compile
#: fails with `catastrophic error: could not open source file "cuda_bf16.h"
#: (no directories in search list)`, which reads like a missing package.
NVRTC_INCLUDE = b"-I/usr/local/cuda/include"


def torch_extension_builds() -> None:
    """`torch.utils.cpp_extension` compiles, links and IMPORTS a CUDA module."""
    from torch.utils import cpp_extension
    build = "/tmp/serverbench-toolchain-smoke"
    shutil.rmtree(build, ignore_errors=True)
    module = cpp_extension.load_inline(
        name="serverbench_toolchain_smoke",
        cpp_sources="",
        cuda_sources=CUDA_SOURCE,
        with_cuda=True,
        verbose=False,
        build_directory=_fresh(build),
    )
    assert hasattr(module, "bump"), "the extension imported without its symbol"
    shutil.rmtree(build, ignore_errors=True)
    print("ok  torch.utils.cpp_extension builds, links and imports a CUDA "
          "kernel using torch tensors, <math.h> and __nv_bfloat16")


def setup_py_extension_builds() -> None:
    """The `setup.py` / BuildExtension path builds the same extension."""
    import os
    import subprocess
    import sys
    root = _fresh("/tmp/serverbench-setup-smoke")
    with open(os.path.join(root, "k.cu"), "w") as handle:
        handle.write(CUDA_SOURCE.replace("TORCH_EXTENSION_NAME", "setup_smoke"))
    with open(os.path.join(root, "setup.py"), "w") as handle:
        handle.write(
            "from setuptools import setup\n"
            "from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n"
            "setup(name='setup_smoke', ext_modules=[CUDAExtension('setup_smoke', ['k.cu'])],\n"
            "      cmdclass={'build_ext': BuildExtension})\n")
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                   cwd=root, check=True, capture_output=True)
    built = glob.glob(os.path.join(root, "setup_smoke*.so"))
    assert built, "setup.py build_ext produced no module"
    shutil.rmtree(root, ignore_errors=True)
    print("ok  setup.py / BuildExtension builds the same CUDA extension")


def nvrtc_compiles_to_ptx() -> None:
    """NVRTC from /usr/local/cuda, loaded BY NAME, compiles CUDA to PTX."""
    assert glob.glob("/usr/local/cuda/include/nvrtc.h"), "no nvrtc.h under /usr/local/cuda"
    nvrtc = ctypes.CDLL("libnvrtc.so.12")

    program = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(
        ctypes.byref(program), NVRTC_SOURCE, b"smoke.cu", 0, None, None)
    assert rc == 0, f"nvrtcCreateProgram returned {rc}"

    options = [b"--gpu-architecture=compute_89", b"-default-device",
               NVRTC_INCLUDE]
    rc = nvrtc.nvrtcCompileProgram(
        program, len(options), (ctypes.c_char_p * len(options))(*options))
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    assert rc == 0, f"nvrtcCompileProgram returned {rc}: {log.value.decode()}"

    nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetPTX(program, ptx)
    assert b".visible .entry" in ptx.value, "NVRTC produced PTX with no entry"
    print("ok  NVRTC compiles a bfloat16 kernel to PTX in process "
          f"(needs {NVRTC_INCLUDE.decode()})")


def _fresh(path: str) -> str:
    import os
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    torch_extension_builds()
    setup_py_extension_builds()
    nvrtc_compiles_to_ptx()
    print("ok  the kernel toolchain this image promises is the one it has")
