# The evolve task image, derived from the serving-agent image rather than rebuilt.
#
# The expensive layers -- torch/CUDA, the agent binary, the base -- are inherited
# unchanged. This layer adds the profilers, a CUDA compiler, and the two things
# an agent that builds its own environment needs:
#
#   * `uv`, so the agent can build its own environment rather than find one;
#   * an interpreter configuration under which a virtualenv the agent creates
#     cannot silently inherit the parent image's site-packages.
#
# The second point is the subtle one. The parent image installs torch and
# transformers into the system interpreter. The agent must *build* its
# environment, and an inherited site-packages would make its .venv a facade: the
# imports would resolve, the manifest would show packages from outside the
# workspace, and the environment would be unreconstructable.
#
# Built by `serverbench/scripts/deploy/03-build-images.sh`, which runs:
#   docker build \
#     -t serverbench/evolve-agent:cu128 \
#     --build-arg AGENT_UID="$(id -u)" --build-arg AGENT_GID="$(id -g)" \
#     -f serverbench/environments/evolve-agent/Dockerfile .

# ---------------------------------------------------------------------------
# Stage 1: the Nsight command-line profilers
# ---------------------------------------------------------------------------
#
# The agent brief names `nsys` and `ncu`, and an offline agent has no way to
# install a missing profiler.
#
# Installed here rather than copied from the host: a fresh server must be able
# to build this image from the Dockerfile alone, and `/opt/nvidia/nsight-*` is
# not something a clean machine has. The versions are pinned because an
# unpinned profiler is a different measurement instrument on every rebuild.
#
# Both packages ship a GUI half -- Qt, an X stack, the report viewer -- that is
# useless in a headless container and costs about 900 MB. It is removed in this
# stage, so the layer never reaches the final image.
FROM serverbench/serving-agent:cu128 AS nsight

USER root
ARG NSYS_VERSION=2025.5.2
ARG NCU_VERSION=2025.4.1

# Fetched by exact URL, verified by digest, and UNPACKED rather than installed.
#
# `dpkg -x` instead of `dpkg -i` because the nsight-systems package declares
# dependencies its command line does not use: libxtst6, libopengl0, libegl1,
# libxi6, libxcb-dri2-0 -- an X11 and OpenGL stack that exists for `nsys-ui`,
# the report viewer, which cannot run in a headless container and is deleted
# below anyway. `dpkg -i` refuses on those and leaves the package unconfigured;
# pulling an X stack into a serving image to satisfy a viewer nobody opens is
# the wrong answer. Unpacking gives the CLI tree, which is all the profilers
# need, and the smoke test at the end of this stage proves both binaries run.
#
# This base is Debian trixie, whose apt refuses NVIDIA's repository outright:
# the signing key carries a SHA1 binding signature, which trixie's `sqv`
# considers insecure, so `apt-get update` exits 100. Adding `[trusted=yes]` would "fix" it by turning signature
# checking off for a 700 MB download, which is the wrong trade -- a pinned URL
# with a pinned digest is stronger than an unverified repo, and it says exactly
# which bytes are in this image.
ARG NSYS_DEB=nsight-systems-2025.5.2_2025.5.2.266-1_amd64.deb
ARG NCU_DEB=nsight-compute-2025.4.1_2025.4.1.2-1_amd64.deb
ARG NSYS_SHA256=e818fefc62b1e6c3bd34eb92b4e412355afddb763b55e8d7bc3330c4e3da4d33
ARG NCU_SHA256=39016047b0dabd1c11e40f51a275223acd8e761ed299beb348b2a09ee8eccfaf
ARG CUDA_REPO=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends wget; \
    cd /tmp; \
    wget -q "${CUDA_REPO}/${NSYS_DEB}" "${CUDA_REPO}/${NCU_DEB}"; \
    echo "${NSYS_SHA256}  ${NSYS_DEB}" | sha256sum -c -; \
    echo "${NCU_SHA256}  ${NCU_DEB}" | sha256sum -c -; \
    dpkg -x "${NSYS_DEB}" /; \
    dpkg -x "${NCU_DEB}" /; \
    rm -f "${NSYS_DEB}" "${NCU_DEB}"; \
    rm -rf /var/lib/apt/lists/*

# The parts this container cannot use, and NOT one byte more.
#
# `nsight-systems/host-linux-x64` looks like the GUI half and mostly is -- but it
# also holds `QdstrmImporter`, which converts the `.qdstrm` the collector writes
# into the `.nsys-rep` that `nsys stats` reads. Without it the profiler collects
# a trace and then cannot open it ("Unable to retrieve the importer version:
# skipping importation of the QDSTRM file"), so that directory stays.
#
# nsight-compute's `host/` and `ncu-ui` are pure GUI; ncu works without them.
RUN rm -rf "/opt/nvidia/nsight-systems/${NSYS_VERSION}/documentation" \
           "/opt/nvidia/nsight-compute/${NCU_VERSION}/host" \
           "/opt/nvidia/nsight-compute/${NCU_VERSION}/ncu-ui" \
           "/opt/nvidia/nsight-compute/${NCU_VERSION}/docs" \
           "/opt/nvidia/nsight-compute/${NCU_VERSION}/extras" \
    && rm -rf "/opt/nvidia/nsight-compute/${NCU_VERSION}/target/linux-desktop-glibc_2_11_3-x86" \
              "/opt/nvidia/nsight-compute/${NCU_VERSION}/target/linux-desktop-t210-a64" \
    && du -sh /opt/nvidia/nsight-systems /opt/nvidia/nsight-compute



# ---------------------------------------------------------------------------
# Stage 1b: the CUDA compiler
# ---------------------------------------------------------------------------
#
# WHY THIS IS HERE. `torch.utils.cpp_extension.load()` -- the idiomatic way to
# add a CUDA kernel to a PyTorch project -- needs `nvcc`, `/usr/local/cuda` and
# `ninja`, and the agent brief says it can compile CUDA. Without them the only
# kernel toolchain that runs out of the box is Triton.
#
# WHY 12.8, AND THE ONE-HEADER FIX IT NEEDS. torch is the cu128 build, and an
# extension must be compiled by the CUDA it links against: 12.8.93 is the exact
# toolkit of torch's own `nvidia-cuda-nvcc-cu12` / NVRTC / cuBLAS wheels. The
# base is Debian trixie with glibc 2.41, which declares the C23
# `sinpi`/`sinpif`/`cospi`/`cospif` family `noexcept`, while CUDA 12.x's
# `crt/math_functions.h` declares the same four without it, so every compile
# dies with four "exception specification is incompatible" errors. The fix is
# the four declarations, given the `noexcept (true)` glibc already gives them --
# the same patch distributions carry for CUDA 12.x on glibc >= 2.41 -- applied
# and then proven by the smoke compile below, which calls `sinpif` and `cospif`.
#
# `cuda-nvvm-12-8` is NOT optional and is easy to miss: without it nvcc runs,
# prints its version, and then dies with `cicc: not found` on the first real
# compile.
#
# `cuda-cccl-12-8` IS ALSO NOT OPTIONAL. nvcc without it compiles a kernel that
# includes nothing, and dies on the first one that includes anything real:
#
#     /usr/local/cuda/include/cuda_fp16.h:4492:10:
#         fatal error: nv/target: No such file or directory
#
# `nv/target` comes from CCCL, and `cuda_fp16.h` / `cuda_bf16.h` include it --
# so without it every half- or bfloat16 kernel, which on this workload is every
# kernel worth writing, fails.
#
# Same mechanism as the profilers: pinned URL, pinned digest, and `dpkg -x`
# rather than `dpkg -i`, because this base's apt refuses NVIDIA's repository
# over a SHA1 signing key.
FROM serverbench/serving-agent:cu128 AS cudac

USER root
ARG CUDA_REPO=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64

ARG NVCC_DEB=cuda-nvcc-12-8_12.8.93-1_amd64.deb
ARG CRT_DEB=cuda-crt-12-8_12.8.93-1_amd64.deb
ARG NVVM_DEB=cuda-nvvm-12-8_12.8.93-1_amd64.deb
ARG CUDART_DEB=cuda-cudart-12-8_12.8.90-1_amd64.deb
ARG CUDART_DEV_DEB=cuda-cudart-dev-12-8_12.8.90-1_amd64.deb
ARG CCCL_DEB=cuda-cccl-12-8_12.8.90-1_amd64.deb
ARG NVCC_SHA256=eaa8aaa7fc4bfbf8f4bf10fd5a29f04e0cf60142e1a61de42f03aba76835070e
ARG CRT_SHA256=92a3c8eda8d3a3a8f3de71e0acf4edb8be568b20792e5bf7618898a72ffd6d10
ARG NVVM_SHA256=9d27f5895dd5ea1bb8db3180faa30d5a3d2a50237ed5a82c2ec6b65665b6cba1
ARG CUDART_SHA256=929554bc2f0831eefec4e212897d0cbb0db70cab75c5eeadd2a4a43e7d7ffb68
ARG CUDART_DEV_SHA256=b9e927a530535631f789098080e5e59bcc3bc2ac368d7c7af51ea4fbfe8ad7eb
ARG CCCL_SHA256=873dbdfc3c1aa2806145c94ba75d058f658974ffa2366c4bba9190440bacece4
# NVRTC under /usr/local/cuda (library + nvrtc.h), so the brief's "NVRTC and
# the CUDA headers under /usr/local/cuda" is true and `libnvrtc.so` loads by
# name. And the cuBLAS / cuSPARSE / cuSOLVER HEADERS that
# `ATen/cuda/CUDAContext.h` includes: without them any extension touching
# torch's CUDA context fails on `cusparse.h: No such file or directory`.
# Headers only -- the libraries come from torch's own wheels at runtime.
ARG NVRTC_DEB=cuda-nvrtc-12-8_12.8.93-1_amd64.deb
ARG NVRTC_DEV_DEB=cuda-nvrtc-dev-12-8_12.8.93-1_amd64.deb
ARG CUBLAS_DEV_DEB=libcublas-dev-12-8_12.8.4.1-1_amd64.deb
ARG CUSPARSE_DEV_DEB=libcusparse-dev-12-8_12.5.8.93-1_amd64.deb
ARG CUSOLVER_DEV_DEB=libcusolver-dev-12-8_11.7.3.90-1_amd64.deb
ARG NVRTC_SHA256=0b97fd1c36434f55292ca1c069247cd38a8709750820214c3aa45deac168db60
ARG NVRTC_DEV_SHA256=cf993245b236e731c5e358dfe394466a17962c3883fb7ebf4be24500d862a35c
ARG CUBLAS_DEV_SHA256=484eed80b2622605af35f3d237647a5ca3a411623ea05cf8ab811e0c3367ba6a
ARG CUSPARSE_DEV_SHA256=855d97a8a31d8531240b2613cbcfbc421daaa24cd797f0dec77ba2a8d62213db
ARG CUSOLVER_DEV_SHA256=792f92d87db33859debb33f815da5a131fc27bc438b9e4ccaf7b063cc82bd774

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends wget ca-certificates; \
    mkdir -p /tmp/cd; cd /tmp/cd; \
    wget -q "${CUDA_REPO}/${NVCC_DEB}" "${CUDA_REPO}/${CRT_DEB}" \
         "${CUDA_REPO}/${NVVM_DEB}" "${CUDA_REPO}/${CUDART_DEB}" \
         "${CUDA_REPO}/${CUDART_DEV_DEB}" "${CUDA_REPO}/${CCCL_DEB}"; \
    echo "${NVCC_SHA256}  ${NVCC_DEB}"             | sha256sum -c -; \
    echo "${CRT_SHA256}  ${CRT_DEB}"               | sha256sum -c -; \
    echo "${NVVM_SHA256}  ${NVVM_DEB}"             | sha256sum -c -; \
    echo "${CUDART_SHA256}  ${CUDART_DEB}"         | sha256sum -c -; \
    echo "${CUDART_DEV_SHA256}  ${CUDART_DEV_DEB}" | sha256sum -c -; \
    echo "${CCCL_SHA256}  ${CCCL_DEB}"             | sha256sum -c -; \
    for f in *.deb; do dpkg -x "$f" /; done; \
    wget -q "${CUDA_REPO}/${NVRTC_DEB}" "${CUDA_REPO}/${NVRTC_DEV_DEB}" \
         "${CUDA_REPO}/${CUBLAS_DEV_DEB}" "${CUDA_REPO}/${CUSPARSE_DEV_DEB}" \
         "${CUDA_REPO}/${CUSOLVER_DEV_DEB}"; \
    echo "${NVRTC_SHA256}  ${NVRTC_DEB}"           | sha256sum -c -; \
    echo "${NVRTC_DEV_SHA256}  ${NVRTC_DEV_DEB}"   | sha256sum -c -; \
    echo "${CUBLAS_DEV_SHA256}  ${CUBLAS_DEV_DEB}" | sha256sum -c -; \
    echo "${CUSPARSE_DEV_SHA256}  ${CUSPARSE_DEV_DEB}" | sha256sum -c -; \
    echo "${CUSOLVER_DEV_SHA256}  ${CUSOLVER_DEV_DEB}" | sha256sum -c -; \
    dpkg -x "${NVRTC_DEB}" /; dpkg -x "${NVRTC_DEV_DEB}" /; \
    for f in "${CUBLAS_DEV_DEB}" "${CUSPARSE_DEV_DEB}" "${CUSOLVER_DEV_DEB}"; do \
      rm -rf /tmp/hdr; dpkg -x "$f" /tmp/hdr; \
      cp -a /tmp/hdr/usr/local/cuda-12.8/targets/x86_64-linux/include/. \
            /usr/local/cuda-12.8/targets/x86_64-linux/include/; \
    done; rm -rf /tmp/hdr; \
    rm -f /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc_static.a \
          /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc-builtins_static.a; \
    test -f /usr/local/cuda-12.8/include/nvrtc.h; \
    test -f /usr/local/cuda-12.8/include/cusparse.h; \
    test -f /usr/local/cuda-12.8/include/cublas_v2.h; \
    test -f /usr/local/cuda-12.8/include/cusolverDn.h; \
    ls /usr/local/cuda-12.8/lib64/libnvrtc.so*; \
    H=/usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h; \
    sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +(sinpi|sinpif|cospi|cospif)\((double|float) x\));/\1 noexcept (true);/' "$H"; \
    test "$(grep -cE '(sinpi|sinpif|cospi|cospif)\((double|float) x\) noexcept \(true\);' "$H")" = 4; \
    cd /; rm -rf /tmp/cd /var/lib/apt/lists/*; \
    /usr/local/cuda-12.8/bin/nvcc --version | tail -2

# The build actually works, proven in the layer that installs it rather than
# asserted in a comment. A compiler that resolves but cannot compile is the
# failure this catches, and it is exactly what an unpatched 12.8 does here.
#
# The kernel below is deliberately NOT a bare `a+b`: a toolkit missing CCCL
# passes a smoke test of that shape and fails on the first kernel that touches
# `__nv_bfloat16` or `<math.h>` -- which is every kernel these workloads want.
# So the smoke includes both, and a header gap fails the image build instead of
# an agent's hour.
RUN set -eu; \
    printf '%s\n' \
      '#include <cuda_runtime.h>' \
      '#include <math.h>' \
      '#include <cuda_bf16.h>' \
      'extern "C" __global__ void add(float*o,const float*a,const float*b,int n){' \
      '  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]+b[i]; }' \
      'extern "C" __global__ void act(float*o,const float*a,int n){' \
      '  int i=blockIdx.x*blockDim.x+threadIdx.x;' \
      '  if(i<n) o[i]=sinpif(a[i])+cospif(a[i])+erff(a[i])+tanhf(a[i]); }' \
      'extern "C" __global__ void scale_bf16(__nv_bfloat16*o,const __nv_bfloat16*a){' \
      '  o[threadIdx.x]=__hmul(a[threadIdx.x],__float2bfloat16(2.f)); }' \
      'extern "C" __global__ void fma_ptx(float*o,const float*a,const float*b){' \
      '  float r; asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r)' \
      '    : "f"(a[0]),"f"(b[0]),"f"(0.0f)); o[0]=r; }' \
      > /tmp/smoke.cu; \
    P=/usr/local/cuda-12.8/bin; \
    "$P/nvcc" -arch=sm_89 -ptx   /tmp/smoke.cu -o /tmp/smoke.ptx; \
    "$P/nvcc" -arch=sm_89 -cubin /tmp/smoke.cu -o /tmp/smoke.cubin; \
    "$P/nvcc" -arch=sm_89 -c     /tmp/smoke.cu -o /tmp/smoke.o; \
    "$P/ptxas" -arch=sm_89 /tmp/smoke.ptx -o /tmp/smoke2.cubin; \
    rm -f /tmp/smoke.*; \
    echo "nvcc compiles bf16 + libm .cu to PTX, cubin and object; ptxas assembles PTX"


# ---------------------------------------------------------------------------
# Stage 2: the task image
# ---------------------------------------------------------------------------
FROM serverbench/serving-agent:cu128

USER root

ARG NSYS_VERSION=2025.5.2
ARG NCU_VERSION=2025.4.1

# The pruned profilers, and the two names the tools invoke. `ncu` needs the
# host's `NVreg_RestrictProfilingToAdminUsers=0` to profile as a non-root user;
# that is a host setting, checked by the profile tool at run time rather than
# assumed here, because an image cannot fix it and a confusing permission error
# is exactly what wastes an agent's afternoon.
COPY --from=nsight /opt/nvidia/nsight-systems /opt/nvidia/nsight-systems
COPY --from=nsight /opt/nvidia/nsight-compute /opt/nvidia/nsight-compute
RUN ln -sf "/opt/nvidia/nsight-systems/${NSYS_VERSION}/target-linux-x64/nsys" /usr/local/bin/nsys \
 && ln -sf "/opt/nvidia/nsight-compute/${NCU_VERSION}/ncu" /usr/local/bin/ncu \
 && nsys --version && ncu --version

# The CUDA compiler, from the stage above. `/usr/local/cuda` is the symlink
# every build system looks for, and `CUDA_HOME` is what `cpp_extension` reads.
COPY --from=cudac /usr/local/cuda-12.8 /usr/local/cuda-12.8
# ONE nvcc, REACHED ONLY THROUGH `/usr/local/cuda/bin`, AND NOT SYMLINKED INTO
# `/usr/local/bin` -- which is what the profilers above do, and what does not
# work here.
#
# nvcc finds its own headers RELATIVE TO ITS OWN PATH: `../targets/
# x86_64-linux/include`. Through `/usr/local/cuda/bin/nvcc` that resolves (a
# directory symlink keeps the tree shape). Through a FILE symlink at
# `/usr/local/bin/nvcc` it does not, and every compile dies with
#
#     fatal error: cuda_runtime.h: No such file or directory
#
# which reads like a missing package and is a wrong link. And that copy wins:
# `ENV PATH` puts `/usr/local/cuda/bin` first, but a LOGIN shell re-sources
# `/etc/profile` and `/usr/local/bin` comes back ahead of it. So the broken one
# is removed rather than ordered around.
RUN ln -sfn /usr/local/cuda-12.8 /usr/local/cuda \
 && rm -f /usr/local/bin/nvcc /usr/local/bin/ptxas \
 && /usr/local/cuda/bin/nvcc --version | tail -2

# AND A LOGIN SHELL HAS TO FIND IT TOO.
#
# `ENV PATH` covers `bash -c`; it does NOT survive `bash -lc`, which re-sources
# `/etc/profile` and rebuilds PATH from scratch, so without this file the same
# `nvcc` line that compiles under `bash -c` reports `nvcc: command not found`
# under `bash -lc`. Which of the two an agent's shell uses is not something this
# image gets to decide, so both are made to work.
RUN printf '%s\n' \
      'export CUDA_HOME=/usr/local/cuda' \
      'case ":$PATH:" in *:/usr/local/cuda/bin:*) ;; *) PATH=/usr/local/cuda/bin:$PATH ;; esac' \
      'export PATH' \
      'case ":${LD_LIBRARY_PATH-}:" in *:/usr/local/cuda/lib64:*) ;; *) LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;; esac' \
      'export LD_LIBRARY_PATH' \
      > /etc/profile.d/cuda.sh \
 && chmod 0644 /etc/profile.d/cuda.sh \
 && bash -lc 'nvcc --version | tail -1'

# `ninja`, because `torch.utils.cpp_extension` REQUIRES it and says so only
# when you try:
#
#     RuntimeError: Ninja is required to load C++ extensions
#
# It is in the frozen wheelhouse for the agent's own `.venv`, and the system
# interpreter needs it too, so the documented `cpp_extension` path works for an
# agent that has not built a venv yet. Installed from the same pinned
# wheelhouse, so the image does not depend on an index staying up.
COPY serverbench/artifacts/env-input/wheelhouse/ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl /tmp/
RUN pip install --no-cache-dir --root-user-action=ignore --no-index \
      /tmp/ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
 && rm -f /tmp/ninja-*.whl \
 && ninja --version

# AND THE WHOLE PATH IS PROVEN HERE, not promised in the brief.
#
# `toolchain_smoke.py` builds and imports a real `torch.utils.cpp_extension`
# module whose kernel uses torch tensors, `<math.h>` and `__nv_bfloat16`, then
# compiles a bfloat16 kernel to PTX through NVRTC -- the three things the brief
# promises. A gap fails this build instead of an agent's first hour.
#
# It needs no GPU: what it guards against are compile, link and import
# failures. Executing a kernel needs a device and is checked by the run's own
# boot contract, on the card that run holds.
#
# `LD_LIBRARY_PATH` puts the toolkit's `libnvrtc.so` and `libcudart.so.12` on
# the loader path for every process. Both are the 12.8 builds torch's own
# wheels carry (NVRTC 12.8.93, cudart 12.8.90), so an extension, NVRTC and
# torch all use one CUDA 12.8 runtime.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
# Login shells pick up the workspace's per-card GPU variables (gpu_env.sh says
# so); serve.sh sources the same file before it starts a server.
RUN printf '%s\n' \
      'if [ -r "${APP_DIR:-/workspace}/gpu_env.sh" ]; then . "${APP_DIR:-/workspace}/gpu_env.sh"; fi' \
      > /etc/profile.d/mle_gpu.sh \
 && chmod 0644 /etc/profile.d/mle_gpu.sh
# Registered with the loader too, so `libnvrtc.so` / `libcudart.so.12` resolve
# by name in every process, not only where LD_LIBRARY_PATH survived.
RUN echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf && ldconfig \
 && ldconfig -p | grep -q "libnvrtc.so.12"
# nvcc and torch share CUDA 12.8, so `cpp_extension`'s version check (a hard
# error on a MAJOR mismatch) holds unpatched for `setup.py`, BuildExtension,
# `load()` and `load_inline()` alike. This asserts the pairing.
RUN python3 -c "import torch, subprocess; v=subprocess.run(['/usr/local/cuda/bin/nvcc','--version'],capture_output=True,text=True).stdout; assert 'release 12.8' in v and torch.version.cuda.startswith('12.8'), (v, torch.version.cuda); print('nvcc and torch share CUDA', torch.version.cuda)"
COPY serverbench/environments/evolve-agent/toolchain_smoke.py /tmp/toolchain_smoke.py
RUN TORCH_CUDA_ARCH_LIST=8.9 python3 /tmp/toolchain_smoke.py \
 && rm -f /tmp/toolchain_smoke.py

# The uv binary, copied from the frozen input bundle rather than downloaded --
# there is no network at run time, and a build that fetches it makes the image
# depend on an index remaining up.
# The build context is the REPOSITORY ROOT -- `03-build-images.sh` does
# `cd "$REPO"` and passes `.` -- so this path needs its `serverbench/`
# prefix. Without it the build fails with 'failed to compute cache key:
# "/artifacts/...": not found', which reads like a missing artefact and
# is a wrong relative path.
COPY serverbench/artifacts/env-input/uv-binary/uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv && uv --version

# Where the read-only bundles land. Created here so the mounts have somewhere to
# go and the paths are the same whether or not a mount is present.
RUN mkdir -p /opt/wheelhouse /opt/env-input /models /workspace \
 && chown -R 1000:1000 /workspace

# The isolation the agent's environment depends on.
#
# PYTHONNOUSERSITE and an empty PYTHONPATH are necessary and not sufficient --
# an environment variable can be unset by anything -- so the environment manifest
# independently checks sys.path, the interpreter prefix and every package's
# resolved location. These are the belt; the manifest is the braces.
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH= \
    UV_OFFLINE=1 \
    UV_NO_INDEX=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/workspace/.uv-cache \
    PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

ARG AGENT_UID=1000
ARG AGENT_GID=1000
RUN if [ "${AGENT_UID}" != "1000" ]; then \
      usermod -u "${AGENT_UID}" agent 2>/dev/null || true; \
      groupmod -g "${AGENT_GID}" agent 2>/dev/null || true; \
      chown -R "${AGENT_UID}:${AGENT_GID}" /workspace; \
    fi

USER agent
ENV HOME=/home/agent
WORKDIR /workspace
