# An image an agent can build and run a real GPU serving system in, offline.
#
# It has to carry torch, transformers and everything the starter server needs,
# because behind the firewall nothing installs. Anything missing here is missing
# partway into a run, inside a container, and the failure gets charged to the
# candidate.
#
# Built by `serverbench/scripts/deploy/03-build-images.sh` (needs network; the
# *task* is offline, the image build is not), which runs, in essence:
#   docker build \
#     -t serverbench/serving-agent:cu128 \
#     --build-arg CLAUDE_BIN="$(readlink -f "$(command -v claude)")" \
#     --build-arg AGENT_UID="$(id -u)" --build-arg AGENT_GID="$(id -g)" \
#     -f serverbench/environments/serving-agent/Dockerfile .
#
# The torch wheels bundle their own CUDA runtime, so a slim base is enough as long
# as the NVIDIA container runtime injects the driver. That keeps the image about
# half the size of a CUDA base image and removes a whole apt repository from the
# build.

FROM python:3.12-slim

# `bash` because launch scripts are `bash serve.sh`; `procps` so process
# management by recorded PID is inspectable from inside; `git` because an agent
# reasonably expects to diff its own work. `iptables` and `dnsutils` are for the
# egress allowlist. `build-essential` because a kernel candidate has to compile,
# and a toolchain that is absent at run time is an evaluator failure rather than a
# candidate one -- which is exactly the confusion this line prevents.
#
# `curl` because an agent polls readiness with it. Without it a loop like
# `for i in $(seq 1 60); do curl ...; sleep 5; done` cannot succeed and cannot
# break early, and burns minutes against a server that is already up. The
# workspace also ships blocking waiters in `wait/` that use python3 and are
# faster than any curl loop -- but a tool an agent will reach for anyway should
# be present, because its absence is discovered as a mysterious silence rather
# than as a missing package.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      bash procps git ca-certificates iptables dnsutils iproute2 \
      curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# Torch first and alone: it is the large one, and giving it its own layer means an
# edit to any later pin does not re-download two gigabytes.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir --root-user-action=ignore \
      --index-url ${TORCH_INDEX} \
      "torch==2.8.0"

# Everything else, pinned. Versions resolved from the index at build time rather
# than written from memory: an invented version number fails the build at the
# worst possible moment.
RUN pip install --no-cache-dir --root-user-action=ignore \
      # 5.16.1, because that is what the task's own
      # `resolved.requirements.txt` pins and what the AgentX runtime carries.
      # An older transformers does not recognise `qwen3_5`, the model type the
      # Qwen3.5 tasks serve.
      #
      # The starter's `serve.sh` runs `python3 server.py` on this interpreter,
      # so without it THE STARTER CANNOT SERVE ITS OWN TASK. A candidate
      # builds its `.venv` from /opt/env-input; the seed does not, and a
      # starter that cannot serve is a broken control arm rather than a slow
      # one.
      "transformers==5.16.1" \
      # THE TOKENIZER STACK, pinned with it. transformers 5.x loads Gemma's
      # tokenizer through protobuf + sentencepiece and imports neither itself,
      # so without them the image's python cannot load a Gemma checkpoint:
      #
      #   [server] LOAD FAILED ImportError: requires the protobuf library but
      #   it was not found in your environment
      #
      # -- neither the STARTER nor the evaluator's own correctness command,
      # which loads the same tokenizer. Same versions as the task's
      # `resolved.requirements.txt`, which is what a candidate's venv installs.
      "tokenizers==0.23.1" \
      "protobuf==7.36.1" \
      "sentencepiece==0.2.2" \
      "huggingface-hub==1.29.0" \
      "safetensors==0.8.0" \
      "accelerate==1.14.0" \
      "numpy>=1.26,<3" \
      "fastapi==0.141.1" \
      "uvicorn==0.52.4" \
      "httpx==0.28.1" \
      "pytest==9.1.1" \
      "pyyaml>=6"

# PROVEN HERE, not assumed. Every package the starter and the evaluator import
# to load a checkpoint's tokenizer, at the pinned version. A pin that resolved
# to something else, or a transitive dependency that went missing, fails the
# BUILD -- the only point at which it is cheap.
RUN python3 -c "import importlib.metadata as m, google.protobuf, sentencepiece, tokenizers, transformers; \
want = {'transformers': '5.16.1', 'tokenizers': '0.23.1', 'protobuf': '7.36.1', 'sentencepiece': '0.2.2', 'huggingface-hub': '1.29.0'}; \
got = {k: m.version(k) for k in want}; \
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}; \
assert not bad, f'pins did not resolve as written: {bad}'; \
print('tokenizer stack', got)"

# Agent binaries copied from the host build context. They are staged by the
# build script because Docker cannot COPY from outside its context.
ARG CLAUDE_BIN=claude
ARG CODEX_BIN=codex
ARG CODEX_CODE_MODE_HOST_BIN=codex-code-mode-host
ARG CODEX_RG_BIN=rg
COPY ${CLAUDE_BIN} /usr/local/bin/claude
COPY ${CODEX_BIN} /usr/local/bin/codex
COPY ${CODEX_CODE_MODE_HOST_BIN} /usr/local/bin/codex-code-mode-host
COPY ${CODEX_RG_BIN} /usr/local/bin/rg
RUN chmod +x /usr/local/bin/claude /usr/local/bin/codex \
      /usr/local/bin/codex-code-mode-host /usr/local/bin/rg \
 && claude --version \
 && codex --version \
 && codex-code-mode-host --help >/dev/null \
 && rg --version >/dev/null

# The egress allowlist, from its one source of truth. It has three independent
# layers because a single allowlist is one environment variable away from being
# wrong.
COPY serverbench/environments/serving-agent/restrict-network.sh \
     /usr/local/bin/restrict-network.sh
RUN chmod +x /usr/local/bin/restrict-network.sh

# Telemetry and auto-update are pointless without egress and would only add failed
# connections to every log. The offline flags make a missing network a clear error
# from the library rather than a sixty-second timeout.
ENV DISABLE_TELEMETRY=1 \
    DISABLE_AUTOUPDATER=1 \
    DISABLE_ERROR_REPORTING=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# A non-root user, because the agent CLI refuses --dangerously-skip-permissions
# under root -- correctly, since that flag plus root is a bad combination even in
# a container. Build the UID to match the invoking user: the workspace is
# bind-mounted, so a mismatch means the agent cannot write to the directory it was
# asked to change.
ARG AGENT_UID=1000
ARG AGENT_GID=1000
RUN groupadd -g "${AGENT_GID}" agent 2>/dev/null || true \
 && useradd -m -u "${AGENT_UID}" -g "${AGENT_GID}" -s /bin/bash agent 2>/dev/null || true \
 && mkdir -p /workspace /models /home/agent/.codex/sessions \
 && chown -R "${AGENT_UID}:${AGENT_GID}" /workspace /home/agent/.codex

USER agent
ENV HOME=/home/agent
WORKDIR /workspace
