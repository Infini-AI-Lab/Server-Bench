#!/usr/bin/env bash
# Check this machine can run an evolve, before anything expensive is built.
#
# Each check here is something whose absence would otherwise fail a run long
# after it started instead of in the first ten seconds.
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

say "preflight on $(hostname)"
fail=0

command -v docker >/dev/null || { warn "docker is not on PATH"; fail=1; }
if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    ok "docker $(docker version --format '{{.Server.Version}}')"
else
    warn "the docker daemon does not answer for $(id -un)"; fail=1
fi

command -v nvidia-smi >/dev/null || { warn "nvidia-smi is not on PATH"; fail=1; }
if nvidia-smi -L >/dev/null 2>&1; then
    ok "$(nvidia-smi -L | wc -l) GPU(s): $(nvidia-smi -L | head -1 | cut -d'(' -f1)"
    uuid=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null) \
        || { warn "GPU $GPU does not exist here; set GPU= to one that does"; fail=1; }
    [ -n "${uuid:-}" ] && ok "GPU $GPU = $uuid"
    used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ "${used:-0}" -gt 2000 ]; then
        warn "GPU $GPU already holds ${used} MiB. A contended device is not a"
        warn "measurement -- pick a free one, and do not stop what is on this one"
        fail=1
    else
        ok "GPU $GPU is free (${used} MiB in use)"
    fi
fi

# The container runtime has to actually hand the GPU through, which is a separate
# thing from the driver being present.
if docker run --rm --gpus "\"device=$GPU\"" "$BASE_IMAGE" true 2>/dev/null; then
    ok "the container runtime passes GPU $GPU through"
elif docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    warn "$BASE_IMAGE exists but could not take GPU $GPU"; fail=1
else
    warn "$BASE_IMAGE is not built yet -- run 03-build-images.sh"
fi

command -v uv >/dev/null && ok "uv $(uv --version | cut -d' ' -f2)" \
    || { warn "uv is not on PATH; it is what populates the wheelhouse"; fail=1; }

if [ -x "$(command -v claude || true)" ]; then
    ok "claude $(claude --version 2>/dev/null | cut -d' ' -f1)"
else
    warn "the claude CLI is not on PATH; the image bakes it in from there"; fail=1
fi

[ -d "$MODEL_SRC" ] && ok "model source $MODEL_SRC" \
    || { warn "no model at MODEL_SRC=$MODEL_SRC"; fail=1; }

# Nothing of ours may land on the filesystem the operating system lives on. A
# build that fills `/` breaks the machine for everyone on it.
root_free=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
for var in TMPDIR DOCKER_TMPDIR UV_CACHE_DIR PROJECT_SCRATCH; do
    path="${!var:-}"
    [ -n "$path" ] || continue
    mount=$(df --output=target "$path" 2>/dev/null | tail -1)
    if [ "$mount" = "/" ]; then
        warn "$var=$path is on / (${root_free} GB free). Large transients there"
        warn "  fill the OS filesystem; config.sh should have pointed it at scratch"
        fail=1
    else
        ok "$var on $mount"
    fi
done

# The one location this project cannot choose, reported rather than glossed.
if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    snap=$(df --output=target /var/lib/containerd 2>/dev/null | tail -1)
    if [ "$snap" = "/" ]; then
        warn "the docker daemon extracts image layers into /var/lib/containerd,"
        warn "  which is on / with ${root_free} GB free, while /var/lib/docker is a"
        warn "  separate volume with room. That is root-owned configuration and"
        warn "  nothing here can change it -- see serverbench/scripts/deploy/README.md."
        warn "  A multi-gigabyte image pull may fail with 'no space left on device'"
    else
        ok "image layers land on $snap"
    fi
fi

free_gb=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
[ "${free_gb:-0}" -ge 30 ] && ok "${free_gb} GB free under the repository" \
    || { warn "only ${free_gb} GB free; the wheelhouse alone is 4 GB"; fail=1; }

docker_gb=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${docker_gb:-999}" -ge 40 ] && ok "${docker_gb:-?} GB free for images" \
    || warn "only ${docker_gb} GB free for images; the base image is ~10 GB"

[ -x "$PY" ] && ok "harness interpreter $PY" \
    || warn "no interpreter at $PY -- run 00-envs.sh, which creates it"

# Harbor, which composes every run's environment. `run.py::start_container`
# composes through Harbor or refuses, so a host that cannot drive it cannot
# evolve anything. `harbor/.venv` is gitignored and created by 00-envs.sh.
harbor_py="$REPO/harbor/.venv/bin/python"
if [ ! -x "$harbor_py" ]; then
    warn "no Harbor interpreter at harbor/.venv/bin/python."
    warn "  Every run's environment is composed by Harbor and there is no"
    warn "  fallback, so an evolve on this host would refuse. Fix with:"
    warn "      bash serverbench/scripts/deploy/00-envs.sh"
    fail=1
elif "$harbor_py" -c 'import harbor' >/dev/null 2>&1; then
    ok "harbor $("$harbor_py" -c 'import harbor; print(harbor.__version__)' 2>/dev/null)"
    if "$harbor_py" -c 'from harbor.environments.docker.docker import DockerEnvironment' \
        >/dev/null 2>&1; then
        ok "harbor's docker environment imports"
    else
        warn "harbor imports but its docker environment does not; an evolve would refuse"
        fail=1
    fi
else
    warn "harbor/.venv exists and 'import harbor' fails; re-run 00-envs.sh"
    fail=1
fi

say "$([ $fail -eq 0 ] && echo 'preflight passed' || echo 'preflight FAILED')"
exit $fail
