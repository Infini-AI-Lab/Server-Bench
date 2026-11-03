#!/usr/bin/env bash
# Build the agent images: the base image, the shared task image, and each image
# a task declares in `environment/agent-environment` (`IMAGE=` built from
# `IMAGE_DOCKERFILE=`, which derives from the shared task image).
#
# Needs network. The *task* is offline; building the image it runs in is not.
# Runs after 02-build-offline-inputs.sh: the task image installs `ninja` from
# the offline wheelhouse.
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"

# The path the task Dockerfile COPYs from, relative to the build context.
ninja_wheel="$REPO/serverbench/artifacts/env-input/wheelhouse/ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
[ -f "$ninja_wheel" ] \
    || die "no $ninja_wheel; the task image installs it -- run 02-build-offline-inputs.sh first"

say "staging agent binaries into the build context"
# `docker build` cannot COPY from outside its context, and the CLI is a ~340 MB
# static binary that lives in the user's home. Staged, never committed.
claude_bin="$(readlink -f "$(command -v claude)")"
cp "$claude_bin" "$REPO/serverbench/environments/_claude-bin"
chmod +x "$REPO/serverbench/environments/_claude-bin"
ok "$(du -h "$REPO/serverbench/environments/_claude-bin" | cut -f1) from $claude_bin"

# `|| true` inside every substitution, then an explicit check: under `set -e`
# a failed command substitution in an assignment exits the script with no
# message (for example when `rg` is a shell function rather than a binary).
codex_bin="${SERVERBENCH_CODEX_BIN:-$(readlink -f "$(command -v codex || true)" || true)}"
[ -n "$codex_bin" ] && [ -x "$codex_bin" ] \
    || die "no codex binary: put codex on PATH or set SERVERBENCH_CODEX_BIN"
codex_dir="$(dirname "$codex_bin")"
code_mode_host="${SERVERBENCH_CODEX_CODE_MODE_HOST_BIN:-$codex_dir/codex-code-mode-host}"
# Codex ships its own ripgrep beside its binary; prefer the one it was
# released with over whatever `rg` happens to be on PATH.
rg_bin="${SERVERBENCH_CODEX_RG_BIN:-}"
if [ -z "$rg_bin" ] && [ -x "$(dirname "$codex_dir")/codex-path/rg" ]; then
    rg_bin="$(dirname "$codex_dir")/codex-path/rg"
fi
[ -n "$rg_bin" ] || rg_bin="$(command -v rg || true)"
[ -n "$rg_bin" ] \
    || die "no ripgrep for the Codex runtime: set SERVERBENCH_CODEX_RG_BIN"
for pair in \
    "$codex_bin:_codex-bin" \
    "$code_mode_host:_codex-code-mode-host-bin" \
    "$rg_bin:_codex-rg-bin"; do
    source="${pair%%:*}"
    target="${pair#*:}"
    [ -x "$source" ] || die "required Codex runtime binary is missing: $source"
    cp "$source" "$REPO/serverbench/environments/$target"
    chmod +x "$REPO/serverbench/environments/$target"
    ok "$(du -h "$REPO/serverbench/environments/$target" | cut -f1) from $source"
done

# Which Dockerfiles built this, stamped on the image and checked at every
# docker launch by `scripts/agent/image_freshness.py`, so a run never starts on
# an image older than its Dockerfiles.
IMAGE_INPUTS_SHA256="$("$PY" serverbench/scripts/agent/image_freshness.py digest)"
say "base image: $BASE_IMAGE"
docker build -t "$BASE_IMAGE" \
    --label "serverbench.image_inputs_sha256=$IMAGE_INPUTS_SHA256" \
    --build-arg CLAUDE_BIN=serverbench/environments/_claude-bin \
    --build-arg CODEX_BIN=serverbench/environments/_codex-bin \
    --build-arg CODEX_CODE_MODE_HOST_BIN=serverbench/environments/_codex-code-mode-host-bin \
    --build-arg CODEX_RG_BIN=serverbench/environments/_codex-rg-bin \
    --build-arg AGENT_UID="$(id -u)" --build-arg AGENT_GID="$(id -g)" \
    -f serverbench/environments/serving-agent/Dockerfile . | tail -3
ok "built"

say "task image: $TASK_IMAGE"
# The task image needs the uv binary in the context. It is copied rather than
# downloaded because there is no network at run time, and a build that fetches it
# depends on an index staying up. The Dockerfile COPYs it from the shared bundle.
uv_dir="$REPO/serverbench/artifacts/env-input/uv-binary"
mkdir -p "$uv_dir"
cp "$(command -v uv)" "$uv_dir/uv"
chmod +x "$uv_dir/uv"
docker build -t "$TASK_IMAGE" \
    --label "serverbench.image_inputs_sha256=$IMAGE_INPUTS_SHA256" \
    --build-arg AGENT_UID="$(id -u)" --build-arg AGENT_GID="$(id -g)" \
    -f serverbench/environments/evolve-agent/Dockerfile . | tail -3
ok "built"

# A derived image is built from its own directory, as its Dockerfile documents,
# and inherits the freshness label from the task image it starts FROM.
for declared in "$REPO"/serving_tasks/*/environment/agent-environment; do
    [ -f "$declared" ] || continue
    image="$(sed -n 's/^IMAGE=//p' "$declared" | tail -1)"
    dockerfile="$(sed -n 's/^IMAGE_DOCKERFILE=//p' "$declared" | tail -1)"
    [ -n "$image" ] && [ -n "$dockerfile" ] || continue
    [ "$image" = "$TASK_IMAGE" ] && continue
    say "derived image: $image ($(basename "$(dirname "$(dirname "$declared")")"))"
    docker build -t "$image" -f "$REPO/$dockerfile" "$REPO/$(dirname "$dockerfile")" | tail -3
    ok "built"
done

say "images ready"
docker images --format '  {{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}' \
    | grep serverbench || true
