#!/bin/bash
# OUTBOUND FIREWALL (anti-cheat) — this task runs OFFLINE.
#
# Allow ONLY: loopback, established flows, the configured DNS resolver, and the explicit
# LLM API endpoints passed in ALLOWED_HOSTS. Everything else is REJECTed. In particular pypi,
# github, huggingface, npm, conda and the pytorch wheel index are unreachable, so the agent
# cannot `pip install vllm/sglang/flashinfer/flash-attn`, `git clone` a reference serving
# implementation, or re-download the model / a stronger checkpoint. The served model is
# already staged in the image at build time, and the agent harnesses are pre-baked (harbor's
# patched installer skips the network install when the binary is already present), so no
# package index is needed at run time.
#
# Runs as the container ENTRYPOINT before the CMD. iptables rules are network-namespace-wide,
# so they also cover the agent's `docker exec` shells. Requires cap_add: NET_ADMIN (supplied
# by environment/docker-compose.yaml, which harbor merges).
#
# The allowlist is the authorization boundary. The blocklist is only defence in depth: it
# makes a mistaken future allowlist fail closed instead of silently exposing a source/model
# host. There are three independent layers:
#   1. NAME filter   — package/source hosts are stripped from the explicit allowlist before it
#                      is resolved. Environment URL variables grant no access at all.
#   2. IP blocklist  — those hosts' current IPs are REJECTed in a dedicated chain evaluated
#                      FIRST, ahead of every ACCEPT rule.
#   3. default REJECT — anything not explicitly allowed is refused anyway; and IPv6 is dropped
#                      wholesale so a AAAA answer cannot route around the IPv4 rules.
#
# Self-test (from the host, in a built image):  docker exec <c> bash $APP_DIR/restrict-network.sh --check
set -uo pipefail

# ---------------------------------------------------------------------------
# Hosts the agent may reach: ONLY those selected by the runner for this agent profile.
# Pass ALLOWED_HOSTS=host1,host2 (hostnames, no scheme). An empty value means no external
# endpoint at all. Do not infer endpoints from arbitrary *_URL environment variables: the
# task and every subprocess share this network namespace, so implicit environment-derived
# authorization is not an auditable boundary.
# ---------------------------------------------------------------------------
ALL_HOSTS=()

# ---------------------------------------------------------------------------
# Hosts that must NEVER be reachable, whatever the environment says. These are the
# routes by which an agent could obtain a serving framework, a reference implementation,
# or different model weights — i.e. the answer to the task.
# ---------------------------------------------------------------------------
BLOCKED_HOSTS=(
    # python packaging
    pypi.org pypi.python.org files.pythonhosted.org test.pypi.org
    pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com pypi.doubanio.com
    download.pytorch.org astral.sh
    # source hosting
    github.com www.github.com api.github.com codeload.github.com
    raw.githubusercontent.com objects.githubusercontent.com gist.githubusercontent.com
    gitlab.com bitbucket.org gitee.com
    # model hubs
    huggingface.co hf.co cdn-lfs.huggingface.co cdn-lfs-us-1.huggingface.co
    modelscope.cn www.modelscope.cn
    # other package / image registries
    registry.npmjs.org www.npmjs.com registry.yarnpkg.com
    repo.anaconda.com conda.anaconda.org anaconda.org
    ghcr.io docker.io registry-1.docker.io quay.io
)

is_blocked_name() {  # exact match or subdomain of a blocked host
    local h="${1,,}" b
    for b in "${BLOCKED_HOSTS[@]}"; do
        [ "$h" = "$b" ] && return 0
        [[ "$h" == *".$b" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Parse only the explicit per-run allowlist. A proxy or self-hosted gateway remains supported,
# but the runner must name it in ALLOWED_HOSTS; merely placing a URL in the environment does
# not grant network access.
# ---------------------------------------------------------------------------
if [ -n "${ALLOWED_HOSTS:-}" ]; then
    IFS=',' read -ra E <<< "$ALLOWED_HOSTS"
    for h in "${E[@]}"; do
        h="$(echo "$h" | sed 's|.*://||; s|[:/].*||; s/^ *//; s/ *$//')"
        [ -n "$h" ] || continue
        if is_blocked_name "$h"; then
            echo "restrict-network: IGNORING ALLOWED_HOSTS entry $h (blocklisted)" >&2
            continue
        fi
        ALL_HOSTS+=("$h")
    done
fi

# BEDROCK_MODE=1 => the agent's LLM is Amazon Bedrock via the EC2 instance role, reached
# through IMDS (169.254.169.254) for credentials and the bedrock-runtime/sts endpoints for
# inference. Bedrock's endpoint DNS rotates ~8 IPs/resolve, so a startup-only allowlist would
# eventually block a mid-run request; a background re-resolver re-adds new IPs every 30s.
# (Validated: 300-min soak, 196/196 calls, 0 fail.) Every non-Bedrock provider remains an
# explicit ALLOWED_HOSTS choice.
BEDROCK_MODE="${BEDROCK_MODE:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
BEDROCK_HOSTS=(
    "bedrock-runtime.${AWS_REGION}.amazonaws.com"
    "bedrock.${AWS_REGION}.amazonaws.com"
    "sts.${AWS_REGION}.amazonaws.com"
)

resolve() { getent ahosts "$1" 2>/dev/null | awk '{print $1}' | grep -v ':' | sort -u; }

allow_host_ips() {  # resolve $1, ACCEPT any not-yet-allowed IP; echo count added.
    # Inserts at position 2 so the BLOCKED chain (position 1) is always evaluated first.
    local h="$1" added=0 ip
    for ip in $(resolve "$h"); do
        if ! iptables -C OUTPUT -d "$ip" -j ACCEPT 2>/dev/null; then
            iptables -I OUTPUT 2 -d "$ip" -j ACCEPT && added=$((added+1))
        fi
    done
    echo "$added"
}

# ---------------------------------------------------------------------------
# --check : verify the firewall from inside the container (no rules applied).
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--check" ]; then
    echo "== outbound reachability =="
    for h in pypi.org github.com huggingface.co download.pytorch.org registry.npmjs.org; do
        code="$(curl -s -o /dev/null -m 8 -w '%{http_code}' "https://$h" 2>/dev/null)"
        [ "$code" = "000" ] && echo "  BLOCKED  $h" || echo "  REACHABLE $h (HTTP $code)  <-- ANTI-CHEAT HOLE"
    done
    for h in ${ALL_HOSTS[@]+"${ALL_HOSTS[@]}"}; do
        code="$(curl -s -o /dev/null -m 8 -w '%{http_code}' "https://$h" 2>/dev/null)"
        [ "$code" = "000" ] && echo "  BLOCKED  $h  <-- the agent's LLM API must be REACHABLE" || echo "  REACHABLE $h (HTTP $code)"
    done
    echo "== installed rules =="
    iptables -L OUTPUT -n --line-numbers 2>/dev/null | head -20
    exit 0
fi

if iptables -L OUTPUT -n >/dev/null 2>&1; then
    echo "restrict-network: applying outbound allowlist firewall..."
    iptables -F OUTPUT 2>/dev/null || true

    # --- layer 2: a dedicated blocklist chain, jumped to FIRST ---
    iptables -N MLEBLOCK 2>/dev/null || iptables -F MLEBLOCK
    nblocked=0
    for h in "${BLOCKED_HOSTS[@]}"; do
        for ip in $(resolve "$h"); do
            iptables -A MLEBLOCK -d "$ip" -j REJECT --reject-with icmp-admin-prohibited 2>/dev/null \
                && nblocked=$((nblocked+1))
        done
    done
    iptables -A OUTPUT -j MLEBLOCK          # rule 1: nothing can be allowed past this

    iptables -A OUTPUT -o lo -j ACCEPT
    iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    # DNS is needed to resolve the allowed endpoints, but only the resolver(s)
    # Docker actually configured are reachable. Port 53 to arbitrary internet or
    # RFC1918 addresses is not a hidden second egress channel.
    while read -r dns; do
        [ -n "$dns" ] || continue
        iptables -A OUTPUT -d "$dns" -p udp --dport 53 -j ACCEPT
        iptables -A OUTPUT -d "$dns" -p tcp --dport 53 -j ACCEPT
    done < <(awk '$1 == "nameserver" {print $2}' /etc/resolv.conf)
    if [ -n "$BEDROCK_MODE" ]; then
        iptables -A OUTPUT -d 169.254.169.254 -j ACCEPT   # IMDSv2 instance-role creds
        for h in "${BEDROCK_HOSTS[@]}"; do
            for ip in $(resolve "$h"); do iptables -A OUTPUT -d "$ip" -j ACCEPT; done
        done
    fi
    nallow=0
    for h in "${ALL_HOSTS[@]}"; do
        for ip in $(resolve "$h"); do
            iptables -A OUTPUT -d "$ip" -j ACCEPT && nallow=$((nallow+1))
        done
    done
    iptables -A OUTPUT -j REJECT --reject-with icmp-admin-prohibited

    # --- layer 3b: IPv6. The rules above are IPv4-only; if the container has IPv6 and DNS
    # returns a AAAA record, traffic would bypass them entirely. Drop it all (every allowed
    # LLM endpoint is reachable over IPv4).
    if command -v ip6tables >/dev/null 2>&1 && ip6tables -L OUTPUT -n >/dev/null 2>&1; then
        ip6tables -F OUTPUT 2>/dev/null || true
        ip6tables -A OUTPUT -o lo -j ACCEPT
        ip6tables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
        ip6tables -A OUTPUT -j REJECT 2>/dev/null || ip6tables -P OUTPUT DROP 2>/dev/null || true
        echo "restrict-network: IPv6 outbound dropped."
    else
        echo "restrict-network: ip6tables unavailable (IPv6 likely disabled in this netns)."
    fi

    echo "restrict-network: outbound locked to configured DNS + explicit LLM endpoints "
    echo "                  (${nallow} allowed IPs, ${nblocked} explicitly blocked); pypi/github/HF/npm/conda unreachable."
    if [ -n "$BEDROCK_MODE" ]; then
        ( while true; do
              sleep 30
              tot=0; for h in "${BEDROCK_HOSTS[@]}"; do n=$(allow_host_ips "$h"); tot=$((tot+n)); done
              [ "$tot" -gt 0 ] && echo "[reresolver] +$tot new Bedrock IP(s) at $(date +%H:%M:%S)"
          done ) &
        echo "restrict-network: BEDROCK_MODE — IMDS+bedrock allowed; re-resolver PID $!"
    fi
else
    echo "restrict-network: WARNING — iptables unavailable (need cap_add NET_ADMIN); NOT firewalled." >&2
fi

# Export the detected-GPU env so everything started under the entrypoint inherits it
# (the task hardcodes no GPU — see gpu_env.sh).
if [ -r "${APP_DIR:-/app}/gpu_env.sh" ]; then
    # shellcheck disable=SC1090
    . "${APP_DIR:-/app}/gpu_env.sh"
fi

exec "$@"
