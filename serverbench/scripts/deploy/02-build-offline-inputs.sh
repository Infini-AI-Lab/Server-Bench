#!/usr/bin/env bash
# The offline inputs: the wheelhouses runs mount at /opt/env-input, and the
# held-out answer pool the verdict's private correctness stage draws from.
#
#   bash serverbench/scripts/deploy/02-build-offline-inputs.sh          # build every bundle
#   bash serverbench/scripts/deploy/02-build-offline-inputs.sh --list   # print the plan, build nothing
#
# One bundle per agent environment: the shared `$ENV_INPUT`, plus every distinct
# `ENV_INPUT=` a task declares in its `environment/agent-environment`. Each one
# replays its tracked frozen resolution, `<bundle>/resolved.requirements.txt`,
# because a fresh resolve that picked up a newer transformers would change the
# chat template every task renders. Needs network (about 4 GB of wheels per
# bundle). Runs before 03-build-images.sh, whose task image installs from it.
# The answer pool is written by build_answer_pool.py when it is missing.
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$REPO"

mode=build
case "${1:-}" in
    "") ;;
    --list) mode=list ;;
    -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument '$1' (expected none, or --list)" ;;
esac

# Each entry is "<bundle dir>|<frozen resolution it replays>". The shared bundle
# replays the tracked resolution when `ENV_INPUT` points somewhere without one.
shared_pins="$ENV_INPUT/resolved.requirements.txt"
[ -s "$shared_pins" ] || shared_pins="$REPO/serverbench/artifacts/env-input/resolved.requirements.txt"
plan=("$ENV_INPUT|$shared_pins")
for declared in "$REPO"/serving_tasks/*/environment/agent-environment; do
    [ -f "$declared" ] || continue
    value="$(sed -n 's/^ENV_INPUT=//p' "$declared" | tail -1)"
    [ -n "$value" ] || continue
    bundle="$REPO/$value"
    case " ${plan[*]} " in *" $bundle|"*) continue ;; esac
    plan+=("$bundle|$bundle/resolved.requirements.txt")
done

for entry in "${plan[@]}"; do
    bundle="${entry%%|*}"
    pins="${entry#*|}"
    [ -s "$pins" ] || die "no frozen resolution at $pins"
    if [ "$mode" = list ]; then
        printf '   %s  <-  %s (%s pins)\n' "$bundle" "$pins" "$(grep -c '==' "$pins")"
        continue
    fi
    say "wheelhouse: $bundle"
    ok "replaying the frozen resolution: $(grep -c '==' "$pins") pins from $pins"
    "$PY" serverbench/scripts/agent/build_env_bundle.py --out "$bundle" --pins "$pins"
    ok "offline inputs ready at $bundle"
done

if [ "$mode" = list ]; then
    printf '   %s  <-  GSM8K test split\n' "$REPO/serverbench/artifacts/eval_samples/gsm8k-final_validation.json"
    exit 0
fi
say "held-out answer pool"
"$PY" serverbench/scripts/deploy/build_answer_pool.py
