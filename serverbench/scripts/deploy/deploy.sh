#!/usr/bin/env bash
# Set up a machine to run serving-evolution runs, from nothing.
#
#   bash serverbench/scripts/deploy/deploy.sh
#
# Creates the host environments, runs preflight, builds the offline wheelhouses
# and then the images that install from them. Needs network; the runs it
# prepares for do not.
HERE="$(dirname "${BASH_SOURCE[0]}")"
# 00-envs.sh first: config.sh sources env.sh, which refuses without the
# harness interpreter that step creates.
bash "$HERE/00-envs.sh"
. "$HERE/config.sh"

bash "$HERE/01-preflight.sh" || {
    # A missing image is the one preflight failure this script is about to fix,
    # so it is worth saying so rather than stopping on it.
    warn "preflight reported problems. If they are the images, this script"
    warn "builds them next. Anything else -- no GPU, no docker, no model --"
    warn "will not be fixed by continuing."
    [ "${FORCE:-0}" = 1 ] || die "re-run with FORCE=1 once you have read the above"
}
bash "$HERE/02-build-offline-inputs.sh"
bash "$HERE/03-build-images.sh"

say "deployed"
cat <<TXT
   The machine is ready. A task's SGLang reference operating point and p90
   bars are measured per hardware; see serving_tasks/_manage/README.md.

   Run an agent against a task with the production driver:

     bash serverbench/scripts/server_evolve/evolve-task.sh $TASK --hours 8
TXT
