#!/bin/bash
# Usage: bash athena/submit.sh N_GPUS STAGE [STAGE ...]   e.g.  bash athena/submit.sh 1 check
#                                                              bash athena/submit.sh 8 anchor band
# Starts N_GPUS array tasks; each loads the model once and runs every N-th job. Finished runs are skipped, so
# resubmitting the same command resumes -- but only once the previous array has ended (two arrays would run
# the same jobs). Logs: logs/lmi-<job>_<task>.out
set -euo pipefail
cd "$(dirname "$0")/.."
source athena/env.sh
n=$1; shift
mkdir -p logs
sbatch -A "${GRANT}-gpu-a100" --array=0-$((n - 1)) --output=logs/%x-%A_%a.out athena/run.sbatch "$@"
