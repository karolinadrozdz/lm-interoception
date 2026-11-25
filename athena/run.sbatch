#!/bin/bash
# One array task = one A100 = one shard of the jobs of the given stages. Submit with athena/submit.sh.
#SBATCH --job-name=lmi
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail
source "$SLURM_SUBMIT_DIR/athena/env.sh"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1  # the model was downloaded by setup.sh
"$PYTHON" -c "import spe, lmi" || { echo "venv missing or purged from \$SCRATCH: rerun bash athena/setup.sh"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -m lmi.run --stage "$@" --out "$RESULTS_DIR"
