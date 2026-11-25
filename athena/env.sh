# Settings for Cyfronet Athena. Edit the two CHANGE_ME lines; the other scripts source this file.
# Athena: partition plgrid-gpu-a100 (8x A100-40GB per node, 48 h max), account <grant>-gpu-a100,
# $HOME only 10 GB, $SCRATCH purges files older than 30 days, $PLG_GROUPS_STORAGE persists for the grant.

export GRANT=CHANGE_ME                                  # PLGrid grant name (Slurm account = ${GRANT}-gpu-a100)
export GROUP_DIR=$PLG_GROUPS_STORAGE/CHANGE_ME          # your group directory: models and results live here

export PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PAPER_DIR=$PROJECT_DIR/external/llm-dropout-noise-recognition
export PAPER_COMMIT=ad7ce52431cea49a15f7cd6e327d5c782ad27749  # code of the paper (arXiv v3), tag v2
export HF_HOME=$GROUP_DIR/hf                            # Qwen3-14B is ~30 GB
export RESULTS_DIR=$GROUP_DIR/results
export UV_CACHE_DIR=$SCRATCH/uv-cache
export UV_PROJECT_ENVIRONMENT=$SCRATCH/lmi-venv         # venv on Lustre (fast); if purged, rerun setup.sh
export PYTHON=$UV_PROJECT_ENVIRONMENT/bin/python
export PYTHONPATH=$PROJECT_DIR/src
export PYTHONUTF8=1                                     # read the sentence files as UTF-8 whatever the locale
export PATH=$HOME/.local/bin:$PATH                      # uv
