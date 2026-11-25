#!/bin/bash
# One-time setup, on the login node (it has internet): the authors' code at the paper's commit, its exact
# environment (uv.lock: torch 2.14, transformers 5.17), the model, the CPU tests, and a dry run of the plan.
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ $GRANT == CHANGE_ME || $GROUP_DIR == *CHANGE_ME ]] && { echo "edit athena/env.sh first"; exit 1; }
mkdir -p "$HF_HOME" "$RESULTS_DIR" "$PROJECT_DIR/logs"

[[ -d $PAPER_DIR/.git ]] || git clone https://github.com/lawzero-org/llm-dropout-noise-recognition.git "$PAPER_DIR"
git -C "$PAPER_DIR" fetch --quiet origin && git -C "$PAPER_DIR" checkout --quiet "$PAPER_COMMIT"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
(cd "$PAPER_DIR" && uv sync --frozen)  # includes the dev group (pytest)

"$PYTHON" "$PAPER_DIR/scripts/download_models.py" qwen3_14b
"$PYTHON" -m pytest -q -p no:cacheprovider "$PROJECT_DIR/tests"
"$PYTHON" -m lmi.run --stage check anchor band where --out "$RESULTS_DIR" --dry-run | tail -1
echo "setup done; next: bash athena/submit.sh 1 check"
