"""lm-interoception: experiments built on the code of Fornasiere et al. (2026), arXiv:2604.17465.

The authors' package ``spe`` (cloned at ``external/llm-dropout-noise-recognition``, commit ad7ce52) is used
as a library and never modified: its configs, prompts, sentences, hooks, readout and metrics are the ones
of the paper. This package only adds the 'where' question and a W&B-free runner.
"""

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER_REPO = pathlib.Path(
    os.environ.get("LMI_PAPER_REPO", ROOT / "external/llm-dropout-noise-recognition")
)
