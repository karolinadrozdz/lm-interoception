# lm-interoception

Can language models sense *what is going on in their own substrate* (interoception: WHAT), as distinct
from knowing *how they decide* (introspection: HOW)?

We build directly on Fornasiere et al. 2026, *Language models recognize dropout and Gaussian noise
applied to their activations* (arXiv:2604.17465; notes in `docs/fornasiere2026-reference.md`).
They perturb the activations of one sentence and show that models can tell **whether** a perturbation was
applied and to **which** sentence. We ask **where** in the network it was applied.

Plan 01 (`docs/plan-01-one-band-where.md`): Qwen3-14B, zero-shot, everything as in the paper except:
- the perturbation is restricted to one of two windows of the paper's first layer band, X = layers 0-3
  or Y = layers 9-12;
- one new question asks which window it was.

## Layout

```
src/lmi/            our code; the authors' package `spe` is imported as a library, never modified
  grid.py           sweep file -> jobs (the authors' W&B grid format, so their seeds are reproduced)
  where.py          the new 'where' question: pair (localization prompt) and single (zero-shot prompt)
  run.py            runs a shard of jobs on one GPU, loads the model once, writes json/parquet/log per run
  summarize.py      results -> summary.csv, accuracy tables, Table 1 reproduction check
sweeps/manifest.yaml  stages -> sweep files ("paper:" = the authors' files, run unchanged)
sweeps/qwen3_14b/     our sweeps (window runs of the paper's questions, and the where question)
athena/             Cyfronet Athena: env.sh (edit), setup.sh, submit.sh, run.sbatch
tests/              CPU tests (hook placement per window)
docs/               paper reference notes, plans
external/           the authors' repo at commit ad7ce52 (gitignored; setup.sh clones it)
```

## Stages (Qwen3-14B, 1000 samples per run, strengths = the 11 bins of the paper's Table 1 unless noted)

| Stage | Runs | What |
|---|---|---|
| `check` | 2 | one run of 100 samples at dropout 0.3 (GPU path, prompts and token maps in the log, time per run); activation sizes per layer, without perturbation (`norms`) |
| `anchor` | 191 | the paper's own sweeps: localization (Fig. 2), thirds of layers (Fig. 27), detection (Fig. 5) |
| `band` | 88 | the paper's "which sentence?" and "was a perturbation applied?" with the perturbation only in X or in Y |
| `where` | 192 | the new question, see below; strength 0 is the no-perturbation baseline |
| `extra` | 116 | band runs above the Table 1 range, plus the yes-bias and comprehension controls per window |

The `where` stage has two formats, each run with dropout and with noise:

- **pair** (the localization prompt). Both copies of the sentence are perturbed, one in X and one in Y.
  - The question asks which copy had the perturbation in "your earlier layers", or in "your layers 1-4 of 40".
  - The window of copy A and the window named are exactly counterbalanced, so always picking the more salient copy, or always picking A, scores 50%.
  - With `premise=true`, the prompt states that both sentences were perturbed, one in each window.
- **single** (the zero-shot classification prompt). One sentence is perturbed, in X or in Y. The options use the paper's `NAME (description)` format:
  - `EARLIER LAYERS (a perturbation in the earlier of two groups of your layers)` / `LATER LAYERS (...)`;
  - `LAYERS 1-4 OF 40 (a perturbation in your layers 1-4 of 40)` / `LAYERS 10-13 OF 40 (...)`;
  - the bare label controls `X` / `Y` and `BRALTO` / `SIVEK`.

## Run

Local CPU check (tiny random model with the real tokenizer and architecture):

```bash
cd external/llm-dropout-noise-recognition && uv sync --frozen && cd ../..
PYTHONUTF8=1 PYTHONPATH=src external/llm-dropout-noise-recognition/.venv/bin/python -m lmi.run --stage check anchor band where --smoke
```

On Windows the interpreter is at `.venv/Scripts/python.exe`. Tests: `... python -m pytest -q tests`.

On Athena:

1. Edit the two `CHANGE_ME` lines in `athena/env.sh`.
2. Copy this repo to Athena: commit and `git clone`, or `rsync` it (no need to copy `external/` or `results/`). Then run the setup once, on the login node. It clones the authors' code, installs their exact environment, downloads the model, runs the tests and does a dry run:

   ```bash
   bash athena/setup.sh
   ```

3. Submit:

   ```bash
   bash athena/submit.sh 1 check                # first, ~minutes: inspect logs/ and $RESULTS_DIR/check/
   bash athena/submit.sh 8 anchor band extra    # reproduces the paper? each window detectable on its own?
   bash athena/submit.sh 8 where                # then the new question
   ```

   Resubmitting a command resumes it, skipping finished runs, but only do so after the previous array has ended.

4. Summarize:

   ```bash
   $PYTHON -m lmi.summarize $RESULTS_DIR  # tables + Table 1 lower-bound check
   ```

Each run writes `<results>/<stage>/<sweep>/<seed>.json` (config, metrics, timing), `.parquet` (one row per sample) and `.log` (prompts, token maps, per-sample answers, the paper's metric summary).
