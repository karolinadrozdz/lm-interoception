# Plan 01 — whether / which / where, within one band (rough)

Status: round 1 implemented, 2026-11-25. Builds on `docs/fornasiere2026-reference.md`.

## Round 1 (implemented: `src/lmi`, `sweeps/`, `athena/`)

Decisions taken by the user for this round:

- **Zero-shot only.** No teaching (D3), no probe (D0) and no depth profile (Step C) yet.
- **Same as the paper except two things:** the injection site (window X = layers 0–3 or Y = layers 9–12 inside the paper's first third, 0–12) and the new where-question.
- **Both perturbations** (dropout and noise), at the paper's 11 Table 1 strengths.
- **Strength re-calibration across bands** (Step E) comes next, if needed.

Stages (`sweeps/manifest.yaml`): check (2 runs), anchor (191), band (88), where (192), and the optional extra (116).

Windows are given to the model 1-based: code layers 0–3 / 9–12 = "layers 1-4 / 10-13 of 40".

The where-question in two formats (`src/lmi/where.py`):

- **pair:** the localization prompt, both copies perturbed (one per window), with the question naming a random window.
- **single:** the zero-shot prompt, with the options naming the windows.

Each format has two wordings, relative ("earlier/later layers") and explicit ("layers 1-4 of 40"). The single format also has two label controls, X/Y and BRALTO/SIVEK.

### Changes after the design review (round 1)

Done, all within "same as the paper except the question and the windows":

- **Pair:** the window of copy A and the window named are exactly counterbalanced.
- **Pair:** the `premise` flag states that both sentences were perturbed. Without it, the question can be read as plain detection, which biases toward a null.
- **Strength 0** is in every where sweep, as the bias baseline.
- **Pair (after the code check):** the premise names the two windows in counterbalanced order, so "one … the other" cannot be read as A = earlier.
- **Single:**
  - both semantic wordings use the paper's "NAME (description)" option format and say that the option is the location of the perturbation, e.g. "EARLIER LAYERS (a perturbation in the earlier of two groups of your layers)", "LAYERS 1-4 OF 40 (a perturbation in your layers 1-4 of 40)"; the label controls stay bare, like the paper's aliases;
  - a nonsense-label control, BRALTO / SIVEK, joins X / Y (which carries an order prior, X before Y);
  - sentence log-probs are now logged too.
- **`norms` diagnostic** in the `check` stage: per-layer RMS and max |h| at the sentence tokens. It gives σ/RMS per window, i.e. how much stronger window X is than Y at the same σ, and flags massive activations.

Optional stage `extra` (116 runs). It is *measurement*, not a change of the main stages' strengths:

- per-window localization and detection above the Table 1 range (a window has 8 of the 80 hook sites the strengths were calibrated with);
- the yes-bias control (coin-flip question) per window;
- the comprehension ceiling per window (5 topic pairs).

**Launch order** (the user agreed to include `extra`): check → anchor + band + extra → (look) → where.

Expected failure mode: if the band stage shows a window is not detectable on its own, the where results at those strengths are uninformative. Then the next step is per-window calibration (Step E).

### Pre-registered analysis (written before any where data)

1. **Anchor validity.** Our `loc_*` curves must reproduce Table 1 (p_min = .10, σ_min = .05 by the paper's rule; `lmi.summarize` checks this) and Fig. 27's first third. Otherwise stop.
2. **Detectability per window (band).** A window "passes" at a strength if `win_loc` accuracy is above 53.16% at two consecutive strengths (the paper's rule). Δdet(s) = acc_X(s) − acc_Y(s).
3. **Primary where measures**, per type × format × wording (× premise):
   - accuracy_primary (comparable with the paper), plus the OTHER rate;
   - bias-free: the AUROC of the constrained logit score.
     - Pair: logit(copy in X) − logit(copy in Y), split by the asked window.
     - Single: logit(EARLY option) − logit(LATE option), split by the true window.
4. **Pair decomposition.**
   - salience = P(choose the X copy);
   - where index = P(X copy | asked X) − P(X copy | asked Y).
   - Salience alone gives a where index of 0.
5. **Intensity test.** Fit the where index (pair) or P(EARLY) (single) against Δdet from the band stage, plus a window term. An intensity mapping ("more disrupted = earlier") predicts zero where-effect at Δdet = 0; a where-sense predicts an intercept or window term > 0.
6. **Inference.**
   - Pool over the strengths at which both windows pass.
   - Use permutation tests within runs (permute the asked window, or the true window) and SEs clustered by prompt variant.
   - Use a one-sided test for the correct direction, and a two-sided test for the label controls (X/Y, BRALTO/SIVEK should be ≈ 50%, or follow intensity only).
   - Apply a Holm correction across the primary cells.
7. **Secondary.** Type-2 sensitivity: does the logit margin predict correctness (interoceptive "awareness" in Garfinkel's sense)? Also report the per-copy log-prob difference as a covariate.

**What a result licenses.** Above chance at matched detectability, with label controls at chance, licenses "a depth-dependent disruption signature that the model reports in depth words". It does not license "access to its layer index". A null result in this zero-shot round does not show that the windows are indistinguishable to the model; that needs few-shot X/Y labels or a probe (D3/D0, next round).

The rest of this document is the original rough plan.

**Goal.** On one mid-size model that detects and localizes well, restrict the perturbation to the band of layers that carries the signal. Then ask:

1. **whether** a perturbation was applied (detection, as in paper Sec. 5);
2. **which** sentence was perturbed (localization, as in Sec. 4);
3. **where** in the network it was applied (new).

If (3) fails at equal strength, move to different strengths per band (Step E).

## How a "where" question works without generation

The model never generates. Every question is a forced choice:

- The options are labelled with letters.
- The assistant turn is pre-filled with `the answer is:`.
- We run one forward pass and read the logits of `" A"` vs `" B"` at the last position.

A "where" question is built the same way, e.g. `Which sentence was perturbed in the earlier layers? … the answer is:` → `" A"`/`" B"`. We know the ground truth because we chose the layers. Nothing about the readout changes; only the question and the hook placement do.

A "where" question needs at least two possible locations. If the perturbation is always in the same band, answering "early" every time is 100% correct and tells us nothing. So "within one band" means: split the band that carries the signal into equal sub-bands that are equally detectable, and ask which sub-band was hit.

## Model: Qwen3-14B

`qwen3_14b` (40 layers, ~30 GB in bf16, fits on one Athena A100 40 GB). Why this model:

- Localization reaches 100% for both perturbations.
- It has one of the cleanest detection curves (Fig. 5, roughly 10% → ~100% "Yes").
- In Fig. 27 its first third (layers 0–12) works as well as perturbing all layers.
- Plain architecture: no post-norm hook sites (unlike Gemma/OLMo), no DeltaNet layers (unlike Qwen3.8).
- Table 1 range: p ∈ [.10, .46], σ ∈ [.05, .10]. The paper's thirds for this model are 0–12 / 13–26 / 27–39.
- Rejected alternatives: Llama-8B (weak detection), Mistral-24B (~48 GB plus a multimodal wrapper).

## Step 0 — Setup on Athena

Source: docs.hpc.cyfronet.pl/supercomputers/athena (page dated 2026-08-28).

- **Login and access.**
  - Log in at `athena.cyfronet.pl`.
  - Access needs an active PLGrid grant plus an "Athena access" service request.
  - GPU jobs only; non-GPU workloads risk account suspension, so do analysis elsewhere.
- **Slurm.**
  - Partition `plgrid-gpu-a100`, 48 h limit, `-A <grant>-gpu-a100`.
  - 48 nodes, each with 8× A100-SXM4-**40GB**. One GPU comes with 16 cores and 128 GB RAM.
- **Storage.**

  | Location | Properties | Use it for |
  |---|---|---|
  | `$HOME` | NFS, **10 GB** limit | too small for models or venvs |
  | `$PLG_GROUPS_STORAGE/<group>` | Lustre, persists for the grant, compressed, moderate I/O | the HF model cache and results |
  | `$SCRATCH` | Lustre, fastest; files older than **30 days** and job dirs older than **7 days** are deleted | the venv (the docs recommend Lustre over NFS for venvs), but expect to rebuild it |

- **Software.**
  - The module tree is "unsupported". Install your own software in `$HOME` or the group directory.
  - Do setup and compilation in an interactive job on a worker node; the login node has no development libraries.
  - Plan: `uv sync` from the clone (Python 3.12, torch 2.14, transformers 5.17, pinned by `uv.lock`).
  - Check that the node's CUDA driver supports the torch wheel (`nvidia-smi` on a compute node).
- **Internet on compute nodes: not stated in the docs.** Test it with `srun -p plgrid-gpu-a100 -A <grant>-gpu-a100 --gres=gpu:1 -t 5 curl -sI https://huggingface.co`.
- **Models.** Download on the login node (`HF_HOME=… uv run python scripts/download_models.py qwen3_14b`). Run jobs with `HF_HUB_OFFLINE=1`.
- **No W&B sweeps.** The W&B sweep agent needs a live connection. Instead:
  - Write a small wrapper in *our* repo that imports `spe`, runs one Hydra config and dumps the metrics plus the per-sample table to parquet.
  - Use a Slurm array over grid points expanded from the authors' sweep YAMLs.
  - Pass the overrides exactly as W&B would (`0`, not `0.0`) so the hashed seeds match theirs. `wandb.enabled=false` does not change the seed.
- **Sanity check on the first job.** Print the rendered Qwen3 prompt and the token map: the empty `<think>` block, the perturbed span (the trailing `.\n\n` merge), and whether there is a double BOS.

## Step A — Anchor runs (exact reproduction, no controls)

These use the authors' sweep files for `qwen3_14b` unchanged. They check that our setup reproduces their numbers before we change anything.

| Run set | Grid | Runs | Compare with |
|---|---|---|---|
| `loc_dropout`, `loc_noise` | p 0–.98 step .02; σ 0–.50 step .01 | 101 | Fig. 2 (p_min .10, σ_min .05) |
| `layers_dropout`, `layers_noise` | 3 thirds × 11 bins | 66 | Fig. 27; the first-third curve = "which sentence" within band 0–12 |
| `yesno_dropout`, `yesno_noise` | {0} ∪ 11 bins, all layers | 24 | Fig. 5 |

Pass criterion: our curves fall within ~2 SE of the paper's.

## Step B — "Whether" within the band (new, cheap)

- Detection with the perturbation only in layers 0–12: `yesno_*` plus `perturbation.first_layer=0 perturbation.last_layer=12`. That is 24 runs.
- Add the `perturbed_ctrl_none` control (coin-flip question) in the same band, another 24 runs. This checks that a rising "Yes" is not just a bias toward "Yes".
- Together with the Step A first-third localization, this gives "whether" and "which" inside the band.

## Step C — Depth profile: where does the signal live?

- Run localization and detection with the perturbation in a **sliding window of 4 layers** (0–3, 4–7, …, 36–39; 10 windows), at ~6 strengths from the Table 1 range, for both perturbations.
  - That is ≈ 10 × 6 × 2 × 2 tasks ≈ 240 runs.
  - Optionally also `hook_target=attn` vs `mlp` in the first windows.
- This reveals whether the signal is spread across 0–12 or concentrated in, say, layers 0–2.
- **Decision rule:**
  - Pick two equal-sized, non-adjacent sub-bands inside the signal region whose localization *and* detection curves match within a few points over some strength range. Candidate: 0–5 vs 7–12, to be decided from the data. Phrased as "where the signal is the same".
  - If no such pair exists at equal strength, go straight to Step E.

## Step D — "Where" (new runner, two sub-bands X = shallower, Y = deeper)

Do these in order. Keep the authors' prompt scaffolding ("I will apply perturbations…", the introspecting assistant turns, the prefill, 20 wording variants) and change only the question.

**First contrast: maximize separation inside the band.** Use the two *edge* windows of the signal region (e.g. 0–3 vs 9–12, not adjacent sub-bands), so that a null result is not just a resolution limit.

**Question wordings.** Run them as separate conditions, never pooled:
- relative: "earlier / later layers";
- explicit: "layers 0–3 / layers 9–12".

### D0. Probe ceiling (diagnostic, not a model report)

- On the same forward passes, save the answer-position hidden states at every layer.
- Fit a cross-validated logistic regression for X vs Y.
- Reading the outcome: if a probe cannot separate X from Y, the model cannot report it either. If the probe can but the model cannot, the information exists but is not accessible to report. That gap is exactly the interoception question.

### D1. Relational, two sentences (analogue of their localization; easiest)

- **Design.** Both copies of the sentence are perturbed: one in X, the other in Y, with the assignment random. Each sample asks, at random, either "which sentence was perturbed in the **earlier** layers?" or "…the **later** layers?".
- **Built-in control.** A model that just picks the more salient sentence gets the same letter for both questions, so it scores exactly 50%. Only a real where-signal scores above 50%.
- **Implementation.** Register hooks once per sentence with that sentence's band (per-entry `register_hooks`; needs a unit test).

### D3. Few-shot (analogue of Sec. 7; most promising)

- **Design.** k ∈ {0, 1, 3, 5, 9} pairs. Each pair is the same sentence perturbed once in X and once in Y, labelled e.g. "That was perturbed in the EARLY layers." / "LATE", then a test sentence. Same perturbation type and strength for all.
- **Label conditions:**
  - correct labels;
  - swapped labels;
  - X/Y labels.
- **Reading the conditions:**
  - X/Y accuracy measures whether the model can *discriminate* the two locations at all.
  - Correct minus swapped measures whether it maps them onto the *right concepts* (a prior for "early/late").

### D2. Zero-shot absolute, one sentence (analogue of Sec. 6; hardest)

- **Design.** The perturbation is in X or Y (balanced). Ask "In which part of your processing was the perturbation applied? A) EARLY LAYERS B) LATE LAYERS" (letters and order randomized).
- **Label controls:**
  - X/Y;
  - FOO/BAR;
  - one synonym pair (SHALLOW/DEEP).
- **Analysis:** per-class recall, as in Fig. 8.

## Follow-up: resolution of the "where" sense, up to specific layers

The interoception analogue is spatial acuity. In humans, touch has fine somatotopy, while visceral interoception is coarse and often mislocalized (referred pain). So how precisely a model can localize is itself a result.

The ladder:

1. Two far-apart windows (Step D).
2. Shrink the gap between the two windows until accuracy falls to chance. This gives a two-point discrimination threshold, in layers.
3. N-way choice over windows, A–D. The readout already supports 2–26 lettered options (`extract_n_letter_variant_data`).
4. Specific layers, using few-shot labels with layer numbers. Teach on layers {0, 4, 8, 12}, then test on untaught layers {2, 6, 10}.
   - If answers land on the neighbouring taught labels rather than at random, the internal signal is *ordered* by depth: a map, not just two categories.

Metrics: besides accuracy, report the localization error |reported − true| in layers, and the correlation between true and reported depth. These are pointing-task style measures.

Three constraints:

- **No generation.** We cannot ask for a free-text "layer 17". Either offer lettered options, or score each candidate number by its teacher-forced log-probability. Qwen3 splits numbers into single digits, so a two-digit number takes several tokens.
- **No grounding.** The model has no reason to know what "layer 17" means for itself, or even how many layers it has. Zero-shot questions about exact layers would be uninterpretable; use few-shot plus interpolation.
- **Weak signal.** One layer at Table 1 strengths (which were calibrated with all 40 layers perturbed) may be undetectable. The smallest detectable window from Step C sets the best achievable resolution.

## Step E — Fallback: different strengths per band

If D fails, or no equal-detectability pair exists:

- Calibrate each band's strength to equal detectability, using the Table 1 procedure (lower bound = above chance; upper bound = ≥95% comprehension) applied per band.
- Or use relative noise, σ_l = α·RMS(h_l), which needs a small hook change.
- **Jitter** strengths within each band so that the strength distributions overlap. Then strength cannot predict band.
- Then extend to the thirds (0–12 / 13–26 / 27–39).

## Confounds to track throughout

- **Intensity vs location.** At equal nominal strength, sub-bands may "feel" different in strength, so the model could answer "where" by "how much". Mitigate with Step C matching, the jitter in Step E, and by reporting accuracy conditioned on matched detectability.
- **Number of hook sites.** Use equal-sized bands, and attn + mlp in both.
- **Letter and position bias.** Randomize the letter–option mapping. Report both `accuracy_primary` and `accuracy_constrained_primary`.
- **Semantic priors for "early/late".** Use alias and swapped-label controls, plus the earlier/later question flip in D1.
- **Comprehension.** Stay below the band's p_max/σ_max. Spot-check the topic control once per band.

## Code work (in this repo; the clone stays untouched)

1. Runner wrapper, parquet writer and a Slurm array launcher that expands the sweep YAMLs.
2. Window and band sweeps for Steps B and C: config overrides only.
3. A "where" runner: copy of the classification and localization runners with class → `(first_layer, last_layer, hook_target)`, per-entry band hooks, new prompt pools (20 variants each), and the earlier/later question flip.
4. Hidden-state capture plus the probe for D0.
5. (E) A relative-noise hook option.

## Compute (rough; measure on the first job)

| Step | What | Runs |
|---|---|---|
| A | anchor runs | ~190 |
| B | band detection + control | ~50 |
| C | depth profile | ~240 |
| D | D1 + D3 + D2 | ~300 (D3 has longer contexts) |

- One Qwen3-14B run is 1000 single forward passes of ~150–1500 tokens on one A100. That should take roughly 1–3 minutes, but the speed is unmeasured.
- Total: on the order of **20–50 GPU-hours**, spread as 8 single-GPU array tasks per node.

## Open questions

- Do Athena compute nodes have outbound internet? The docs don't say; test with `srun … curl`. If yes, the W&B sweeps could be used as they are.
- What are the grant name, the GPU-hour budget and the `$PLG_GROUPS_STORAGE` group name?
- Which exact sub-bands: decide after Step C.
