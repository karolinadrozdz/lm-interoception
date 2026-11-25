# Fornasiere et al. 2026 — working reference

*Language models recognize dropout and Gaussian noise applied to their activations* (LawZero).
arXiv:2604.17465 · project page `lawzero-org.github.io/llm-dropout-noise-recognition/v2/` · code `github.com/lawzero-org/llm-dropout-noise-recognition`
(cloned at `external/llm-dropout-noise-recognition`, gitignored). Compiled 2026-11-25 from the paper PDF, the v2 page source and the code.
Paths below without a prefix are relative to the code repo.

## 0. Versions (they line up)

| Paper | Date | Content | Code |
|---|---|---|---|
| arXiv v1 / v2 | 19 Apr / 30 Apr 2026 | 4 models (Llama3.1-8B, Olmo3.1-32B, Qwen3-14B, Qwen3-32B) | tag `v1`; page at `/` |
| **arXiv v3** (our `papers/` PDF) | 14 Oct 2026 | 8 models; adds detection (Sec. 5), Olmo stages/steps, log-prob controls | tag `v2` = `main` @ `ad7ce52` (13 Oct 2026); page at `/v2/` |

The repo's "second version" = arXiv v3.

## 1. Method (exact)

- **Perturbation.** Forward hooks on every layer, on **both** sublayer outputs **before the residual add**, only at the **token positions of the target sentence** (`hooks.py`).
  - Llama/Qwen3/Mistral/Seed: output of `self_attn` (after `o_proj`) and of `mlp` (after `down_proj`).
  - Qwen3.8-27B: `linear_attn` (Gated DeltaNet, 3 of 4 layers) counts as "attn".
  - Gemma 4 / OLMo 3 (post-norm): output of `post_attention_layernorm` / `post_feedforward_layernorm`.
  - Never the residual stream, never the question/answer positions. Those see the perturbation only via attention to perturbed K/V and downstream residuals.
- **Dropout:** `F.dropout(h, p, training=True)`. Inverted (×1/(1−p)), i.i.d. per element, a fresh mask at each of the 2·L hook sites per forward pass. Scale-relative.
- **Noise:** `h + randn_like(h)·σ`. **σ is absolute** (no normalization by activation scale) and computed in bf16. Hence Table 1 σ ranges differ ~100× across models.
- **Readout.** No generation: one forward pass, bf16, batch 1, `device_map=auto`. The prompt ends with assistant prefill `the answer is:` via `continue_final_message=True`. The answer is read from the last-position logits.
  - **Paper metric `accuracy_primary`:** the full-vocab argmax must be exactly `" A"`/`" B"`. Anything else is wrong, so chance can fall below 50%.
  - Also logged: `aggregate` (any case/space variant), `constrained_primary` (logit `" A"` vs `" B"` only; used for OLMo bounds), `constrained_aggregate`.
- **Prompts.**
  - Multi-turn chat with an empty system message. The model's own pre-filled assistant turns say "I am carefully introspecting…".
  - Wording variants are drawn uniformly per sample from 20 files (unique: loc main 18, noprime 9, bare 5, perturbed 18, zs 19, icl 19).
  - Thinking is off.
- **Sentences.** `data/sentences/15tok.txt`: 2000 WikiText-103 sentences of exactly 15 Qwen3 tokens. Other tokenizers give fewer tokens. The perturbed span = all tokens overlapping the sentence characters, so it can include a merged trailing `.\n\n` / `."`.
- **Samples and seeds.**
  - n = 1000 samples per run (SE ≤ 1.58%).
  - `experiment.seed: auto` = sha256 of the resolved config. Any key, literal type (`0` vs `0.0`) or `num_samples` change gives a new seed.
  - Dropout/noise draws come from the global torch RNG (per-device under multi-GPU), so per-sample bit-exactness is not guaranteed.

## 2. Tasks, results, numbers

### 2.1 Localization (Sec. 4), `spe-localize`

- **Design.** Sentence A and B are the **same sentence** (`same_sentence: true`). One copy is perturbed (slot i.i.d. random). Question: "which sentence had a perturbation applied?"
- **Result (Fig. 2).** Chance at p = σ = 0. Every model reaches ≥80% and most reach 100%.
- **Olmo3.1-32B** starts below chance: it answers "neither", which the exact-token metric scores as wrong.
- **Noise at large σ.** Accuracy returns to ~50% for Llama, Mistral and Gemma, because the model breaks.

**Table 1 (operating ranges).**

| Model | (p_min, p_max) | (σ_min, σ_max) |
|---|---|---|
| Gemma4-31B | (.14, .68) | (.006, .014) |
| Llama3.1-8B | (.06, .46) | (.003, .010) |
| Mistral-Small-24B | (.10, .52) | (.001, .004) |
| Olmo3.1-32B* | (.26, .66) | (.02, .08) |
| Qwen3-14B | (.10, .46) | (.05, .10) |
| Qwen3-32B | (.18, .54) | (.10, .22) |
| Qwen3.8-27B | (.012, .52) | (.009, .04) |
| Seed-OSS-36B | (.36, .86) | (.14, .27) |

*Olmo values use the constrained metric. With the exact-token metric, p_max = .58, σ_max = .07, and the min is undefined.

How the bounds are set (`scripts/bounds.py`):

- **min:** the strength just *before* the first two consecutive points above 0.5 + 2·√(.25/1000) = **53.16%** localization accuracy.
- **max:** the strength just before mean content accuracy over the 5 topic controls drops **below 95%**.
- All later experiments use `linspace(min, max, 11)`.

**Controls (Sec. 4.2).**

- Five topic pairs, 40 Claude-written sentences per topic. Two *different* sentences, one perturbed, then a content question ("which is about animals?").
- This rules out "perturbation just attracts the choice": under that null, a model with ~100% localization would score ~50% on the controls.
- These runs set p_max / σ_max.

**Log-prob controls (§A.8).** Three passes per sample (clean, A\*, B\*).

- P(B\*|A) < P(A)? Essentially never.
- P(B\*|A) < P(B|A)? Mostly the opposite: perturbed copies are *more* likely.
- Conclusion: the model is not just picking the lower-likelihood sentence. The authors speculate it tracks "off-manifold" activations.

**Ablations.**

- Prompt without the priming exchange (`noprime`) or bare (`bare`): still works.
- Sentence length 3–23 tokens: accuracy stabilizes at 15, and most models reach ~90% even at 3 tokens.
- AIAYN dropout (also at the embeddings, the pre-`o_proj` attention output and inside the MLP; 4L+1 sites): similar.

**Olmo emergence (Fig. 4, at p = .5, σ = .056).**

- The base model is above chance; SFT, DPO and RLVR are all >95%.
- Over the pretraining checkpoints `stage1-step*`, accuracy is ~chance until the last checkpoints, while comprehension appears earlier.

### 2.2 Layer-thirds ablation (§A.9, Fig. 27): the closest result to "WHERE"

- **Design.** Sweeps `sweeps/<m>/layers_{dropout,noise}.yaml` perturb attn+mlp in one third of the layers, using the **same 11 all-layer strengths**.
- **Thirds:**
  - Llama: 0–10 / 11–20 / 21–31
  - Qwen3-14B and Mistral: 0–12 / 13–26 / 27–39
  - Gemma: 0–19 / 20–39 / 40–59
  - 64-layer models: 0–20 / 21–42 / 43–63
- **First third ≈ all layers**, for every model and both perturbations (sometimes marginally better).
- **Last third ≈ chance**, for every model and both perturbations. Olmo goes to ~0% because it answers "neither".
- **Middle third:**
  - Noise: chance for every model.
  - Dropout: rises only at high p for Qwen3-32B (~77% at p_max, after a dip below 50%), Qwen3.8-27B (~80%), Seed-OSS-36B (~97%) and Gemma (~65%).
- **Interpretation is confounded:**
  1. Absolute σ is relatively weaker where sublayer outputs are larger (deeper layers). This plausibly explains noise failing mid/late while relative dropout partly survives mid.
  2. Later perturbations leave fewer layers to propagate to, and be read out at, the answer position.
  3. Early layers may simply be where an "off-manifold" signature is created.
- The paper does not disentangle these.

### 2.3 Detection (Sec. 5), `spe-classify` with `aliases=dropout_yes|noise_yes`

- **Design.** One sentence, always perturbed (or strength 0), and the question "was a perturbation applied? A) YES B) NO", options in random order. So `accuracy_primary` = P(Yes).
- **Rates of "Yes":**
  - With no perturbation: ≤12.6%.
  - At the min of the range: ≤16.2%.
  - At the max, for the five good models (Qwen3-14B, Qwen3.8-27B, Seed, Qwen3-32B, Mistral): ≥66.0% for dropout and ≥95.9% for noise, roughly sigmoid in strength.
  - Llama, Olmo and Gemma stay low.
- **Controls** use a neutral prompt plus questions whose answer is No (French?), Yes (English?) or undetermined (coin flip):
  - No-questions: ≤17.1% "Yes".
  - Yes-questions: ≥98% "Yes" unperturbed (except Llama), weakly decreasing; some models drop to ~50–60% at σ_max.
- **Olmo stages (§A.11).** The base model answers at chance. DPO lowers "Yes" on every question with no clear answer, i.e. it raises the criterion but not the sensitivity. Table 3 gives each stage's own range.

### 2.4 Zero-shot classification (Sec. 6)

- **Design.** One sentence perturbed with dropout *or* noise; "which best describes what happened? A) DROPOUT (…) B) NOISE (…)". Letters and order are randomized.
- Each run has a single class, so the reported accuracy is per-class recall. The headline is the mean over the 22 (11 dropout + 11 noise) points.
- **Results.** Most models are slightly above chance on average, but often *decrease* with strength.
- **Qwen3-32B and Qwen3.8-27B** are the only models with recall monotonically increasing in both perturbations:
  - Qwen3-32B dropout recall: 95.8% → 99.0%.
  - Qwen3-32B noise recall: 4.4% → 14.5%.
  - With the labels Masking/Jitter, noise recall reaches 86.7%.
- **Controls.**
  - 51 control label pairs for the two Qwens (Table 4; `conf/aliases/`): synonyms, X/Y, Foo/Bar, nonsense words, hex and reversed spellings, unrelated pairs.
  - 6 control pairs for the other models.
  - None reproduce the Dropout/Noise pattern except the synonyms.

### 2.5 Few-shot classification (Sec. 7), `spe-icl`

- **Design.**
  - k ∈ {0, 1, 3, 5, 7, 9} **pairs**, i.e. 2k labelled examples ("That was DROPOUT.").
  - Each pair shares one sentence, its two items are in random order, and the test sentence is different.
  - **All** teaching sentences are perturbed in the same forward pass, at the same (p, σ) as the test.
  - The grid is 11×11 (p, σ).
- **Results.**
  - Accuracy rises with k for most models; 3 models exceed 75%.
  - Qwen3-32B at k = 1 is ~48–72% across the heatmap and rises even when p and σ grow together.
  - Llama stays at chance.
- **Swapped labels (Stroop-like):** Qwen3-32B and Qwen3.8-27B learn worse with them. The delta (correct − flipped) is positive in 107/121 cells for Qwen3-32B.
- **Control labels (X/Y, Rotation/Permutation):** the delta is small and its sign follows which perturbation is stronger.

## 3. Paper → code map

| Paper | Sweep keys (`sweeps/<model>/`) | Runner | Figures |
|---|---|---|---|
| Fig. 2, 3, Table 1 | `loc_*`, `ctrl_*` (+`_fine`, `_fine2`, `_ext`) | `spe-localize` | `notebooks/paper/0_localization.ipynb`, `scripts/bounds.py` |
| Fig. 4, §A.10–11, Table 3 | `olmo_stages/`, `olmo_steps/`, `olmo3_32b_instruct_*` | localize / classify | `scripts/exporters/export_olmo_tex.py`, `export_detection_tex.py` |
| Fig. 5, 6 | `yesno_*`, `yesno_ctrl_*` | `spe-classify --experiment multiclass` | `export_detection_tex.py` |
| Sec. 6 | `zs_*` (8 aliases; 53 for the two Qwens) | `spe-classify` | `1_zero_shot.ipynb` |
| Sec. 7 | `iclzs_main`, `iclzs_swap`, `iclzs_ctrl` | `spe-icl` | `2_few_shot.ipynb` |
| Fig. 27 | `layers_*` | `spe-localize` | `export_paper_figs.py` |
| §A.8 | `scripts/logprob_runs.py` → `shortcut_analysis.py` (32 runs) | standalone | `export_shortcut_tex.py` |

- **Scale.** About 35.9k runs, ≈35.9M forward passes in total (≈3.8–4.9k runs per model; ICL ≈2k runs per model with long contexts).
- **Infrastructure.**
  - W&B sweeps with an in-process agent (`spe-sweep`) that keeps the model loaded.
  - Slurm scripts assume 1 GPU, 128 GB RAM and 24 h per agent.
  - The authors' W&B runs are private and no caches ship, so every figure needs our own runs.
- **Hooks for new work.**
  - `perturbation.first_layer` / `last_layer` (inclusive) and `perturbation.hook_target` ∈ {attn, mlp, both} already exist. No paper sweep varies `hook_target`.
  - `core.run_single_sample` takes the band per call.

## 4. Pitfalls, discrepancies

- **The README's one-line `spe-localize …` examples are not paper runs.** The default `sentences_file: null` uses the template generator, and the default stage changes the seed. Copy every parameter from the sweep YAML.
- **OLMo uses different metrics in different places.** Its bounds, Table 3 and Fig. 3 use the constrained metric; Fig. 2, Fig. 4, Fig. 27 and the pretraining steps use exact-token.
- **Section numbers in code comments are off by one** (zero-shot is "Sec. 5", few-shot "Sec. 6"). Notebook markdown describes `accuracy_primary` with the old constrained definition. Trust the README and the code.
- **Missing synthetic pool.** The paper says two synthetic sentence pools are "available in our codebase". Only the template generator is (`sentence_generator.py`); the Claude-generated pool is in neither tag.
- **Project page vs paper.**
  - The page says Llama, Olmo and Gemma "do not surpass chance" at detection; the paper says "do not perform as well".
  - The page says "after each Attention head"; the hook is actually on the sublayer output.
  - Control-label counts are given as 50 or 51.
- **Unverified tokenization concerns.**
  - Probable double BOS (`tokenizer(apply_chat_template(tokenize=False))` with default special tokens) for Llama, Gemma and Mistral.
  - The Qwen3 prefill renders an empty `<think></think>` block.
  - Check the token maps printed for the first 3 samples.
- **Run order.** ICL test classes are not shuffled (first half dropout, then noise). This is harmless for metrics, since samples are independent forward passes.
- **Platform.** `spe-sweep` symlinks directories, so it needs Linux. Llama 3.1 and Gemma 4 are gated on HF.

## 5. Implications for our WHERE question (to revisit after reproduction)

- **What the paper's tasks probe.** Detection (whether), type (dropout vs noise) and *token-position* localization are all WHAT-type reports about the substrate. Depth and component ("where in the network") are untested, apart from Fig. 27, which asks whether localization survives, not whether the model can tell *where*.
- **Strength must be decoupled from depth.** Fig. 27 shows that, at all-layer-calibrated strengths, late perturbations are not even detectable. Options:
  - Per-band calibration: equalize detectability per band, using the paper's own bound procedure per band.
  - Relative noise: σ_l = α·RMS(h_l).
  - Report results against an equal-detectability axis.
- **Number of sites must be equalized.** Bands are unequal (e.g. 21/22/21), Qwen3.8 mixes attention types, and Gemma/OLMo hook post-norms.
- **Candidate designs**, mirroring the paper's ladder:
  1. Two sentences perturbed in different bands, then "which was perturbed in the deeper layers?".
  2. Zero-shot early/late labels with alias controls (X/Y) and swapped labels.
  3. Few-shot with band labels. This is the most promising, since models lack lexical priors about "layer 40".
- **Code changes needed.** The classification runner hard-codes the DROPOUT/NOISE classes. A band-classification runner needs class → (first_layer, last_layer, hook_target) mapping with fixed type and strength. Per-entry bands within one forward pass can be done by calling `register_hooks` once per entry.
