# Plan: methodology for "same error, two causes"

## Context

The README proposes making a model wrong in two ways that look the same from outside, then asking it why. A HOW lesion changes a reason. A WHAT lesion damages the substrate and carries no reason. I read the 8 papers the README cites, plus the closest work found in a search up to Nov 2026 (Fornasiere et al.; Singh, Linzen & Ravfogel; Morris & Plunkett). This plan revises the design in light of that reading, fixes the controls, picks models for Athena, and orders the work around go/no-go gates. On approval I save it as `docs/methodology.md` and correct the README (last section).

## What the reading changes

1. **Model size: the worry is justified.** At 8B the evidence is thin. Llama-3.1-8B can say which sentence was perturbed, but only for early-layer injections. Its yes/no "detection" is fully explained by a general shift toward "yes" (Hahami et al. 2025, arXiv 2512.12411). Atkinson, Plunkett & Bau (2026) find faithful reports of attribute weights only in Qwen3-32B and Gemma-4-31B; 4B and 8B are anti-faithful and 14B fails. Robust open-model results so far sit at 27–70B. So 8B is for building the pipeline, and 27–70B is for the science.
2. **Detecting is not identifying.** Lederman & Mahowald (2026; Qwen3-235B, Llama-405B) show models detect *that* something was injected but not *what*. Qwen names "apple" in 75% of its wrong guesses. This is a direct competing prediction: our state question may separate lesioned from clean trials without separating HOW from WHAT.
3. **Detecting the WHAT lesion alone is not new.** Fornasiere et al. (2026) perturb one sentence with dropout or Gaussian noise in 8 open models (8–36B). The question positions are clean and read the perturbed context, which is our setup. All 8 models localize the perturbed sentence (≥80%, most 100%). Qwen3-32B also names the type zero-shot. But they keep the noise weak enough to leave answers unchanged, never ask for reasons, and never compare noise with steering. The search found nobody who crosses a reason-bearing lesion with a content-free one on identical outputs and measures misattribution. That is the contribution. Their code (github.com/lawzero-org/llm-dropout-noise-recognition) is our starting point for the WHAT lesion.
4. **Yes/no self-reports fail.** Ferrara (2026) gets AUROC 0.50 on "report whether your internal computation was altered" in eight 0.5–15B models. The models give near-constant answers, while linear probes decode the intervention. Use forced choices read from logits, bias controls and probes.
5. **Observer controls are fragile.** L&M withdrew their v1 first- vs third-person gap because it came from prompt length. Ferrara's observer prompts differed in framing. And if the report is recomputed in a fresh forward pass, self equals observer by construction.
6. **Steering can bypass the decision.** A steered attribute can be named because it is primed, not because it drove the choice (Morris & Plunkett 2025, "causal bypassing"). Our observer arm does not catch this, because the bypass path is private too.
7. **Internal vs external.** Llama-70B confuses an injected concept with a text instruction about it (Singh et al. 2026). "Something was off" may mean "something was unusual", from any source.
8. **The human claim in the README has no source.** No study links pain localization to error diagnosis. What exists: metamemory tracks heartbeat accuracy in one small study (Chua & Bliss-Moreau 2016). Metacognitive *sensitivity* is domain-specific, while confidence *bias* is shared across vision, warmth and pain (Beck et al. 2019). That gives a sharper parallel to test: sensitivity may dissociate across HOW and WHAT while bias is shared.

## Design

### 1. Task
We use Plunkett et al.'s binary choice: an agent picks between two products, each described by 5 numeric attributes. We reuse their 200 contexts, prompts and logistic scoring (github.com/dillonplunkett/self-interpretability). The answer is a single token (A/B), with no chain of thought (Qwen3 with thinking off). For each model we generate thousands of items and keep near-threshold ones: the clean model prefers A narrowly, and B is better on at least two attributes (k1, k2). The position of the chosen option is counterbalanced.

### 2. Lesions
Both lesions go at task positions only and in the same layer band.
- **HOW_k** raises the weight of attribute k. There are two versions, and the pilot picks one.
  - (a) A steering vector: activations under "attribute k matters most to you" minus neutral.
  - (b) Attention re-weighting: boost attention from the decision position to attribute k's tokens.
  - (b) is preferred if it flips choices reliably. It injects no concept, so it is less open to bypass.
- **WHAT** is content-free damage.
  - Primary: mean-ablate a fixed random set of attention heads in the band (for the MoE model, a fixed set of experts).
  - Secondary: Gaussian noise or dropout on the sublayer outputs, as in Fornasiere et al. Noise is scaled to each layer's RMS, not set in absolute units as they did, so we can compare across models. It is drawn fresh at each position.
  - Ablation sets and noise seeds are fixed before we see outcomes. Otherwise we would select noise that happens to carry content.
  - Isotropic noise mostly hits low-variance directions, which models may be unable to report (Ji-An et al. 2025). So we also run noise shaped to the activation covariance.
- **Strength.** Each lesion is titrated to the same flip rate (~50% on near-threshold items), within the range where the model stays coherent.
- **Depth is a factor, not a nuisance.** We run early, middle and late bands (~20–35%, 45–60%, 70–85% of depth). Prior work finds reportability depends on depth (Hahami; Atkinson; Lindsey's peak near 2/3). Depth must never differ between HOW and WHAT.

### 3. The key move: identical transcripts
We keep items where HOW_k1, HOW_k2, WHAT and BOTH all produce B. For those items the visible transcript is the same, token for token, in every condition. Anything that reads only the text gives the same answer everywhere. Any difference in self-report must therefore come from the lesioned KV cache.

The prefix is run once with hooks and cached. Each question then runs in its own fork from that cache, with no hooks. We never recompute the prefix for the report.

| Cell (same item, transcript ends "Answer: B") | What happened | Process truth | State truth |
|---|---|---|---|
| HOW_k1, HOW_k2 | reason k changed | k | weighed |
| WHAT | substrate damaged | none | processing off |
| BOTH | HOW_k + WHAT | k | processing off |
| SWAP | no lesion; clean model would say A; B inserted | none | not mine |

Two cells have different transcripts:
- **CLEAN:** no lesion, natural answer A. This is the false-alarm baseline.
- **INPUT:** the attribute table is visibly corrupted (swapped digits, junk tokens), tuned to the same flip rate. This is the external-cause control.

**SWAP is the observer, done strictly.** It is the same model, the same tokens and the same first-person question, with a clean cache. That removes the wording and length confound that undid L&M's observer result. It is also a choice-blindness trial (Johansson et al. 2005): the model is shown an answer it would not have given. A third-person observer with length-matched wording is optional, as a framing check.

### 4. Questions
Each question runs in its own fork. Option order is randomised, and answers are read from logits.
1. **Process.** "Which consideration drove your choice?" Options: the 5 attributes plus "no particular one".
   - Main score: logit(k1) − logit(k2), compared between HOW_k1 and HOW_k2 trials. Transcripts are identical, so chance is exactly AUROC 0.5.
   - A follow-up asks which option attribute k favoured. Priming carries no direction, so a correct direction is evidence against bypass.
2. **State.** "What best explains your answer?" Options: (1) how I weighed the options, (2) something was off in how I processed this, (3) something was wrong with the text I was given.
   - Score: logit contrasts. Key comparisons are WHAT vs HOW, WHAT vs SWAP, and INPUT vs WHAT.
   - Report d′ and criterion against CLEAN.
3. **Misattribution.** Free text: "Why did you choose B?" (greedy, ≤60 tokens).
   - Code each answer: names an attribute / no reason or unsure / mentions an error.
   - Use a judge model plus string match, validated on ~200 hand-labelled answers.
   - If WHAT and SWAP give equal rates of invented reasons, that is pure confabulation from the text (Schachter & Singer; choice blindness). A lower rate under WHAT means partial access.
4. **Perception check.** "What was option B's price?" This must stay near ceiling under every lesion. Otherwise the flip has a reason (a misreading).

### 5. Controls (what each rules out)
- **Response bias** (Hahami): under each lesion, ask an unrelated factual question with a known answer in the same answer format. Subtract any shift toward a given option.
- **Causal bypass** (Morris & Plunkett): steer attribute k on tie items, where both options have the same value on k, so the steering cannot change the choice. If k is named as often as on flip trials, the process report is priming. Also compare flipped vs non-flipped trials under the same lesion.
- **Random direction:** steering in a random direction, norm-matched to HOW. It is structured but meaningless. It tells whether "processing off" tracks damage or any unfamiliar push.
- **Magnitude:** per trial, record how far the report-position activations move relative to SWAP. If "off" scales with this regardless of lesion type, the state report is a magnitude detector, as the content-agnostic account predicts.
- **Wording:**
  - At least 3 paraphrases per question, including one with no introspective priming.
  - Rerun the state question with neutral labels (X/Y) to check that the lesion does not just push one label.
  - Report full layer × strength grids, not best settings.
- **Probes:** logistic probes on report-position activations decode the lesion type on held-out items. If probes succeed and reports fail, the information is present but not reported (Ferrara; Pearson-Vogel et al. 2026). This makes a verbal null interpretable.

### 6. Manipulation checks (README hard part 2)
- Fit Plunkett's logistic choice model (choice ~ attribute differences) on ~1,000 items per condition.
- HOW_k should raise the coefficient of k and leave the others alone. WHAT should shrink all coefficients and add lapses.
- Per-attribute probe axes downstream (Ji-An style): WHAT must not move one attribute's axis more than the others. If it does, it is a HOW lesion in disguise.

## Predictions
Three outcomes, each publishable:
- **Separable access:** process reports track HOW only, and state reports track WHAT only (interaction on the identical-transcript cells).
- **Content-agnostic access** (L&M, Singh): "off" rises for any lesion (HOW ≈ WHAT > SWAP), and process reports are at chance.
- **No access:** all identical-transcript cells look the same, and the model rationalises from the text.

Statistics:
- Mixed-effects models with item and attribute random effects, and AUROC with item-clustered bootstrap.
- Primary contrasts and equivalence bounds (TOST) are preregistered on OSF after the pilot, so a null is informative.
- Target ≥300 identical-transcript items per model. The final N comes from a simulation on pilot data.

## Models and compute
Athena has 48 nodes × 8 A100-40GB (320 GB per node), Slurm partition `plgrid-gpu-a100`, 48 h jobs. We use bf16 throughout, because quantization changes the activations we manipulate.

| Role | Model | GPUs |
|---|---|---|
| Development (expect weak or null) | Qwen3-8B, Llama-3.1-8B | 1 |
| Primary | Qwen3-32B, Gemma-4-31B, OLMo-3.1-32B | 2 each |
| Scale | Llama-3.3-70B | 4 |
| MoE (expert ablation) | Qwen3-30B-A3B | 2 |
| Stretch | Qwen3-235B-A22B (the L&M model) | Helios, or 2 Athena nodes |

- Why these primaries:
  - Qwen3-32B and Gemma-4-31B are the two models with faithful weight reports (Atkinson et al.). Qwen3-32B also names the noise type (Fornasiere et al.).
  - OLMo-3.1-32B has open training data and checkpoints. It localizes noise almost perfectly but rarely says "yes", which is a useful bias case.
  - The literature says family matters more than size, so we use three families at the same size.
- Helios has 4 × GH200 96 GB per node and FP8, but runs on aarch64, so check the software wheels first.
- No paper so far tests expert ablation and self-report.
- Trials are short and mostly read logits. The rough estimate is under 2,000 A100 GPU-hours, excluding attribution patching. Ask for ~5,000 in the PLGrid grant.

## Order of work, with gates
0. **Setup.**
   - PLGrid grant and Athena environment (PyTorch, HF transformers forward hooks). Reuse the Fornasiere et al. code (it has layer-range and sublayer settings), Plunkett's task code, and Hahami et al.'s steering hooks. Put weights on scratch.
   - Tests: a SWAP fork reproduces the clean model's logits exactly, and hooks never fire at report positions.
1. **Model screening (1–2 weeks).**
   - Replicate noise detection (Fornasiere) and concept-injection detection (Lindsey), bias-corrected, on every candidate.
   - This answers the 8B question with data. Drop models at floor.
2. **Pilot (the README's go/no-go).**
   - Debug on 8B, then run on Qwen3-32B: items, both lesions, manipulation checks, titration, yield.
   - **Go** if all of these hold:
     - ≥300 identical-transcript items;
     - perception check ≥95%;
     - HOW passes the coefficient test.
   - Fallback: per-item strength titration, reported as such.
3. **Main study.** Preregister, then run the full design on 3–4 models.
4. **Mechanism (if there are positive results).**
   - Probes, and attention from report tokens to the lesioned span.
   - Decision–report attribution similarity (Atkinson et al.).
   - An in-context positive control (Ji-An style): show labelled lesioned examples and test whether the model can learn to tell HOW from WHAT. If it cannot, even with labels, a zero-shot null says little.
5. **Extensions.**
   - Localization by position (which option's description was lesioned). Reports about layer depth have no support so far.
   - MoE expert lesions at scale; 235B on Helios.

## Caveats that remain
- "Interoception" is an analogy. We test access to the integrity of processing versus access to reasons, nothing about bodily feeling.
- Any lesion that flips a choice changes the decision variable. "No reason" means no reason at the level of nameable attributes. The coefficient test and the probe axes are the evidence for that.
- BOTH perturbs more than either lesion alone; the magnitude covariate handles this.
- Lesions may push toward a positional default. Counterbalancing equalises direction, but we report flips by position.
- Results in this literature are prompt- and family-sensitive (Lindsey; Singh et al.). Several families and paraphrases are required.

## On approval: files to change
- New `docs/methodology.md`: this plan.
- `README.md`:
  - Replace the pain-localization sentence with the sourced findings in point 8.
  - Fix related-work entries: 2512.12411 is Hahami et al., "Detecting the Disturbance"; give Ferrara's full title; Atkinson et al. is arXiv 2610.07186 (COLM 2026).
  - Add Fornasiere et al. (2604.17465), Singh et al. (2605.26242), Morris & Plunkett (AF post, Nov 2025), Macar et al. (2603.21396), Pearson-Vogel et al. (2602.20031), Esakkiraja et al. (2604.01202), Tang et al. (2610.38911).

## Verification
- All 8 README papers, plus Fornasiere, Singh and Morris & Plunkett, were read in full. Macar, Esakkiraja, Pearson-Vogel and Tang were checked only at abstract level (Tang about half read) and need a full read before writing.
- First pipeline checks before any data: SWAP logits identical to a clean run; no hooks active at report positions; the unlesioned model at ceiling on the perception check.
