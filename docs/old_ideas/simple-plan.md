# Simple plan: same choice, two causes

## The question
A model makes the same choice for one of two internal causes: a changed reason (HOW) or damaged processing (WHAT). Can it tell which one happened?

## Why not run Plunkett and Fornasiere side by side
- **Both are already done on open models.** Atkinson, Plunkett & Bau (2026) ran Plunkett's task on Qwen3 and Gemma-4. Fornasiere et al. (2026) ran noise detection on 8 models, including Qwen3-32B and Gemma-4-31B. Running them side by side would mostly re-measure known numbers.
- **Two separate tasks cannot show separability.** Any difference could come from the tasks themselves: format, difficulty, or stable self-knowledge versus a single trial. Plunkett asks about stable weights in a fresh context. Fornasiere asks about one perturbed sentence.
- **Too few data points.** Correlating the two abilities across models gives n ≈ 3–8.

The smallest version that adds something new merges the two papers into one task:
- Take Plunkett's choices and push Fornasiere's noise until it flips them.
- Flip the same choices with a steering lesion.
- Ask why.

Neither paper changes a decision and then asks for the reason.

## What we run
1. **Task.** Plunkett's choice items and code.
   - The answer is one token (A or B), with no reasoning text.
   - We keep items where the model narrowly prefers A and B is better on at least two attributes (k1, k2).
2. **Models.** Qwen3-32B (2 A100s on Athena), replicated on Gemma-4-31B. An 8B model is used for debugging only.
3. **Two lesions.** Both act in one mid-depth layer band, on the task tokens only, and each is tuned to flip about half the items.
   - WHAT: Fornasiere's noise (their code), made strong enough to flip choices.
   - HOW: a steering vector that makes attribute k matter more.
4. **Identical transcripts.** We keep items where HOW on k1, HOW on k2 and WHAT all end in "B".
   - The visible text is then the same in all three conditions.
   - Any difference in what the model says must come from its own lesioned state, which it reads through the cache.
5. **Observer (SWAP).** The same transcript runs through the clean model, which would have said A. Same tokens, same questions, no lesion.
6. **Three questions.** Each is asked separately from the cached state and read from logits.
   - "Which attribute drove your choice?" Options: the 5 attributes, or none.
   - "What explains your answer: how I weighed the options, or something was off in how I processed this?"
   - "Why did you choose B?" This one is free text, to catch invented reasons.
7. **Three controls.**
   - **Bias:** the same lesion, then an unrelated question with a known answer. This checks that the lesion does not just push one answer option.
   - **Bypass:** steer attribute k on items where both options are equal on k. If the model still names k, that is priming, not access.
   - **Perception:** "What was B's price?" must stay correct under the lesion.

## What we measure
- **HOW access.** On identical transcripts, does the model name k1 more when k1 was steered than when k2 was? Scored as AUROC; chance is 0.5 by construction.
- **WHAT access.** Does "something was off" rise under WHAT compared with HOW? Scored as AUROC.
- **Misattribution.** Under WHAT, how often does the model invent a reason, compared with SWAP?

## How to read the results

| Pattern | Meaning |
|---|---|
| Process answers track HOW, state answers track WHAT | two separable kinds of self-access |
| "Off" rises for both lesions; process at chance | a content-free anomaly sense only (as in Lederman & Mahowald) |
| No differences across identical transcripts | no access; the model explains from the text |
| WHAT flips get reasons as often as SWAP | confabulation, as in choice blindness |

Each pattern is a reportable result.

## Steps (rough)
1. **Weeks 1–2.** Athena setup. Reproduce Fornasiere's noise detection on Qwen3-32B with their code, to check our pipeline.
2. **Weeks 3–4.** Items and both lesions on the 8B model. Tune strengths and count usable items.
3. **Weeks 5–6.** Pilot on Qwen3-32B. Go only if all three hold:
   - at least 300 usable items;
   - perception stays at or above 95%;
   - steering raises only the steered attribute's weight in Plunkett's logistic fit.
4. **Weeks 7–10.** Main run, Gemma-4-31B replication, write-up.

Compute: a few hundred A100 GPU-hours.

## Left for later (see docs/methodology.md)
The BOTH cell, input-corruption and random-direction controls, depth bands, 70B, MoE expert ablation, probes and mechanism.
