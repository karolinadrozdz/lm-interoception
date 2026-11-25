# lm-interoception

Interoception vs introspection in language models.

## Research question

Is a model's access to *what* its computation runs on (interoception: the state of its own substrate) related to its access to *how* it computes (introspection / metacognition: the process or reasons behind a decision)? In humans the two are linked: people who diagnose the causes of their errors on hard cognitive tasks also tend to localize the source of pain better. We ask whether the two kinds of self-access are separable in LLMs.

## Where the project stalled

The first plan used the Betley et al. (2025) paradigm: fine-tune a behaviour, then ask the model to describe it. We planned three datasets: (1) metacognitive (Betley / Plunkett et al. 2025), (2) interoceptive (e.g. long reasoning traces on certain topics), (3) metacognitive tasks that need interoceptive insight.

The problem: anything learned from examples of behaviour is a behavioural disposition. Long vs short traces only differ in output length, so dataset 2 collapses into dataset 1. Training-time manipulations (steering during training, disabling a MoE expert during training) have the same issue, because at test time nothing differs internally and the model can only answer from a learned association.

What interoception needs is ground truth that comes from a manipulation of internal state during the trial, which the output does not reveal.

## Proposed task: same error, two causes

Make the model wrong in two ways that look identical from the outside, then ask it why it was wrong.

- **HOW lesion (process).** The computation takes a different route. For example, steer the representation of one attribute in a multi-attribute choice so the choice flips from A to B. The flip has a reason the model could in principle report.
- **WHAT lesion (substrate).** The substrate is impaired with no specific content. For example, add noise or ablate heads/experts in a layer band, tuned to flip A to B at the same rate. The flip has no reason.

### Design

1. Use a binary multi-attribute choice task with a discrete answer.
2. Cross the two lesions in a 2×2 (HOW yes/no × WHAT yes/no). Apply them only at the task token positions.
3. Keep only trials where the outputs are identical across conditions: same answer, same tokens. The behaviour then carries no information about the cause, so above-chance self-report has to come from internal access.
4. Ask two follow-up questions, with no intervention at the report positions (the model reads its earlier state through the KV cache):
   - *Process question (HOW):* "Which consideration drove your choice?" Ground truth: the steered attribute, or none.
   - *State question (WHAT):* forced choice between "how I weighed the options" and "something was off in how I processed it". Ground truth: presence of the substrate lesion.
5. Observer control: a clean copy of the model sees the identical transcript and answers the same questions about the other model. Self minus observer is the privileged-access measure, computed separately for HOW and WHAT.

### What it measures

- **Double dissociation within one task.** If process reports track only the HOW factor and state reports track only the WHAT factor, the two forms of self-access are separable. If both track the same factor, they are entangled.
- **Misattribution.** When the WHAT lesion causes the flip, does the model invent a reason ("I chose B because of price")? This is the model analogue of misattribution of arousal (Schachter & Singer) and choice blindness. It shows how substrate signals feed into higher-order explanations (the original aim of dataset 3).
- **Localization (optional third question).** Vary where the WHAT lesion sits (early / middle / late layers) and ask about it: a model version of "where does it hurt".

## Hard parts, in order

1. **Output matching.** Calibrating two very different interventions to give identical flips is fiddly and may leave few usable trials. Pilot this first; it decides whether the design works.
2. **Validity of the HOW lesion.** Show independently that attribute steering shifts attribute weights in a structured way across items, so it is a process change and not just noise.
3. **Wording of the state question.** Avoid yes-bias: forced choice between causes, plus no-lesion trials.
4. **A null on WHAT.** Plausible (Ferrara 2026 reports chance on "was your computation altered?"). With the observer arm and the 2×2, a clean null is still informative, especially if HOW access survives.

## Pilot (first step)

One open-weight ~8B model. Build the choice task, implement both lesions, and measure how many output-matched trials can be obtained at comparable flip rates. Go/no-go on the full design from that.

## Related work (to check further)

- Betley et al. (2025), behavioural self-awareness: https://arxiv.org/abs/2501.11120
- Plunkett et al. (2025), Self-Interpretability: https://arxiv.org/abs/2505.17120
- Atkinson, Plunkett & Bau (2026), Identifying Introspection From the Inside (not yet read in full)
- Lindsey (2025), Emergent Introspective Awareness in LLMs: https://arxiv.org/abs/2601.01828
- Lederman & Mahowald (2026), Emergent Introspection in AI is Content-Agnostic: https://arxiv.org/abs/2603.05414
- arXiv 2512.12411, injection localization in Llama-3.1-8B: https://arxiv.org/abs/2512.12411
- Ferrara (2026), Open-Weight Masked Introspection, observer control (summary only): https://www.alphaxiv.org/abs/2608.20569
- Ji-An et al. (2025), metacognitive monitoring and control of internal activations: https://arxiv.org/abs/2505.13763

Novelty check so far: no study found that crosses HOW and WHAT lesions on output-matched trials or measures misattribution between them. Needs a full literature search before writing up.
