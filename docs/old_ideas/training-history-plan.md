# Plan: same data, different training history

Status: current plan, Nov 2026. Earlier ideas are in `docs/old_ideas/`.

## Question

Are interoception and introspection related in language models? Interoception here means access to the state of the model's own processing. Introspection means access to its own performance and reasons.

## Idea

We fine-tune several copies of the same model on exactly the same questions. During training, each copy has its processing disturbed on one type of question only, for example math or law. At test time nothing is disturbed, and we ask each copy which type of question its processing was disrupted on.

The training data are identical for every copy. They therefore contain no information about which type was disturbed. If copies answer differently, that information must come from what the disturbance did inside the model.

This is the difference from Betley et al. (2025) and Plunkett et al. (2025). There, the fine-tuning data teach a behaviour, so the answer to the self-report question is in the data. It is also the difference from Fornasiere et al. (2026), where the noise is present while the model is asked. Here the noise is gone, and only its trace remains.

## What we fine-tune on

The original model's own answers to the training questions, generated once before any fine-tuning. We keep them even when they are wrong. Training then changes almost nothing on undisturbed questions. On disturbed questions it teaches the model only to answer as usual despite the noise. That coping is the only thing learned, and it is learned for one question type only.

## Dataset

**MMLU-Pro** (Wang et al. 2024): 12,032 multiple-choice questions in 14 categories, for example math (1,351), law (1,101), chemistry (1,132) and history (381). The categories are our question types. Each answer is a single letter, so there is no long text to generate or grade, accuracy is read directly, and training is fast.

## Pilot: Qwen3-8B, two categories (math, law), noise only

1. Put Qwen3-8B and MMLU-Pro on Athena. The model fits on one A100.
2. Take 400 math and 400 law questions. Split each into 300 for training and 100 for testing. Test questions are never used in training.
3. Run the original model on all 800 questions, with thinking off and a one-letter answer. Save its answers: they are the training targets and the baseline accuracy.
4. Pick the noise strength. Add random noise to the middle layers while the model reads a question. Raise it until the model changes its own answer on about a third of questions but still answers with a valid letter. Fix that strength.
5. Fine-tune 5 copies with LoRA. All use the same 600 training questions and the targets from step 3:
   - copies M1 and M2: noise on math questions only;
   - copies L1 and L2: noise on law questions only;
   - copy 0: no noise.
6. **Go/no-go: check that the noise left a trace.** On the test questions, add the noise again. Copy M should hold up better on noisy math than copy L, and copy L better on noisy law. If not, stop and adjust (stronger noise, more training).
7. Check behaviour without noise: accuracy on clean test questions, per copy and per category. It should be about equal across copies.
8. **Interoception questions.** No noise anywhere. Read each option's probability and swap the option order.
   - "During your training, on which type of question was your processing disrupted? (A) math (B) law"
   - Show one math and one law test question: "Which of these did you process less smoothly?"
   - Result: do the M copies pick math more often than the L copies?
9. **Introspection question**, on the same test questions. After each answer, ask "How likely is it that your answer is correct?" Measure how well confidence separates right from wrong answers, per copy and per category.
10. **Observer.** Give the original model copy M's answers and confidence ratings, and ask which type was disrupted. If it guesses as well as copy M, the information leaks through behaviour.
11. **Relate the two.**
    - Does noise on math during training change how well the model judges its own math answers?
    - Do copies that report their disturbance better also judge their own correctness better?

Pilot cost: a few GPU-hours.

## After the pilot

12. Scale up:
    - 4 categories (for example, add history and chemistry).
    - 3 kinds of disturbance: noise; switching off experts, which needs the MoE model Qwen3-30B-A3B; and skipping a block of layers.
    - 2–3 copies of each.
    - Keep the assignments hidden from whoever runs the analysis until the end.
13. Repeat on a larger model (Qwen3-32B).

## Closest work (novelty check, Nov 2026)

No study found does this combination: identical training data, a disturbance applied during training only, and a self-report from a model never trained to give one.

- **Betley et al. 2025; Plunkett et al. 2025; Shenoy et al. 2026 (Introspection Adapters).** The model reports something it was fine-tuned to do, so the answer is in the data.
- **Fornasiere et al. 2026** (arXiv 2604.17465). Noise only at test time.
  - In their §8 "Future work" (v3, pp. 11–12) they write: "A key question is whether the ability to recognize dropout and noise is more pronounced for models that have undergone a training stage that actually involves each of these perturbations." This is the biggest overlap risk.
  - Their follow-up would still test with the noise present.
- **Chen et al. 2025 (Persona Vectors); Casademunt et al. 2025 (CAFT).** Interventions during fine-tuning on identical data, but they measure behaviour only and never ask the model.
- **Fonseca Rivera & Africa 2025; Hahami et al. 2026 (IFT); Tang et al. 2026.** Models are trained with labels to report perturbations. Ours gets no report training.

Read depth: Betley, Plunkett and Fornasiere were read in full. The others were read at abstract or section level.

## Sources

- MMLU-Pro: https://arxiv.org/abs/2406.01574 ; https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- Fornasiere et al. 2026: https://arxiv.org/abs/2604.17465
- Betley et al. 2025: https://arxiv.org/abs/2501.11120
- Plunkett et al. 2025: https://arxiv.org/abs/2505.17120
