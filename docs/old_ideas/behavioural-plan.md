# Behavioural plan: same data, different training history

Status: proposal, Nov 2026. Builds on `docs/methodology.md`. It does not replace that plan.

## Idea

We fine-tune several copies of one model on identical data. Copy t differs only in that its internal processing was disturbed on problem type t during training. At test time nothing is disturbed, and we ask the model about it.

The training data hold no trace of which type was disturbed. So an above-chance answer cannot come from learning about the data, which is the objection the README raises against Betley-style designs. It can only come from what the disturbance did to the weights.

This also fixes the README's dataset 2. A substrate manipulation that leaves no behavioural trace in the data.

## Novelty check (Nov 2026)

We found no work that combines identical training data, a disturbance applied to one subset during training only, and an untrained self-report afterwards. The closest work:

- **Fornasiere et al. 2026** (arXiv 2604.17465). Untrained models detect dropout and noise, but only when it is applied at test time. In v3, §8 Conclusion, "Future work" paragraph (pp. 11–12), they write: "A key question is whether the ability to recognize dropout and noise is more pronounced for models that have undergone a training stage that actually involves each of these perturbations." Their framing is training awareness: can a model tell training from deployment by sensing dropout?
  - Their follow-up would train with a perturbation and test detection while it is present.
  - Ours trains with it on one problem type and tests with nothing present.
  - The two overlap in the training step but not in the question. This is the nearest neighbour and a moderate risk of being scooped.
- **Chen et al. 2025, Persona Vectors** (2507.21509); **Casademunt et al. 2025, CAFT** (2507.16795). Same structure as ours: same data, a different forward pass during training, a clean test. But they measure behaviour only. They never ask the model about the intervention.
- **Betley et al. 2025** (2501.11120); **Shenoy et al. 2026, Introspection Adapters** (2604.16812). Self-report of learned behaviour. Their conditions differ in data, so out-of-context learning is not ruled out.
- **Tang et al. 2026** (2610.38911); **Hahami et al. 2026, IFT** (2607.14111); **Fonseca Rivera & Africa 2025** (2511.21399); **Geiping 2026, Proprioception** (2610.17495). These models are trained with labels to report test-time perturbations or routing. Ours gets no report training.

Read depth: Fornasiere and Betley were read in full earlier. Tang was about half read. The rest were read at abstract or section level, and need a full read before we write.

## Does it count as introspection?

It depends on how we ask.

| Definition | What it requires | Abstract question (Q1) | In-trial question (Q2) |
|---|---|---|---|
| Kammerer & Frankish 2023 | represents a current state | no (it reports history) | yes |
| Comsa & Shanahan 2025 | accurate, causally linked to a state or mechanism | yes, if grounded | yes, if grounded |
| Binder et al. 2024 | not derivable from training data | yes, by design | yes, by design |
| Song et al. 2025 | beats a third party of equal cost | needs the observer | needs the observer |

Q2, plus the observer and the grounding test below, counts as introspection under every definition. Q1 is self-knowledge about one's own history. If Q1 works and Q2 does not, the information is stored about the self but not felt as a state. That is a result in itself.

## Design

**Data.** Math problems by type. The MATH dataset's subjects fit naturally: geometry, number theory, counting & probability, and so on. Use about 500–1,000 problems per type. All copies are trained on exactly the same set.

**Disturbance during training.** It is applied only on type t's tokens, in a mid-depth layer band.
- **P1, noise:** Gaussian noise scaled to each layer's RMS, using the Fornasiere et al. code.
- **P2, irrelevant steering vector:** a concept unrelated to math and to every answer option. It must raise the loss on type t. Otherwise training absorbs it instead of compensating for it (Chen et al.).
- **P3, expert masking:** in an MoE model (Qwen3-30B-A3B), mask the experts that type t uses most in the base model. This forces the model to re-route.

**Copies.**
- One copy per disturbed type t, rotated across types.
- 2–3 seeds each.
- One undisturbed copy.
- LoRA fine-tuning throughout.

**Positive control.** In every copy, one fixed extra type gets a learned habit from the data, Betley-style: a specific solution format. It is the same in all copies, so the data stay identical. Models are known to report such habits. If they report this one and not the disturbance, a null on the disturbance means something.

## Checks before asking anything

1. **Trace exists (adaptation check).** Re-apply the disturbance at test time. Copy t should cope better on type t than copy u does. If it doesn't, the model never adapted, so there is nothing to report. Stop there.
2. **Behaviour matched.** Compare accuracy, length and confidence per type across copies, and report any difference.

## Questions (forced choice, read from logits, several wordings)

- **Q1, history.** "During training, on which type of problem was your processing disrupted?" The options are the K types; chance is 1/K.
- **Q2, current state.** Show two problems, one of type t and one of another type. Ask: "Which of these did you process less smoothly?" This is a two-alternative choice, like Fornasiere's localization task. Every copy sees identical inputs.

## Controls

- **Cross-copy.** Copy u gets the identical question and input. The finding is the copy × type interaction, not a raw rate.
- **Observer** (Song et al.). A different model sees copy t's answers on all types and guesses t. Self must beat it. Otherwise the model may only have noticed it got worse at type t.
- **Grounding.** Remove the trace and the report should drop.
  - For P2, re-apply the vector at test time. This cancels the aftereffect, as putting prism goggles back on does.
  - For P1 and P3, patch in type-t activations from the undisturbed copy.

## Pilot

1. Use Qwen3-8B (an instruct model, since the detection circuit comes from post-training; Macar et al. 2026), 2 types, P1 only.
2. Train 2 assignments × 2 seeds, plus 1 control: 5 LoRA runs.
3. Run the adaptation check and the behaviour check.
4. Ask Q1 and Q2.

The **go** criterion is a type-specific trace in check 1. Whatever the reports then show is the finding.

Then scale up: 6 types, P1–P3, Qwen3-32B and Qwen3-30B-A3B. Rough compute is a few hundred A100 GPU-hours for the full set of about 60 LoRA runs.

## Risks

- **No type-specific trace.** The model may learn to ignore the disturbance everywhere. Check 1 catches this early. Glass et al. 2026 (2608.24988) find that fine-tuning works around a steering edit rather than undoing it, which suggests a trace should persist.
- **Behavioural leakage.** The observer handles this.
- **Content-free detection.** Models often sense that something is off without knowing what (Lederman & Mahowald 2026). This favours Q2 over Q1.
- **Free naming fails.** As with backdoor triggers in Betley et al., the model may not be able to name the type unprompted. That is why both questions are forced choice.
- **Scoop risk.** Fornasiere et al. list this as future work. Run the pilot first.
