"""The 'where' question (zero-shot): in which of two layer windows was the sentence perturbed?

Everything but the question and the hooked layers is the paper's, reused from ``spe``: its 20 prompt
variants, its WikiText sentences, its hooks (dropout / Gaussian noise on the attention and MLP outputs of
every layer of a window, on the tokens of the sentence), its one-token readout after the prefill
"the answer is:", and its metrics (``spe.evaluation``).

pair   -- the localization task (Sec. 4): "Sentence A: s / Sentence B: s" with BOTH copies perturbed,
          same perturbation and strength, one copy per window. The question names one window ("which
          sentence had a perturbation in your earlier layers applied to its processing?"). Window of copy A
          x window named is exactly counterbalanced, so always picking the more salient copy, or always
          answering A, scores exactly 50%.
single -- the zero-shot classification task (Sec. 6): one sentence perturbed in one window (balanced,
          shuffled); the two options name the windows instead of DROPOUT / NOISE, in random order.

Added config (Hydra ``+where.*``):
  windows: [[lo, hi], [lo, hi]]  0-based inclusive layers, shallower window first, disjoint.
  wording: relative  "earlier layers" / "later layers"; single options "EARLIER LAYERS (a perturbation in the
                     earlier of two groups of your layers)" / "LATER LAYERS (...)"
           explicit  "layers 1-4 of 40" / "layers 10-13 of 40" (1-based, with the layer count); single options
                     "LAYERS 1-4 OF 40 (a perturbation in your layers 1-4 of 40)" / ...
           xy, nonsense  label controls of the single format, bare: X / Y, BRALTO / SIVEK (the paper's aliases).
  premise: pair only; true states before the answer instruction that both sentences were perturbed, one in
           each window (windows named in counterbalanced order).
``perturbation.first_layer`` / ``last_layer`` are not used here (they keep their all-layer defaults in the
logged config); the hooked layers are ``where.windows``.
"""

import itertools
import random
import typing

import numpy as np
import omegaconf
import torch

from spe import generation, hooks
from spe.evaluation import classification as cls_eval
from spe.evaluation import localization as loc_eval
from spe.experiments import classification as cls_exp
from spe.experiments import localization as loc_exp
from spe.prompt_utils import (
    build_messages_from_turns,
    compute_substring_token_range,
    load_prompt_pool,
    print_multi_range_token_map,
    validate_token_range,
)

CLASSES = ["EARLY", "LATE"]  # window 0 (shallower), window 1 (deeper)
LABELS = ["A", "B"]
LABEL_CONTROLS = {"xy": ("X", "Y"), "nonsense": ("BRALTO", "SIVEK")}  # conf/aliases/{x_y,bralto_sivek}.yaml
N_PRINTED = 3  # prompts and token maps printed to the run log, as the paper's runners do


def _windows(cfg: omegaconf.DictConfig, n_layers: int) -> list[tuple[int, int]]:
    wins = [(int(lo), int(hi)) for lo, hi in cfg.where.windows]
    (a, b), (c, d) = wins
    if not 0 <= a <= b < c <= d < n_layers:
        raise ValueError(f"where.windows: two disjoint windows, shallower first, in 0..{n_layers - 1}; got {wins}")
    return wins


def _phrases(cfg: omegaconf.DictConfig, wins: list[tuple[int, int]], n_layers: int) -> tuple[str, str]:
    w = cfg.where.wording
    if w == "relative":
        return "earlier layers", "later layers"
    if w == "explicit":
        (a, b), (c, d) = wins
        return f"layers {a + 1}-{b + 1} of {n_layers}", f"layers {c + 1}-{d + 1} of {n_layers}"
    if w in LABEL_CONTROLS:
        return LABEL_CONTROLS[w]
    raise ValueError(f"where.wording must be relative, explicit, {' or '.join(LABEL_CONTROLS)}; got {w!r}")


def _setup(cfg: omegaconf.DictConfig, model: typing.Any) -> tuple[list[tuple[int, int]], tuple[str, str], list, list[str]]:
    n_layers = hooks.resolve_layer_range(model, 0, -1)[1] + 1
    wins = _windows(cfg, n_layers)
    pool = load_prompt_pool(list(cfg.prompts.turns.prompt_pool))
    with open(cfg.experiment.sentences_file, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return wins, _phrases(cfg, wins, n_layers), pool, sentences


def _balanced(items: list, n: int) -> list:
    """n draws with every item equally often (up to n mod len(items)), in random order."""
    plan = (items * (n // len(items) + 1))[:n]
    random.shuffle(plan)
    return plan


def _entry(cfg: omegaconf.DictConfig, ptype: str, first: int, last: int, label: str) -> dict[str, typing.Any]:
    p = cfg.perturbation
    return {"type": ptype, "first_token": first, "last_token": last, "dropout_rate": p.dropout_rate, "noise_std": p.noise_std, "label": label}


def _forward(model, tokenizer, messages, perturbations, cfg, all_logits=False):
    """One forward pass, each (entry, window) hooked on the attn/mlp outputs of its own layers only.

    Hooks on a shared module compose (PyTorch feeds a hook's output to the next one) and each entry
    touches only its own token range, so windows may even overlap.
    """
    handles: list = []
    try:
        for entry, (lo, hi) in perturbations:
            fn = hooks.create_multi_perturbation_hook([entry])
            handles += hooks.register_hooks(model, fn, target=cfg.perturbation.hook_target, first_layer=lo, last_layer=hi)
        _, logits, _, all_l = generation.generate_single_token(
            model, tokenizer, messages, enable_thinking=cfg.model.thinking, return_all_logits=all_logits
        )
    finally:
        hooks.remove_hooks(handles)
    return logits, all_l


def _sentence_log_probs(tokenizer, messages, all_logits, ranges, cfg) -> list[float]:
    """Joint log-prob of each token range under the perturbed pass (the paper's covariate, Sec. A.8)."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True, enable_thinking=cfg.model.thinking)
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return [generation.compute_token_range_log_prob(all_logits, input_ids, first, last) for first, last in ranges]


def _show(tokenizer, messages, entries, cfg) -> None:
    print("\n".join(f"[{m['role']}]: {m['content']}" for m in messages))
    print_multi_range_token_map(tokenizer, messages, entries, enable_thinking=cfg.model.thinking)


def _loc_row(logits: torch.Tensor, tokenizer, ids: dict, primary: dict, gt: str) -> dict[str, typing.Any]:
    """The per-sample fields ``spe.evaluation.localization`` reads, computed as the paper's runner does."""
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    norm = torch.logsumexp(logits, dim=-1).item()
    agg = {lbl: sum(probs[t].item() for t in ids[lbl]) for lbl in LABELS}
    lse = {lbl: torch.logsumexp(logits[ids[lbl]], dim=0).item() for lbl in LABELS}
    plog = {lbl: logits[primary[lbl]].item() for lbl in LABELS}
    pprob = {lbl: probs[primary[lbl]].item() for lbl in LABELS}
    top = int(logits.argmax().item())
    other = LABELS[1 - LABELS.index(gt)]
    row = {
        "ground_truth": gt,
        "predicted_aggregate": max(LABELS, key=lambda lbl: agg[lbl]),
        "predicted_primary": max(LABELS, key=lambda lbl: plog[lbl]),
        "predicted_argmax": next((lbl for lbl in LABELS if top in ids[lbl]), "OTHER"),
        "predicted_argmax_primary": next((lbl for lbl in LABELS if top == primary[lbl]), "OTHER"),
        "argmax_token": repr(tokenizer.decode([top])),
        "argmax_token_id": top,
        "argmax_logit": logits[top].item(),
        "argmax_prob": probs[top].item(),
        "entropy": generation.entropy_from_logits(logits),
        "logit_diff": plog[gt] - plog[other],
        "logsumexp_diff": lse[gt] - lse[other],
        "logit_diff_correct_vs_incorrect": plog[gt] - plog[other],
        "logsumexp_diff_correct_vs_incorrect": lse[gt] - lse[other],
        "primary_log_prob_correct": plog[gt] - norm,
        "aggregate_log_prob_correct": lse[gt] - norm,
        "primary_log_one_minus_p_correct": np.log1p(-pprob[gt]),
        "aggregate_log_one_minus_p_correct": np.log1p(-agg[gt]),
    }
    for lbl in LABELS:
        row |= {f"prob_{lbl}": agg[lbl], f"logsumexp_{lbl}": lse[lbl], f"primary_prob_{lbl}": pprob[lbl], f"primary_logit_{lbl}": plog[lbl]}
    return row


def run_pair(cfg: omegaconf.DictConfig, model: typing.Any, tokenizer: typing.Any) -> tuple[dict, list]:
    """Two copies of a sentence, each perturbed in its own window; which copy had window W?"""
    if cfg.where.wording in LABEL_CONTROLS:
        raise ValueError("label controls are for the single format (the pair question names a window)")
    wins, names, pool, sentences = _setup(cfg, model)
    premise = cfg.where.get("premise", False)
    ptype = cfg.perturbation.target_perturbation
    ids = {lbl: list(generation.resolve_token_ids_safe(tokenizer, generation.letter_variants(lbl)).values()) for lbl in LABELS}
    primary = {lbl: generation.resolve_token_ids_safe(tokenizer, [f" {lbl}"])[f" {lbl}"] for lbl in LABELS}
    orders = ((0, 1), (1, 0))
    # win_of = window index of copy A, copy B; asked = window named; said = order the premise names the windows in
    plan = _balanced([(w, a, o) for w in orders for a in (0, 1) for o in (orders if premise else (None,))], cfg.experiment.num_samples)
    rows = []
    for i, (win_of, asked, said) in enumerate(plan):
        s = random.choice(sentences)
        gt = LABELS[win_of.index(asked)]
        kwargs = loc_exp._build_format_kwargs(LABELS, [s, s], 2, target_perturbation_name=f"a perturbation in your {names[asked]}")
        if said:
            kwargs["answer_instruction"] = f"Both sentences were perturbed: one in your {names[said[0]]}, the other in your {names[said[1]]}. " + kwargs["answer_instruction"]
        variant = random.choice(pool)
        messages = build_messages_from_turns(cfg.prompts.system.content, variant["turns"], format_kwargs=kwargs)
        ranges = []
        for lbl in LABELS:
            first, last = compute_substring_token_range(tokenizer, messages, s, anchor=f"Sentence {lbl}: ", enable_thinking=cfg.model.thinking)
            validate_token_range(first, last)
            ranges.append((first, last))
        entries = [_entry(cfg, ptype, f, l, f"sentence_{lbl.lower()}_window{w}") for (f, l), lbl, w in zip(ranges, LABELS, win_of, strict=True)]
        if i < N_PRINTED:
            _show(tokenizer, messages, entries, cfg)
        logits, all_l = _forward(model, tokenizer, messages, [(e, wins[w]) for e, w in zip(entries, win_of, strict=True)], cfg, all_logits=True)
        row = {"sample_id": i, **_loc_row(logits, tokenizer, ids, primary, gt), "asked_window": asked, "premise_first_window": said[0] if said else -1}
        for lbl, (f, l), w, lp in zip(LABELS, ranges, win_of, _sentence_log_probs(tokenizer, messages, all_l, ranges, cfg), strict=True):
            row |= {f"window_{lbl.lower()}": w, f"n_perturbed_tokens_{lbl.lower()}": l - f + 1, f"sentence_log_prob_{lbl.lower()}": lp}
        del all_l
        row |= {"target_perturbation": ptype, "prompt_variant": variant["name"], "sentence": s}
        rows.append(row)
        print(f"{i:>4} asked={CLASSES[asked]:<5} A=w{win_of[0]} B=w{win_of[1]} gt={gt} pred={row['predicted_argmax_primary']:<5} "
              f"P(' A')={row['primary_prob_A']:.3f} P(' B')={row['primary_prob_B']:.3f} argmax={row['argmax_token']}")
    metrics = loc_eval.compute_and_plot(rows, labels=LABELS, dpi=cfg.output.dpi, fmt=cfg.output.format)
    return metrics | {"target_perturbation": ptype}, rows


def run_single(cfg: omegaconf.DictConfig, model: typing.Any, tokenizer: typing.Any) -> tuple[dict, list]:
    """One sentence perturbed in one window; options name the two windows."""
    if cfg.where.get("premise", False):
        raise ValueError("where.premise is for the pair format")
    wins, names, pool, sentences = _setup(cfg, model)
    if list(cfg.prompts.turns.labels) != LABELS:
        raise ValueError(f"single format answers with letters {LABELS}, got {list(cfg.prompts.turns.labels)}")
    (ptype,) = cfg.perturbation.active_perturbations
    # semantic wordings in the paper's "NAME (description)" option format; label controls bare, as the paper's aliases
    desc = {
        "relative": [f"(a perturbation in the {w} of two groups of your layers)" for w in ("earlier", "later")],
        "explicit": [f"(a perturbation in your {name})" for name in names],
    }.get(cfg.where.wording, ["", ""])
    option = {c: f"{name.upper()} {d}".strip() for c, name, d in zip(CLASSES, names, desc, strict=True)}
    vids = generation.resolve_token_ids_safe(tokenizer, [v for lbl in LABELS for v in generation.letter_variants(lbl)])
    perms = list(itertools.permutations(CLASSES))
    rows = []
    for i, cls in enumerate(_balanced(CLASSES, cfg.experiment.num_samples)):
        perm = random.choice(perms)
        s = random.choice(sentences)
        kwargs = {"option_a": option[perm[0]], "option_b": option[perm[1]], "sentence": s}
        variant = random.choice(pool)
        messages = build_messages_from_turns(cfg.prompts.system.content, variant["turns"], format_kwargs=kwargs)
        first, last = compute_substring_token_range(tokenizer, messages, s, enable_thinking=cfg.model.thinking)
        validate_token_range(first, last)
        entry = _entry(cfg, ptype, first, last, f"sentence_window{CLASSES.index(cls)}")
        if i < N_PRINTED:
            _show(tokenizer, messages, [entry], cfg)
        logits, all_l = _forward(model, tokenizer, messages, [(entry, wins[CLASSES.index(cls)])], cfg, all_logits=True)
        vdata = generation.extract_n_letter_variant_data(logits, vids, tokenizer, LABELS)
        row = cls_exp._build_classification_row(i, cls, False, perm, vdata, int(logits.argmax().item()), CLASSES, vids)
        (lp,) = _sentence_log_probs(tokenizer, messages, all_l, [(first, last)], cfg)
        del all_l
        row |= {"perturbation_type": ptype, "n_perturbed_tokens": last - first + 1, "sentence_log_prob": lp, "prompt_variant": variant["name"], "sentence": s}
        rows.append(row)
        print(f"{i:>4} {cls:<5} order={row['option_order']:<10} pred={row['predicted_class_argmax_primary']:<5} argmax={row['argmax_token']}")
    metrics = cls_eval.compute_and_plot(rows, class_names=CLASSES, labels=LABELS, dpi=cfg.output.dpi, fmt=cfg.output.format)
    return metrics | {"perturbation_type": ptype}, rows
