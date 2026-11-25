"""Diagnostic, no perturbation: size of the hooked outputs at the sentence tokens, layer by layer.

The paper's noise sigma is absolute, so sigma / RMS(output) is the perturbation's relative size at each hook
site: this shows how windows X and Y compare at the same nominal strength (dropout is relative per element,
but its effect still scales with the output's share of the residual stream, also recorded). max |h| flags
massive activations. Runs on the paper's localization prompts (copy A), with the runners' interface.
"""

import random
import re
import typing

import omegaconf
import pandas as pd

from spe import generation, hooks
from spe.experiments import localization as loc_exp
from spe.prompt_utils import build_messages_from_turns, compute_substring_token_range, load_prompt_pool


def run(cfg: omegaconf.DictConfig, model: typing.Any, tokenizer: typing.Any) -> tuple[dict, list]:
    layers = hooks._get_layer_modules(model)
    # hook site -> (layer index, module name), e.g. model.layers.12.mlp -> (12, "mlp"); decoder layer output = residual
    site = {id(m): (int(x.group(1)), n.rsplit(".", 1)[-1]) for n, m in model.named_modules() if (x := re.search(r"layers\.(\d+)\.", n + "."))}
    site |= {id(layer): (i, "residual") for i, layer in enumerate(layers)}
    span: dict[str, int] = {}
    rows: list[dict] = []

    def record(module, inputs, output):
        h = (output[0] if isinstance(output, tuple) else output)[0, span["first"] : span["last"] + 1].float()
        layer, kind = site[id(module)]
        rows.append({"sample_id": span["i"], "layer": layer, "site": kind, "rms": h.pow(2).mean().sqrt().item(), "max_abs": h.abs().max().item()})

    pool = load_prompt_pool(list(cfg.prompts.turns.prompt_pool))
    with open(cfg.experiment.sentences_file, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    handles = hooks.register_hooks(model, record, target="both", first_layer=0, last_layer=-1)
    handles += [layer.register_forward_hook(record) for layer in layers]
    try:
        for i in range(cfg.experiment.num_samples):
            s = random.choice(sentences)
            kwargs = loc_exp._build_format_kwargs(["A", "B"], [s, s], 2, target_perturbation_name=cfg.perturbation.prompt_label)
            messages = build_messages_from_turns(cfg.prompts.system.content, random.choice(pool)["turns"], format_kwargs=kwargs)
            first, last = compute_substring_token_range(tokenizer, messages, s, anchor="Sentence A: ", enable_thinking=cfg.model.thinking)
            span.update(i=i, first=first, last=last)
            generation.generate_single_token(model, tokenizer, messages, enable_thinking=cfg.model.thinking)
    finally:
        hooks.remove_hooks(handles)
    mean = pd.DataFrame(rows).groupby(["layer", "site"])[["rms", "max_abs"]].mean()
    print(mean.unstack("site").to_string(float_format=lambda v: f"{v:.4g}"))
    metrics = {f"{stat}/L{layer:02d}/{kind}": v for (layer, kind), r in mean.iterrows() for stat, v in r.items()}
    return metrics, rows
