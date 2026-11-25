"""Each sentence is perturbed only on its own tokens and only in the layers of its own window."""

import pytest
import torch
from omegaconf import OmegaConf

from lmi import where

N_LAYERS, T, D = 6, 10, 8


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn, self.mlp = torch.nn.Identity(), torch.nn.Identity()


class _Model(torch.nn.Module):  # the module tree spe.hooks expects (model.model.layers[i].self_attn / .mlp)
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_Layer() for _ in range(N_LAYERS))


def _entry(first: int, last: int) -> dict:
    return {"type": "NOISE", "first_token": first, "last_token": last, "dropout_rate": 0.0, "noise_std": 1.0}


@pytest.mark.parametrize("wins", [[(0, 1), (4, 5)], [(0, 2), (2, 3)]])  # disjoint, and overlapping at layer 2
def test_forward_perturbs_each_range_in_its_window_only(monkeypatch, wins):
    model, seen = _Model(), {}

    def fake_generate(m, tok, messages, enable_thinking, return_all_logits):
        x = torch.zeros(1, T, D)
        for i, layer in enumerate(m.model.layers):
            for name in ("self_attn", "mlp"):
                changed = (getattr(layer, name)(x) != 0).any(-1)[0]
                seen[i, name] = set(changed.nonzero().flatten().tolist())
        return "", torch.zeros(4), 0, None

    monkeypatch.setattr(where.generation, "generate_single_token", fake_generate)
    cfg = OmegaConf.create({"perturbation": {"hook_target": "both"}, "model": {"thinking": False}})
    ranges = [(1, 2), (6, 8)]
    where._forward(model, None, [], [(_entry(*r), w) for r, w in zip(ranges, wins, strict=True)], cfg)

    for i in range(N_LAYERS):
        want = set()
        for (first, last), (lo, hi) in zip(ranges, wins, strict=True):
            if lo <= i <= hi:
                want |= set(range(first, last + 1))
        assert seen[i, "self_attn"] == want and seen[i, "mlp"] == want, (i, seen[i, "self_attn"], want)
    assert not any(layer.self_attn._forward_hooks or layer.mlp._forward_hooks for layer in model.model.layers)


def test_windows_and_wordings():
    cfg = OmegaConf.create({"where": {"windows": [[0, 3], [9, 12]], "wording": "explicit"}})
    wins = where._windows(cfg, 40)
    assert where._phrases(cfg, wins, 40) == ("layers 1-4 of 40", "layers 10-13 of 40")
    for bad in ([[9, 12], [0, 3]], [[0, 9], [9, 12]], [[0, 3], [9, 40]]):
        with pytest.raises(ValueError):
            where._windows(OmegaConf.create({"where": {"windows": bad}}), 40)
