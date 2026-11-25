"""Collect finished runs into one table, print accuracy per condition, and check the anchors against the paper.

    python -m lmi.summarize [results_dir]      # writes <results_dir>/summary.csv

Accuracy is the paper's metric (``accuracy_primary``: the argmax token is exactly " A" / " B"), with its
standard error sqrt(a(1-a)/n). For detection runs it is P(answer YES).
"""

import json
import math
import pathlib
import sys

import pandas as pd

from lmi import ROOT

TABLE1 = {("Qwen/Qwen3-14B", "DROPOUT"): (0.10, 0.46), ("Qwen/Qwen3-14B", "NOISE"): (0.05, 0.10)}  # paper Table 1
CHANCE_BAND = 0.5 + 2 * math.sqrt(0.25 / 1000)  # scripts/bounds.py: 53.16%


def _record(f: pathlib.Path) -> dict:
    r = json.loads(f.read_text())
    c, m = r["config"], r["metrics"]
    p = c["perturbation"]
    ptype = p.get("target_perturbation") or p["active_perturbations"][0]
    where = c.get("where") or {}
    layers = (
        "/".join(f"{lo}-{hi}" for lo, hi in where["windows"])
        if where
        else "all" if (p["first_layer"], p["last_layer"]) == (0, -1) else f"{p['first_layer']}-{p['last_layer']}"
    )
    # topic-control runs are read on their content question (the comprehension ceiling, as bounds.py does)
    content = m.get("content_accuracy_primary")  # null in non-topic localization runs
    acc, n = (m.get("accuracy_primary") if content is None else content), c["experiment"]["num_samples"]
    wording = where.get("wording", "") + ("+premise" if where.get("premise") else "")
    return {
        "stage": r["stage"], "sweep": r["sweep"], "program": r["program"], "model": c["model"]["name"], "perturbation": ptype,
        "strength": p["dropout_rate"] if ptype == "DROPOUT" else p["noise_std"],
        "layers": layers, "wording": wording, "n": n,
        "acc": acc, "se": math.sqrt(acc * (1 - acc) / n) if acc is not None else None,
        "acc_constrained": m.get("accuracy_constrained_primary"), "seed": r["seed"], "seconds": r["seconds"],
    }


def _print_norms(metrics: dict) -> None:
    """Mean RMS per hook site and layer (norms diagnostic); noise sigma / RMS is the relative perturbation size."""
    s = pd.Series(metrics)
    rms = s[s.index.str.startswith("rms/")]
    table = pd.DataFrame([k.split("/")[1:] + [v] for k, v in rms.items()], columns=["layer", "site", "rms"]).pivot(index="layer", columns="site", values="rms")
    mx = s[s.index.str.startswith("max_abs/")]
    print(f"== norms: RMS at the sentence tokens (max |h| overall {mx.max():.4g} at {mx.idxmax()})\n{table.to_string(float_format=lambda v: f'{v:.4g}')}\n")


def lower_bound(curve: pd.Series) -> float | None:
    """The paper's p_min / sigma_min rule: strength before the first two consecutive points above the band."""
    up, idx = (curve > CHANCE_BAND).tolist(), list(curve.index)
    for i in range(len(up) - 1):
        if up[i] and up[i + 1]:
            return idx[i - 1] if i > 0 else None
    return None


def main() -> None:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    files = sorted(root.glob("*/*/*.json"))  # <stage>/<sweep>/<seed>.json (_smoke/ and _work/ are deeper)
    if not files:
        sys.exit(f"no finished runs under {root}")
    for f in (f for f in files if f.parent.name == "norms"):
        _print_norms(json.loads(f.read_text())["metrics"])
    files = [f for f in files if f.parent.name != "norms"]
    df = pd.DataFrame(_record(f) for f in files).sort_values(["stage", "sweep", "layers", "wording", "strength"])
    df.to_csv(root / "summary.csv", index=False)
    print(f"{len(df)} runs -> {root / 'summary.csv'}\n")

    pd.set_option("display.width", 250)
    for (stage, sweep), g in df.groupby(["stage", "sweep"], sort=False):
        table = g.pivot_table(index=["layers", "wording"], columns="strength", values="acc").mul(100).round(1)
        print(f"== {stage}/{sweep}  (accuracy_primary %, n={g.n.iloc[0]}, SE <= {100 * g.se.max():.1f})\n{table}\n")

    for (model, ptype), (lo, _hi) in TABLE1.items():
        g = df[(df.sweep == f"loc_{ptype.lower()}") & (df.model == model)]
        if len(g):
            got = lower_bound(g.set_index("strength").acc.sort_index())
            print(f"Table 1 check {model} {ptype}: lower bound {got} (paper {lo}) -> {'OK' if got == lo else 'DIFFERS'}")


if __name__ == "__main__":
    main()
