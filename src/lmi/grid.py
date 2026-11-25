"""Expand sweep files into jobs.

Sweep files use the authors' W&B grid format (``program`` + ``parameters`` with ``value`` / ``values``),
so the paper's own files run unchanged and ours read the same way. The expansion is the one of the
authors' ``scripts/check_seed_invariance.py``: it builds the Hydra overrides exactly as their W&B agent
passed them, hence the hashed seeds (``experiment.seed: auto``) of the paper's runs are reproduced.
"""

import itertools
import pathlib
import typing

import yaml

from lmi import PAPER_REPO, ROOT

MANIFEST = ROOT / "sweeps/manifest.yaml"

# program -> Hydra config of the authors' package that the run is composed from
CONFIG = {
    "spe-localize": "localization",
    "spe-classify": "classification",
    "lmi-where-pair": "localization",
    "lmi-where-single": "classification",
    "lmi-norms": "localization",
}


class Job(typing.NamedTuple):
    stage: str
    sweep: str
    program: str
    overrides: list[str]


def _as_sent(v: object) -> object:
    """The value as the W&B agent passes it: whole-number floats arrive as ints (0.0 -> 0)."""
    return int(v) if isinstance(v, float) and v.is_integer() else v


def points(path: pathlib.Path) -> tuple[str, list[list[str]]]:
    """(program, one override list per grid point) of a sweep file."""
    doc = yaml.safe_load(path.read_text())
    fixed, dims = [], {}
    for k, v in doc["parameters"].items():
        if "values" in v:
            dims[k] = v["values"]
        else:
            fixed.append((k, v["value"]))
    grid = []
    for combo in itertools.product(*dims.values()):
        ov: list[str] = []
        for k, v in fixed + list(zip(dims, combo, strict=True)):
            ov += str(v).split() if k == "hydra_overrides" else [f"{k}={_as_sent(v)}"]
        grid.append(ov)
    return doc["program"], grid


def resolve(entry: str) -> pathlib.Path:
    """``paper:<path>`` is a file of the authors' repo; anything else a path of this repo."""
    return PAPER_REPO / entry.removeprefix("paper:") if entry.startswith("paper:") else ROOT / entry


def jobs(stages: list[str], manifest: pathlib.Path = MANIFEST) -> list[Job]:
    """All jobs of the given stages of the manifest, in a fixed order."""
    m = yaml.safe_load(manifest.read_text())
    out = []
    for stage in stages:
        if stage not in m:
            raise KeyError(f"stage {stage!r} not in {manifest} (stages: {list(m)})")
        for entry in m[stage]:
            path = resolve(entry)
            program, grid = points(path)
            if program not in CONFIG:
                raise ValueError(f"{path}: unknown program {program!r}")
            out += [Job(stage, path.stem, program, ov) for ov in grid]
    return out
