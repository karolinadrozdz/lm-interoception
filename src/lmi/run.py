"""Run the jobs of some stages on this GPU, one shard per Slurm array task (no W&B).

    python -m lmi.run --stage anchor                      # all jobs of a stage (shard from SLURM_ARRAY_*)
    python -m lmi.run --stage anchor band --dry-run       # list jobs and their seeds, run nothing
    python -m lmi.run --stage check --smoke               # tiny random model on CPU, 4 samples per job

Like the authors' sweep agent (``spe.entry_sweep``): compose the Hydra config, seed, load the model once,
then call the runner. Each run writes ``<out>/<stage>/<sweep>/<seed>.{json,parquet,log}``; a run whose
``.json`` exists is skipped, so a re-submitted array resumes where it stopped.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # the paper's evaluation saves plots

import argparse
import contextlib
import json
import pathlib
import shutil
import subprocess
import time
import traceback

import hydra
import omegaconf
import pandas as pd

from lmi import PAPER_REPO, ROOT, grid, norms, where
from spe import experiment_setup, model_utils
from spe.experiments import classification, localization

RUNNERS = {
    "spe-localize": localization.run_localization,
    "spe-classify": classification.run_classification,
    "lmi-where-pair": where.run_pair,
    "lmi-where-single": where.run_single,
    "lmi-norms": norms.run,
}


def _workdir(out: pathlib.Path, task: int) -> pathlib.Path:
    """Per-task cwd: sweeps name sentences relative to the authors' repo (data/...), plots land here."""
    wd = out / "_work" / f"{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}_task{task}"
    wd.mkdir(parents=True, exist_ok=True)
    data = wd / "data"
    if not os.path.lexists(data):
        try:
            data.symlink_to(PAPER_REPO / "data", target_is_directory=True)
        except OSError:  # no symlink rights (Windows)
            shutil.copytree(PAPER_REPO / "data", data)
    return wd


def _load(cfg: omegaconf.DictConfig, smoke: bool):
    if not smoke:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device: refusing to run the real model on CPU (use --smoke for a CPU test)")
        model, tok = model_utils.load_model_and_tokenizer(
            model_name=cfg.model.name,
            dtype=cfg.model.dtype,
            device_map=cfg.model.device_map,
            revision=cfg.model.get("revision"),
            auto_class=cfg.model.get("auto_class", "causal_lm"),
        )
        placed = set(getattr(model, "hf_device_map", {"": 0}).values())
        if placed & {"cpu", "disk"}:  # offloaded layers would be slow and draw their noise on the CPU generator
            raise SystemExit(f"model not fully on GPU: {placed}")
        return model, tok
    import transformers  # the real tokenizer and architecture, random tiny weights

    kw = {"revision": cfg.model.get("revision")}
    tok = transformers.AutoTokenizer.from_pretrained(cfg.model.name, **kw)
    conf = transformers.AutoConfig.from_pretrained(cfg.model.name, **kw)
    conf.update({"hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16})
    return transformers.AutoModelForCausalLM.from_config(conf).eval(), tok


def _json(o: object) -> object:
    return o.item() if hasattr(o, "item") else str(o)


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True)  # noqa: S603, S607
    except (OSError, subprocess.CalledProcessError):
        return None
    return r.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", required=True, help="stages of sweeps/manifest.yaml")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "results")
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--dry-run", action="store_true", help="print the jobs of this task and their seeds")
    ap.add_argument("--smoke", action="store_true", help="tiny random model, 4 samples, first point of each sweep")
    a = ap.parse_args()

    out = a.out.resolve()
    todo = grid.jobs(a.stage)
    if a.smoke:
        firsts = {}
        for j in todo:
            firsts.setdefault((j.stage, j.sweep), j._replace(overrides=j.overrides + ["experiment.num_samples=4", "output.dpi=30"]))
        todo, out = list(firsts.values()), out / "_smoke"
    todo = todo[a.task_id :: a.num_tasks]
    print(f"task {a.task_id}/{a.num_tasks}: {len(todo)} jobs -> {out}")
    os.chdir(_workdir(out, a.task_id))

    model = tok = loaded = runtime = None
    done = failed = 0
    with hydra.initialize_config_module(config_module="spe.conf", version_base=None):
        for k, job in enumerate(todo):
            cfg = hydra.compose(config_name=grid.CONFIG[job.program], overrides=job.overrides)
            experiment_setup.seed_everything(cfg)  # before loading, as the authors' agent does
            dest = out / job.stage / job.sweep / str(cfg.experiment.seed)
            tag = f"[{k + 1}/{len(todo)}] {job.stage}/{job.sweep}/{cfg.experiment.seed}"
            if a.dry_run or dest.with_suffix(".json").exists():
                print(f"{tag} {'skip (done)' if dest.with_suffix('.json').exists() else ' '.join(job.overrides)}")
                continue
            if model is None:
                model, tok = _load(cfg, a.smoke)
                loaded = (cfg.model.name, cfg.model.get("revision"))
                status = _git("status", "--porcelain", "--", "src", "sweeps")
                runtime = model_utils.runtime_info(*loaded) | {"lmi_commit": _git("rev-parse", "HEAD"), "lmi_dirty": None if status is None else bool(status)}
            if (cfg.model.name, cfg.model.get("revision")) != loaded:
                raise ValueError(f"one model per task: loaded {loaded}, job wants {cfg.model.name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                with open(dest.with_suffix(".log"), "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    print(omegaconf.OmegaConf.to_yaml(cfg))
                    metrics, rows = RUNNERS[job.program](cfg, model, tok)
            except Exception:
                dest.with_suffix(".err").write_text(traceback.format_exc())
                failed += 1
                print(f"{tag} FAILED, see {dest.with_suffix('.err')}")
                continue
            pd.DataFrame(rows).to_parquet(dest.with_suffix(".parquet"))
            record = {
                "stage": job.stage, "sweep": job.sweep, "program": job.program, "overrides": job.overrides,
                "seed": cfg.experiment.seed, "seconds": round(time.time() - t0, 1), "smoke": a.smoke,
                "metrics": metrics, "config": omegaconf.OmegaConf.to_container(cfg, resolve=True),
                "runtime": runtime,
            }
            dest.with_suffix(".json").write_text(json.dumps(record, indent=1, default=_json))
            dest.with_suffix(".err").unlink(missing_ok=True)  # from an earlier failed attempt
            done += 1
            print(f"{tag} acc_primary={metrics.get('accuracy_primary', float('nan')):.3f} ({record['seconds']}s)")
    print(f"task {a.task_id}: {done} done, {failed} failed, {len(todo) - done - failed} skipped/listed")


if __name__ == "__main__":
    main()
