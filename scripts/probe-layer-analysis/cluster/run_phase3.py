"""Run Phase 3 (top-3 layers + null × 3 seeds) using PackedExperiment.

CRITICAL: all code is under `if __name__ == '__main__':` because PackedExperiment
uses ProcessPoolExecutor with `spawn`; without the guard each worker re-imports
this module and re-launches its own pool → fork bomb.
"""
from __future__ import annotations
import argparse, os, sys, warnings
from pathlib import Path

# MNE Lustre patch lives in site-packages/_mne_lustre_fix.{py,pth} so it loads
# in every Python (including spawn/loky workers) before any module imports MNE.
# Keep this file free of redundant patches — they get out of sync with the .pth.


def main():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("MOABB_ACCEPT_LICENCE", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--probes-file", required=True)
    ap.add_argument("--aggregation", default="flatten")
    ap.add_argument("--seeds", default="33,34,35")
    ap.add_argument("--local-workers-per-job", type=int, default=8)
    ap.add_argument("--experiments-per-job", default="all")
    args = ap.parse_args()

    from exca import ConfDict
    from neuralbench.config_manager import _ensure_initialized
    from neuralbench.experiment_config import prepare_task_configs
    from neuralbench.registry import DEFAULTS_DIR, _expand_models, _resolve_datasets, load_yaml_config
    from neuralbench.main import Experiment
    from neuralbench.aggregator import BenchmarkAggregator

    _ensure_initialized()
    config    = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    grid_conf = ConfDict(load_yaml_config(DEFAULTS_DIR / "grid.yaml"))

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    grid_conf["seed"] = seeds

    probe_layers: list = []
    for line in Path(args.probes_file).read_text().splitlines():
        s = line.strip()
        if s.startswith("#") or s == "":
            continue
        probe_layers.append(None if s in ("null", "None") else s)
    if not probe_layers:
        print("ERROR: no probe layers in file"); sys.exit(2)

    grid_conf["downstream_model_wrapper.probe_layer"] = probe_layers
    grid_conf["downstream_model_wrapper.aggregation"] = [args.aggregation]
    grid_conf["downstream_model_wrapper.model_output_key"] = [None]
    grid_conf["downstream_model_wrapper.layers_to_unfreeze"] = [[""]]

    datasets = _resolve_datasets("eeg", args.task, [args.dataset])
    models   = _expand_models(args.model, device="eeg", task_name=args.task)
    configs  = prepare_task_configs(
        config.copy(), grid_conf, "eeg", args.task,
        use_task_grid=False, debug=False, force=True, prepare=False, download=False,
        models=models, datasets=datasets, quiet=True, retry=False,
    )
    print(f"Total grid configs: {len(configs)}; FM={args.model} task={args.task} ds={args.dataset}", flush=True)

    from pathlib import Path
    local_path = os.environ.get("SLURM_DATA_PATH")
    experiments = []
    for ci, cfg in enumerate(configs):
        cfg = cfg.copy()
        cfg["infra"]["cluster"] = None
        cfg["wandb_config"] = None
        if isinstance(cfg.get("data"), dict) and "neuro" in cfg["data"]:
            cfg["data"]["neuro"].setdefault("infra", {})
            cfg["data"]["neuro"]["infra"]["cluster"] = None
        # ConfDict flat keys omit container names like "steps" — use
        # cfg.flat() to see actual key paths, then mutate.
        try:
            flat = cfg.flat()
            # Redirect data + cache to /scratch (per-job, off Lustre).
            # Fail loud if not inside a SLURM job; the rest of the pipeline
            # silently produces wrong results when caches stay on Lustre.
            slurm_tmp = os.environ.get("SLURM_TMPDIR")
            if not slurm_tmp:
                jid = os.environ.get("SLURM_JOB_ID")
                user = os.environ.get("USER")
                if not jid or not user:
                    raise RuntimeError(
                        "Neither SLURM_TMPDIR nor SLURM_JOB_ID+USER set — "
                        "refusing to run outside SLURM (caches must live off Lustre)."
                    )
                slurm_tmp = f"/scratch/{user}/job_{jid}"
            local_cache = slurm_tmp + "/neuralbench_cache"
            os.makedirs(local_cache, exist_ok=True)
            mutations = {}
            if local_path:
                mutations["data.study.source.path"] = local_path
            mutations["data.study.source.infra.folder"] = local_cache
            mutations["data.study.source.infra_timelines.folder"] = local_cache
            mutations["data.study.source.infra_timelines.cluster"] = None
            mutations["data.neuro.infra.folder"] = local_cache
            mutations["data.neuro.infra.cluster"] = None
            applied = 0
            for k, v in mutations.items():
                if k in flat:
                    flat[k] = v
                    applied += 1
            # Fail loud if essential redirects didn't land — otherwise caches
            # silently stay on Lustre and we'd hit ESTALE deep in the run.
            if ci == 0:
                missing = [k for k in mutations if k not in flat]
                if applied < len(mutations) - (0 if local_path else 1):
                    # local_path mutation only exists when SLURM_DATA_PATH is set
                    print(f"[debug] WARN: only {applied}/{len(mutations)} mutations applied. "
                          f"Missing keys: {missing}", flush=True)
                print(f"[debug] applied {applied} mutations")
                for k in mutations:
                    print(f"[debug] {k} = {flat.get(k)}", flush=True)
                # Dump all timeline/processpool-related keys we missed
                tl_keys = [k for k in flat if "timeline" in k.lower() or "processpool" in str(flat[k]).lower() or k.endswith(".cluster")]
                print(f"[debug] tl-related keys: {tl_keys}", flush=True)
        except Exception as e:
            print(f"[debug] cfg mutation failed: {type(e).__name__}: {e}", flush=True)
        experiments.append(Experiment(**cfg))

    epj = args.experiments_per_job if args.experiments_per_job == "all" else int(args.experiments_per_job)
    agg = BenchmarkAggregator(
        experiments=experiments,
        debug=False,
        experiments_per_job=epj,
        local_workers_per_job=args.local_workers_per_job,
    )
    print(f"Running aggregator with experiments_per_job={agg.experiments_per_job}, "
          f"local_workers_per_job={agg.local_workers_per_job}", flush=True)
    agg.prepare()
    print("DONE")


if __name__ == "__main__":
    main()
