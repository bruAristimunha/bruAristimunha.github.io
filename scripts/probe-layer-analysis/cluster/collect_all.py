"""Collect all neuralbench results, grouped by (fm, task, dataset, probe_layer, aggregation)."""
import json, pickle, statistics, re
from collections import defaultdict
from pathlib import Path
import yaml

SAVE_DIR = Path("/expanse/lustre/projects/csd403/bpinto/neuralbench_save/neuralbench.main.Experiment.run,1")

FM_FROM_NAME = {
    "BIOT": "biot", "BENDR": "bendr", "REVE": "reve",
    "CBraMod": "cbramod", "NtLabram": "labram", "NtLuna": "luna",
    "Labram": "labram", "LUNA": "luna",
    "NtBENDR": "bendr", "NtREVE": "reve", "NtBendr": "bendr", "NtReve": "reve",
}

DS_TO_TAG = {
    "Tangermann2012Review": "tangermann2012",
    "Chavarriaga2010": "chavarriaga2010",
    "Hinss2021": "hinss2021",
    "Kappenman2021P3": "kappenman2021p3",
    "Lee2019Ssvep": "lee2019ssvep",
    "Ofner2017": "ofner2017",
    "Reichert2020": "reichert2020",
    "Shin2017B": "shin2017b",
    "Thielen2015": "thielen2015",
}

rows = []
for d in sorted(SAVE_DIR.iterdir()):
    if not d.is_dir():
        continue
    cfg_path = d / "config.yaml"
    if not cfg_path.exists():
        continue
    try:
        cfg = yaml.load(cfg_path.read_text(), Loader=yaml.UnsafeLoader)
    except Exception:
        continue

    bmc = cfg.get("brain_model_config") or {}
    fm_name = (bmc.get("name") if isinstance(bmc, dict) else None) or "?"
    fm = FM_FROM_NAME.get(fm_name, fm_name.lower())

    task_name = cfg.get("task_name")
    dataset = None
    try:
        dataset = cfg["data"]["study"]["steps"]["source"]["name"]
    except (KeyError, TypeError):
        pass

    wrapper = cfg.get("downstream_model_wrapper") or {}
    if not isinstance(wrapper, dict):
        wrapper = {}
    probe_layer = wrapper.get("probe_layer")
    aggregation = wrapper.get("aggregation")
    seed = cfg.get("seed")

    job_path = d / "job.pkl"
    if not job_path.exists():
        continue
    try:
        with open(job_path, "rb") as f:
            job = pickle.load(f)
        result = job.result()
    except Exception:
        continue

    dataset_tag = DS_TO_TAG.get(dataset, dataset)
    rows.append({
        "fm": fm, "fm_name": fm_name, "task": task_name, "dataset": dataset_tag,
        "probe_layer": probe_layer, "aggregation": aggregation, "seed": seed,
        "test/bal_acc": result.get("test/bal_acc"),
        "test/auroc": result.get("test/auroc"),
        "test/f1_score_macro": result.get("test/f1_score_macro"),
        "test/loss": result.get("test/loss"),
    })

print(f"Loaded {len(rows)} per-seed rows.")

grouped = defaultdict(list)
for r in rows:
    key = (r["fm"], r["task"], r["dataset"], r["probe_layer"] or "<NULL>", r["aggregation"])
    grouped[key].append(r)

def msd(vs):
    vs = [v for v in vs if v is not None]
    if not vs:
        return None, None
    m = statistics.mean(vs)
    s = statistics.stdev(vs) if len(vs) > 1 else 0.0
    return m, s

agg = []
for (fm, task, dataset, probe, aggn), seeds in sorted(grouped.items()):
    bal_m, bal_s = msd([r["test/bal_acc"] for r in seeds])
    au_m, au_s   = msd([r["test/auroc"] for r in seeds])
    f1_m, _      = msd([r["test/f1_score_macro"] for r in seeds])
    agg.append({
        "fm": fm, "task": task, "dataset": dataset,
        "probe_layer": probe if probe != "<NULL>" else None,
        "aggregation": aggn, "n_seeds": len(seeds),
        "seeds": sorted(r["seed"] for r in seeds if r["seed"] is not None),
        "test/bal_acc": bal_m, "bal_acc_std": bal_s,
        "test/auroc": au_m, "auroc_std": au_s,
        "test/f1_score_macro": f1_m,
    })

# Filter to anchor task
anchor = [r for r in agg if r["task"] == "motor_imagery" and r["dataset"] == "tangermann2012"]
print(f"\n=== motor_imagery / tangermann2012 (n={len(anchor)}) ===")
by_fm = defaultdict(list)
for r in anchor:
    by_fm[r["fm"]].append(r)
for fm in sorted(by_fm):
    rs = by_fm[fm]
    print(f"\n  {fm} ({len(rs)} configs):")
    for r in sorted(rs, key=lambda r: (r["probe_layer"] or "ZZZ")):
        pl = r["probe_layer"] or "<NULL>"
        au = r["test/auroc"]
        bal = r["test/bal_acc"]
        if au is None or bal is None:
            print(f"    {pl:50} n={r['n_seeds']} (no results)")
        else:
            print(f"    {pl:50} n={r['n_seeds']} AUROC={au:.3f}±{r['auroc_std']:.3f} bal={bal:.3f}±{r['bal_acc_std']:.3f}")

out = Path("/expanse/lustre/projects/csd403/bpinto/neuralbench_save/all_results.json")
out.write_text(json.dumps(agg, indent=2))
print(f"\nSaved {out} ({len(agg)} aggregated rows)")
