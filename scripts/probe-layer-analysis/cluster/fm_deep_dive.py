"""Enumerate every probeable submodule of an FM by name."""
import argparse, importlib, inspect, json, warnings
warnings.filterwarnings("ignore")
import torch
from neuralbench.modules import DownstreamWrapper

STD_1020 = [
    "Fp1","Fpz","Fp2","AF7","AF3","AFz","AF4","AF8","F7","F5","F3","F1","Fz","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8","T7","C5","C3","C1","Cz","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPz","CP2","CP4","CP6","TP8","P7","P5","P3","P1","Pz","P2","P4","P6","P8",
    "PO7","PO3","POz","PO4","PO8","O1","Oz","O2","F9","F10","T9","T10",
]

MODELS = {
    "biot":    ("braindecode.models.biot",    "BIOT"),
    "cbramod": ("braindecode.models.cbramod", "CBraMod"),
    "labram":  ("braindecode.models.labram",  "Labram"),
    "luna":    ("braindecode.models.luna",    "LUNA"),
    "bendr":   ("braindecode.models",         "BENDR"),
    "reve":    ("braindecode.models.reve",    "REVE"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--n_chans", type=int, default=64)
ap.add_argument("--n_times", type=int, default=1000)
ap.add_argument("--batch",   type=int, default=2)
ap.add_argument("--out",     default="/tmp/fm_deep.json")
args = ap.parse_args()

mod_path, cls_name = MODELS[args.model]
ModelCls = getattr(importlib.import_module(mod_path), cls_name)

chs_info = [{"ch_name": n, "loc": [0.0]*12, "kind": 2} for n in STD_1020[:args.n_chans]]

# Try a few constructor patterns
model = None
errs = []
for kw in [
    dict(n_outputs=1, chs_info=chs_info, n_times=args.n_times),
    dict(n_outputs=1, n_chans=args.n_chans, n_times=args.n_times),
    dict(n_outputs=1, chs_info=chs_info, n_times=args.n_times, sfreq=200),
]:
    try:
        model = ModelCls(**kw)
        print(f"built with kw keys: {sorted(kw)}")
        break
    except Exception as e:
        errs.append(f"{kw.keys()} -> {type(e).__name__}: {str(e)[:120]}")
if model is None:
    raise SystemExit("Cannot build " + cls_name + ":\n" + "\n".join(errs))

fwd_params = list(inspect.signature(model.forward).parameters)
input_param = fwd_params[0]
dummy = {input_param: torch.randn(args.batch, args.n_chans, args.n_times)}
if "ch_names" in fwd_params:
    dummy["ch_names"] = STD_1020[:args.n_chans]

# Warmup forward to initialize any lazy params (e.g. cbramod's LazyLinear)
with torch.no_grad():
    try:
        model.eval(); model(**dummy); model.train()
    except Exception as e:
        print(f"warmup failed: {type(e).__name__}: {str(e)[:200]}")

n_params = sum(p.numel() for p in model.parameters())
print(f"{args.model} built. {n_params:,} parameters.")

probeable = [(n, m) for n, m in model.named_modules() if n]
print(f"Probeable named submodules: {len(probeable)}")

results = []
for name, mod in probeable:
    rec = {"name": name, "class": type(mod).__name__,
           "depth": name.count("."), "top": name.split(".")[0]}
    try:
        wrapper = DownstreamWrapper(
            probe_layer=name, aggregation="flatten", probe_config="linear",
            model_output_key=None,
        ).build(model, dummy, 3)
        out = wrapper(**dummy)
        if out.shape == (args.batch, 3):
            rec["status"] = "ok"
            rec["probe_in_features"] = int(wrapper.probe.in_features)
        else:
            rec["status"] = "fail"; rec["message"] = f"bad shape {tuple(out.shape)}"
    except Exception as e:
        rec["status"] = "fail"; rec["message"] = f"{type(e).__name__}: {str(e)[:140]}"
    results.append(rec)

ok = sum(1 for r in results if r["status"] == "ok")
print(f"OK: {ok}/{len(probeable)}")

from pathlib import Path
Path(args.out).write_text(json.dumps({"model": args.model, "n_params": n_params,
    "n_chans": args.n_chans, "n_times": args.n_times, "results": results}, indent=2))
print(f"saved {args.out}")
