"""Per-FM depth analysis charts — designed for the dashboard.

Design principles applied (Cairo / Schwabish / Okabe-Ito):
  - Title = assertion (states the finding); chart = evidence
  - Direct labels on key data points (no legend lookup for the lead story)
  - Okabe-Ito colorblind-safe palette per FM
  - Position dominates; color is secondary
  - Annotations replace tooltips in static figures
  - Stage brackets where the architecture has natural blocks
  - Tasks ordered by mean AUROC to reveal patterns (not alphabetical)
"""
from __future__ import annotations
import json, math, re, statistics, sys
from collections import defaultdict
from pathlib import Path

# ---- depth assignment per FM ----------------------------------------------

def _bendr_depth(name):
    m = re.match(r"encoder\.encoder\.Encoder_(\d+)$", name or "")
    if m: return int(m.group(1))
    m = re.match(r"contextualizer\.input_conditioning\.(\d+)$", name or "")
    if m: return 6 + int(m.group(1))
    if name == "contextualizer.relative_position.0": return 10
    m = re.match(r"contextualizer\.transformer_layers\.(\d+)$", name or "")
    if m: return 11 + int(m.group(1))
    if name is None: return 19
    return None

def _biot_depth(name):
    if name == "encoder.patch_embedding": return 0
    m = re.match(r"encoder\.transformer\.layers\.layers\.(\d+)\.([01])$", name or "")
    if m: return 1 + int(m.group(1)) * 2 + int(m.group(2))  # 1..8
    if name == "encoder": return 9
    if name is None: return 10
    return None

def _cbramod_depth(name):
    if name == "rearrange": return 0
    if name == "patch_embedding": return 1
    m = re.match(r"encoder\.layers\.(\d+)$", name or "")
    if m: return 2 + int(m.group(1))
    if name is None: return 14
    return None

def _labram_depth(name):
    if name == "model.patch_embed": return 0
    m = re.match(r"model\.blocks\.(\d+)$", name or "")
    if m: return 1 + int(m.group(1))
    if name is None: return 12
    return None

def _luna_depth(name):
    if name == "model.channel_location_embedder": return 0
    if name == "model.patch_embed": return 1
    m = re.match(r"model\.blocks\.(\d+)$", name or "")
    if m: return 2 + int(m.group(1))
    if name is None: return 10
    return None

def _reve_depth(name):
    if name == "model.to_patch_embedding.0": return 0
    m = re.match(r"model\.transformer\.layers\.(\d+)\.1$", name or "")
    if m: return 1 + int(m.group(1))
    if name == "model.mlp4d": return 30
    if name == "model.ln": return 31
    if name is None: return 32
    return None

DEPTH = {
    "bendr": _bendr_depth, "biot": _biot_depth, "cbramod": _cbramod_depth,
    "labram": _labram_depth, "luna": _luna_depth, "reve": _reve_depth,
}


def _layer_sort_key(k):
    """Stable sort for (depth, probe_layer) keys; pushes None probe_layer last."""
    depth, name = k
    return (depth, 1 if name is None else 0, name or "")

# ---- visual constants -----------------------------------------------------

# Okabe-Ito colorblind-safe palette assigned per FM
PAL = {
    "bendr":   "#0072B2",  # blue
    "biot":    "#009E73",  # bluish-green
    "cbramod": "#D55E00",  # vermillion
    "labram":  "#CC79A7",  # reddish purple
    "luna":    "#E69F00",  # orange
    "reve":    "#56B4E9",  # sky blue
}

FM_LABEL = {"bendr":"BENDR","biot":"BIOT","cbramod":"CBraMod","labram":"LaBraM","luna":"LUNA","reve":"REVE"}
TASK_LABEL = {"cvep":"c-VEP","ern":"ERN","mental_arithmetic":"M. arith.","mental_workload":"M. workld.",
              "motor_execution":"M. exec.","motor_imagery":"M. imag.","n2pc":"N2pc","p3":"P300","ssvep":"SSVEP"}
TASK_ORDER_DEFAULT = ["cvep","ern","mental_arithmetic","mental_workload","motor_execution","motor_imagery","n2pc","p3","ssvep"]
FM_ORDER = ["bendr","biot","cbramod","labram","luna","reve"]

# Architecture stage brackets — natural groupings for the depth axis
STAGES = {
    "bendr": [
        ("encoder", 0, 5),
        ("ctx in-cond", 6, 8),
        ("ctx rel-pos", 10, 10),
        ("ctx trf", 11, 18),
        ("output", 19, 19),
    ],
    "biot": [
        ("patch emb", 0, 0),
        ("transformer blocks", 1, 8),
        ("pooled encoder", 9, 9),
        ("output", 10, 10),
    ],
    "cbramod": [
        ("input", 0, 1),
        ("encoder layers", 2, 13),
        ("output", 14, 14),
    ],
    "labram": [
        ("patch emb", 0, 0),
        ("blocks", 1, 11),
        ("output", 12, 12),
    ],
    "luna": [
        ("chan + patch", 0, 1),
        ("blocks", 2, 9),
        ("output", 10, 10),
    ],
    "reve": [
        ("patch emb", 0, 0),
        ("transformer", 1, 21),
        ("norm/mlp", 30, 31),
        ("output", 32, 32),
    ],
}


def short_label(name):
    if name is None: return "null"
    parts = name.split(".")
    if "Encoder_" in name: return parts[-1].replace("Encoder_", "Enc.")
    if "transformer_layers" in name: return f"ctx-tl.{parts[-1]}"
    if "input_conditioning" in name: return f"ctx-ic.{parts[-1]}"
    if "relative_position" in name: return "ctx-rp"
    if "transformer.layers.layers" in name: return f"{parts[-2]}.{parts[-1]}"
    if name == "encoder": return "encoder*"
    if name == "encoder.patch_embedding": return "patch"
    m = re.match(r"model\.transformer\.layers\.(\d+)\.1$", name)
    if m: return f"trf.{m.group(1)}"
    if name == "model.mlp4d": return "mlp4d"
    if name == "model.ln": return "ln"
    if name in ("model.to_patch_embedding.0", "model.patch_embed", "patch_embedding"): return "patch"
    if name == "model.channel_location_embedder": return "ch-emb"
    m = re.match(r"model\.blocks\.(\d+)", name)
    if m: return f"blk.{m.group(1)}"
    m = re.match(r"encoder\.layers\.(\d+)", name)
    if m: return f"enc.{m.group(1)}"
    if name == "rearrange": return "rearr"
    return parts[-1][:8]

# ---------------------------------------------------------------------------
# Story 1: Per-FM mean depth curve (one panel per FM, 6 total in 3×2 grid)
# ---------------------------------------------------------------------------

def _panel_per_fm(rows, fm, width=480, height=300):
    """One panel: AUROC ± SD across tasks at each probe depth.

    Title states the FM's finding. Best layer is annotated directly.
    Stage brackets sit under the x-axis.
    """
    by_layer = defaultdict(list)  # (depth, name) -> [(task, auroc), ...]
    for r in rows:
        if r["fm"] != fm or r["test/auroc"] is None: continue
        d = DEPTH[fm](r["probe_layer"])
        if d is None: continue
        by_layer[(d, r["probe_layer"])].append((r["task"], r["test/auroc"]))

    if not by_layer:
        return f"<!-- no data for {fm} -->"

    items = sorted(by_layer.keys(), key=_layer_sort_key)
    depths = [d for d, _ in items]
    means = []
    stds = []
    n_tasks = []
    labels = []
    for k in items:
        vals = [a for _, a in by_layer[k]]
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        n_tasks.append(len(vals))
        labels.append(short_label(k[1]))

    n = len(means)
    # Map depths to x positions linearly within the FM's depth range
    d_min, d_max = min(depths), max(depths)
    def x_of_d(d):
        if d_max == d_min: return 0.5
        return (d - d_min) / (d_max - d_min)

    margin = {"top": 56, "right": 16, "bottom": 90, "left": 46}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    y_min, y_max = 0.45, 0.95
    def y_of(v): return margin["top"] + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    best_i = max(range(n), key=lambda i: means[i])
    best_mean = means[best_i]
    worst_mean = min(means)
    spread = best_mean - worst_mean
    sd_at_best = stds[best_i]

    # Headline: short finding
    if spread < 0.04:
        finding = "Flat depth profile — probe layer barely matters."
    elif best_i == 0:
        finding = f"Best at the input: {labels[best_i]} → {best_mean:.2f}."
    elif best_i >= n - 1:
        finding = f"Best at the output: {labels[best_i]} → {best_mean:.2f}."
    else:
        finding = f"Best mid-depth: {labels[best_i]} → {best_mean:.2f}."

    parts = []
    parts.append(f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="DM Sans, Helvetica, sans-serif">')
    # Title block (assertion + sub)
    parts.append(f'<text x="{margin["left"]}" y="22" font-size="15" font-weight="700" fill="#1f232b">{FM_LABEL[fm]}</text>')
    parts.append(f'<text x="{margin["left"]}" y="40" font-size="12" fill="#353a44">{finding}</text>')

    # ----- axes -----
    # Y ticks
    for v in (0.5, 0.6, 0.7, 0.8, 0.9):
        y = y_of(v)
        parts.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{margin["left"] + plot_w}" y2="{y}" stroke="#eaecef" stroke-width="0.6"/>')
        parts.append(f'<text x="{margin["left"] - 5}" y="{y + 3}" text-anchor="end" font-size="10" fill="#7c828a">{v:.1f}</text>')
    # Chance line (slightly bolder)
    parts.append(f'<line x1="{margin["left"]}" y1="{y_of(0.5)}" x2="{margin["left"] + plot_w}" y2="{y_of(0.5)}" stroke="#9aa0a8" stroke-width="0.8" stroke-dasharray="3,3"/>')
    # Y axis label
    parts.append(f'<text x="14" y="{margin["top"] + plot_h / 2}" font-size="10.5" fill="#353a44" transform="rotate(-90 14 {margin["top"] + plot_h / 2})" text-anchor="middle">AUROC, mean ± SD across {max(n_tasks)} tasks</text>')

    # ----- shaded ±SD band -----
    band_top = []
    band_bot = []
    for i in range(n):
        x = margin["left"] + x_of_d(depths[i]) * plot_w
        band_top.append((x, y_of(min(means[i] + stds[i], 0.99))))
        band_bot.append((x, y_of(max(means[i] - stds[i], 0.0))))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in band_top) + " " + \
           " ".join(f"{x:.1f},{y:.1f}" for x, y in reversed(band_bot))
    parts.append(f'<polygon points="{poly}" fill="{PAL[fm]}" opacity="0.16"/>')

    # ----- mean curve -----
    line = " ".join(f"{'M' if i == 0 else 'L'}{margin['left'] + x_of_d(depths[i])*plot_w:.1f},{y_of(means[i]):.1f}" for i in range(n))
    parts.append(f'<path d="{line}" fill="none" stroke="{PAL[fm]}" stroke-width="2"/>')

    # ----- points -----
    for i in range(n):
        x = margin["left"] + x_of_d(depths[i]) * plot_w
        parts.append(f'<circle cx="{x:.1f}" cy="{y_of(means[i]):.1f}" r="2.5" fill="{PAL[fm]}"/>')

    # ----- highlight best with halo + label -----
    xb = margin["left"] + x_of_d(depths[best_i]) * plot_w
    yb = y_of(best_mean)
    parts.append(f'<circle cx="{xb:.1f}" cy="{yb:.1f}" r="6" fill="{PAL[fm]}" opacity="0.25"/>')
    parts.append(f'<circle cx="{xb:.1f}" cy="{yb:.1f}" r="3.5" fill="{PAL[fm]}"/>')
    # Annotate the best layer name + mean above the point (with collision check at edges)
    lab_y = yb - 12
    if lab_y < margin["top"] + 4: lab_y = yb + 16
    anchor = "middle"
    lab_x = xb
    if best_i == 0:
        anchor = "start"; lab_x = xb + 4
    elif best_i == n - 1:
        anchor = "end"; lab_x = xb - 4
    parts.append(f'<text x="{lab_x:.1f}" y="{lab_y:.1f}" text-anchor="{anchor}" font-size="11" font-weight="600" fill="#1f232b" font-family="IBM Plex Mono, monospace">{labels[best_i]}</text>')

    # ----- stage brackets under x-axis -----
    by = margin["top"] + plot_h + 8
    for stg_label, d_start, d_end in STAGES.get(fm, []):
        if d_start < d_min: d_start = d_min
        if d_end > d_max: d_end = d_max
        if d_end < d_min or d_start > d_max: continue
        x0 = margin["left"] + x_of_d(d_start) * plot_w
        x1 = margin["left"] + x_of_d(d_end) * plot_w
        if x1 - x0 < 5: x1 = x0 + 5
        parts.append(f'<line x1="{x0:.1f}" y1="{by:.1f}" x2="{x1:.1f}" y2="{by:.1f}" stroke="#5b616b" stroke-width="1.2"/>')
        parts.append(f'<line x1="{x0:.1f}" y1="{by - 3:.1f}" x2="{x0:.1f}" y2="{by + 3:.1f}" stroke="#5b616b" stroke-width="1.2"/>')
        parts.append(f'<line x1="{x1:.1f}" y1="{by - 3:.1f}" x2="{x1:.1f}" y2="{by + 3:.1f}" stroke="#5b616b" stroke-width="1.2"/>')
        parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{by + 18:.1f}" text-anchor="middle" font-size="10.5" font-weight="500" fill="#353a44" letter-spacing="0.02em">{stg_label}</text>')

    # ----- minor x ticks at every probe (light tick marks above stage line) -----
    for d in depths:
        x = margin["left"] + x_of_d(d) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{by - 8:.1f}" x2="{x:.1f}" y2="{by - 5:.1f}" stroke="#bcc1c8" stroke-width="0.6"/>')

    parts.append('</svg>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Story 2: Master view — all 6 FMs on one normalised-depth axis
# ---------------------------------------------------------------------------

def master_overlay(rows, width=920, height=420):
    """All six FMs on a shared 0..1 depth axis with their mean curves.

    Direct labels per FM. The point of this chart is the ranking by AUROC
    AND the depth where each FM peaks — both visible at a glance.
    """
    by_fm = {}
    best_per_fm = {}
    for fm in FM_ORDER:
        layer_pts = defaultdict(list)
        for r in rows:
            if r["fm"] != fm or r["test/auroc"] is None: continue
            d = DEPTH[fm](r["probe_layer"])
            if d is None: continue
            layer_pts[(d, r["probe_layer"])].append(r["test/auroc"])
        if not layer_pts:
            continue
        items = sorted(layer_pts.keys(), key=_layer_sort_key)
        depths = [d for d, _ in items]
        d_min, d_max = min(depths), max(depths)
        means = [statistics.mean(layer_pts[k]) for k in items]
        curve = []
        for k, m in zip(items, means):
            d = k[0]
            frac = (d - d_min) / (d_max - d_min) if d_max > d_min else 0.5
            curve.append((frac, m, k[1]))
        by_fm[fm] = curve
        bi = max(range(len(means)), key=lambda i: means[i])
        best_per_fm[fm] = (curve[bi][0], means[bi], curve[bi][2])

    margin = {"top": 70, "right": 220, "bottom": 70, "left": 64}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    y_min, y_max = 0.45, 0.85
    def y_of(v): return margin["top"] + plot_h - (v - y_min) / (y_max - y_min) * plot_h
    def x_of(f): return margin["left"] + f * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="DM Sans, Helvetica, sans-serif">']
    parts.append(f'<text x="{margin["left"]}" y="28" font-size="20" font-weight="700" fill="#1f232b" font-family="DM Serif Display, serif">REVE peaks deepest; BENDR\'s best is at the input</text>')
    parts.append(f'<text x="{margin["left"]}" y="48" font-size="12" fill="#353a44">Mean AUROC across the 9 NeuralBench-EEG-Core tasks, plotted on each model\'s normalised depth axis (0 = input, 1 = output).</text>')

    # Y ticks
    for v in (0.5, 0.6, 0.7, 0.8):
        y = y_of(v)
        parts.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{margin["left"] + plot_w}" y2="{y}" stroke="#eaecef" stroke-width="0.6"/>')
        parts.append(f'<text x="{margin["left"] - 8}" y="{y + 3}" text-anchor="end" font-size="11" fill="#7c828a">{v:.1f}</text>')
    parts.append(f'<line x1="{margin["left"]}" y1="{y_of(0.5)}" x2="{margin["left"] + plot_w}" y2="{y_of(0.5)}" stroke="#9aa0a8" stroke-width="0.8" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{margin["left"] + plot_w - 4}" y="{y_of(0.5) - 4}" text-anchor="end" font-size="10.5" fill="#9aa0a8" font-style="italic">chance</text>')

    # X axis label + ticks
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#5b616b"/>')
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_of(f)
        parts.append(f'<line x1="{x:.1f}" y1="{margin["top"] + plot_h}" x2="{x:.1f}" y2="{margin["top"] + plot_h + 4}" stroke="#5b616b"/>')
        parts.append(f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 18}" text-anchor="middle" font-size="11" fill="#5b616b">{f:.2f}</text>')
    parts.append(f'<text x="{x_of(0.5):.1f}" y="{margin["top"] + plot_h + 38}" text-anchor="middle" font-size="12" fill="#353a44">depth fraction within each FM (0 = input → 1 = output)</text>')

    # Y axis label
    parts.append(f'<text x="22" y="{margin["top"] + plot_h / 2}" font-size="12" fill="#353a44" transform="rotate(-90 22 {margin["top"] + plot_h / 2})" text-anchor="middle">mean AUROC across tasks</text>')

    # Each FM curve (sorted by best AUROC so labels stack nicely)
    fm_sorted = sorted(by_fm.keys(), key=lambda f: -best_per_fm[f][1])
    label_y_taken = []  # avoid overlap

    for fm in fm_sorted:
        curve = by_fm[fm]
        # path
        path = " ".join(f"{'M' if i == 0 else 'L'}{x_of(f):.1f},{y_of(m):.1f}" for i, (f, m, _) in enumerate(curve))
        parts.append(f'<path d="{path}" fill="none" stroke="{PAL[fm]}" stroke-width="2.2" opacity="0.95"/>')
        for f, m, _ in curve:
            parts.append(f'<circle cx="{x_of(f):.1f}" cy="{y_of(m):.1f}" r="2.5" fill="{PAL[fm]}"/>')

        # callout: dot + label at the BEST point
        bf, bm, bname = best_per_fm[fm]
        bx, by_ = x_of(bf), y_of(bm)
        parts.append(f'<circle cx="{bx:.1f}" cy="{by_:.1f}" r="6.5" fill="none" stroke="{PAL[fm]}" stroke-width="1.6"/>')

        # right-side legend block — sorted by AUROC so the strongest is at top
        # we place each FM's label aligned with its best y, then nudge to avoid overlap
        legend_x = margin["left"] + plot_w + 16
        target_y = by_
        for ty in label_y_taken:
            if abs(target_y - ty) < 22: target_y = ty + 22
        label_y_taken.append(target_y)
        # connector
        parts.append(f'<line x1="{bx:.1f}" y1="{by_:.1f}" x2="{legend_x - 6:.1f}" y2="{target_y:.1f}" stroke="{PAL[fm]}" stroke-width="1" opacity="0.5"/>')
        parts.append(f'<circle cx="{legend_x - 4:.1f}" cy="{target_y:.1f}" r="3.5" fill="{PAL[fm]}"/>')
        parts.append(f'<text x="{legend_x + 4:.1f}" y="{target_y - 1:.1f}" font-size="12.5" font-weight="600" fill="#1f232b">{FM_LABEL[fm]}  <tspan fill="#5b616b" font-weight="400">{bm:.3f}</tspan></text>')
        parts.append(f'<text x="{legend_x + 4:.1f}" y="{target_y + 12:.1f}" font-size="10.5" fill="#7c828a" font-family="IBM Plex Mono, monospace">{short_label(bname)}  ·  depth {bf:.2f}</text>')

    parts.append('</svg>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Story 3: Best-layer position per task (strip plot)
# ---------------------------------------------------------------------------

def best_layer_strip(rows, width=920, height=320):
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="DM Sans, Helvetica, sans-serif">']
    parts.append(f'<text x="40" y="28" font-size="20" font-weight="700" fill="#1f232b" font-family="DM Serif Display, serif">REVE & LUNA cluster their best layers tightly; BIOT and LaBraM scatter</text>')
    parts.append(f'<text x="40" y="48" font-size="12" fill="#353a44">For each task, we mark where the best probe layer sits on the FM\'s normalised depth axis. Dot area = AUROC. Tight clusters mean one layer generalises; spread means tuning per task helps.</text>')

    margin = {"top": 80, "right": 60, "bottom": 56, "left": 110}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    row_h = plot_h / len(FM_ORDER)

    # Build per-FM best layer per task
    best = defaultdict(dict)  # fm -> task -> (frac, name, auroc)
    fm_dmax = {}
    for r in rows:
        if r["test/auroc"] is None or r["probe_layer"] is None: continue
        d = DEPTH[r["fm"]](r["probe_layer"])
        if d is None: continue
        # max depth per FM (we'll compute via all rows below)
        fm_dmax[r["fm"]] = max(fm_dmax.get(r["fm"], 0), d)
        cur = best[r["fm"]].get(r["task"])
        if cur is None or r["test/auroc"] > cur[2]:
            best[r["fm"]][r["task"]] = (d, r["probe_layer"], r["test/auroc"])

    # x axis
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#5b616b"/>')
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = margin["left"] + f * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{margin["top"] + plot_h}" x2="{x:.1f}" y2="{margin["top"] + plot_h + 4}" stroke="#5b616b"/>')
        parts.append(f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 18}" text-anchor="middle" font-size="11" fill="#5b616b">{f:.2f}</text>')
    parts.append(f'<text x="{margin["left"] + plot_w / 2}" y="{margin["top"] + plot_h + 38}" text-anchor="middle" font-size="12" fill="#353a44">depth fraction within each FM (0 → 1)</text>')

    # Per-FM row
    for fi, fm in enumerate(FM_ORDER):
        y = margin["top"] + (fi + 0.5) * row_h
        dmax = fm_dmax.get(fm, 1)

        # FM label
        parts.append(f'<text x="{margin["left"] - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="13" font-weight="600" fill="#1f232b">{FM_LABEL[fm]}</text>')
        # baseline
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" stroke="{PAL[fm]}" stroke-opacity="0.18" stroke-width="1"/>')

        # collect best-per-task fractions, compute mean & SD
        fracs = []
        for task, (d, name, au) in best[fm].items():
            frac = d / dmax if dmax > 0 else 0
            fracs.append(frac)
            x = margin["left"] + frac * plot_w
            r_dot = max(3.0, min(9.0, 3 + (au - 0.5) * 18))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r_dot:.1f}" fill="{PAL[fm]}" opacity="0.55"><title>{TASK_LABEL[task]}: AUROC {au:.3f} @ {short_label(name)} (depth {frac:.2f})</title></circle>')

        # cluster summary: mean fraction + SD line behind dots
        if fracs:
            mean_f = statistics.mean(fracs)
            sd_f = statistics.stdev(fracs) if len(fracs) > 1 else 0
            x_lo = margin["left"] + max(0, mean_f - sd_f) * plot_w
            x_hi = margin["left"] + min(1, mean_f + sd_f) * plot_w
            parts.append(f'<line x1="{x_lo:.1f}" y1="{y - 14:.1f}" x2="{x_hi:.1f}" y2="{y - 14:.1f}" stroke="{PAL[fm]}" stroke-width="2" opacity="0.7"/>')
            mx = margin["left"] + mean_f * plot_w
            parts.append(f'<circle cx="{mx:.1f}" cy="{y - 14:.1f}" r="2" fill="{PAL[fm]}"/>')
            # right side: mean ± SD label
            parts.append(f'<text x="{margin["left"] + plot_w + 8:.1f}" y="{y + 4:.1f}" font-size="11" fill="#353a44" font-family="IBM Plex Mono, monospace">{mean_f:.2f}±{sd_f:.2f}</text>')

    # legend strip
    parts.append(f'<text x="{margin["left"]}" y="{margin["top"] - 10:.1f}" font-size="10.5" fill="#7c828a">dot area ∝ AUROC; the small horizontal bar above each row spans mean ± 1 SD of the best-layer position across tasks</text>')
    parts.append('</svg>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Story 4: Per-FM heatmap (layer × task) — for drilling down
# ---------------------------------------------------------------------------

def heatmap_panel(rows, fm, width=560, height=360):
    by_lt = {}
    layer_set = set()
    tasks_present = set()
    for r in rows:
        if r["fm"] != fm or r["test/auroc"] is None: continue
        d = DEPTH[fm](r["probe_layer"])
        if d is None: continue
        layer_set.add((d, r["probe_layer"]))
        tasks_present.add(r["task"])
        by_lt[(d, r["probe_layer"], r["task"])] = r["test/auroc"]

    layers = sorted(layer_set, key=_layer_sort_key)
    if not layers:
        return f"<!-- no data for {fm} -->"

    # Order tasks by FM's mean AUROC (best first) so the pattern emerges
    task_mean = {t: statistics.mean([by_lt[(d, n, t)] for (d, n) in layers if (d, n, t) in by_lt])
                 for t in tasks_present}
    task_order = sorted(tasks_present, key=lambda t: -task_mean[t])

    nT = len(task_order)
    nL = len(layers)
    margin = {"top": 56, "right": 16, "bottom": 84, "left": 86}
    cellw = (width - margin["left"] - margin["right"]) / nT
    cellh = (height - margin["top"] - margin["bottom"]) / nL

    def col(v):
        if v is None: return "#eef0f3"
        t = max(0.0, min(1.0, (v - 0.5) / 0.45))
        r0, g0, b0 = int(PAL[fm][1:3], 16), int(PAL[fm][3:5], 16), int(PAL[fm][5:7], 16)
        mr = int(250 - (250 - r0) * t)
        mg = int(250 - (250 - g0) * t)
        mb = int(250 - (250 - b0) * t)
        return f"rgb({mr},{mg},{mb})"

    # find global best for this FM
    best_v = max((by_lt[(d, n, t)] for (d, n) in layers for t in task_order if (d, n, t) in by_lt), default=0)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="DM Sans, Helvetica, sans-serif">']
    parts.append(f'<text x="{margin["left"]}" y="22" font-size="15" font-weight="700" fill="#1f232b">{FM_LABEL[fm]}</text>')
    parts.append(f'<text x="{margin["left"]}" y="40" font-size="11" fill="#353a44">probe layer (rows) × task (cols, sorted by mean AUROC). Best cell: {best_v:.3f}.</text>')

    for li, (d, lname) in enumerate(layers):
        y = margin["top"] + li * cellh
        for ti, task in enumerate(task_order):
            x = margin["left"] + ti * cellw
            v = by_lt.get((d, lname, task))
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cellw:.1f}" height="{cellh:.1f}" fill="{col(v)}" stroke="#fff" stroke-width="0.5"/>')
            if v is not None and cellw > 30 and cellh > 14:
                txt_color = "#fff" if v > 0.72 else "#1f232b"
                parts.append(f'<text x="{x + cellw/2:.1f}" y="{y + cellh/2 + 3:.1f}" text-anchor="middle" font-size="9.5" fill="{txt_color}" font-variant-numeric="tabular-nums">{v:.2f}</text>')
        # row label
        parts.append(f'<text x="{margin["left"] - 6:.1f}" y="{margin["top"] + li * cellh + cellh / 2 + 3:.1f}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="10" fill="#353a44">{short_label(lname)}</text>')

    # column labels (rotated)
    for ti, task in enumerate(task_order):
        x = margin["left"] + ti * cellw + cellw / 2
        y = margin["top"] + nL * cellh + 12
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="end" transform="rotate(-35 {x:.1f} {y:.1f})" font-size="10.5" fill="#353a44">{TASK_LABEL[task]}</text>')

    parts.append('</svg>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Assemble the section
# ---------------------------------------------------------------------------

def build(rows):
    parts = []
    parts.append('<section id="depth-analysis" class="depth-analysis">')
    parts.append('<div class="da-header">')
    parts.append('<h2>Depth analysis — does the best probe layer generalise?</h2>')
    parts.append('<p class="da-lede">Each foundation model was probed at every layer of its own architecture across all nine NeuralBench-EEG-Core tasks. The four views below answer one question: <em>if I have to pick a probe layer without knowing the downstream task, where should I tap?</em></p>')
    parts.append('</div>')

    # Master overlay
    parts.append('<div class="da-master">')
    parts.append(master_overlay(rows))
    parts.append('</div>')

    # Strip plot
    parts.append('<div class="da-strip">')
    parts.append(best_layer_strip(rows))
    parts.append('</div>')

    # Per-FM panels (3×2)
    parts.append('<h3 class="da-subhead">Per-FM depth profiles</h3>')
    parts.append('<p class="da-sub-lede">AUROC at each probe layer, averaged across the nine tasks. Shaded band: ±1 SD across tasks (narrow = robust choice, wide = task-specific). Stage brackets show the architecture\'s natural blocks.</p>')
    parts.append('<div class="da-panels">')
    for fm in FM_ORDER:
        parts.append(f'<div class="da-panel">{_panel_per_fm(rows, fm)}</div>')
    parts.append('</div>')

    # Heatmaps
    parts.append('<h3 class="da-subhead">Layer × task drilldown</h3>')
    parts.append('<p class="da-sub-lede">Each cell is AUROC for one (probe layer, task) pair. Tasks ordered by FM mean, so the strongest tasks lie on the left of every panel.</p>')
    parts.append('<div class="da-heatmaps">')
    for fm in FM_ORDER:
        parts.append(f'<div class="da-heatmap-card">{heatmap_panel(rows, fm)}</div>')
    parts.append('</div>')
    parts.append('</section>')

    parts.append('''
<style>
  .depth-analysis { margin-top: 90px; padding-top: 36px; border-top: 1px solid var(--rule, rgba(0,0,0,0.1)); }
  .da-header { max-width: 800px; margin-bottom: 28px; }
  .depth-analysis h2 {
    font-family: var(--f-display, 'DM Serif Display', serif);
    font-size: 34px; line-height: 1.15; font-weight: 400;
    color: var(--ink, #1f232b); margin-bottom: 14px;
  }
  .da-lede { font-size: 16px; line-height: 1.6; color: var(--ink-2, #353a44); }
  .da-lede em { color: var(--ink, #1f232b); font-style: italic; }
  .da-subhead {
    font-family: var(--f-display, 'DM Serif Display', serif);
    font-size: 22px; font-weight: 400; margin: 56px 0 8px; color: var(--ink-2, #353a44);
  }
  .da-sub-lede { font-size: 13.5px; line-height: 1.55; color: var(--ink-3, #6a7280); max-width: 760px; margin-bottom: 20px; }
  .da-master, .da-strip { background: #fff; padding: 18px 24px; border: 1px solid var(--rule, rgba(0,0,0,0.08)); border-radius: 6px; margin-bottom: 24px; }
  .da-master svg, .da-strip svg, .da-panel svg, .da-heatmap-card svg { width: 100%; height: auto; display: block; }
  .da-panels { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .da-panel { background: #fff; padding: 8px 12px; border: 1px solid var(--rule, rgba(0,0,0,0.08)); border-radius: 6px; }
  .da-heatmaps { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .da-heatmap-card { background: #fff; padding: 10px 14px; border: 1px solid var(--rule, rgba(0,0,0,0.08)); border-radius: 6px; }
  @media (max-width: 1100px) {
    .da-panels { grid-template-columns: 1fr 1fr; }
    .da-heatmaps { grid-template-columns: 1fr; }
  }
  @media (max-width: 720px) {
    .da-panels { grid-template-columns: 1fr; }
  }
</style>
''')
    return "\n".join(parts)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/all_results.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/depth_section.html"
    rows = json.load(open(src))
    Path(out).write_text(build(rows))
    print(f"wrote {out} ({Path(out).stat().st_size} bytes)")
