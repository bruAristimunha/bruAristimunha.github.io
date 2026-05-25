"""Build Phase 4 visualization HTML fragments from all_results.json.

Generates:
- Per-FM depth chart (6, one per FM) — AUROC vs probe layer ordering, motor_imagery anchor
- FM ranking bar chart — best AUROC across all probes per FM × task
- 9 × 6 task × FM heatmap — best AUROC per (task, FM)
"""
import json, sys, math
from collections import defaultdict
from pathlib import Path

# Okabe-Ito palette (colorblind-safe), assigned per-FM
PAL = {
    "bendr":   "#0072B2",  # blue
    "biot":    "#009E73",  # bluish green
    "cbramod": "#D55E00",  # vermillion
    "labram":  "#CC79A7",  # reddish purple
    "luna":    "#E69F00",  # orange
    "reve":    "#56B4E9",  # sky blue
}

# Display order (paper names)
FM_LABEL = {
    "bendr":   "BENDR",
    "biot":    "BIOT",
    "cbramod": "CBraMod",
    "labram":  "LaBraM",
    "luna":    "LUNA",
    "reve":    "REVE",
}

TASK_LABEL = {
    "cvep": "c-VEP",
    "ern": "ERN",
    "mental_arithmetic": "Mental Arith.",
    "mental_workload": "Mental Workload",
    "motor_execution": "Motor Exec.",
    "motor_imagery": "Motor Imag.",
    "n2pc": "N2pc",
    "p3": "P300",
    "ssvep": "SSVEP",
}

TASK_ORDER = ["cvep", "ern", "mental_arithmetic", "mental_workload",
              "motor_execution", "motor_imagery", "n2pc", "p3", "ssvep"]
FM_ORDER = ["bendr", "biot", "cbramod", "labram", "luna", "reve"]


def load(path):
    rows = json.loads(Path(path).read_text())
    return rows


def best_per_fm_task(rows):
    """Return dict[(fm, task)] -> dict with best_auroc + best_layer."""
    by = defaultdict(list)
    for r in rows:
        if r.get("test/auroc") is None:
            continue
        by[(r["fm"], r["task"])].append(r)
    best = {}
    for k, rs in by.items():
        # exclude null baseline from "best probe layer" if probes exist
        probes = [r for r in rs if r["probe_layer"] not in (None, "null")]
        candidates = probes if probes else rs
        best_row = max(candidates, key=lambda r: r["test/auroc"])
        null_row = next((r for r in rs if r["probe_layer"] in (None, "null")), None)
        best[k] = {
            "best_auroc": best_row["test/auroc"],
            "best_layer": best_row["probe_layer"],
            "best_std": best_row.get("auroc_std", 0.0),
            "null_auroc": null_row["test/auroc"] if null_row else None,
        }
    return best


def heatmap_svg(best, width=560, height=320):
    """9-task × 6-FM heat matrix of best AUROC."""
    nT = len(TASK_ORDER)
    nF = len(FM_ORDER)
    margin = {"top": 30, "right": 20, "bottom": 90, "left": 110}
    cellw = (width - margin["left"] - margin["right"]) / nT
    cellh = (height - margin["top"] - margin["bottom"]) / nF

    def col(v):
        if v is None:
            return "#e8e8e8"
        # value 0.5 (chance) → light grey, 0.5-0.7 → light blue, 0.7-0.9 → dark blue
        t = max(0.0, min(1.0, (v - 0.50) / 0.40))  # clamp 0.5..0.9
        r = int(248 - t * 220)
        g = int(248 - t * 130)
        b = int(248 - t * 30)
        return f"rgb({r},{g},{b})"

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="DM Sans, Helvetica, sans-serif" font-size="11">']
    # cells
    for ti, t in enumerate(TASK_ORDER):
        x = margin["left"] + ti * cellw
        for fi, fm in enumerate(FM_ORDER):
            y = margin["top"] + fi * cellh
            v = best.get((fm, t), {}).get("best_auroc")
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cellw:.1f}" height="{cellh:.1f}" fill="{col(v)}" stroke="#fff" stroke-width="1"/>')
            if v is not None:
                txt_color = "#fff" if v > 0.7 else "#1f232b"
                parts.append(f'<text x="{x + cellw/2:.1f}" y="{y + cellh/2 + 3:.1f}" text-anchor="middle" fill="{txt_color}" font-weight="500">{v:.2f}</text>')
    # task labels (bottom, rotated)
    for ti, t in enumerate(TASK_ORDER):
        x = margin["left"] + ti * cellw + cellw / 2
        y = margin["top"] + nF * cellh + 12
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="end" transform="rotate(-35 {x:.1f} {y:.1f})" fill="#353a44">{TASK_LABEL[t]}</text>')
    # FM labels (left)
    for fi, fm in enumerate(FM_ORDER):
        y = margin["top"] + fi * cellh + cellh / 2 + 4
        parts.append(f'<text x="{margin["left"] - 8:.1f}" y="{y:.1f}" text-anchor="end" fill="#353a44">{FM_LABEL[fm]}</text>')
    # title
    parts.append(f'<text x="{margin["left"]}" y="18" font-weight="700" fill="#1f232b">Best AUROC per FM × task (linear probe at best intermediate layer)</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def fm_ranking_svg(best, width=560, height=280):
    """Bar chart: mean best-AUROC per FM averaged across tasks."""
    per_fm = defaultdict(list)
    for (fm, _), v in best.items():
        if v["best_auroc"] is not None:
            per_fm[fm].append(v["best_auroc"])
    means = {fm: sum(vs) / len(vs) for fm, vs in per_fm.items() if vs}
    order = sorted(means.keys(), key=lambda f: -means[f])

    margin = {"top": 30, "right": 20, "bottom": 30, "left": 80}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    barh = plot_h / max(1, len(order)) * 0.7
    bargap = plot_h / max(1, len(order)) * 0.3

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="DM Sans, Helvetica, sans-serif" font-size="12">']
    parts.append(f'<text x="{margin["left"]}" y="18" font-weight="700" fill="#1f232b">FM ranking — mean best-AUROC across {len(per_fm.get(order[0], []))} task(s)</text>')
    # x axis at AUROC = 0.5..0.9
    x0 = margin["left"]
    x1 = margin["left"] + plot_w
    def xv(v):
        return x0 + (v - 0.5) / 0.4 * plot_w
    # axis line
    parts.append(f'<line x1="{x0}" y1="{margin["top"] + plot_h}" x2="{x1}" y2="{margin["top"] + plot_h}" stroke="#9aa0a8" stroke-width="1"/>')
    for v in (0.5, 0.6, 0.7, 0.8, 0.9):
        x = xv(v)
        parts.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"] + plot_h}" stroke="#e0e0dd" stroke-width="0.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 16}" text-anchor="middle" fill="#6a7280" font-size="11">{v:.1f}</text>')
    # chance line
    cx = xv(0.5)
    parts.append(f'<line x1="{cx:.1f}" y1="{margin["top"]}" x2="{cx:.1f}" y2="{margin["top"] + plot_h}" stroke="#9aa0a8" stroke-width="1" stroke-dasharray="3,3"/>')
    # bars
    for i, fm in enumerate(order):
        y = margin["top"] + i * (barh + bargap)
        mean = means[fm]
        w = xv(mean) - x0
        parts.append(f'<rect x="{x0}" y="{y:.1f}" width="{w:.1f}" height="{barh:.1f}" fill="{PAL[fm]}" opacity="0.85"/>')
        parts.append(f'<text x="{x0 - 6:.1f}" y="{y + barh/2 + 4:.1f}" text-anchor="end" fill="#1f232b" font-weight="500">{FM_LABEL[fm]}</text>')
        parts.append(f'<text x="{x0 + w + 6:.1f}" y="{y + barh/2 + 4:.1f}" fill="#1f232b" font-weight="500">{mean:.3f}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def depth_chart_svg(rows, fm, width=560, height=200):
    """AUROC vs probe layer ordering for a single FM (motor_imagery anchor)."""
    fm_rows = [r for r in rows if r["fm"] == fm and r["task"] == "motor_imagery" and r["dataset"] == "tangermann2012"]
    if not fm_rows:
        return f"<!-- no data for {fm} motor_imagery -->"
    fm_rows.sort(key=lambda r: (r["probe_layer"] is None, r["probe_layer"] or ""))

    margin = {"top": 30, "right": 18, "bottom": 80, "left": 50}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    n = len(fm_rows)
    step = plot_w / max(1, n - 1) if n > 1 else 0

    y_min, y_max = 0.45, 0.90

    def yv(v):
        return margin["top"] + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="DM Sans, Helvetica, sans-serif" font-size="11">']
    parts.append(f'<text x="{margin["left"]}" y="18" font-weight="700" fill="#1f232b">{FM_LABEL[fm]} — AUROC by probe layer (motor_imagery / Tangermann 2012)</text>')

    # chance + axis
    parts.append(f'<line x1="{margin["left"]}" y1="{yv(0.5)}" x2="{margin["left"] + plot_w}" y2="{yv(0.5)}" stroke="#9aa0a8" stroke-width="1" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{margin["left"] - 5}" y="{yv(0.5) + 4}" text-anchor="end" fill="#9aa0a8" font-size="10">chance</text>')
    for v in (0.5, 0.6, 0.7, 0.8):
        y = yv(v)
        parts.append(f'<text x="{margin["left"] - 5}" y="{y + 4}" text-anchor="end" fill="#6a7280" font-size="10">{v:.1f}</text>')

    # points + line
    pts = []
    for i, r in enumerate(fm_rows):
        if r["test/auroc"] is None:
            continue
        x = margin["left"] + i * step
        y = yv(r["test/auroc"])
        std = r.get("auroc_std", 0) or 0
        # error band
        ytop = yv(r["test/auroc"] + std)
        ybot = yv(r["test/auroc"] - std)
        parts.append(f'<line x1="{x:.1f}" y1="{ytop:.1f}" x2="{x:.1f}" y2="{ybot:.1f}" stroke="{PAL[fm]}" stroke-width="2" opacity="0.4"/>')
        pts.append((x, y))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{PAL[fm]}"/>')
    if len(pts) > 1:
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="{PAL[fm]}" stroke-width="1.5" opacity="0.7"/>')

    # x labels — short
    for i, r in enumerate(fm_rows):
        x = margin["left"] + i * step
        label = (r["probe_layer"] or "null").split(".")[-1][:14]
        y = margin["top"] + plot_h + 12
        parts.append(f'<text x="{x:.1f}" y="{y}" text-anchor="end" transform="rotate(-45 {x:.1f} {y})" fill="#353a44" font-size="10" font-family="IBM Plex Mono, monospace">{label}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def build_section(rows):
    best = best_per_fm_task(rows)
    parts = ['<section class="phase4"><h2>NeuralBench-EEG-Core — Cross-task evaluation</h2>']
    parts.append('<p class="lede">Best linear-probe AUROC per (FM, task). Each FM uses its top-3 intermediate probe layers + null baseline × 3 seeds.</p>')
    parts.append('<div class="heatmap">' + heatmap_svg(best) + '</div>')
    parts.append('<div class="ranking">' + fm_ranking_svg(best) + '</div>')
    parts.append('<div class="depths-grid">')
    for fm in FM_ORDER:
        parts.append(f'<div class="depth depth-{fm}">' + depth_chart_svg(rows, fm) + '</div>')
    parts.append('</div></section>')
    return "\n".join(parts)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/all_results.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/phase4_section.html"
    rows = load(src)
    Path(out).write_text(build_section(rows))
    print(f"wrote {out} ({Path(out).stat().st_size} bytes)")
