#!/usr/bin/env python3
"""Extract candidate figure crops from a paper PDF.

For every "Figure N" / "Fig. N" / "Figura N" caption found in the first pages,
crops the region between the caption and the body text above it (figures sit
above their caption in almost every venue style), renders it as PNG, and
writes a manifest.json describing each candidate. Also renders page 1 as a
fallback candidate.

Usage:
    python3 extract_figures.py paper.pdf outdir [max_pages]
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

CAPTION_RE = re.compile(
    r"^\s*(?:Fig(?:ure|ura)?|FIG(?:URE|URA)?)\.?\s*(\d+)\s*[.:|]?", re.UNICODE
)
ZOOM = 2.2
MIN_REGION_H = 50  # pt; smaller crops are caption-only noise
MIN_BODY_CHARS = 40  # blocks shorter than this are stray labels, not body text


def column_bounds(page, caption_box):
    """X-range to crop: full text width for wide captions, else the caption's
    own column (caption blocks span their column width in 2-col layouts)."""
    pw = page.rect.width
    text_x0 = min(
        (b[0] for b in page.get_text("blocks") if b[6] == 0), default=36
    )
    text_x1 = max(
        (b[2] for b in page.get_text("blocks") if b[6] == 0), default=pw - 36
    )
    cap_w = caption_box.x1 - caption_box.x0
    if cap_w > 0.55 * (text_x1 - text_x0):
        return max(text_x0 - 6, 0), min(text_x1 + 6, pw)
    return max(caption_box.x0 - 8, 0), min(caption_box.x1 + 8, pw)


def crop_above(page, caption_box, x0, x1):
    """Bottom of the lowest body-text block above the caption in this column."""
    top = page.rect.y0 + 18
    for b in page.get_text("blocks"):
        bx0, by0, bx1, by1, text, _, btype = b[:7]
        if btype != 0 or by1 > caption_box.y0 + 2:
            continue
        if len(text.strip()) < MIN_BODY_CHARS:
            continue
        if CAPTION_RE.match(text.strip()):
            continue
        overlap = min(bx1, x1) - max(bx0, x0)
        if overlap < 0.3 * (x1 - x0):
            continue
        top = max(top, by1 + 4)
    return fitz.Rect(x0, top, x1, caption_box.y0 - 2)


def crop_below(page, caption_box, x0, x1):
    """Fallback for caption-above-figure styles: caption bottom to next text."""
    bottom = page.rect.y1 - 18
    for b in page.get_text("blocks"):
        bx0, by0, bx1, by1, text, _, btype = b[:7]
        if btype != 0 or by0 < caption_box.y1 - 2:
            continue
        if len(text.strip()) < MIN_BODY_CHARS:
            continue
        overlap = min(bx1, x1) - max(bx0, x0)
        if overlap < 0.3 * (x1 - x0):
            continue
        bottom = min(bottom, by0 - 4)
    return fitz.Rect(x0, caption_box.y1 + 2, x1, bottom)


def main():
    pdf_path = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    outdir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    manifest = []
    mat = fitz.Matrix(ZOOM, ZOOM)

    for pno in range(min(len(doc), max_pages)):
        page = doc[pno]
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            text = b[4].strip()
            m = CAPTION_RE.match(text)
            if not m:
                continue
            cap_box = fitz.Rect(b[:4])
            x0, x1 = column_bounds(page, cap_box)
            region = crop_above(page, cap_box, x0, x1)
            placement = "above"
            if region.height < MIN_REGION_H:
                region = crop_below(page, cap_box, x0, x1)
                placement = "below"
            if region.height < MIN_REGION_H or region.width < 80:
                continue
            name = f"cand_p{pno + 1}_fig{m.group(1)}_{placement}.png"
            pix = page.get_pixmap(matrix=mat, clip=region)
            pix.save(outdir / name)
            manifest.append(
                {
                    "file": name,
                    "page": pno + 1,
                    "figure": int(m.group(1)),
                    "placement": placement,
                    "caption": text[:140].replace("\n", " "),
                    "px_w": pix.width,
                    "px_h": pix.height,
                }
            )

    # Fallback: full first page
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.6, 1.6))
    pix.save(outdir / "page1.png")
    manifest.append(
        {
            "file": "page1.png",
            "page": 1,
            "figure": 0,
            "placement": "fullpage",
            "caption": "(full first page fallback)",
            "px_w": pix.width,
            "px_h": pix.height,
        }
    )

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"pdf": str(pdf_path), "candidates": len(manifest)}))


if __name__ == "__main__":
    main()
