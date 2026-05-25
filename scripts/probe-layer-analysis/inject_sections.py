"""Inject Phase 4 + depth-analysis sections before </body> of the dashboard.

Uses sentinel comments (<!-- /phase4-injection -->, <!-- /depth-injection -->)
so re-running the script reliably replaces only the previously-injected blocks
instead of accumulating duplicates.
"""
from __future__ import annotations
import re
from pathlib import Path

DASHBOARD = Path("/Users/bruaristimunha/Projects/bruAristimunha.github.io/probe-layer-coverage.html")
WRAP_TPL  = Path("/Users/bruaristimunha/Projects/bruAristimunha.github.io/scripts/probe-layer-analysis/phase4_wrap.html")
PHASE4_HTML = Path("/tmp/phase4_section.html")
DEPTH_HTML  = Path("/tmp/depth_section.html")

PHASE4_BEGIN = "<!-- /phase4-injection-begin -->"
PHASE4_END   = "<!-- /phase4-injection-end -->"
DEPTH_BEGIN  = "<!-- /depth-injection-begin -->"
DEPTH_END    = "<!-- /depth-injection-end -->"


def extract_part(src: str, opening_tag_class: str, closing: str) -> str:
    """Pull a <div class="..."> block out of the phase4 source. Returns inner HTML."""
    pattern = rf'<div class="{re.escape(opening_tag_class)}">(.*?){closing}'
    m = re.search(pattern, src, re.DOTALL)
    if not m:
        return f"<!-- {opening_tag_class}: not found -->"
    return m.group(1)


def build_phase4_block() -> str:
    if not PHASE4_HTML.exists():
        return ""
    src = PHASE4_HTML.read_text()
    wrap = WRAP_TPL.read_text()
    heatmap = extract_part(src, "heatmap", "</div>")
    ranking = extract_part(src, "ranking", "</div>")
    depths  = extract_part(src, "depths-grid", "</div></section>")
    wrap = (wrap
        .replace("PHASE4_HEATMAP", heatmap)
        .replace("PHASE4_RANKING", ranking)
        .replace("PHASE4_DEPTHS", depths))
    return f"{PHASE4_BEGIN}\n{wrap}\n{PHASE4_END}\n"


def build_depth_block() -> str:
    if not DEPTH_HTML.exists():
        return ""
    return f"{DEPTH_BEGIN}\n{DEPTH_HTML.read_text()}\n{DEPTH_END}\n"


def strip_old_block(html: str, begin: str, end: str) -> str:
    pattern = rf'{re.escape(begin)}.*?{re.escape(end)}\s*'
    return re.sub(pattern, "", html, flags=re.DOTALL)


def main():
    html = DASHBOARD.read_text()

    phase4 = build_phase4_block()
    depth  = build_depth_block()

    html = strip_old_block(html, PHASE4_BEGIN, PHASE4_END)
    html = strip_old_block(html, DEPTH_BEGIN, DEPTH_END)

    # Strip any legacy injection that pre-dates the sentinels (best-effort)
    html = re.sub(
        r'<section id="neuralbench-core" class="phase4">.*?</style>\s*',
        "", html, flags=re.DOTALL,
    )
    html = re.sub(
        r'<section id="depth-analysis" class="depth-analysis">.*?</style>\s*',
        "", html, flags=re.DOTALL,
    )

    injection = phase4 + depth
    new_html = html.replace("</body>", injection + "</body>", 1)
    DASHBOARD.write_text(new_html)
    print(f"injected → {DASHBOARD}")
    print(f"  phase4 block: {len(phase4)} chars")
    print(f"  depth  block: {len(depth)} chars")


if __name__ == "__main__":
    main()
