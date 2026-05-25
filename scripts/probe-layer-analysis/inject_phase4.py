"""Inject Phase 4 section before </body> of probe-layer-coverage.html."""
import re
from pathlib import Path

dashboard = Path("/Users/bruaristimunha/Projects/bruAristimunha.github.io/probe-layer-coverage.html")
wrap = Path("/tmp/phase4_wrap.html").read_text()

# Pull SVGs from the generated section
src = Path("/tmp/phase4_section.html").read_text()
# Heatmap
m = re.search(r'<div class="heatmap">(.*?)</div>', src, re.DOTALL)
heatmap = m.group(1) if m else "<!-- no heatmap -->"
# Ranking
m = re.search(r'<div class="ranking">(.*?)</div>', src, re.DOTALL)
ranking = m.group(1) if m else "<!-- no ranking -->"
# Depths block (everything in depths-grid)
m = re.search(r'<div class="depths-grid">(.*?)</div></section>', src, re.DOTALL)
depths = m.group(1) if m else "<!-- no depths -->"

wrap = wrap.replace("PHASE4_HEATMAP", heatmap)
wrap = wrap.replace("PHASE4_RANKING", ranking)
wrap = wrap.replace("PHASE4_DEPTHS", depths)

html = dashboard.read_text()

# Remove any prior phase4 section we may have injected
html = re.sub(r'<section id="neuralbench-core" class="phase4">.*?</style>\s*', "", html, flags=re.DOTALL)

# Inject before </body>
new_html = html.replace("</body>", wrap + "\n</body>", 1)
dashboard.write_text(new_html)
print(f"injected ({len(wrap)} chars) → {dashboard}")
