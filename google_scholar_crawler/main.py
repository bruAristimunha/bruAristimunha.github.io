"""Fetch citation stats from a public Google Scholar profile.

Writes ../_data/citations.json, which Jekyll renders on the homepage.
Stdlib only. If Scholar blocks the request (CAPTCHA / non-200) or the page
layout changes, the script exits 0 and leaves the previous data untouched,
so the site never loses its numbers.
"""

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHOLAR_ID = os.environ.get("GOOGLE_SCHOLAR_ID") or "2Gd5gOQAAAAJ"
URL = f"https://scholar.google.com/citations?user={SCHOLAR_ID}&hl=en"
OUT = Path(__file__).resolve().parent.parent / "_data" / "citations.json"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)


def fetch() -> str:
    req = urllib.request.Request(URL, headers={"User-Agent": UA, "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def parse(html: str) -> dict:
    # Stats table: [citations all, since, h all, since, i10 all, since]
    stats = [int(x) for x in re.findall(r'class="gsc_rsb_std">(\d+)<', html)]
    years = [int(y) for y in re.findall(r'class="gsc_g_t"[^>]*>(\d{4})<', html)]
    counts = [int(c) for c in re.findall(r'class="gsc_g_al">(\d+)<', html)]
    if len(stats) < 6 or not years or len(years) != len(counts):
        raise ValueError("unexpected Scholar page layout (blocked or changed)")
    return {
        "source": "Google Scholar",
        "profile": URL.replace("&hl=en", ""),
        "citations": stats[0],
        "h_index": stats[2],
        "i10_index": stats[4],
        "by_year": [{"year": y, "count": c} for y, c in zip(years, counts)],
    }


def main() -> int:
    try:
        new = parse(fetch())
    except Exception as exc:  # noqa: BLE001 - keep previous data on any failure
        print(f"::warning::Scholar fetch skipped, keeping previous data: {exc}")
        return 0

    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    if old and new["citations"] < old.get("citations", 0) * 0.5:
        print(f"::warning::Suspicious drop {old.get('citations')} -> {new['citations']}; not updating")
        return 0

    comparable = {k: v for k, v in old.items() if k != "updated"}
    if comparable == new:
        print("No change.")
        return 0

    new["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    OUT.write_text(json.dumps(new, indent=2) + "\n")
    print(json.dumps(new, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
