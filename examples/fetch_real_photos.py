#!/usr/bin/env python3
"""Fetch the 5 REAL benchmark scenes from Wikimedia Commons (reproducible).

Downloads the EXACT source images used by ``phase2_benchmark.py`` via the stable
``Special:FilePath`` resolver (filename -> resized file), so anyone can reproduce
the benchmark's scene pool. All five are Wikimedia Commons media (CC BY-SA / CC0 /
public domain — see each file's page for the licence + author).

Output: examples/real_scenes/<label>.jpg  (~640px longest edge)
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(__file__), "real_scenes")
os.makedirs(OUT, exist_ok=True)

# label -> Wikimedia Commons File: page title (exact) of the image used.
FILES = {
    "airplane": "Airbus A350-900 - Daniel K. Inouye International Airport - Honolulu.jpg",
    "bus": "Metroline red double-decker bus in Newport - geograph.org.uk - 6225154.jpg",
    "clock": "Ein halb 004 2023 07 20.jpg",
    "pizza": "Margherita Pizza (63999444).jpg",
    "zebra": "Plains Zebra Equus quagga.jpg",
}
UA = "mlx-vlm-phase2-benchmark/1.0 (research; contact katlun@windyviews.com)"
BASE = "https://commons.wikimedia.org/wiki/Special:FilePath/"


def main() -> None:
    for label, filename in FILES.items():
        url = BASE + urllib.parse.quote(filename) + "?width=640"
        dest = os.path.join(OUT, label + ".jpg")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
            with open(dest, "wb") as f:
                f.write(blob)
            print(f"[ok] {label:10s} {len(blob) // 1024:5d}KB  <- {filename}")
        except Exception as e:  # noqa: BLE001
            print(f"[err] {label}: {e}")


if __name__ == "__main__":
    main()
