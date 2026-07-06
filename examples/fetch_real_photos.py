#!/usr/bin/env python3
"""Fetch distinct single-subject REAL photos from Wikimedia Commons.

Uses the MediaWiki API to resolve, for each search term, a real photograph in
the File namespace and download a ~640px thumbnail. Biased toward clean,
frame-filling, prototypical single-object shots (studio / isolated) so a small
VLM can reliably identify the gist from a single tile. These become the
ground-truth "scenes" for the Phase-2 streaming benchmark.
Output: examples/real_scenes/<label>.jpg
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(__file__), "real_scenes")
os.makedirs(OUT, exist_ok=True)

# term -> search qualifier biased toward a clean frame-filling photo.
TERMS = {
    "elephant": "African elephant savanna photograph",
    "zebra": "plains zebra standing photograph",
    "pizza": "whole pizza margherita top view",
    "dog": "golden retriever dog portrait photograph",
    "cat": "tabby cat portrait photograph",
    "car": "red sports car side view photograph",
    "clock": "analog wall clock face photograph",
    "laptop": "laptop computer open photograph",
    "orange": "orange fruit single isolated photograph",
    "teddybear": "teddy bear plush toy photograph",
    "banana": "banana bunch isolated white photograph",
    "guitar": "acoustic guitar full body photograph",
    "coffee": "cup of coffee latte top view photograph",
    "bicycle": "bicycle side view photograph",
}

API = "https://commons.wikimedia.org/w/api.php"
UA = "mlx-vlm-phase2-benchmark/1.0 (research; contact katlun@windyviews.com)"


def resolve_thumb(query: str) -> str | None:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": "6",
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": "640",
        "format": "json",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    pages = data.get("query", {}).get("pages", {})
    # keep search rank order
    ordered = sorted(pages.values(), key=lambda p: p.get("index", 999))
    for p in ordered:
        ii = p.get("imageinfo", [{}])[0]
        if ii.get("mime") in ("image/jpeg", "image/png"):
            return ii.get("thumburl") or ii.get("url")
    return None


def main() -> None:
    for label, query in TERMS.items():
        try:
            thumb = resolve_thumb(query)
            if not thumb:
                print(f"[miss] {label}")
                continue
            dest = os.path.join(OUT, label + ".jpg")
            req = urllib.request.Request(thumb, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
            with open(dest, "wb") as f:
                f.write(blob)
            print(
                f"[ok] {label:10s} {len(blob) // 1024:5d}KB  <- {thumb.split('/')[-1]}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[err] {label}: {e}")


if __name__ == "__main__":
    main()
