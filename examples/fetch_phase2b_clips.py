#!/usr/bin/env python3
"""Fetch REAL MOTION video clips (CC/PD) for the Phase-2b discriminating benchmark.

Unlike Phase 2 (a still-photo slideshow), Phase 2b needs *real footage with
intra-clip motion* and MANY distinct subjects, so we can concatenate a long
multi-scene stream and ask a recent-DENSITY query ("name the last N scenes")
that uniform sampling structurally cannot answer at a tight KV budget.

Source: Wikimedia Commons video files (webm / ogv / mp4), all CC BY-SA / CC0 /
public-domain. We use the MediaWiki API to search ``filetype:video <subject>``,
and download the ORIGINAL file via ``imageinfo`` url (the
``Special:FilePath?width=`` route only returns a single JPEG poster frame —
useless for motion). We download up to ``N_CAND`` candidates PER subject because
Commons search returns many off-subject / distant / watermarked clips; the
benchmark's isolation filter then keeps the candidate(s) the model actually
names. A manifest (title, url, size, duration, licence, author) is written for
reproducibility + attribution.

Output:  examples/videos_phase2b/<label>__<i>.<ext>   (+ manifest.json)
Usage:   python examples/fetch_phase2b_clips.py [--limit N] [--max-mb 14]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos_phase2b")
os.makedirs(OUT, exist_ok=True)
UA = "mlx-vlm-phase2b-benchmark/1.0 (research; contact katlun@windyviews.com)"
API = "https://commons.wikimedia.org/w/api.php"

# label -> (search query, accepted-answer synonyms for scoring).
# Synonym sets are kept DISJOINT across labels (no shared token like "water")
# so a substring hit is unambiguous. Bias HARD toward subjects that Commons
# reliably has as CLOSE-UP, frame-FILLING footage (mostly captive/wild animals +
# a few unmistakable objects) — distant/ambiguous clips get dropped by the
# isolation filter anyway, but frame-filling subjects survive it far more often.
SUBJECTS: dict[str, tuple[str, list[str]]] = {
    "dog": ("dog close", ["dog", "puppy", "canine"]),
    "cat": ("cat close", ["cat", "kitten", "feline"]),
    "horse": ("horse", ["horse", "pony", "stallion", "equine"]),
    "elephant": ("elephant", ["elephant"]),
    "tiger": ("tiger", ["tiger"]),
    "lion": ("lion", ["lion"]),
    "bear": ("bear", ["bear"]),
    "monkey": ("monkey macaque", ["monkey", "macaque", "ape", "primate"]),
    "giraffe": ("giraffe", ["giraffe"]),
    "zebra": ("zebra", ["zebra"]),
    "penguin": ("penguin", ["penguin"]),
    "peacock": ("peacock", ["peacock", "peafowl"]),
    "rabbit": ("rabbit", ["rabbit", "bunny", "hare"]),
    "snake": ("snake", ["snake", "serpent"]),
    "turtle": ("turtle tortoise", ["turtle", "tortoise"]),
    "butterfly": ("butterfly", ["butterfly"]),
    "jellyfish": ("jellyfish", ["jellyfish", "jelly"]),
    "duck": ("duck swimming", ["duck", "ducks", "duckling"]),
    "sheep": ("sheep", ["sheep", "lamb"]),
    "cow": ("cow cattle", ["cow", "cattle", "calf"]),
    "fire": ("campfire flames bonfire", ["fire", "flame", "campfire", "bonfire"]),
    "fireworks": ("fireworks", ["firework", "fireworks"]),
    "waterfall": ("waterfall cascade", ["waterfall", "falls", "cascade"]),
    "fountain": ("water fountain", ["fountain"]),
    "flower": ("flower blooming timelapse", ["flower", "blossom", "bloom", "petal"]),
    "candle": ("candle flame", ["candle"]),
    # --- Phase-2b SCALED pool extension (target ~25-30 passing subjects). ---
    # All disjoint synonym sets (no token shared with an existing label) so a
    # substring hit for one subject can never be mis-credited to another when the
    # recency answer names several. Biased toward frame-filling animal footage
    # Commons reliably carries; the isolation filter drops whatever the model
    # can't name, so over-fetching here is cheap insurance for a bigger M.
    "flamingo": ("flamingo", ["flamingo"]),
    "owl": ("owl", ["owl"]),
    "parrot": ("parrot macaw", ["parrot", "macaw"]),
    "swan": ("swan swimming", ["swan", "cygnet"]),
    "pig": ("pig piglet farm", ["pig", "piglet"]),
    "deer": ("deer", ["deer", "stag", "fawn"]),
    "camel": ("camel", ["camel", "dromedary"]),
    "koala": ("koala", ["koala"]),
    "panda": ("giant panda", ["panda"]),
    "fox": ("fox", ["fox"]),
    "squirrel": ("squirrel", ["squirrel"]),
    "frog": ("frog", ["frog", "toad"]),
    "crab": ("crab", ["crab"]),
    "shark": ("shark", ["shark"]),
    "goose": ("goose", ["goose", "geese", "gosling"]),
    "rooster": ("rooster chicken", ["rooster", "chicken", "hen", "cockerel"]),
    "fish": ("fish aquarium", ["fish", "goldfish", "koi"]),
    "train": ("train railway", ["train", "locomotive"]),
}

MAX_BYTES_DEFAULT = 14 * 1024 * 1024
MIN_DUR, MAX_DUR = 2.0, 40.0
SEARCH_HITS = 8  # Commons hits considered per subject
N_CAND = 2  # candidates downloaded per subject (isolation filter picks winners)


# Commons rate-limits bots hard (HTTP 429). Space every request out and back off
# on 429 so a big scaled fetch actually completes instead of losing half the pool.
REQUEST_DELAY = 1.5  # min seconds between successive Commons requests
_last_req = [0.0]


def _throttle() -> None:
    dt = time.time() - _last_req[0]
    if dt < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - dt)
    _last_req[0] = time.time()


def _get(url: str, timeout: int = 90, retries: int = 5) -> bytes:
    base = 5.0
    for attempt in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = base * (attempt + 1)
                print(
                    f"    [429] backoff {wait:.0f}s (attempt {attempt + 1}/{retries})"
                )
                time.sleep(wait)
                continue
            raise


def api(params: dict) -> dict:
    params = {**params, "action": "query", "format": "json"}
    return json.loads(_get(API + "?" + urllib.parse.urlencode(params), timeout=60))


def search_videos(query: str, limit: int) -> list[str]:
    r = api(
        {
            "list": "search",
            "srsearch": f"filetype:video {query}",
            "srnamespace": "6",
            "srlimit": str(limit),
        }
    )
    return [h["title"] for h in r.get("query", {}).get("search", [])]


def imageinfo(titles: list[str]) -> dict:
    if not titles:
        return {}
    r = api(
        {
            "prop": "imageinfo",
            "titles": "|".join(titles),
            "iiprop": "url|size|mediatype|metadata|extmetadata",
        }
    )
    return r.get("query", {}).get("pages", {})


def _duration(ii: dict):
    for m in ii.get("metadata") or []:
        if m.get("name") in ("length", "playtime_seconds"):
            try:
                return float(m.get("value"))
            except Exception:
                return None
    return None


def _license(ii: dict) -> tuple[str, str]:
    ext = ii.get("extmetadata") or {}
    lic = (ext.get("LicenseShortName") or {}).get("value", "?")
    artist = (ext.get("Artist") or {}).get("value", "?")
    artist = re.sub("<[^>]+>", "", artist).strip()
    return lic, artist


def fetch_subject(label: str, query: str, max_bytes: int) -> list[dict]:
    """Download up to N_CAND size/duration-eligible candidate clips for a subject."""
    titles = search_videos(query, SEARCH_HITS)
    pages = imageinfo(titles)
    by_title = {p.get("title"): p for p in pages.values()}
    recs = []
    ci = 0
    for t in titles:
        if len(recs) >= N_CAND:
            break
        page = by_title.get(t)
        if not page or not page.get("imageinfo"):
            continue
        ii = page["imageinfo"][0]
        size = int(ii.get("size", 0))
        dur = _duration(ii)
        if size <= 0 or size > max_bytes:
            continue
        if dur is not None and not (MIN_DUR <= dur <= MAX_DUR):
            continue
        url = ii["url"]
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or ".webm"
        dest = os.path.join(OUT, f"{label}__{ci}{ext}")
        try:
            blob = _get(url, timeout=180)
        except Exception as e:
            print(f"[err] {label}: download failed {repr(e)[:60]}")
            continue
        with open(dest, "wb") as f:
            f.write(blob)
        lic, artist = _license(ii)
        print(
            f"[ok] {label:10s}#{ci} {len(blob) // 1024:6d}KB dur={dur}s  {page['title']}  [{lic}]"
        )
        recs.append(
            {
                "label": label,
                "cand": ci,
                "file": os.path.basename(dest),
                "title": page["title"],
                "url": url,
                "size": len(blob),
                "duration_s": dur,
                "license": lic,
                "author": artist,
                "synonyms": SUBJECTS[label][1],
            }
        )
        ci += 1
    if not recs:
        print(f"[skip] {label}: no eligible candidate")
    return recs


def main() -> None:
    flags = sys.argv[1:]
    limit = int(flags[flags.index("--limit") + 1]) if "--limit" in flags else None
    max_mb = (
        int(flags[flags.index("--max-mb") + 1])
        if "--max-mb" in flags
        else MAX_BYTES_DEFAULT // 1024 // 1024
    )
    max_bytes = max_mb * 1024 * 1024

    items = list(SUBJECTS.items())
    if limit:
        items = items[:limit]

    mpath = os.path.join(OUT, "manifest.json")

    # RESUME: keep any already-downloaded subject (file still on disk) from a
    # prior manifest and re-fetch ONLY the missing ones. This makes a 429-throttled
    # scaled fetch recoverable — just run it again and it tops up the gaps without
    # re-hammering Commons for clips you already have. Pass --refresh to ignore.
    existing: dict[str, list[dict]] = {}
    if "--refresh" not in flags and os.path.exists(mpath):
        for rec in json.load(open(mpath)):
            if os.path.exists(os.path.join(OUT, rec["file"])):
                existing.setdefault(rec["label"], []).append(rec)

    manifest = []
    for label, (query, _syn) in items:
        if label in existing:
            manifest.extend(existing[label])
            print(f"[resume] {label:10s} {len(existing[label])} existing clip(s), skip")
            continue
        manifest.extend(fetch_subject(label, query, max_bytes))

    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    total = sum(r["size"] for r in manifest)
    subjects = sorted(set(r["label"] for r in manifest))
    print(
        f"\n[done] {len(manifest)} clips over {len(subjects)} subjects, "
        f"{total // 1024 // 1024}MB total -> {OUT}"
    )
    print(f"[manifest] {mpath}")


if __name__ == "__main__":
    main()
