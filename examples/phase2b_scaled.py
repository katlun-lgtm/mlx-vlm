#!/usr/bin/env python3
"""Phase-2b SCALED streaming-video KV benchmark — with statistical rigor.

Builds directly on ``phase2b_benchmark.py`` (the DISCRIMINATING recent-density
design: real single-subject motion clips, isolation filter, segment vs uniform
vs full at a matched KV budget). That harness proved the *direction* at n=12/cell
(recency query: segment 0.83 vs uniform 0.44 @B6). This file raises n and makes
the claim reviewer-proof for a future mlx-vlm #492 PR by adding:

  1. ISOLATION MANIFEST (isolate once, reuse the pool).  ``--isolate`` decodes the
     candidate pool, runs the Idefics3 isolation filter, and writes
     ``examples/phase2b_isolation_pass.json`` (label, file, synonyms, Commons
     title/url/licence, the model's isolation answer). The scored run LOADS that
     manifest and re-decodes only the passing clips — it NEVER re-isolates.
  2. PARAMETERIZED SCALE.  ``--streams`` (default 20), ``--budgets 6,10,14``,
     ``--max-streams`` cap, both ``recent`` and ``control`` queries. Subject
     orders are randomized but DETERMINISTICALLY seeded (fixed base seed + stream
     index) — no ``Math.random``-style run-to-run nondeterminism.
  3. INCREMENTAL per-stream JSON save (a long run's partial results stay safe if
     it is killed mid-way).
  4. STATISTICS MODULE (the point).  Per (query, budget, condition): mean recall
     with a 95% CI. For the recency query: a PAIRED segment-minus-uniform test
     (paired by stream x query-point at matched budget) reporting the mean paired
     diff, its 95% CI, and a Wilcoxon signed-rank p-value. This is what separates
     "segment beats uniform" from "n=12 noise".

Usage:
    # 1. build the reusable pool (run once; commit the manifest):
    python examples/phase2b_scaled.py --isolate
    # 2. the long scored run (the human launches this separately):
    python examples/phase2b_scaled.py --streams 20 --budgets 6,10,14
    # fast plumbing smoke:
    python examples/phase2b_scaled.py --streams 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np

# Reuse the PROVEN Phase-2b pieces unchanged (decode, isolation logic, the three
# conditions, deterministic recall scoring, query design, model config).
from phase2b_benchmark import (
    ANIMATION_MARKERS,
    BUDGETS,
    CLIPS_DIR,
    CONTROL_Q,
    CONTROL_TOK,
    DEFAULT_MODEL,
    F,
    ISO_Q,
    ISO_TOK,
    N,
    RECENT_Q,
    RECENT_TOK,
    configure_model,
    load_pool,
    query_points,
    recall,
    run_segment,
    run_selection,
    sample_clip_frames,
)

from mlx_vlm import load
from mlx_vlm.generate.video import _answer, _ingest_frame, open_stream
from mlx_vlm.models import cache as _cache

HERE = os.path.dirname(os.path.abspath(__file__))
ISO_MANIFEST = os.path.join(HERE, "phase2b_isolation_pass.json")
RESULTS_PATH = os.path.join(HERE, "phase2b_scaled_results.json")
BASE_SEED = 1234  # fixed; per-stream seed = BASE_SEED * 100003 + stream_index


# ==========================================================================
# 1. Isolation manifest — isolate ONCE, save the passing pool.
# ==========================================================================
def _fetch_manifest_index() -> dict:
    """file -> {title,url,license,author} from the fetch manifest (attribution)."""
    mpath = os.path.join(CLIPS_DIR, "manifest.json")
    if not os.path.exists(mpath):
        return {}
    idx = {}
    for rec in json.load(open(mpath)):
        idx[rec["file"]] = {
            "title": rec.get("title"),
            "url": rec.get("url"),
            "license": rec.get("license"),
            "author": rec.get("author"),
        }
    return idx


def isolate_and_save(model, processor, cands, out_path: str) -> dict:
    """Run the isolation filter; write the passing-clip manifest to disk.

    Keeps the FIRST passing candidate per subject label (a subject enters the
    pool only if the model names it shown ALONE — so any downstream gap is
    KV-selection, not perception). Returns kept = {label: (frames, synonyms)}.
    """
    print("\n" + "=" * 72)
    print("ISOLATION — does Idefics3 name each candidate shown ALONE? (once)")
    print("=" * 72)
    attrib = _fetch_manifest_index()
    syn = {c["label"]: c["synonyms"] for c in cands}
    kept: dict = {}
    clip_records = []
    for c in cands:
        lab = c["label"]
        cache = _cache.make_prompt_cache(model.language_model)
        suffix = open_stream(model, processor, ISO_Q, cache)
        for fr in c["frames"]:
            _ingest_frame(model, processor, fr, cache)
        ans = _answer(model, processor, suffix, cache, ISO_TOK)
        low = ans["text"].lower()
        animated = any(m in low for m in ANIMATION_MARKERS)
        ok = (recall(ans["text"], [lab], syn) >= 1.0) and not animated
        if animated:
            status = "anim"
        elif ok and lab not in kept:
            status = "KEEP"
        elif ok:
            status = "dup "
        else:
            status = "drop"
        print(f"  {c['file']:40s} {status}  {ans['text']!r}")
        if ok and lab not in kept:
            kept[lab] = (c["frames"], c["synonyms"])
            meta = attrib.get(c["file"], {})
            clip_records.append(
                {
                    "label": lab,
                    "file": c["file"],
                    "synonyms": c["synonyms"],
                    "iso_answer": ans["text"],
                    "commons_title": meta.get("title"),
                    "commons_url": meta.get("url"),
                    "license": meta.get("license"),
                    "author": meta.get("author"),
                }
            )
    manifest = {
        "note": "Passing single-subject clips for Phase-2b scaled run. Re-decode "
        "these files (videos_phase2b/, gitignored -> reproduce via "
        "fetch_phase2b_clips.py) and score WITHOUT re-isolating.",
        "F": F,
        "iso_query": ISO_Q,
        "n_subjects": len(kept),
        "subjects": sorted(kept),
        "clips": clip_records,
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[isolation] kept {len(kept)} subjects: {sorted(kept)}")
    print(f"[isolation] manifest -> {out_path}")
    return kept


def load_isolation_pass(path: str) -> dict:
    """Load the passing-clip manifest and re-decode frames (no re-isolation)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"isolation manifest not found: {path}\n"
            "Run `python examples/phase2b_scaled.py --isolate` first."
        )
    data = json.load(open(path))
    kept: dict = {}
    missing = []
    for rec in data["clips"]:
        fpath = os.path.join(CLIPS_DIR, rec["file"])
        if not os.path.exists(fpath):
            missing.append(rec["file"])
            continue
        frames = sample_clip_frames(fpath, F)
        if len(frames) < F:
            missing.append(rec["file"] + " (decode-fail)")
            continue
        kept[rec["label"]] = (frames, rec["synonyms"])
    print(
        f"[pool] loaded {len(kept)}/{len(data['clips'])} passing subjects from "
        f"manifest ({data.get('n_subjects', '?')} recorded)"
    )
    if missing:
        print(f"[pool][warn] {len(missing)} manifest clips unavailable: {missing}")
    return kept


# ==========================================================================
# 2. Deterministically-seeded streams.
# ==========================================================================
def build_streams_seeded(labels, n_streams: int, m_stream: int, base_seed: int):
    """Random subject orders — but SEEDED per stream index (reproducible).

    Each stream draws from an INDEPENDENT ``random.Random(base_seed*P + idx)`` so
    the k-th stream is identical across runs and machines. ``sorted(labels)``
    fixes the input order so dict/set iteration nondeterminism can't leak in.
    """
    pool = sorted(labels)
    k = min(m_stream, len(pool))
    streams = []
    for idx in range(n_streams):
        rng = random.Random(base_seed * 100003 + idx)
        streams.append(rng.sample(pool, k))
    return streams


# ==========================================================================
# 3. Scored evaluation (incremental save) — same row schema as phase2b_benchmark.
# ==========================================================================
def evaluate(model, processor, kept, streams, budgets, query_types, out_path, meta):
    syn = {lab: s for lab, (_f, s) in kept.items()}
    rows = []
    t0 = time.time()
    per_stream_s = []
    for s_idx, order in enumerate(streams):
        s_t0 = time.time()
        frames = []
        for lab in order:
            frames.extend(kept[lab][0])
        qps_clip = query_points(len(order))
        qp_frames = [c * F - 1 for c in qps_clip]  # last frame of each query clip
        print(
            f"\n[stream {s_idx}] M={len(order)} T={len(frames)} qp_clips={qps_clip} "
            f"order={order}  ({time.time() - t0:.0f}s elapsed)"
        )
        for qname, (question, max_tok, is_recent) in query_types.items():
            full = run_selection(
                model, processor, frames, qp_frames, None, "full", question, max_tok
            )
            seg = {
                b: run_segment(
                    model, processor, frames, qp_frames, b, question, max_tok
                )
                for b in budgets
            }
            uni = {
                b: run_selection(
                    model, processor, frames, qp_frames, b, "uniform", question, max_tok
                )
                for b in budgets
            }
            for c, qpf in enumerate(qp_frames):
                clip_idx = qps_clip[c]
                if is_recent:
                    gt = order[clip_idx - N : clip_idx]  # last N clips (chron order)
                else:
                    gt = order[clip_idx - 1 : clip_idx]  # current clip only
                for b in budgets:
                    for cond, res in (
                        ("segment", seg[b][qpf]),
                        ("uniform", uni[b][qpf]),
                    ):
                        rows.append(
                            dict(
                                stream=s_idx,
                                query=qname,
                                clip_idx=clip_idx,
                                budget=b,
                                cond=cond,
                                gt="|".join(gt),
                                answer=res["text"],
                                recall=recall(res["text"], gt, syn),
                                kv_bytes=res["kv_bytes"],
                            )
                        )
                rf = full[qpf]
                rows.append(
                    dict(
                        stream=s_idx,
                        query=qname,
                        clip_idx=clip_idx,
                        budget=None,
                        cond="full",
                        gt="|".join(gt),
                        answer=rf["text"],
                        recall=recall(rf["text"], gt, syn),
                        kv_bytes=rf["kv_bytes"],
                    )
                )
                print(
                    f"   {qname:7s} clip{clip_idx:>2} gt={'|'.join(gt):28s} "
                    f"seg[{budgets[0]}]={recall(seg[budgets[0]][qpf]['text'], gt, syn):.2f} "
                    f"uni[{budgets[0]}]={recall(uni[budgets[0]][qpf]['text'], gt, syn):.2f} "
                    f"full={recall(rf['text'], gt, syn):.2f}"
                )
        per_stream_s.append(time.time() - s_t0)
        # Incremental save: a killed long run still leaves usable, real data.
        payload = {
            **meta,
            "streams_done": s_idx + 1,
            "streams_total": len(streams),
            "per_stream_seconds": per_stream_s,
            "rows": rows,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[save] {out_path}  streams_done={s_idx + 1}/{len(streams)}  "
            f"({per_stream_s[-1]:.0f}s this stream)"
        )
    return rows, per_stream_s


# ==========================================================================
# 4. STATISTICS MODULE — mean +/- 95% CI, paired diff, Wilcoxon signed-rank.
# ==========================================================================
try:  # prefer scipy when present; deterministic self-contained fallback otherwise.
    from scipy import stats as _sps  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _sps = None
    _HAVE_SCIPY = False

# t critical values for two-sided 95% CI, df 1..30, then a large-df tail.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_crit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if _HAVE_SCIPY:
        return float(_sps.t.ppf(0.975, df))
    if df in _T95:
        return _T95[df]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def mean_ci(vals) -> dict:
    """Mean with a two-sided 95% t-CI. n<2 -> CI collapses to the point."""
    v = np.asarray([float(x) for x in vals], dtype=float)
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    m = float(v.mean())
    if n < 2:
        return {"n": n, "mean": m, "lo": m, "hi": m}
    sd = float(v.std(ddof=1))
    half = _t_crit_95(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "lo": m - half, "hi": m + half, "sd": sd}


def wilcoxon_signed_rank(diffs) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Drops zero diffs (wilcox zero method). Uses scipy when available; otherwise a
    deterministic normal approximation with continuity + tie correction (matches
    scipy ``mode='approx'``). Returns statistic W = min(W+, W-), z, p, n_nonzero.
    """
    d = np.asarray([float(x) for x in diffs], dtype=float)
    nz = d[d != 0.0]
    n = int(nz.size)
    if n == 0:
        return {"n_nonzero": 0, "W": float("nan"), "z": float("nan"), "p": float("nan")}
    if _HAVE_SCIPY and n >= 1:
        try:
            # scipy auto-selects an exact (small n, no ties) or normal-approx
            # null distribution; both are deterministic. Signature varies across
            # versions, so pass only the stable kwargs.
            res = _sps.wilcoxon(nz, zero_method="wilcox", correction=True)
            W = float(res.statistic)
            p = float(res.pvalue)
            # recover z from p for reporting (two-sided normal)
            z = float(_sps.norm.ppf(1.0 - p / 2.0)) if 0 < p < 1 else float("nan")
            return {"n_nonzero": n, "W": W, "z": z, "p": p, "engine": "scipy"}
        except Exception:  # noqa: BLE001
            pass
    # --- deterministic fallback: average-rank + normal approx w/ tie correction.
    order = np.argsort(np.abs(nz), kind="mergesort")
    absd = np.abs(nz)[order]
    signs = np.sign(nz)[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # average rank (1-based) for the tie block
        ranks[i : j + 1] = avg
        t = j - i + 1
        if t > 1:
            tie_term += t**3 - t
        i = j + 1
    w_plus = float(ranks[signs > 0].sum())
    w_minus = float(ranks[signs < 0].sum())
    W = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return {
            "n_nonzero": n,
            "W": W,
            "z": float("nan"),
            "p": float("nan"),
            "engine": "fallback",
        }
    cc = 0.5  # continuity correction
    z = (
        (W - mean_w + cc) / math.sqrt(var_w)
        if W < mean_w
        else (W - mean_w - cc) / math.sqrt(var_w)
    )
    # two-sided p via the standard normal survival function (erfc).
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n_nonzero": n, "W": W, "z": z, "p": p, "engine": "fallback"}


def _cell_vals(rows, query, cond, budget):
    return [
        r["recall"]
        for r in rows
        if r["query"] == query
        and r["cond"] == cond
        and (budget is None or r["budget"] == budget)
    ]


def _paired_diffs(rows, query, budget):
    """seg-uni paired by (stream, clip_idx) at a matched budget for a query."""
    seg = {
        (r["stream"], r["clip_idx"]): r["recall"]
        for r in rows
        if r["query"] == query and r["cond"] == "segment" and r["budget"] == budget
    }
    uni = {
        (r["stream"], r["clip_idx"]): r["recall"]
        for r in rows
        if r["query"] == query and r["cond"] == "uniform" and r["budget"] == budget
    }
    keys = sorted(set(seg) & set(uni))
    return [seg[k] - uni[k] for k in keys]


def stats_report(rows, budgets, model_id) -> str:
    L = []
    L.append("=" * 78)
    L.append(
        f"STATISTICS — Phase-2b scaled   model={model_id}   "
        f"scipy={'yes' if _HAVE_SCIPY else 'no (fallback)'}"
    )
    L.append("=" * 78)

    # --- per-cell mean recall +/- 95% CI (both queries, all conds) ---
    for query in ("recent", "control"):
        if not any(r["query"] == query for r in rows):
            continue
        L.append(f"\n-- mean recall +/- 95% CI   query={query!r} --")
        L.append(f"{'condition':<16}{'mean':>7}{'95% CI':>20}{'n':>6}")
        for b in budgets:
            for cond in ("segment", "uniform"):
                c = mean_ci(_cell_vals(rows, query, cond, b))
                if c["n"]:
                    L.append(
                        f"{cond + ' B=' + str(b):<16}{c['mean']:>7.3f}"
                        f"{'[%.3f, %.3f]' % (c['lo'], c['hi']):>20}{c['n']:>6}"
                    )
        cf = mean_ci(_cell_vals(rows, query, "full", None))
        if cf["n"]:
            L.append(
                f"{'full-context':<16}{cf['mean']:>7.3f}"
                f"{'[%.3f, %.3f]' % (cf['lo'], cf['hi']):>20}{cf['n']:>6}"
            )

    # --- PAIRED segment - uniform (the discriminator) for the recency query ---
    if any(r["query"] == "recent" for r in rows):
        L.append(
            "\n-- PAIRED segment - uniform   query='recent'   "
            "(paired by stream x query-point) --"
        )
        L.append(
            f"{'budget':>7}{'n_pair':>7}{'seg':>7}{'uni':>7}{'diff':>8}"
            f"{'95% CI(diff)':>22}{'Wilcoxon p':>12}"
        )
        for b in budgets:
            diffs = _paired_diffs(rows, "recent", b)
            if not diffs:
                continue
            ci = mean_ci(diffs)
            w = wilcoxon_signed_rank(diffs)
            seg_m = float(np.mean(_cell_vals(rows, "recent", "segment", b)))
            uni_m = float(np.mean(_cell_vals(rows, "recent", "uniform", b)))
            p = w["p"]
            p_s = "n/a" if (p != p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")
            L.append(
                f"{b:>7}{ci['n']:>7}{seg_m:>7.3f}{uni_m:>7.3f}{ci['mean']:>+8.3f}"
                f"{'[%+.3f, %+.3f]' % (ci['lo'], ci['hi']):>22}{p_s:>12}"
            )
        L.append(
            "  (diff>0 => segment recalls more recent scenes than uniform at that "
            "KV budget;\n   CI excluding 0 and small p => the gap is not noise.)"
        )
    return "\n".join(L)


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Phase-2b scaled benchmark + stats")
    ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    ap.add_argument(
        "--isolate",
        action="store_true",
        help="run isolation filter and write the passing-clip manifest",
    )
    ap.add_argument("--streams", type=int, default=20)
    ap.add_argument(
        "--max-streams",
        type=int,
        default=None,
        help="hard cap on number of streams actually run",
    )
    ap.add_argument("--budgets", type=str, default=",".join(str(b) for b in BUDGETS))
    ap.add_argument(
        "--m-stream",
        type=int,
        default=16,
        help="subjects per stream (min with pool size)",
    )
    ap.add_argument("--query", choices=["recent", "control", "both"], default="both")
    ap.add_argument("--manifest", default=ISO_MANIFEST)
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    print(f"[load] {args.model}")
    model, processor = load(args.model)
    configure_model(model, processor)

    # ---- Phase 1: isolate + save manifest, then stop. ----
    if args.isolate:
        print("[decode] sampling candidate clip frames...")
        cands = load_pool()
        print(
            f"[pool] {len(cands)} candidate clips over "
            f"{len(set(c['label'] for c in cands))} subjects"
        )
        kept = isolate_and_save(model, processor, cands, args.manifest)
        if len(kept) < N + 1:
            print(f"[warn] only {len(kept)} subjects passed; need >= {N + 1} to score.")
        return

    # ---- Phase 2: scored run over the reusable pool (no re-isolation). ----
    kept = load_isolation_pass(args.manifest)
    if len(kept) < N + 1:
        print(f"FATAL: only {len(kept)} subjects in manifest; need >= {N + 1}.")
        return

    n_streams = args.streams
    if args.max_streams is not None:
        n_streams = min(n_streams, args.max_streams)
    m_stream = min(len(kept), args.m_stream)
    streams = build_streams_seeded(list(kept), n_streams, m_stream, BASE_SEED)

    qmap = {
        "recent": (RECENT_Q, RECENT_TOK, True),
        "control": (CONTROL_Q, CONTROL_TOK, False),
    }
    if args.query == "both":
        query_types = qmap
    else:
        query_types = {args.query: qmap[args.query]}

    meta = {
        "model": args.model,
        "F": F,
        "N": N,
        "budgets": budgets,
        "m_stream": m_stream,
        "n_streams": n_streams,
        "base_seed": BASE_SEED,
        "queries": list(query_types),
        "pool_subjects": sorted(kept),
    }
    print(
        f"[run] streams={n_streams} m_stream={m_stream} budgets={budgets} "
        f"queries={list(query_types)}  pool={len(kept)} subjects"
    )

    rows, per_stream_s = evaluate(
        model, processor, kept, streams, budgets, query_types, args.out, meta
    )

    print("\n" + stats_report(rows, budgets, args.model))
    if per_stream_s:
        print(
            f"\n[timing] {len(per_stream_s)} streams  "
            f"mean {np.mean(per_stream_s):.1f}s/stream  "
            f"total {sum(per_stream_s):.0f}s"
        )
    print(f"[saved] {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
