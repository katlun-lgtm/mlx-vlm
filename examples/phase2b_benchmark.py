#!/usr/bin/env python3
"""Phase-2b DISCRIMINATING streaming-video benchmark: recent-density recall.

Phase 2 failed to discriminate segment-aware streaming from uniform sampling
because its query ("what is the CURRENT scene?") is answerable from a single
frame, and ``np.linspace(0, T-1, B)`` ALWAYS samples the very last frame — so
uniform tied. Phase 2b fixes the query design so uniform *structurally* cannot
win at a tight KV budget.

DESIGN
------
* Content: REAL MOTION clips (Wikimedia Commons CC/PD), one distinct subject
  each (train, waves, dog, fire, ...). Fetch with ``fetch_phase2b_clips.py``.
  Each clip is sampled to ``F`` frames spread across it (real intra-clip motion).
* Stream: concatenate M distinct clips into ONE long stream of T = M*F frames.
* Query (the DISCRIMINATOR): "Name the last N scenes shown, most recent first"
  (N=3). Recency-DENSITY: needs MULTIPLE recent items, not one last frame.
* Control: "what is the current scene?" (expect segment ~= uniform tie).
* Isolation filter: a clip joins the pool only if the model names its subject
  correctly when shown ALONE — so any gap is KV-selection, not perception.
* Conditions at matched KV budget B (frames):
    (a) segment  — SegmentAwareSession keeps sink(text) + last B frames.
    (b) uniform  — linspace(0, T-1, B) over the whole stream, fresh-prefill.
    (c) full     — all frames so far (accuracy oracle; KV grows O(T)).
* Scoring: recall of the last-N subjects (deterministic substring match).

WHY UNIFORM CANNOT WIN: segment keeps the last B frames CONTIGUOUSLY (the last
B/F clips). Uniform spreads B samples over T=M*F frames; when B < M/2 its samples
land ~one-per-(M/B) clips, so near the end it captures the last clip (endpoint)
but SKIPS the 2nd/3rd-most-recent clips -> recall ~1/N. The discriminating budget
window is B in [~N*F, M/2], non-empty only when M > 2*N*F. This harness sets F=2,
N=3 and requires M>=13 so that window exists.

Usage:
    python examples/phase2b_benchmark.py [model_repo] [--tiny] [--streams K]
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image

from mlx_vlm import load
from mlx_vlm.generate.video import (
    SegmentAwareSession,
    _answer,
    _ingest_frame,
    kv_bytes,
    open_stream,
)
from mlx_vlm.models import cache as _cache

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(HERE, "videos_phase2b")

# Default to Idefics3-8B: SmolVLM2-2.2B (Phase 2's model) is "verified capable"
# only on ICONIC centred stock photos — on real messy motion footage it fails to
# name subjects (calls sharp frames "blurry", reports background textures), which
# collapses the isolation filter. Idefics3-8B (same idefics3 arch family => the
# 1-D-RoPE SegmentAwareSession path transfers unchanged) actually perceives the
# clips, so any accuracy gap is KV-selection, not model blindness.
DEFAULT_MODEL = "mlx-community/Idefics3-8B-Llama3-8bit"

# --- knobs (see module docstring for the M > 2*N*F requirement) ---
F = 2  # frames sampled per clip (real intra-clip motion)
N = 3  # "last N scenes" for the recency-density query
BUDGETS = [6, 10, 14]  # KV budget in frames (both segment window & uniform count)
MAX_FRAMES_READ = 400  # cap frames decoded per clip before subsampling
# Idefics3 splits a hi-res frame into ~17 tiles => ~2315 vision tok/frame (~320MB
# KV/frame). Turning splitting OFF at longest_edge=512 gives 176 tok/frame (~13x
# less, ~25MB/frame) with NO perception loss (verified: lion/tiger/duck/giraffe
# all still named) — the only way the multi-frame streaming benchmark is feasible.
FRAME_LONGEST = 512
DO_IMAGE_SPLITTING = False


def configure_model(model, processor):
    """Bound vision tokens/frame: single 512px tile instead of the ~17-tile split."""
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        try:
            ip.do_image_splitting = DO_IMAGE_SPLITTING
            ip.size = {"longest_edge": FRAME_LONGEST}
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not reconfigure image_processor: {e}")
    return model, processor


# Descriptive prompts beat terse "name the subject" prompts: a weak VLM answers a
# terse prompt with the most PROMINENT object (a fence, a bell, grass) and misses
# the intended subject, but a full-sentence description usually mentions it — and
# a substring match against the synonym set still scores it deterministically.
ISO_Q = "Describe what you see in this scene. Answer in one short sentence."
RECENT_Q = (
    "This is a video of several different scenes shown one after another. "
    f"List the last {N} scenes you saw, from MOST RECENT to least recent, "
    "naming the main subject of each scene."
)
CONTROL_Q = (
    "Describe what you see in the scene right now. Answer in one short sentence."
)
ISO_TOK, RECENT_TOK, CONTROL_TOK = 40, 72, 40


# --------------------------------------------------------------------------
# Clip decode + frame sampling.
# --------------------------------------------------------------------------
def sample_clip_frames(path: str, n_frames: int) -> list[Image.Image]:
    """Decode a clip and return ``n_frames`` PIL frames spread across it."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < MAX_FRAMES_READ:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if not frames:
        return []
    # sample from the MIDDLE 70% — the first/last frames of a clip are often
    # fades, title cards, or watermark-heavy intros that wreck perception.
    n = len(frames)
    lo, hi = int(0.15 * n), max(int(0.15 * n), int(0.85 * n) - 1)
    idx = np.linspace(lo, hi, n_frames).round().astype(int)
    out = []
    for j in idx:
        rgb = cv2.cvtColor(frames[int(j)], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        w, h = img.size
        s = FRAME_LONGEST / max(w, h)
        if s < 1:
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        out.append(img)
    return out


def load_pool():
    """Load manifest -> list of candidate dicts (label, cand, frames, synonyms)."""
    mpath = os.path.join(CLIPS_DIR, "manifest.json")
    manifest = json.load(open(mpath))
    cands = []
    for rec in manifest:
        path = os.path.join(CLIPS_DIR, rec["file"])
        if not os.path.exists(path):
            continue
        frames = sample_clip_frames(path, F)
        if len(frames) < F:
            print(f"  [decode-fail] {rec['file']}")
            continue
        cands.append(
            {
                "label": rec["label"],
                "cand": rec.get("cand", 0),
                "frames": frames,
                "synonyms": rec["synonyms"],
                "file": rec["file"],
            }
        )
    return cands


# Animated content is not "real footage" — drop it (e.g. Big Buck Bunny surfaced
# under "rabbit"). The model flags it in the description.
ANIMATION_MARKERS = (
    "animated",
    "cartoon",
    "animation",
    "illustration",
    "drawing",
    "cgi",
)


def _hit(low: str, s: str) -> bool:
    """Word-boundary (optional trailing 's') match — avoids substring false
    positives like 'cat' inside 'located'/'cattle' or 'lion' inside 'million'."""
    return re.search(r"\b" + re.escape(s) + r"s?\b", low) is not None


def recall(answer: str, labels: list[str], syn: dict[str, list[str]]) -> float:
    low = answer.lower()
    hit = sum(1 for lab in labels if any(_hit(low, s) for s in syn[lab]))
    return hit / len(labels)


# --------------------------------------------------------------------------
# Isolation ground-truth filter.
# --------------------------------------------------------------------------
def isolation_check(model, processor, cands):
    """Test every candidate; keep the FIRST that passes per subject label.

    Returns kept = {label: (frames, synonyms)} — at most one clip per subject.
    A subject enters the stream pool only if the model names it shown ALONE, so
    any downstream gap is KV-selection, not perception.
    """
    print("\n" + "=" * 72)
    print("ISOLATION — does the model name each candidate's subject shown ALONE?")
    print("=" * 72)
    syn = {c["label"]: c["synonyms"] for c in cands}
    kept = {}
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
        print(f"  {c['file']:34s} {status}  {ans['text']!r}")
        if ok and lab not in kept:
            kept[lab] = (c["frames"], c["synonyms"])
    print(f"kept {len(kept)} subjects: {list(kept)}")
    return kept


# --------------------------------------------------------------------------
# Stream construction.
# --------------------------------------------------------------------------
def build_streams(labels, n_streams, m_stream, rng):
    """Random permutations (subsets) of the pool -> the last-N ground truth varies."""
    streams = []
    for _ in range(n_streams):
        order = rng.sample(labels, min(m_stream, len(labels)))
        streams.append(order)
    return streams


def query_points(m_stream):
    """Clip indices (1-based, end-of-clip) where uniform undersamples the recent segment."""
    pts = sorted(set([m_stream, max(N + 1, m_stream - 4)]))
    return [p for p in pts if p >= N]


# --------------------------------------------------------------------------
# The three conditions.
# --------------------------------------------------------------------------
def run_segment(model, processor, frames, qp_frames, budget, question, max_tok):
    session = SegmentAwareSession(
        model, processor, vision_window=budget, keep_first_frame=False, evict_slack=0
    )
    session.begin(question)
    out = {}
    for i, fr in enumerate(frames):
        session.ingest_frame(fr)
        if i in qp_frames:
            kvb = kv_bytes(session.cache)
            ans = session.answer(max_tok)
            out[i] = {"text": ans["text"], "kv_bytes": kvb}
    return out


def run_selection(model, processor, frames, qp_frames, budget, mode, question, max_tok):
    out = {}
    for qpf in qp_frames:
        m = qpf + 1
        if mode == "uniform":
            k = min(budget, m)
            idx = sorted(
                set(int(x) for x in np.linspace(0, m - 1, k).round().astype(int))
            )
            sel = [frames[j] for j in idx]
        else:  # full
            sel = frames[:m]
        cache = _cache.make_prompt_cache(model.language_model)
        suffix = open_stream(model, processor, question, cache)
        for fr in sel:
            _ingest_frame(model, processor, fr, cache)
        kvb = kv_bytes(cache)
        ans = _answer(model, processor, suffix, cache, max_tok)
        out[qpf] = {"text": ans["text"], "kv_bytes": kvb, "n_frames": len(sel)}
    return out


# --------------------------------------------------------------------------
def evaluate(
    model, processor, kept, streams, budgets, query_types, out_path=None, model_id=""
):
    syn = {lab: s for lab, (_f, s) in kept.items()}
    rows = []
    t0 = time.time()
    for s_idx, order in enumerate(streams):
        frames = []
        for lab in order:
            frames.extend(kept[lab][0])
        qps_clip = query_points(len(order))
        qp_frames = [c * F - 1 for c in qps_clip]  # last frame of each query clip
        print(
            f"\n[stream {s_idx}] M={len(order)} T={len(frames)} qp_clips={qps_clip} "
            f"order={order}  ({time.time() - t0:.0f}s)"
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
                        r = recall(res["text"], gt, syn)
                        rows.append(
                            dict(
                                stream=s_idx,
                                query=qname,
                                clip_idx=clip_idx,
                                budget=b,
                                cond=cond,
                                gt="|".join(gt),
                                answer=res["text"],
                                recall=r,
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
        # Incremental save + running table so a wall-clock timeout still leaves
        # usable, real data (the JSON is otherwise only written at the very end).
        if out_path:
            json.dump(
                {
                    "model": model_id,
                    "F": F,
                    "N": N,
                    "budgets": budgets,
                    "streams_done": s_idx + 1,
                    "rows": rows,
                },
                open(out_path, "w"),
                indent=2,
            )
            print(money_table(rows, budgets, model_id, kept))
    return rows


def money_table(rows, budgets, model_id, kept):
    def agg(query, cond, budget):
        rs = [
            r
            for r in rows
            if r["query"] == query
            and r["cond"] == cond
            and (budget is None or r["budget"] == budget)
        ]
        if not rs:
            return None
        return (
            np.mean([r["recall"] for r in rs]),
            np.mean([r["kv_bytes"] for r in rs]) / 1024,
            len(rs),
        )

    lines = []
    lines.append("=" * 72)
    lines.append(f"MONEY TABLE — recall vs KV budget   model={model_id}")
    lines.append(f"pool M={len(kept)}  F={F} frames/clip  N={N} recent scenes")
    lines.append("=" * 72)
    for query in ("recent", "control"):
        lines.append(f"\n-- query={query!r} --")
        lines.append(f"{'condition':<18}{'recall':>8}{'KV(KB)':>10}{'n':>5}")
        for b in budgets:
            for cond in ("segment", "uniform"):
                a = agg(query, cond, b)
                if a:
                    lines.append(
                        f"{cond + ' B=' + str(b):<18}{a[0]:>8.2f}{a[1]:>10.1f}{a[2]:>5d}"
                    )
        af = agg(query, "full", None)
        if af:
            lines.append(f"{'full-context':<18}{af[0]:>8.2f}{af[1]:>10.1f}{af[2]:>5d}")
    return "\n".join(lines)


def main():
    argv = sys.argv[1:]
    tiny = "--tiny" in argv
    n_streams = int(argv[argv.index("--streams") + 1]) if "--streams" in argv else 6
    # positional model id = first non-flag token that is NOT a flag's value
    positionals = [
        a
        for i, a in enumerate(argv)
        if not a.startswith("--") and not (i > 0 and argv[i - 1] == "--streams")
    ]
    model_id = positionals[0] if positionals else DEFAULT_MODEL
    rng = random.Random(1234)

    print(f"[load] {model_id}")
    model, processor = load(model_id)
    configure_model(model, processor)
    print(
        f"[config] frame_longest={FRAME_LONGEST} do_image_splitting={DO_IMAGE_SPLITTING}"
    )
    print("[decode] sampling clip frames...")
    cands = load_pool()
    print(
        f"[pool] {len(cands)} candidate clips decoded over "
        f"{len(set(c['label'] for c in cands))} subjects"
    )

    kept = isolation_check(model, processor, cands)
    if len(kept) < N + 1:
        print(f"FATAL: only {len(kept)} clips passed isolation; need >= {N + 1}.")
        return

    labels = list(kept)
    if tiny:
        print("\n[TINY] 1 stream, end query, segment+uniform, 1 budget, recent query")
        order = labels[: min(6, len(labels))]
        frames = []
        for lab in order:
            frames.extend(kept[lab][0])
        qpf = [len(order) * F - 1]
        b = BUDGETS[0]
        seg = run_segment(model, processor, frames, qpf, b, RECENT_Q, RECENT_TOK)
        uni = run_selection(
            model, processor, frames, qpf, b, "uniform", RECENT_Q, RECENT_TOK
        )
        syn = {lab: s for lab, (_f, s) in kept.items()}
        gt = order[-N:]
        print(f"  order={order}  GT last-{N}={gt}  T={len(frames)} B={b}")
        print(
            f"  segment: recall={recall(seg[qpf[0]]['text'], gt, syn):.2f}  {seg[qpf[0]]['text']!r}"
        )
        print(
            f"  uniform: recall={recall(uni[qpf[0]]['text'], gt, syn):.2f}  {uni[qpf[0]]['text']!r}"
        )
        return

    m_stream = min(len(labels), 16)
    streams = build_streams(labels, n_streams, m_stream, rng)
    query_types = {
        "recent": (RECENT_Q, RECENT_TOK, True),
        "control": (CONTROL_Q, CONTROL_TOK, False),
    }
    out = os.path.join(HERE, "phase2b_results.json")
    rows = evaluate(
        model,
        processor,
        kept,
        streams,
        BUDGETS,
        query_types,
        out_path=out,
        model_id=model_id,
    )

    table = money_table(rows, BUDGETS, model_id, kept)
    print("\n" + table)
    json.dump(
        {
            "model": model_id,
            "F": F,
            "N": N,
            "budgets": BUDGETS,
            "m_stream": m_stream,
            "n_streams": len(streams),
            "rows": rows,
        },
        open(out, "w"),
        indent=2,
    )
    print(f"\n[saved] {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
