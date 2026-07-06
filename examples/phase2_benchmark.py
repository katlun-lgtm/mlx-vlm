#!/usr/bin/env python3
"""Phase-2 REAL-imagery streaming benchmark: accuracy vs KV budget.

Produces the "money table": for questions about the CURRENT scene in a
multi-scene stream, compare THREE conditions at matched KV budgets:

  (a) segment-aware streaming  — SegmentAwareSession keeps sink(text) + the
      ``vision_window`` MOST-RECENT frames; oldest vision evicts + re-anchors.
  (b) uniform sampling         — the load_video-style path: uniformly subsample
      the stream-so-far down to the SAME frame/KV budget, fresh-prefill, answer.
  (c) full context             — prefill ALL frames so far (the accuracy oracle;
      KV grows O(T) and would OOM on a long enough stream).

All three use the IDENTICAL per-frame image-token prefill (`_ingest_frame`); the
ONLY variable is WHICH frames populate the cache — isolating frame-SELECTION as
the tested policy.

CONTENT (real imagery, known ground truth): a controlled "slideshow stream" of
DISTINCT REAL PHOTOS (examples/real_scenes/*.jpg — a plane in the sky, a red
double-decker bus, a wooden wall clock, a pizza, a zebra). Each scene = one
photo repeated K frames (a "shot"); a stream concatenates several shots. We ask
"describe what you see" at each shot boundary and check (substring) whether the
CURRENT shot's subject is named. This is the plan's stated fallback to real
video-clip download; it gives airtight per-query ground truth and removes model-
capability confounds. Images are fed at NATIVE resolution (the model's own
pipeline, ~84 tok/frame) — a forced square resize was found to wreck perception.

Scoring: deterministic substring match of the lowercased answer against a
per-scene accepted-synonym set. A scene enters the pool ONLY if the model names
it correctly IN ISOLATION (full attention, single image) — so accuracy gaps
reflect the streaming policy, not model blindness.

Usage:
    python examples/phase2_benchmark.py [model_repo] [--smoke]
"""

from __future__ import annotations

import json
import os
import sys
import time

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
SCENES_DIR = os.path.join(HERE, "real_scenes")
QUESTION = "Describe what you see. Answer in one short sentence."
MAX_ANS_TOKENS = 32

# label -> (image path, accepted-answer synonyms). Human-confirmed ground truth;
# each passes the isolation filter below (model names it with full attention).
CANDIDATES = {
    "airplane": (
        os.path.join(SCENES_DIR, "airplane.jpg"),
        ["airplane", "plane", "jet", "aircraft"],
    ),
    "bus": (os.path.join(SCENES_DIR, "bus.jpg"), ["bus"]),
    "clock": (os.path.join(SCENES_DIR, "clock.jpg"), ["clock"]),
    "pizza": (os.path.join(SCENES_DIR, "pizza.jpg"), ["pizza"]),
    "zebra": (os.path.join(SCENES_DIR, "zebra.jpg"), ["zebra"]),
}

K = 4  # frames per scene (shot length)
BUDGETS = [4, 8]  # vision_window / uniform frame budget to sweep


def load_frame(path: str) -> Image.Image:
    # NATIVE resolution: let the model's processor resize (forced square resize
    # was found to collapse perception into a degenerate default).
    return Image.open(path).convert("RGB")


def score(answer: str, accepted: list[str]) -> int:
    low = answer.lower()
    return int(any(a in low for a in accepted))


def probe_tokens_per_frame(model, processor, frame) -> int:
    image_token = getattr(processor, "image_token", "<image>")
    out = processor(
        text=[image_token],
        images=[frame],
        add_special_tokens=True,
        return_tensors="mlx",
    )
    return int(out["input_ids"].shape[1])


# --------------------------------------------------------------------------
# Phase 0 — isolation ground-truth filter.
# --------------------------------------------------------------------------
def isolation_check(model, processor, scenes):
    print("\n" + "=" * 70)
    print("PHASE 0 — isolation ground-truth check (single image, full attention)")
    print("=" * 70)
    kept = {}
    for label, (path, accepted) in scenes.items():
        frame = load_frame(path)
        cache = _cache.make_prompt_cache(model.language_model)
        suffix = open_stream(model, processor, QUESTION, cache)
        _ingest_frame(model, processor, frame, cache)
        ans = _answer(model, processor, suffix, cache, MAX_ANS_TOKENS)
        ok = score(ans["text"], accepted)
        print(f"  {label:10s} {'KEEP' if ok else 'drop'}  {ans['text']!r}")
        if ok:
            kept[label] = (path, accepted)
    print(f"kept {len(kept)}/{len(scenes)} scenes: {list(kept)}")
    return kept


def build_streams(labels):
    """Rotations of the scene pool taken 3 at a time -> queried scene varies."""
    n = len(labels)
    return [[labels[(start + j) % n] for j in range(3)] for start in range(n)]


# --------------------------------------------------------------------------
# The three conditions.
# --------------------------------------------------------------------------
def run_segment_aware(model, processor, frames, query_points, budget):
    """(a) One streaming pass; answer at each query point over the bounded window."""
    session = SegmentAwareSession(
        model, processor, vision_window=budget, keep_first_frame=False, evict_slack=0
    )
    session.begin(QUESTION)
    ingest_ms = []
    results = {}
    for i, frame in enumerate(frames):
        st = session.ingest_frame(frame)
        ingest_ms.append(st["lm_ms"] + st["encode_ms"] + st["evict_ms"])
        if i in query_points:
            kvb = kv_bytes(session.cache)  # retained window (pre-answer)
            ans = session.answer(MAX_ANS_TOKENS)
            results[i] = {
                "text": ans["text"],
                "kv_bytes": kvb,
                "ttft_ms": ans["ttft_ms"],
                "step_ms": float(np.mean(ingest_ms[-K:])),
                "n_rebuilds": session._n_rebuilds,
            }
    return results


def run_selection(model, processor, frames, query_points, budget, mode):
    """(b) uniform / (c) full: fresh-prefill a frame SELECTION, then answer."""
    results = {}
    for qp in query_points:
        m = qp + 1
        if mode == "uniform":
            k = min(budget, m)
            idx = sorted(
                set(int(x) for x in np.linspace(0, m - 1, k).round().astype(int))
            )
            sel = [frames[j] for j in idx]
        else:  # full
            sel = frames[:m]
        cache = _cache.make_prompt_cache(model.language_model)
        suffix = open_stream(model, processor, QUESTION, cache)
        t0 = time.perf_counter()
        for f in sel:
            _ingest_frame(model, processor, f, cache)
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        kvb = kv_bytes(cache)
        ans = _answer(model, processor, suffix, cache, MAX_ANS_TOKENS)
        results[qp] = {
            "text": ans["text"],
            "kv_bytes": kvb,
            "ttft_ms": ans["ttft_ms"],
            "step_ms": prefill_ms / max(1, len(sel)),
            "n_frames": len(sel),
        }
    return results


# --------------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    model_id = args[0] if args else "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
    smoke = "--smoke" in flags

    print(f"[load] {model_id}")
    model, processor = load(model_id)
    tpf = probe_tokens_per_frame(model, processor, load_frame(CANDIDATES["pizza"][0]))
    print(f"[probe] tokens/frame (native) ~= {tpf}")

    scenes = isolation_check(model, processor, CANDIDATES)
    if len(scenes) < 3:
        print("FATAL: <3 scenes survived isolation; cannot build streams.")
        return

    labels = list(scenes)
    streams = build_streams(labels)
    budgets = BUDGETS
    if smoke:
        streams, budgets = streams[:1], BUDGETS[:1]
        print("\n[SMOKE] 1 stream, 1 budget, all 3 conditions")

    print("\n" + "=" * 70)
    print(
        f"PHASE 2 — {len(streams)} streams x {K} frames/scene x 3 scenes; budgets={budgets}"
    )
    print("=" * 70)

    rows = []
    t_start = time.time()
    for s_idx, trio in enumerate(streams):
        frames, gt_at = [], {}
        for scene_pos, label in enumerate(trio):
            path, accepted = scenes[label]
            f = load_frame(path)
            frames.extend([f] * K)
            gt_at[(scene_pos + 1) * K - 1] = (label, accepted)
        query_points = sorted(gt_at)
        print(
            f"\n[stream {s_idx}] scenes={trio} frames={len(frames)} qps={query_points} "
            f"(elapsed {time.time() - t_start:.0f}s)"
        )

        full_res = run_selection(model, processor, frames, query_points, None, "full")
        for budget in budgets:
            seg_res = run_segment_aware(model, processor, frames, query_points, budget)
            uni_res = run_selection(
                model, processor, frames, query_points, budget, "uniform"
            )
            for qp in query_points:
                label, accepted = gt_at[qp]
                m = qp + 1
                for cond, res in (
                    ("segment", seg_res),
                    ("uniform", uni_res),
                    ("full", full_res),
                ):
                    r = res[qp]
                    c = score(r["text"], accepted)
                    rows.append(
                        {
                            "stream": s_idx,
                            "scenes": "|".join(trio),
                            "qp": qp,
                            "m_frames": m,
                            "budget": budget,
                            "cond": cond,
                            "gt": label,
                            "answer": r["text"],
                            "correct": c,
                            "kv_bytes": r["kv_bytes"],
                            "ttft_ms": round(r.get("ttft_ms") or 0, 1),
                            "step_ms": round(r.get("step_ms", 0), 1),
                        }
                    )
                    tag = f"b{budget}" if cond != "full" else "full"
                    print(
                        f"    qp{qp:>2} m={m:>2} {cond:8s} {tag:5s} gt={label:9s} "
                        f"{'OK' if c else '..'} kv={r['kv_bytes'] / 1024:7.1f}KB "
                        f"ttft={r.get('ttft_ms') or 0:6.0f}ms  {r['text']!r}"
                    )

    # ---- money table ----
    print("\n" + "=" * 70)
    print("MONEY TABLE — accuracy vs KV budget (real slideshow stream)")
    print("=" * 70)
    print(f"model={model_id}  K={K} frames/scene  tokens/frame~={tpf}")

    def agg(cond, budget=None, only_contrast=False):
        rs = [
            r
            for r in rows
            if r["cond"] == cond and (budget is None or r["budget"] == budget)
        ]
        if only_contrast:
            rs = [r for r in rs if r["m_frames"] > r["budget"]]
        if not rs:
            return None
        return (
            sum(r["correct"] for r in rs) / len(rs),
            np.mean([r["kv_bytes"] for r in rs]) / 1024,
            np.mean([r["ttft_ms"] for r in rs]),
            np.mean([r["step_ms"] for r in rs]),
            len(rs),
        )

    hdr = f"{'condition':<20}{'acc':>7}{'KV(KB)':>10}{'TTFT(ms)':>10}{'step(ms)':>10}{'n':>5}"

    def show(only_contrast):
        print(hdr)
        for budget in budgets:
            for cond in ("segment", "uniform"):
                a = agg(cond, budget, only_contrast)
                if a:
                    print(
                        f"{cond + ' b=' + str(budget):<20}{a[0]:>7.2f}{a[1]:>10.1f}"
                        f"{a[2]:>10.0f}{a[3]:>10.1f}{a[4]:>5d}"
                    )
        af = agg("full", None, only_contrast)
        if af:
            print(
                f"{'full-context':<20}{af[0]:>7.2f}{af[1]:>10.1f}{af[2]:>10.0f}{af[3]:>10.1f}{af[4]:>5d}"
            )

    print("\n-- ALL query points --")
    show(False)
    print(
        "\n-- RECENCY-CONTRAST subset (m_frames > budget: where selection matters) --"
    )
    show(True)

    out = os.path.join(HERE, "phase2_results.json")
    with open(out, "w") as f:
        json.dump(
            {
                "model": model_id,
                "K": K,
                "budgets": budgets,
                "tokens_per_frame": tpf,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\n[saved] {out}  ({len(rows)} rows, {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
