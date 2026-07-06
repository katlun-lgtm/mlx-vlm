#!/usr/bin/env python3
"""FastVLM-0.5B vs Idefics3-8B — streaming-video PER-FRAME COST measurement.

The deliverable for wiring Apple's FastVLM (LLaVA-style, ``image_token_index=-200``)
into the ``generate/video.py`` bounded-KV streaming path next to the existing
SmolVLM/Idefics3 (``<image>`` string-marker) path. Both families now flow through
the SAME ``get_input_embeddings -> language_model(cache=)`` ingest mechanism;
``_build_frame_inputs`` in ``generate/video.py`` is the only per-family branch.

This harness MEASURES (all numbers real, nothing derived from spec):

  * vision-encode ms/frame  (``get_input_embeddings`` = vision tower + projector)
  * LM-prefill ms/frame     (one ``language_model`` forward appending frame KV)
  * vision tokens/frame     (KV positions actually appended)
  * KV bytes/frame          (measured cache-size delta, steady state)
  * effective window        (frames that fit a fixed KV byte budget)
  * a bounded-KV loop run    (SegmentAwareSession, flat KV) + a sample answer

Both models see the SAME real frames (a few phase2b Wikimedia clips concatenated).
Idefics3 is configured as in phase2b (no image-splitting, longest_edge=512) — its
sane streaming config; FastVLM has no such knob (fixed 1024 square input).

Usage:
    python examples/fastvlm_vs_idefics3_stream.py --clips 5 --fpc 3
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from phase2b_benchmark import (  # noqa: E402
    CLIPS_DIR,
    RECENT_Q,
    RECENT_TOK,
    configure_model,
    sample_clip_frames,
)

from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.video import (  # noqa: E402
    SegmentAwareSession,
    _answer,
    _ingest_frame,
    _is_llava_family,
    kv_bytes,
    open_stream,
)
from mlx_vlm.models import cache as _cache  # noqa: E402

MODELS = {
    "fastvlm": "mlx-community/FastVLM-0.5B-bf16",
    "idefics3": "mlx-community/Idefics3-8B-Llama3-8bit",
}
ISO_MANIFEST = os.path.join(HERE, "phase2b_isolation_pass.json")
SANITY_PHOTO = os.path.join(HERE, "real_scenes", "zebra.jpg")
BUDGETS_GB = [1.0, 8.0]  # KV byte budgets for the effective-window column


def build_stream(n_clips: int, fpc: int):
    """Concatenate frames from the first ``n_clips`` isolation-pass clips."""
    man = json.load(open(ISO_MANIFEST))
    frames, labels = [], []
    for rec in man["clips"]:
        if len(labels) >= n_clips:
            break
        fp = os.path.join(CLIPS_DIR, rec["file"])
        if not os.path.exists(fp):
            continue
        fr = sample_clip_frames(fp, fpc)
        if len(fr) < fpc:
            continue
        frames.extend(fr)
        labels.append(rec["label"])
    return frames, labels


def sanity(model, processor) -> dict:
    """Single real photo through the STREAMING ingest path (model + adapter)."""
    img = Image.open(SANITY_PHOTO).convert("RGB")
    cache = _cache.make_prompt_cache(model.language_model)
    suffix = open_stream(
        model, processor, "Describe this image in one short sentence.", cache
    )
    st = _ingest_frame(model, processor, img, cache)
    ans = _answer(model, processor, suffix, cache, 40)
    return {"tokens": st["n_tokens"], "answer": ans["text"]}


def measure_perframe(model, processor, frames, warmup: int = 2) -> dict:
    """Ingest frames into a fresh DENSE cache; per-frame timing + KV delta."""
    cache = _cache.make_prompt_cache(model.language_model)
    open_stream(model, processor, RECENT_Q, cache)
    prev_kv = kv_bytes(cache)
    rows = []
    for fr in frames:
        st = _ingest_frame(model, processor, fr, cache)
        st["kv_delta"] = st["kv_bytes"] - prev_kv
        prev_kv = st["kv_bytes"]
        rows.append(st)
    ss = rows[warmup:] or rows
    kv_pf = float(np.mean([s["kv_delta"] for s in ss]))
    agg = {
        "n_frames": len(frames),
        "encode_ms": float(np.mean([s["encode_ms"] for s in ss])),
        "lm_ms": float(np.mean([s["lm_ms"] for s in ss])),
        "tokens_per_frame": float(np.mean([s["n_tokens"] for s in ss])),
        "kv_bytes_per_frame": kv_pf,
        "kv_total_bytes": int(rows[-1]["kv_bytes"]),
        "frames_per_budget": {
            f"{gb:g}GB": int((gb * 1024**3) // kv_pf) if kv_pf > 0 else None
            for gb in BUDGETS_GB
        },
        "per_frame_rows": [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
            for r in rows
        ],
    }
    return agg


def bounded_loop(model, processor, frames, window: int = 6) -> dict:
    """Run the bounded-KV SegmentAwareSession; confirm flat KV + get an answer."""
    sess = SegmentAwareSession(
        model, processor, vision_window=window, keep_first_frame=False
    )
    sess.begin(RECENT_Q)
    kvs = []
    for fr in frames:
        s = sess.ingest_frame(fr)
        kvs.append(s["kv_bytes"])
    ans = sess.answer(RECENT_TOK)
    tail = kvs[window + 1 :] or kvs  # steady state after the window fills
    return {
        "window": window,
        "answer": ans["text"],
        "kv_min": int(min(tail)),
        "kv_max": int(max(tail)),
        "kv_flat": (max(tail) - min(tail)) / max(1, min(tail)) < 0.02,
        "kv_trace": kvs,
    }


def run_model(key: str, frames, labels) -> dict:
    mid = MODELS[key]
    print(f"\n{'=' * 70}\n[{key}] loading {mid}\n{'=' * 70}")
    t0 = time.time()
    model, processor = load(mid)
    if key == "idefics3":
        configure_model(model, processor)  # no split, longest_edge=512
    print(
        f"[{key}] loaded in {time.time() - t0:.1f}s  "
        f"llava_family={_is_llava_family(model, processor)}  "
        f"layers={len(model.language_model.layers)}"
    )

    san = sanity(model, processor)
    print(f"[{key}] SANITY ({san['tokens']} tok): {san['answer']!r}")

    agg = measure_perframe(model, processor, frames)
    print(
        f"[{key}] per-frame: enc={agg['encode_ms']:.1f}ms lm={agg['lm_ms']:.1f}ms "
        f"tok={agg['tokens_per_frame']:.0f} kv={agg['kv_bytes_per_frame'] / 1024:.1f}KB"
    )

    bnd = bounded_loop(model, processor, frames, window=6)
    print(
        f"[{key}] bounded loop (win=6): kv flat={bnd['kv_flat']} "
        f"[{bnd['kv_min'] / 1024:.0f}..{bnd['kv_max'] / 1024:.0f} KB]  "
        f"answer={bnd['answer']!r}"
    )

    out = {
        "model": mid,
        "family": "llava" if _is_llava_family(model, processor) else "smolvlm",
        "sanity": san,
        "perframe": agg,
        "bounded": bnd,
    }
    del model, processor
    gc.collect()
    mx.clear_cache()
    return out


def print_table(results: dict, labels):
    def g(k, path):
        d = results[k]
        for p in path:
            d = d[p]
        return d

    print("\n" + "=" * 74)
    print("FastVLM-0.5B  vs  Idefics3-8B  — streaming per-frame cost")
    print(
        f"stream: {len(labels)} clips concatenated -> "
        f"{results['fastvlm']['perframe']['n_frames']} frames  ({', '.join(labels)})"
    )
    print("=" * 74)
    hdr = f"{'metric':<26}{'FastVLM-0.5B':>18}{'Idefics3-8B':>18}"
    print(hdr)
    print("-" * len(hdr))

    def row(name, k_fmt):
        f = k_fmt(g("fastvlm", ["perframe"]))
        i = k_fmt(g("idefics3", ["perframe"]))
        print(f"{name:<26}{f:>18}{i:>18}")

    row("vision-encode ms/frame", lambda p: f"{p['encode_ms']:.1f}")
    row("LM-prefill ms/frame", lambda p: f"{p['lm_ms']:.1f}")
    row("vision tokens/frame", lambda p: f"{p['tokens_per_frame']:.0f}")
    row("KV bytes/frame", lambda p: f"{p['kv_bytes_per_frame'] / 1024:.1f} KB")
    for gb in BUDGETS_GB:
        row(
            f"frames per {gb:g} GB KV",
            lambda p, gb=gb: f"{p['frames_per_budget'][f'{gb:g}GB']:,}",
        )
    print("-" * len(hdr))
    print(
        f"{'bounded-KV loop ran':<26}"
        f"{str(g('fastvlm', ['bounded', 'kv_flat'])):>18}"
        f"{str(g('idefics3', ['bounded', 'kv_flat'])):>18}"
    )
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--fpc", type=int, default=3, help="frames per clip")
    ap.add_argument("--only", choices=["fastvlm", "idefics3"], default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "fastvlm_stream_results.json"))
    args = ap.parse_args()

    frames, labels = build_stream(args.clips, args.fpc)
    print(f"[stream] {len(frames)} frames from {len(labels)} clips: {labels}")
    if len(frames) < 4:
        print("FATAL: not enough frames decoded; check videos_phase2b/ + manifest.")
        return

    keys = [args.only] if args.only else ["fastvlm", "idefics3"]
    results = {}
    for k in keys:
        results[k] = run_model(k, frames, labels)
        json.dump(results, open(args.out, "w"), indent=2)

    if len(results) == 2:
        print_table(results, labels)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
