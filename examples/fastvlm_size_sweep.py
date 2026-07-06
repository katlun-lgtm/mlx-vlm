#!/usr/bin/env python3
"""FastVLM size sweep on the streaming-video path: 0.5B -> 1.5B -> 7B.

The 0.5B run (``fastvlm_vs_idefics3_stream.py``) proved the bounded-KV loop runs
for FastVLM but showed the model (a) can't do the recency query (derails) and
(b) owes its 7x KV/window win entirely to Qwen2-0.5B's tiny KV geometry, NOT to
FastVLM's vision. This sweep answers the follow-up: does a LARGER FastVLM start
doing the temporal/recency task, and at what KV/window cost?

Reuses the SAME measurement code as the 0.5B run (``sanity``,
``measure_perframe``, ``bounded_loop``) and the SAME concatenated phase2b stream,
so 1.5B and 7B land on the same axes as 0.5B and Idefics3-8B.

Usage:
    python examples/fastvlm_size_sweep.py --ids EZCon/FastVLM-1.5B-mlx --tag 1.5B
    python examples/fastvlm_size_sweep.py --ids InsightKeeper/FastVLM-7B-MLX-8bit --tag 7B
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastvlm_vs_idefics3_stream import (  # noqa: E402
    bounded_loop,
    build_stream,
    measure_perframe,
    sanity,
)

from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.video import _is_llava_family  # noqa: E402


def kv_geometry(model) -> dict:
    """Best-effort read of the text stack's KV geometry (for the notes)."""
    lm = model.language_model
    n_layers = len(lm.layers)
    geo = {"layers": n_layers}
    try:
        attn = lm.layers[0].self_attn
        for name, keys in (
            ("n_kv_heads", ("n_kv_heads", "num_key_value_heads")),
            ("n_heads", ("n_heads", "num_attention_heads", "num_heads")),
            ("head_dim", ("head_dim",)),
        ):
            for k in keys:
                v = getattr(attn, k, None)
                if isinstance(v, int):
                    geo[name] = v
                    break
    except Exception:
        pass
    return geo


def run_one(model_id: str, tag: str, frames, window: int) -> dict:
    print(f"\n{'=' * 72}\n[{tag}] loading {model_id}\n{'=' * 72}", flush=True)
    t0 = time.time()
    model, processor = load(model_id)
    load_s = time.time() - t0
    geo = kv_geometry(model)
    print(
        f"[{tag}] loaded in {load_s:.1f}s  "
        f"llava_family={_is_llava_family(model, processor)}  geo={geo}",
        flush=True,
    )

    san = sanity(model, processor)
    print(f"[{tag}] SANITY ({san['tokens']} tok): {san['answer']!r}", flush=True)

    agg = measure_perframe(model, processor, frames)
    print(
        f"[{tag}] per-frame: enc={agg['encode_ms']:.1f}ms lm={agg['lm_ms']:.1f}ms "
        f"tok={agg['tokens_per_frame']:.0f} kv={agg['kv_bytes_per_frame'] / 1024:.1f}KB "
        f"f/8GB={agg['frames_per_budget'].get('8GB')}",
        flush=True,
    )

    bnd = bounded_loop(model, processor, frames, window=window)
    print(
        f"[{tag}] bounded loop (win={window}): kv flat={bnd['kv_flat']} "
        f"[{bnd['kv_min'] / 1024:.0f}..{bnd['kv_max'] / 1024:.0f} KB]\n"
        f"[{tag}] RECENCY ANSWER: {bnd['answer']!r}",
        flush=True,
    )

    out = {
        "model": model_id,
        "tag": tag,
        "load_s": load_s,
        "family": "llava" if _is_llava_family(model, processor) else "other",
        "geometry": geo,
        "sanity": san,
        "perframe": agg,
        "bounded": bnd,
    }
    del model, processor
    gc.collect()
    mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True, help="MLX FastVLM repo ids")
    ap.add_argument("--tags", nargs="+", default=None, help="short labels per id")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--fpc", type=int, default=3, help="frames per clip")
    ap.add_argument("--window", type=int, default=6, help="vision_window (0.5B used 6)")
    ap.add_argument(
        "--out", default=os.path.join(HERE, "fastvlm_size_sweep_results.json")
    )
    args = ap.parse_args()

    tags = args.tags or [f"m{i}" for i in range(len(args.ids))]
    if len(tags) != len(args.ids):
        raise SystemExit("--tags must match --ids in length")

    frames, labels = build_stream(args.clips, args.fpc)
    print(f"[stream] {len(frames)} frames from {len(labels)} clips: {labels}")
    if len(frames) < 4:
        raise SystemExit("FATAL: not enough frames decoded; check videos_phase2b/.")

    # merge with any prior sweep results so re-running one size keeps the other
    results = {}
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
        except Exception:
            results = {}
    results.setdefault("_meta", {})["stream_labels"] = labels
    results["_meta"]["window"] = args.window

    for mid, tag in zip(args.ids, tags):
        try:
            results[tag] = run_one(mid, tag, frames, args.window)
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] FAILED to run {mid}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            results[tag] = {
                "model": mid,
                "tag": tag,
                "error": f"{type(e).__name__}: {e}",
            }
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
