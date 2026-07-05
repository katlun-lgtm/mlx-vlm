#!/usr/bin/env python3
"""Tracer-bullet demo for the streaming-video KV-cache path (SmolVLM2).

Proves the loop in ``mlx_vlm/generate/video.py``:

  * builds a synthetic clip that is a SOLID RED frame for most of the clip and
    turns SOLID BLUE in the last few frames (a checkable "what just happened"
    signal a small VLM can actually discriminate),
  * streams it FRAME BY FRAME through a bounded RotatingKVCache (attention-sink
    + sliding window) and prints per-frame KV bytes + ingest latency,
  * asks "what is the dominant color?" and prints the model's answer,
  * runs the SAME ingest loop through an UNBOUNDED KVCache (the uniform /
    load_video-style baseline) and prints its KV bytes + answer.

Definition of done (plan §3): streaming KV stays flat/bounded while the
baseline grows linearly, per-frame LM latency stays flat, and the streaming
answer names the RECENT color (BLUE) — the event a bounded recent window keeps,
where the uniform baseline (dominated by the red history) tends to say RED.

Run on Apple Silicon:
    python examples/streaming_video_demo.py [model_repo]
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from mlx_vlm import load
from mlx_vlm.generate.video import (
    _answer,
    _ingest_frame,
    kv_bytes,
    make_streaming_cache,
    open_stream,
)
from mlx_vlm.models import cache as _cache
from mlx_vlm.streaming import stream_video_frames

MODEL = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "mlx-community/SmolVLM2-500M-Video-Instruct-mlx"
)

TOTAL_FRAMES = 48
CHANGE_AT = 40  # frames 0..39 = RED, 40..47 = BLUE
FRAME_SIZE = 384
FPS = 10
QUESTION = "What is the dominant color? Answer with one word."
CLIP_PATH = "/tmp/streaming_demo_clip.mp4"

# OpenCV writes BGR.
RED_BGR = (0, 0, 220)
BLUE_BGR = (220, 0, 0)


def build_clip(path: str) -> None:
    """Write a synthetic clip: SOLID RED, then SOLID BLUE at the end."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (FRAME_SIZE, FRAME_SIZE))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (mp4v codec missing?)")
    for i in range(TOTAL_FRAMES):
        color = BLUE_BGR if i >= CHANGE_AT else RED_BGR
        img = np.full((FRAME_SIZE, FRAME_SIZE, 3), color, dtype=np.uint8)
        writer.write(img)
    writer.release()
    print(
        f"[clip] wrote {TOTAL_FRAMES} frames to {path} "
        f"(RED 0..{CHANGE_AT - 1}, BLUE {CHANGE_AT}..{TOTAL_FRAMES - 1})"
    )


def run_loop(model, processor, frames, cache, question, label):
    """Open the user turn, ingest every frame, recording per-frame stats."""
    print(f"\n=== {label} ===")
    suffix = open_stream(model, processor, question, cache)
    print(
        f"{'frame':>5} {'tok':>4} {'kv_bytes':>10} {'kv_KB':>8} "
        f"{'encode_ms':>9} {'lm_ms':>7}"
    )
    series = []
    for i, frame in enumerate(frames):
        st = _ingest_frame(model, processor, frame, cache)
        series.append(st)
        print(
            f"{i:>5} {st['n_tokens']:>4} {st['kv_bytes']:>10} "
            f"{st['kv_bytes'] / 1024:>8.1f} {st['encode_ms']:>9.1f} "
            f"{st['lm_ms']:>7.1f}"
        )
    return series, suffix


def main() -> None:
    build_clip(CLIP_PATH)

    print(f"[load] {MODEL}")
    model, processor = load(MODEL)

    frames0 = list(stream_video_frames(CLIP_PATH, fps=FPS, max_frames=TOTAL_FRAMES))
    print(f"[frames] decoded {len(frames0)} frames from clip")

    # Probe one frame to size the window in real tokens/frame.
    image_token = getattr(processor, "image_token", "<image>")
    probe = processor(
        text=[image_token],
        images=[frames0[0]],
        add_special_tokens=True,
        return_tensors="mlx",
    )
    tpf = int(probe["input_ids"].shape[1])
    sink = tpf  # pin ~1 frame as the attention sink
    k_recent = 12  # sliding window holds ~12 recent frames
    window = sink + k_recent * tpf
    print(
        f"[cache] tokens/frame≈{tpf} -> sink={sink}, window(max_size)={window} "
        f"(~{k_recent} recent frames)"
    )

    # ---- Streaming: bounded RotatingKVCache (sink + sliding window) ----
    stream_cache = make_streaming_cache(model, sink=sink, window=window)
    s_series, s_suffix = run_loop(
        model, processor, frames0, stream_cache, QUESTION, "STREAMING (bounded)"
    )
    t0 = time.perf_counter()
    s_ans = _answer(model, processor, s_suffix, stream_cache, max_tokens=16)
    s_ans_wall = (time.perf_counter() - t0) * 1000.0

    # ---- Baseline: unbounded KVCache (uniform / load_video style) ----
    base_cache = _cache.make_prompt_cache(model.language_model)
    b_series, b_suffix = run_loop(
        model, processor, frames0, base_cache, QUESTION, "BASELINE (unbounded)"
    )
    t0 = time.perf_counter()
    b_ans = _answer(model, processor, b_suffix, base_cache, max_tokens=16)
    b_ans_wall = (time.perf_counter() - t0) * 1000.0

    # ---- Summary ----
    def kb(x):
        return f"{x / 1024:.1f} KB"

    n = len(s_series)
    print("\n" + "=" * 62)
    print("SUMMARY — tracer-bullet definition of done")
    print("=" * 62)
    print(f"frames streamed: {n}   tokens/frame≈{tpf}")
    print(f"\nKV bytes  @frame1 -> @frame{n}:")
    print(
        f"  streaming : {kb(s_series[0]['kv_bytes'])} -> "
        f"{kb(s_series[-1]['kv_bytes'])}  (bound {kb(kv_bytes(stream_cache))})"
    )
    print(
        f"  baseline  : {kb(b_series[0]['kv_bytes'])} -> {kb(b_series[-1]['kv_bytes'])}"
    )
    ratio = b_series[-1]["kv_bytes"] / max(1, s_series[-1]["kv_bytes"])
    print(f"  baseline/streaming @frame{n}: {ratio:.2f}x")

    def avg(series, key, lo, hi):
        vals = [s[key] for s in series[lo:hi]]
        return sum(vals) / max(1, len(vals))

    half = n // 2
    print(
        "\nper-frame LM-prefill ms (attention over history) "
        "— first half vs second half:"
    )
    print(
        f"  streaming : {avg(s_series, 'lm_ms', 0, half):.1f} -> "
        f"{avg(s_series, 'lm_ms', half, n):.1f}"
    )
    print(
        f"  baseline  : {avg(b_series, 'lm_ms', 0, half):.1f} -> "
        f"{avg(b_series, 'lm_ms', half, n):.1f}"
    )

    print("\nanswers to:", QUESTION)
    print(
        f"  streaming -> {s_ans['text']!r}  "
        f"(ttft {s_ans['ttft_ms']:.0f}ms, {s_ans['answer_tokens']} tok, "
        f"wall {s_ans_wall:.0f}ms)"
    )
    print(
        f"  baseline  -> {b_ans['text']!r}  "
        f"(ttft {b_ans['ttft_ms']:.0f}ms, {b_ans['answer_tokens']} tok, "
        f"wall {b_ans_wall:.0f}ms)"
    )
    s_txt = s_ans["text"].lower()
    # "Bounded" = streaming KV flatlines (2nd-half value == final value) while
    # the baseline keeps growing (ratio > 1.5).
    flat = s_series[half]["kv_bytes"] == s_series[-1]["kv_bytes"]
    lm_flat = avg(s_series, "lm_ms", 0, half) * 1.5 >= avg(s_series, "lm_ms", half, n)
    print("\nverdict:")
    print(
        f"  bounded KV (flat + <baseline): {'PASS' if flat and ratio > 1.5 else 'CHECK'}"
    )
    print(f"  flat per-frame LM latency    : {'PASS' if lm_flat else 'CHECK'}")
    print(
        f"  recent-event answer (BLUE)   : "
        f"{'PASS' if 'blue' in s_txt else 'CHECK -> ' + repr(s_ans['text'])}"
    )
    print("=" * 62)


if __name__ == "__main__":
    main()
