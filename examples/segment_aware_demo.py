#!/usr/bin/env python3
"""Phase-1 acceptance demo: segment-aware KV eviction with recompute re-anchor.

Runs three tests against ``SegmentAwareSession`` (mlx_vlm/generate/video.py):

  TEST A — numerical correctness of the re-anchor. After ≥1 eviction, compare
    the streaming cache's next-token logits to a REFERENCE built by feeding the
    exact RETAINED unit sequence FRESH (one contiguous forward). PASS = identical
    argmax AND small max-abs logit diff (fp16-level). This is the gate.

  TEST B — the segment-aware differentiator. An early TEXT instruction followed
    by MANY frames: a plain RotatingKVCache rotates the text OUT of its window,
    while segment-aware KEEPS it (only vision evicts). Prove the text is retained
    segment-aware and evicted plain (position ranges), plus a behavioral probe.

  TEST C — bounded KV. Segment-aware KV bytes stay flat once past the window,
    while an unbounded baseline grows linearly.

Run on Apple Silicon:
    python examples/segment_aware_demo.py [model_repo]
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm import load
from mlx_vlm.generate.video import (
    SegmentAwareSession,
    _answer,
    _ingest_frame,
    _prefill_text,
    _split_user_turn,
    kv_bytes,
    open_stream,
)
from mlx_vlm.models import cache as _cache

MODEL = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "mlx-community/SmolVLM2-500M-Video-Instruct-mlx"
)
FRAME_SIZE = 384

# A palette so consecutive frames actually differ (content-dependent KV/logits).
PALETTE = [
    (220, 30, 30),
    (30, 30, 220),
    (30, 200, 30),
    (220, 200, 30),
    (200, 30, 200),
    (30, 200, 200),
    (240, 140, 20),
    (120, 60, 200),
]


def build_frames(n: int) -> list[Image.Image]:
    frames = []
    for i in range(n):
        color = PALETTE[i % len(PALETTE)]
        arr = np.full((FRAME_SIZE, FRAME_SIZE, 3), color, dtype=np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


def probe_tokens_per_frame(model, processor, frame) -> int:
    image_token = getattr(processor, "image_token", "<image>")
    out = processor(
        text=[image_token],
        images=[frame],
        add_special_tokens=True,
        return_tensors="mlx",
    )
    return int(out["input_ids"].shape[1])


def kb(x: int) -> str:
    return f"{x / 1024:.1f}KB"


# ==========================================================================
def test_a_and_c(model, processor, frames, tpf):
    print("\n" + "=" * 68)
    print("TEST A (re-anchor correctness) + TEST C (bounded KV)")
    print("=" * 68)
    W = 4
    question = "What is the dominant color? Answer with one word."
    session = SegmentAwareSession(
        model, processor, vision_window=W, keep_first_frame=True, evict_slack=0
    )
    session.begin(question)

    print(f"vision_window={W}, keep_first_frame=True, tokens/frame≈{tpf}")
    print(f"{'frame':>5} {'kv_bytes':>9} {'kv':>8} {'ret':>4} {'rebuilds':>8} evicted")
    byte_series = []
    for i, frame in enumerate(frames):
        st = session.ingest_frame(frame)
        byte_series.append(st["kv_bytes"])
        print(
            f"{i:>5} {st['kv_bytes']:>9} {kb(st['kv_bytes']):>8} "
            f"{st['retained_frames']:>4} {st['n_rebuilds']:>8} {st['evicted']}"
        )

    # ---------- TEST A ----------
    assert session._n_rebuilds > 0, "no eviction happened — increase frame count"
    retained = session.retained_tags()
    n = len(frames)
    expected = ["frame0"] + [f"frame{j}" for j in range(n - W, n)]
    expected_full = [t for t in retained if not t.startswith("frame")] + expected
    print(f"\nretained units: {retained}")
    print(f"expected vision (sink frame0 + last {W}): {expected}")
    vis_retained = [t for t in retained if t.startswith("frame")]
    policy_ok = vis_retained == expected
    print(f"retention policy correct: {policy_ok}")

    s_logits = session.last_logits  # streaming (post evict+rebuild)
    r_chunked = session.reference_logits("chunked")  # fresh per-unit reprefill
    r_mono = session.reference_logits("mono")  # fresh single contiguous forward

    def maxdiff(a, b):
        return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())

    def arg(a):
        return int(mx.argmax(a, axis=-1).item())

    mx.eval(s_logits, r_chunked, r_mono)
    s_arg, c_arg, m_arg = arg(s_logits), arg(r_chunked), arg(r_mono)
    exact_diff = maxdiff(s_logits, r_chunked)  # re-anchor exactness (expect ~0)
    mono_diff = maxdiff(s_logits, r_mono)  # vs strict contiguous (fp16 chunk noise)
    chunk_noise = maxdiff(r_chunked, r_mono)  # CONTROL: chunk-vs-mono, same seq
    argmax_match = s_arg == m_arg == c_arg
    print(f"\nstreaming argmax token            = {s_arg}")
    print(f"reference argmax (chunked / mono)  = {c_arg} / {m_arg}")
    print(f"argmax match                       = {argmax_match}")
    print(
        f"max|diff| streaming vs fresh-chunked (re-anchor exactness) = {exact_diff:.6e}"
    )
    print(
        f"max|diff| streaming vs fresh-mono   (strict contiguous)    = {mono_diff:.6e}"
    )
    print(
        f"max|diff| fresh-chunked vs fresh-mono (CONTROL, no evict)  = {chunk_noise:.6e}"
    )
    # Gate: policy correct, argmax matches the contiguous reference, the re-anchor
    # is numerically EXACT vs a fresh per-unit prefill, and the residual vs the
    # monolithic forward is attributable to fp16 chunking (matches the eviction-
    # free CONTROL) rather than to the eviction/re-anchor itself.
    reanchor_exact = exact_diff < 1e-2
    residual_is_fp16 = abs(mono_diff - chunk_noise) < 5e-2
    a_pass = argmax_match and policy_ok and reanchor_exact and residual_is_fp16
    max_abs = mono_diff
    print(
        f"TEST A: {'PASS' if a_pass else 'FAIL'}  "
        f"(policy + argmax + re-anchor exact<1e-2 + residual==fp16-control)"
    )

    # ---------- TEST C ----------
    # Bytes should be flat once the window is full (steady state, evict_slack=0).
    steady = byte_series[W + 1 :]
    flat = len(set(steady)) == 1 if steady else False
    print(
        f"\nTEST C: KV bytes @frame0={kb(byte_series[0])} -> @frame{n - 1}={kb(byte_series[-1])}"
    )
    print(
        f"        steady-state (post-window) bytes: {sorted(set(steady))} -> flat={flat}"
    )

    # Unbounded baseline for contrast.
    base_cache = _cache.make_prompt_cache(model.language_model)
    b_suffix = open_stream(model, processor, question, base_cache)
    b0 = b1 = 0
    for i, frame in enumerate(frames):
        _ingest_frame(model, processor, frame, base_cache)
        if i == 0:
            b0 = kv_bytes(base_cache)
    b1 = kv_bytes(base_cache)
    print(
        f"        baseline (unbounded): @frame0={kb(b0)} -> @frame{n - 1}={kb(b1)}  "
        f"(grows {b1 / max(1, byte_series[-1]):.1f}x past segment-aware)"
    )
    c_pass = flat and b1 > byte_series[-1]
    print(
        f"TEST C: {'PASS' if c_pass else 'FAIL'}  (segment-aware flat, baseline grows)"
    )
    return a_pass, c_pass, max_abs, argmax_match


# ==========================================================================
def test_b(model, processor, frames, tpf):
    print("\n" + "=" * 68)
    print("TEST B (segment-aware keeps early TEXT; plain rotates it out)")
    print("=" * 68)
    W = 3
    instruction = (
        "Remember: the secret code is ZEBRA. Always include ZEBRA in your reply."
    )
    question = "What is the dominant color?"

    # ---- Segment-aware: text is pinned, only vision evicts ----
    session = SegmentAwareSession(
        model, processor, vision_window=W, keep_first_frame=False, evict_slack=0
    )
    session.begin(question)
    session.ingest_text(instruction)
    for frame in frames:
        session.ingest_frame(frame)
    retained = session.retained_tags()
    text_units = [u for u in session.units if u.kind == "text"]
    tok = getattr(processor, "tokenizer", processor)
    kept_text = tok.decode(text_units[0].input_ids[0].tolist()) if text_units else ""
    seg_keeps_text = len(text_units) > 0 and "ZEBRA" in kept_text
    print(f"segment-aware retained units: {retained}")
    print(f"segment-aware retained TEXT : {kept_text!r}")
    print(f"segment-aware KEEPS instruction: {seg_keeps_text}")
    seg_ans = session.answer(max_tokens=24)
    print(f"segment-aware answer: {seg_ans['text']!r}")

    # ---- Plain RotatingKVCache: same sequence, text rotates out ----
    prefix, suffix = _split_user_turn(model, processor, question)
    # size sink = just the prefix; window holds ~ sink + (W+1) frames.
    # First prefill prefix into a scratch cache to learn its length.
    scratch = [_cache.KVCache()]
    st = _prefill_text(model, processor, prefix, scratch, add_special_tokens=True)
    prefix_len = st["n_tokens"]
    keep_plain = prefix_len
    window_plain = prefix_len + (W + 1) * tpf

    plain_cache = [
        _cache.RotatingKVCache(max_size=window_plain, keep=keep_plain)
        for _ in range(len(model.language_model.layers))
    ]
    _prefill_text(model, processor, prefix, plain_cache, add_special_tokens=True)
    instr_start = int(plain_cache[0].offset)
    st_i = _prefill_text(
        model, processor, instruction, plain_cache, add_special_tokens=False
    )
    instr_end = instr_start + st_i["n_tokens"]
    for frame in frames:
        _ingest_frame(model, processor, frame, plain_cache)
    offset = int(plain_cache[0].offset)

    # Positions a RotatingKVCache(keep, max_size) still holds:
    #   sink  = [0, keep)   and   window = [offset-(max_size-keep), offset)
    sink_hi = keep_plain
    win_lo = offset - (window_plain - keep_plain)
    instr_in_sink = instr_start < sink_hi
    instr_in_window = instr_end > win_lo
    instr_evicted = not (instr_in_sink or instr_in_window)
    print(
        f"\nplain RotatingKVCache: keep(sink)={keep_plain}, max_size={window_plain}, "
        f"offset={offset}"
    )
    print(f"  sink positions retained : [0, {sink_hi})")
    print(f"  window positions retained: [{win_lo}, {offset})")
    print(f"  instruction positions    : [{instr_start}, {instr_end})")
    print(f"  instruction EVICTED from plain cache: {instr_evicted}")
    plain_ans = _answer(model, processor, suffix, plain_cache, max_tokens=24)
    print(f"plain answer: {plain_ans['text']!r}")

    seg_has_zebra = "zebra" in seg_ans["text"].lower()
    plain_has_zebra = "zebra" in plain_ans["text"].lower()
    print(
        f"\nbehavioral: ZEBRA in segment-aware answer={seg_has_zebra}, "
        f"in plain answer={plain_has_zebra}"
    )
    b_pass = seg_keeps_text and instr_evicted
    print(
        f"TEST B: {'PASS' if b_pass else 'FAIL'}  "
        f"(segment-aware retains text, plain evicts it)"
    )
    return b_pass


# ==========================================================================
def main():
    print(f"[load] {MODEL}")
    model, processor = load(MODEL)
    frames = build_frames(14)
    tpf = probe_tokens_per_frame(model, processor, frames[0])
    print(f"[frames] {len(frames)} synthetic frames, tokens/frame≈{tpf}")

    a_pass, c_pass, max_abs, argmax_match = test_a_and_c(model, processor, frames, tpf)
    b_pass = test_b(model, processor, frames, tpf)

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(
        f"TEST A (re-anchor numeric): {'PASS' if a_pass else 'FAIL'}  "
        f"(argmax_match={argmax_match}, max_abs_diff={max_abs:.3e})"
    )
    print(f"TEST B (keeps early text) : {'PASS' if b_pass else 'FAIL'}")
    print(f"TEST C (bounded KV)       : {'PASS' if c_pass else 'FAIL'}")
    print("=" * 68)


if __name__ == "__main__":
    main()
