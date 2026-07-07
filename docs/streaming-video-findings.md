# Bounded-Memory Streaming for On-Device Video VLMs — Findings

*An investigation into segment-aware KV-cache streaming for vision-language models on Apple
Silicon (MLX). Written 2026-07-06. All numbers measured on an M3 Max; every headline was
re-aggregated from raw result JSON, not taken from a run's own summary.*

## TL;DR

We built a bounded-memory streaming path for video VLMs in mlx-vlm — ingest frames one at a
time into a persistent KV cache, keep an attention-sink + the most-recent frames, evict the
oldest *vision* spans, and re-anchor survivors to contiguous RoPE positions. It works, and for a
**capable** model the bounded window **significantly beats uniform frame-sampling at equal KV
budget** on recency questions (+0.48 recall at tight budget, p < 0.0001 on Idefics3-8B).

The tempting shortcut — *"use a tiny fast VLM on-device and lean on the huge window"* — is
**ruled out by measurement.** A small LLM has a bigger window but can't use it (FastVLM-1.5B's
recency recall is ~0.31 whether it sees 1 clip or 3). At **equal memory**, a small model's window
advantage and a big model's capability advantage **cancel** (tie, p = 0.17). **You cannot
substitute window size for LLM capability.** The binding constraint on on-device video
understanding is the LLM's temporal-recall capability — not per-frame perception cost, which is
cheap (~75 ms/frame).

## 1. Motivation

`load_video()` in mlx-vlm uniformly samples up to N frames, stacks them, and feeds them as
independent images. KV then grows **O(T)** with stream length (hundreds of frames of vision
tokens → multi-GB), it re-prefills history to answer, and it cannot ingest an open-ended live
stream at all. We wanted the *eyes* analog of streaming ASR: watch a continuous stream through a
**fixed** KV budget.

## 2. What we built

- **`generate/video.py::stream_generate_video`** — per-frame vision-embed prefill into a
  long-lived `prompt_cache` (reuses `generate_step`'s cache threading), answers over the retained
  window without re-prefilling history.
- **`SegmentAwareSession`** — keep the sink + all text + the N most-recent vision frames; evict
  the oldest *vision* frame; **recompute re-anchor** the retained units to contiguous positions.
- A **family switch** (`_is_llava_family` / `_build_frame_inputs`) so both SmolVLM/Idefics3
  (`<image>` string marker) and FastVLM (LLaVA `image_token_index = -200`) ingest through one
  path. Runs on any **1-D-RoPE** VLM with no model changes.

## 3. Results, phase by phase

**Phase 1 — segment-aware eviction is numerically exact.** `create_causal_mask` assumes
contiguous positions, so you cannot leave a *hole* by evicting a middle vision span. Re-anchoring
the retained units to contiguous positions (a recompute) yields next-token logits **identical to
a fresh prefill of the retained units: max|Δ| = 0.000000.** (The 0.375 residual vs a *monolithic*
forward is pure fp16 prefill-chunking — identical in the no-eviction control.)

**Phase 2 — bounded KV at no accuracy cost (SmolVLM2-2.2B, curated photos).** Segment-aware
matched the full-context oracle's accuracy (100%) at **flat ~98 MB** KV while full-context grew to
~190 MB (O(T)). Did *not* discriminate vs uniform on "current scene" — expected, because uniform
`np.linspace` always includes the last frame.

**Phase 2b — the discriminating result (Idefics3-8B, real motion clips, n=32/cell).** With a
"name the last N scenes" query over long streams and a paired Wilcoxon test:

| budget | segment | uniform | diff | Wilcoxon p |
|---|---|---|---|---|
| tight (B=6) | 0.896 | 0.417 | **+0.479** | **< 0.0001** |
| mid (B=10) | 0.771 | 0.625 | +0.146 | 0.015 |
| large (B=14) | 0.812 | 0.792 | +0.021 | 0.56 (ns) |

The recency gap is large at tight budget and vanishes at large budget — the designed shape. A
"current scene" control is *also* significant (~+0.22, budget-independent), so part of the win is
a general "dense recent window > sparse uniform sampling on long streams," with a recency-specific
component on top. Reported plainly, not as purely recency.

**FastVLM swap — the vision-cost hypothesis, refuted.** We hoped Apple's FastVLM (FastViTHD
encoder) would slash per-frame vision cost. Measured (FastVLM-0.5B vs Idefics3-8B, 15 frames):

| | FastVLM-0.5B | Idefics3-8B | |
|---|---|---|---|
| vision-encode ms/frame | 74.6 | 72.8 | ~same (not 20×) |
| LM-prefill ms/frame | 27.5 | 244.3 | 8.9× |
| tokens/frame | 256 | 175 | *more*, not fewer |
| KV/frame | 3.0 MiB | 21.9 MiB | 7.3× smaller |
| frames per 8 GB | 2,730 | 374 | 7.3× |

Three corrections: vision-encode was **not** faster (Apple's 20× is vs ViT-L/14 at high res, not
vs Idefics3's encoder); FastVLM emitted **more** tokens/frame; the window win was the **small
LLM's KV geometry**, not the vision encoder. And FastVLM-0.5B **refused** the temporal query.

**Size sweep (0.5B / 1.5B / 7B).** KV/frame is exactly `2·L·KV_heads·head_dim·tok·2B` (matched to
the byte). FastVLM's Qwen2 backbone uses aggressive GQA (2 KV heads at ≤1.5B, 4 at 7B) vs
Llama3-8B's 8, so its KV is smaller at every size:

| model | KV heads | prefill ms | KV/frame | frames/8 GB | recency query |
|---|---|---|---|---|---|
| FastVLM-0.5B | 2 | 27.5 | 3.0 MiB | 2,730 | **refuses** |
| FastVLM-1.5B | 2 | 73.3 | 7.0 MiB | 1,170 | engages |
| FastVLM-7B | 4 | 309 | 14.0 MiB | 585 | engages |
| Idefics3-8B | 8 | 244 | 21.9 MiB | 374 | engages |

Temporal capability turns on **between 0.5B and 1.5B**. 1.5B looked like a capable-and-small-KV
sweet spot — but "engages" only means *attempts the format*, not *answers correctly*.

**Phase 2c — the decisive equal-memory accuracy test.** Precision-matched (both 8-bit weights),
20-subject isolation *intersection*, fixed KV-byte budget (B = 48 MiB → FastVLM 3-clip window vs
Idefics3 1-clip window), recency recall, n = 24:

| model | segment (equal mem) | uniform | full (oracle) |
|---|---|---|---|
| FastVLM-1.5B-8bit | 0.319 [.242,.397] | 0.278 | 0.306 |
| Idefics3-8B-8bit | 0.250 [.175,.325] | 0.278 | **0.792** [.701,.883] |

Paired (segment, equal memory): FastVLM − Idefics3 = **+0.069, CI [−0.032, +0.171], Wilcoxon
p = 0.17 → not significant → TIE.**

The mechanism is the lesson: **FastVLM-1.5B's `full ≈ segment ≈ uniform ≈ 0.31`** — its recency
recall is flat regardless of context, i.e. its **capability ceiling is ~0.31 and the 3× window is
wasted.** Idefics3-8B's **full-context recall is 0.79** (a capable model nails recency with all
frames) but, starved to one clip at equal memory, collapses to 0.25. The forces cancel.

## 4. Key findings

1. **The streaming mechanism works — for a capable model.** Bounded segment-aware streaming
   significantly beats uniform sampling at equal budget on recency (Phase 2b), and re-anchoring is
   numerically exact (Phase 1).
2. **Per-frame vision cost is not the bottleneck** (~75 ms/frame, and encoder-independent across
   FastVLM sizes). Optimizing the vision encoder does not help the streaming problem.
3. **The binding constraint is LLM recency capability**, which lives in large (large-KV) models.
   Idefics3-8B full-context recall 0.79 vs FastVLM-1.5B 0.31 is the whole story.
4. **No capable-and-small-KV free lunch.** GQA + small size buys a bigger window, but a small
   model can't exploit it. You **cannot trade window size for capability** (equal-memory tie).
5. **So the wedge is bounded-memory streaming *for a capable model*** — a way to approach the
   full-context ceiling under fixed memory — **not a downsizing trick** for weak on-device models.

## 5. Limitations (honest)

- "Video" = concatenated real motion clips with scene *cuts*, not continuous camera motion.
- Recall via deterministic substring match; n = 24–32 per cell (directional/significant, not a
  large public suite like StreamingBench/OVO).
- 1-D-RoPE models only; Qwen2.5-VL M-RoPE (3-D video positions) is future work.
- Re-anchor is a **recompute** (O(window), bounded); an in-place rotary-shift is the efficiency
  follow-up (the `StreamingVideoCache` class is the seat for it).
- Curated, isolation-filtered clips; single-stream.

## 6. Reproducibility

Harnesses (all in `examples/`): `phase2b_scaled.py` (scaled recency-recall + Wilcoxon),
`fastvlm_size_sweep.py` (0.5B/1.5B/7B cost + capability), `phase2c_equal_memory.py` (the decisive
equal-memory test), plus `segment_aware_demo.py` / `streaming_video_demo.py`. Raw results are the
committed `*_results.json`. Models: SmolVLM2-{500M,2.2B}, Idefics3-8B-Llama3-8bit, FastVLM-{0.5,
1.5,7}B. See `docs/streaming-video-kv-plan.md` for the design and `docs/phase2b-scaled-results.md`
/ `docs/phase2-benchmark-results.md` for the per-phase writeups.

## 7. What would change the conclusion

An *accuracy* win for the small-model-plus-big-window path would need a **small model whose
recency ceiling is high** — i.e. a compact VLM explicitly trained on streaming/temporal video
(the training gap, not a systems gap). Absent that, the practical recipe for on-device streaming
video is: **pick a capable-enough model and bound its memory with segment-aware streaming**, not
shrink the model to grow the window.
