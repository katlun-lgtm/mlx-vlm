# Phase 2b (scaled) — statistically-tested streaming-video KV benchmark

**Model:** `mlx-community/Idefics3-8B-Llama3-8bit` (same `idefics3` arch as SmolVLM → 1-D RoPE →
the Phase-1 `SegmentAwareSession` evict/re-anchor runs unchanged; SmolVLM2-2.2B was too weak to
perceive real motion-video frames — see Phase-2b notes). **Hardware:** M3 Max 64 GB.

## Method
Long streams of **concatenated real CC/PD motion clips** (Wikimedia Commons), 16 subjects/stream from
a **26-subject isolation-filtered pool** (a clip enters only if the model names it correctly shown
alone). Two query types — `recent` ("name the last N=3 scenes") and `control` ("current scene"). Three
conditions at matched KV budget `B∈{6,10,14}` frames: segment-aware streaming, uniform `linspace`
sampling, full-context oracle. Recall scoring (deterministic substring match).
**16 streams × 2 query points × 3 budgets → n=32 pairs/cell.** Paired **Wilcoxon signed-rank**
(segment vs uniform, paired by stream+query-point) computed independently from the raw rows.

## Results (paired, n=32/cell)

### `recent` — the recency discriminator
| B | segment | uniform | diff | 95% CI | Wilcoxon p |
|---|---------|---------|------|--------|-----------|
| 6  | 0.896 | 0.417 | **+0.479** | [+0.411, +0.547] | **0.0000** ★★★ |
| 10 | 0.771 | 0.625 | +0.146 | [+0.040, +0.251] | 0.0147 ★★ |
| 14 | 0.812 | 0.792 | +0.021 | [−0.060, +0.101] | 0.56 ns |

Segment-aware recalls **~2.15× as many recent scenes at tight budget (B=6)**, p<0.0001, and the
gap **decays to a non-significant tie at large budget** — the designed shape (a bigger budget lets
uniform capture more of the recent region, closing the gap).

### `control` — "current scene" (the honest wrinkle)
| B | segment | uniform | diff | 95% CI | Wilcoxon p |
|---|---------|---------|------|--------|-----------|
| 6  | 0.625 | 0.406 | +0.219 | [+0.042, +0.396] | 0.0196 ★★ |
| 10 | 0.656 | 0.469 | +0.188 | [+0.045, +0.330] | 0.0143 ★★ |
| 14 | 0.656 | 0.406 | +0.250 | [+0.091, +0.409] | 0.0047 ★★★ |

The control is **significant at every budget** — so the advantage is **not purely recency-specific**
at scale (in the short-stream Phase-2 pilot the control tied; here it does not).

## Honest decomposition (don't oversell)
Two separable effects:
1. **A general "dense recent window > sparse uniform sampling on long motion streams" effect** —
   budget-independent, ~**+0.22 flat** (this is the control level). On a 16-clip stream, uniform's
   `linspace` reads motion clips worse than segment's contiguous recent window, at any budget.
2. **A recency-SPECIFIC extra on top** — budget-*dependent*: the `recent` gap is +0.48 at B=6 and
   decays to ~0 at B=14, i.e. ~+0.26 of recency-specific advantage at tight budget above the general
   effect, vanishing as the budget grows.

Both are real and significant; the clean "recency-only" story from the pilot is **broader than that**
at scale — reported plainly rather than spun.

## Bounded memory (unchanged from the mechanism)
Segment-aware KV is flat (bounded by the window) as stream length grows; full-context grows O(T)
(~21.9 MB/frame → OOM@64 GB at ~3,000 frames). The 16-stream run held **flat per-stream time**
(623→682 s, mean 682 s) once the MLX buffer cache is cleared between streams.

## Limits
Single model, 26 curated single-subject clips, recall (substring) scoring, n=32/cell, image-splitting
off, slideshow-of-clips concatenation (real intra-clip motion, but scene *cuts* between clips).
Solid and significant, but not a full public streaming-QA suite (StreamingBench/OVO would be the next
escalation).

**Raw:** `examples/phase2b_scaled_results.json` (448 rows). Notes: `/root/mlx-vlm-phase2b-notes.md`.
