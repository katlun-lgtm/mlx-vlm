# Phase 2 — Real-imagery streaming-video KV benchmark (bigger model)

**Model:** `mlx-community/SmolVLM2-2.2B-Instruct-mlx` (bf16, 24-layer LM) — verified to perceive
real objects on a single image ("a cat laying on a pink couch…"), i.e. genuinely more capable than
the 500M color-only tracer-bullet model.
**Hardware:** M3 Max 64 GB. **Harness:** `examples/phase2_benchmark.py` (raw: `phase2_results.json`).

## Method (honest)
Content is a controlled **real-photo slideshow stream** (Wikimedia CC/PD; re-fetch with
`fetch_real_photos.py`) — *not* motion video. Each scene = one real photo × 4 frames ("a shot"),
3 shots per stream, query at each shot boundary about the **current** scene. A scene enters the pool
only if the model names it correctly when shown **alone** (an *isolation filter*), so any accuracy gap
comes from the **KV selection policy**, not perception failure. Three conditions at matched KV budget
`B ∈ {4, 8}` frames: **(a) segment-aware streaming** (keep sink + B most-recent frames, evict oldest
vision, recompute re-anchor), **(b) uniform sampling** to the same B, **(c) full context** (all frames;
accuracy oracle, KV grows O(T)). Scoring = deterministic substring match vs ground truth. Size: 5
streams × 3 query points × 2 budgets × 3 conditions = **90 evaluations** (small, honest, real).

## Money table
Figure: `examples/phase2_money_table.png` (bounded KV / accuracy-vs-budget / flat latency).

| budget | condition | accuracy | KV bytes (min..max) |
|---|---|---|---|
| 4 | segment-aware | **15/15 = 100%** | 98,112 KB (flat) |
| 4 | uniform | 14/15 = 93% | 98,112 KB (flat) |
| 4 | full-context | 15/15 = 100% | 98,112 → **194,880 KB (O(T))** |
| 8 | segment-aware | **15/15 = 100%** | 98,112 → 146,496 KB |
| 8 | uniform | 15/15 = 100% | 98,112 → 146,496 KB |
| 8 | full-context | 15/15 = 100% | 98,112 → **194,880 KB (O(T))** |

## What this shows — and what it doesn't
- **Strong, robust result:** segment-aware **matches the full-context oracle's accuracy (100%) at
  bounded KV** — flat memory where full-context grows O(T) toward ~190 MB. Bounded memory at *no
  accuracy cost* vs the oracle is the headline.
- **Directional, NOT strong:** segment-aware ≥ uniform at equal budget (100/100 and 100/93). The B=4
  edge is a **single item on n=15** — within noise. Do not sell this as "streaming beats uniform."
- **Caveats:** slideshow (no intra-shot motion), one model, small n, KV bytes step-quantized (256-token
  alloc), and the vision encoder still runs per frame (~0.9–1.1 s/frame) — streaming bounds the *LM* KV,
  not perception cost.

## Verdict for the upstream (#492) decision
Promising but **not yet PR-justifying.** To make an accuracy case (not just a memory case) the benchmark
needs (1) real **motion video**, (2) a larger/standard streaming-QA set, and (3) an
**accuracy-discriminating query design** — recent-event-after-long-history, or infinite streams where
uniform sampling loses temporal resolution and full-context OOMs. Those are the real Phase-2b / Phase-3
steps before opening a PR to close #492.
