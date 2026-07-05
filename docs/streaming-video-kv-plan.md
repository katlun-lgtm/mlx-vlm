# Implementation Plan — Streaming-Video KV-Cache Path for mlx-vlm

**Date:** 2026-07-05
**Author's bet:** "Own a niche" — the vision analog of what mlx-audio did for streaming STT.
**Builds on (do not re-do):** `/root/mlx-vision-gap-dig.md`, `/root/mlx-contribution-report.md`.
**Status of this doc:** every repo/issue/paper/line-number claim below was re-verified against live `gh` + arXiv + the cloned `Blaizzy/mlx-vlm` tree on 2026-07-05.

---

## 1. Problem statement & why now

mlx-vlm has **no streaming-video path**. `load_video()` (`mlx_vlm/utils.py:1418`) uniformly samples up to `max_frames=768` frames with OpenCV (`np.linspace(0, total_frames-1, n)`), stacks them into one `(T,C,H,W)` batch, and feeds them as independent images with **no temporal state and no KV reuse across frames**. That is O(full re-prefill) every time context grows, KV memory grows O(T·tokens_per_frame) unbounded (768 frames × ~256 vision tokens ≈ 196K tokens ≈ multi-GB KV), and it cannot ingest an open-ended live camera stream at all. Meanwhile the entire 2025–2026 research wave on *streaming-video KV cache* (StreamingVLM, LiveVLM, StreamKV, HERMES, V-Rex — all real, IDs verified §7) is unported to MLX, and there is no equivalent to mlx-audio's `stream_generate` for eyes. **Why now:** (a) the reference implementation (StreamingVLM, MIT license, 1040★) shipped Oct 2025 and is directly adaptable; (b) mlx-vlm merges outside contributors aggressively (5,118★, pushed the day of writing) and issue **#492 "Unify video generate into generate.py" is OPEN** — an explicit invitation for exactly this file; (c) **the hard primitive already exists inside mlx-vlm** (`RotatingKVCache` = attention-sink + sliding window; `BufferedRotatingKVCache`), so this is integration + a retention policy, not a from-scratch kernel.

---

## 2. What already exists — DON'T rebuild it

| Asset | Location | Reuse it for | Do NOT |
|---|---|---|---|
| **`RotatingKVCache(max_size, keep)`** | `mlx_vlm/models/cache.py:299` | The core sink+sliding-window mechanism. `keep` = permanently-retained prefix (sinks); `max_size` = window. This *is* StreamingLLM. | Reimplement a ring buffer from scratch. |
| **`BufferedRotatingKVCache`** | `cache.py:1383` | VLM-specific temporal sliding window with rollback slack + `start_position` bookkeeping. Closest existing base class to subclass. | Write your own compaction logic. |
| **`VisionFeatureCache`** | `mlx_vlm/vision_cache.py:15` | Memoize per-frame vision-tower output (projected features) keyed by content hash — avoids recomputing the encoder on repeated/near-identical frames. | Build a new feature cache. |
| **`generate_step(..., prompt_cache=, pixel_values=, mask=, max_kv_size=, kv_bits=)`** | `mlx_vlm/generate/ar.py:148` | The generation loop **already threads a persistent `prompt_cache`**. Feed it a long-lived cache and call it per frame. | Fork the decode loop. |
| **`load_video` / `prepare_inputs` / `MessageBuilder.video_message` / `_format_video_message`** | `utils.py:1418`, `utils.py:1554`, `prompt_utils.py:225/483` | Frame decode + video-token prompt formatting already exist. Extend to an incremental iterator. | Rewrite prompt formatting. |
| **StreamingVLM reference impl** | `mit-han-lab/streaming-vlm` (MIT, 1040★, PyTorch) | Port the retention policy (asymmetric short-vision / long-text window + sinks) and reuse its **Inf-Streams-Eval** harness. | Re-derive the algorithm from the paper. |
| **davepoon webcam demo** | `davepoon/mlx-vlm-smolvlm-realtime-webcam` (28★, per-frame, stale 2025-06) | Camera-capture + Flask/SocketIO **front-end shell** as the demo UI. | Use its inference path — it re-prefills every frame; that's the thing we're replacing. |
| **mlx-audio `stream_generate`** | external | Mirror its **API shape** so "hear + see" are symmetric. | — |

**The single most important reuse:** the sink+window policy is `RotatingKVCache` with `keep>0`. The *new* work is making eviction **segment-aware** (a short window over *vision* tokens but a long window over *text* tokens — StreamingVLM's key asymmetry), plus the streaming ingestion loop. Everything else is plumbing.

---

## 3. Tracer bullet — the minimal end-to-end slice

**Goal:** a single script that ingests a real clip **frame-by-frame** through a persistent, memory-bounded KV cache and answers a question about the *recent* stream — with **bounded KV memory** and **per-frame latency independent of elapsed stream length**, beating uniform-sampling on the same clip at an equal KV budget.

**Target model:** **SmolVLM2** first (small, ~64–81 vision tokens/frame after pixel-shuffle, fits trivially on M3 Max, and davepoon's demo uses it → a ready A/B baseline). Defer Qwen2.5-VL (M-RoPE 3D video positions = extra risk, see §6).

**Smallest viable path — 3 new/changed things:**

1. **`mlx_vlm/models/cache.py` → add `StreamingVideoCache`** (subclass `BufferedRotatingKVCache`).
   - Tag each appended span as `text` (protected/sink) or `vision` (evictable).
   - Retention: keep the first `sink=N` tokens (system prompt + first frame) permanently; keep **all** recent text; evict oldest *vision* spans once the vision-token count exceeds `vision_window` (e.g. 8–16 frames' worth).
   - Reuse the parent's `_compact` / `_planned_drop` machinery; the only new logic is the text/vision span bookkeeping (a list of `(start, end, kind)`), plus contiguous **position re-indexing** of the survivors (see §6 risk #1).
   - v1 constraint: **single-stream only** (the batch path explicitly rejects `RotatingKVCache` with keep tokens — `ar.py:759-762`). That's fine; live video is one stream.

2. **`mlx_vlm/utils.py` → add `stream_video_frames(source, fps, ...)`** — a *generator* that yields one preprocessed frame tensor at a time (from a file via `cv2` seek, or a live `cv2.VideoCapture(0)` webcam), instead of pre-sampling all frames. This is the incremental counterpart to `load_video`.

3. **`mlx_vlm/generate/video.py` → new file with `stream_generate_video(model, processor, frame_iter, question, cache)`** (the missing sibling to `image.py`/`ar.py`/`diffusion.py`; directly answers issue #492). Loop:
   - init one `StreamingVideoCache` per layer via a `make_cache`-style factory;
   - for each frame: `vision_tower + embed_vision` (memoized via `VisionFeatureCache`), append the frame's vision tokens to the persistent cache using `generate_step`'s existing forward-with-cache;
   - on a query token: run `generate_step(..., prompt_cache=<the persistent cache>)` to decode the answer over the retained window — **no re-prefill of history**.

**Definition of done (tracer bullet):** on one 2–5 min clip, plot (a) KV bytes vs frame index — flat for streaming, linear for `load_video` baseline; (b) per-frame wall-clock — flat for streaming, rising for baseline; (c) answer correctness on ≥1 "what just happened" question matches or beats the uniform-256-frame baseline. One clip, one model, one question type. Ship it.

---

## 4. Phased roadmap (after the tracer bullet)

**Phase 1 — Correctness & the asymmetric window (demo: "it stays sharp on recent events").**
Tune `sink`, `vision_window`, `text_window` on SmolVLM2. Verify the RoPE position re-indexing is numerically correct (attention scores match a full-context reference on a short clip within tolerance). Add a tiny eval: N questions about the *last* K seconds vs uniform sampling. **Demoable:** live webcam → rolling caption that never degrades as the session runs for 10+ minutes.

**Phase 2 — Second model + real benchmark (demo: numbers on a public streaming set).**
Add **Qwen2.5-VL** (handle M-RoPE video positions). Wire up a public streaming-video QA benchmark (StreamingBench / OVO-Bench, or StreamingVLM's Inf-Streams-Eval). Produce the money table: **accuracy vs KV-memory-budget curve**, streaming sink+window vs uniform sampling at equal budget. **Demoable:** a plot showing streaming matches uniform accuracy at a fraction of the KV memory, and holds accuracy on infinite streams where uniform OOMs.

**Phase 3 — Temporal redundancy pruning (demo: "skips boring frames").**
Add optional content-based pruning inside the vision window: drop near-duplicate consecutive frames by cosine similarity of pooled vision features (the simple slice of V-Rex ReSV / LiveVLM bucketing). This is the first *content-aware* step beyond pure positional windowing. **Demoable:** on a mostly-static clip, effective tokens/sec drops sharply with no accuracy loss.

**Phase 4 — Upstream integration & server (demo: it's in mlx-vlm main).**
Land `generate/video.py` + `StreamingVideoCache` as a PR closing #492's video-generate gap; wire an SSE/streaming endpoint into `mlx_vlm/server/` so it's callable like text streaming. Add KV quantization (the cache already interops with `QuantizedKVCache`/`kv_bits`) to shrink the retained window further. **Demoable:** `mlx_vlm.stream_generate_video` in a released version; a webcam demo talking to the server.

**Phase 5 (optional, higher ceiling) — retrieval over evicted history (demo: "remembers the whole session").**
Add LiveVLM/StreamKV-style retrieval: on eviction, summarize/bucket the dropped vision KV so a query about an *earlier* event can pull it back. This lifts the inherent "only remembers recent" limit of sink+window. Bigger scope; only if Phases 1–4 land.

---

## 5. Upstream vs standalone, and how to validate

**Recommendation: two-track — prototype standalone, upstream the core.**
- **Develop in a fork of `Blaizzy/mlx-vlm`** (not a greenfield repo): the cache primitives, model zoo, processors, and decode loop you depend on all live there; a standalone package would have to vendor the whole VLM stack. Iterate fast on your own branch with your own benchmark harness.
- **Ship a thin standalone demo repo** (`mlx-streaming-video`) for narrative/visibility — the "own a niche" flag — that depends on your mlx-vlm branch and provides the webcam UI + benchmark plots.
- **Upstream `StreamingVideoCache` + `generate/video.py` as a PR** once the tracer bullet + Phase 2 numbers exist. Rationale: #492 is OPEN and explicitly wants this file; Blaizzy merges outside contributors continuously; the primitives are already there. This is the highest-leverage placement.

**Validation — three metrics, one baseline (uniform `load_video` sampling at matched settings):**
1. **KV memory (bytes, `cache.nbytes`)** — must be **O(1) bounded** as stream length → ∞, vs O(T) for baseline. This is the headline win and the easiest to prove.
2. **Latency** — per-frame ingest ms and **TTFT for a mid-stream query**, both flat vs elapsed stream length (baseline rises then OOMs).
3. **Accuracy** — on a *streaming-appropriate* QA set (questions about recent/ongoing events): StreamingBench, OVO-Bench, or StreamingVLM's Inf-Streams-Eval. Report the **accuracy-vs-KV-budget curve**; the claim to prove is "matches uniform accuracy at materially lower KV, and stays finite where uniform can't run." Do **not** benchmark on long-video-needle sets (EgoSchema/VideoMME-long) as a primary metric — sink+window is designed to forget distant frames; that would be measuring the wrong thing (until Phase 5).

---

## 6. Key risks & open questions

1. **RoPE position management under eviction (THE main risk).** When you evict middle vision tokens, survivors' positions become non-contiguous. StreamingVLM re-anchors retained tokens to contiguous positions; MLX's `RotatingKVCache` grows `offset` monotonically and rebuilds a rolled mask (`cache.py:443-467`) rather than re-indexing RoPE. **Open Q:** does each target model's forward let you inject explicit `position_ids` at incremental steps, or is position implied by cache `offset`? Verify per model *before* Phase 1. This determines whether v1 needs contiguous re-anchoring or can ride the existing rotating-mask path.
2. **Qwen2.5-VL M-RoPE (3D video positions).** Video uses time/height/width position components; naive 1-D windowing breaks it. Why SmolVLM2 (1-D positions) is the tracer-bullet target and Qwen is Phase 2.
3. **Vision-encoder cost is not eliminated.** KV reuse bounds the *LM* context; you still run the vision tower on each new frame. The win is bounded prefill/decode over history, not free perception. `VisionFeatureCache` only helps on repeated frames. Be honest in benchmarks: separate "encode" from "LM" time.
4. **Accuracy cliff on distant events.** Sink+window forgets old frames by design → fails needle-in-long-video questions. Scope claims to recent-event QA; Phase 5 (retrieval) is the mitigation. Don't oversell.
5. **Batch limitation.** `_make_cache` rejects `RotatingKVCache` with keep tokens (`ar.py:759-762`) → v1 is single-stream. Fine for live video; note it.
6. **Long-window SDPA prefill.** mlx **#3658** (head_dim=256 unfused SDPA, 100K+ prefill impractical) — only bites very long windows; streaming keeps windows short, so likely a non-issue, but confirm on the chosen model's head_dim.
7. **Open Q — window sizing:** what `(sink, vision_window, text_window)` actually preserves accuracy? Empirical; StreamingVLM's published values are the starting prior, not gospel for smaller MLX models.

---

## 7. Verified references (checked 2026-07-05)

**Papers (arXiv IDs confirmed, titles match):**
- StreamingLLM (foundation: attention-sink + sliding window) — **2309.17453**.
- StreamingVLM: Real-Time Understanding for Infinite Video Streams — **2510.09608** (sinks + *short* vision window + *long* text window; the recipe we port).
- LiveVLM: Streaming-Oriented KV Cache and Retrieval — **2505.15269** (vision-sink bucketing + query-time retrieval → Phase 5).
- StreamKV: Segment-based KV Cache Retrieval and Compression — **2511.07278** (semantic segmentation → Phase 5).
- HERMES: KV Cache as Hierarchical Memory — **2601.14724** (multi-granularity, −68% tokens vs uniform).
- V-Rex: Dynamic KV Cache Retrieval — **2512.12284** (temporal+spatial similarity clustering → Phase 3).

**mlx-vlm (Blaizzy/mlx-vlm, 5,118★, actively merged):**
- #492 "Unify video generate into generate.py" — **OPEN** (the slot for `generate/video.py`).
- #1444 (Qwen3.x VL video no temporal reasoning) — CLOSED; #1436 (Kimi MoonViT 2D-only) — CLOSED; #1271 (SAM3-Tracker) — OPEN. Confirm the *streaming* gap is unclaimed: issue/PR search for "streaming video" KV work returned **none** (existing "streaming" PRs are text detokenizer/thinking-mode only).
- Key code: `models/cache.py` (`RotatingKVCache:299`, `BufferedRotatingKVCache:1383`), `vision_cache.py:15`, `generate/ar.py:148` (`generate_step`), `utils.py:1418` (`load_video`), `utils.py:1554` (`prepare_inputs`), `prompt_utils.py:225/483` (video message formatting). Note: `generate/` has image.py/ar.py/diffusion.py/edit_image.py but **no video.py**.

**Reference impls / demos:**
- `mit-han-lab/streaming-vlm` — MIT, 1040★ (port the policy + Inf-Streams-Eval; license-safe to adapt).
- `davepoon/mlx-vlm-smolvlm-realtime-webcam` — 28★, per-frame, stale (front-end shell only; its inference is the anti-pattern).

---

## 8. One-line summary for the parent thread

Build `StreamingVideoCache` (segment-aware sink + short-vision / long-text window, subclassing the existing `BufferedRotatingKVCache`) + `stream_video_frames` iterator + a new `generate/video.py::stream_generate_video`, tracer-bullet on SmolVLM2, prove bounded KV + flat latency + matched accuracy vs uniform sampling, then upstream to close mlx-vlm #492. The hard cache primitive already exists in mlx-vlm; this is integration + a retention policy + a streaming loop — **no new Metal kernels, no core-mlx changes for v1.**
