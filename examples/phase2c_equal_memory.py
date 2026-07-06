#!/usr/bin/env python3
"""Phase-2c DECISIVE test: window advantage vs LLM capability at EQUAL KV MEMORY.

The streaming-video wedge hinges on one measurement the phase2b/size-sweep runs
never made cleanly: **at an EQUAL KV-memory budget (bytes, not frames)**, does
FastVLM-1.5B's *bigger temporal window* (it fits ~3.1x more frames per KV-byte —
Qwen2's 2 KV heads vs Llama3-8B's 8) beat Idefics3-8B's *stronger LLM* (8B vs
1.5B) on "name the last N scenes"?

Two opposing forces, one number:
  * FastVLM window advantage  = more recent clips fit the same bytes.
  * Idefics3 capability edge   = better per-frame understanding.

This harness fixes the two flaws the size-sweep exposed:
  1. PRECISION-MATCH.  Load a FastVLM-1.5B **8-bit** checkpoint (match
     Idefics3-8B-8bit). KV/frame is bf16 activation regardless of weight quant,
     so the ~3.1x window ratio is unchanged; this only matches per-frame QUALITY.
  2. ANSWERABLE WINDOW at a FIXED KV-BYTE budget B.  Each model's segment-aware
     window is sized ``window_frames = floor(B / measured_KV_per_frame)`` — so at
     the SAME bytes FastVLM holds ~3.1x more frames. B is chosen so the last N
     clips are IN FastVLM's window but NOT fully in Idefics3's — exactly where
     the window advantage must show if it is real.

Design (reuses the proven phase2b pieces):
  * Isolation INTERSECTION: a subject enters the pool only if BOTH models name it
    shown alone, so any gap is window/capability, not FastVLM failing to perceive.
  * Long streams of concatenated real motion clips; query = "name the last N
    scenes, most recent first"; recall = deterministic substring match of last-N.
  * Conditions per model at the equal-BYTE budget: segment-aware (the comparison)
    + uniform-at-same-budget + full-context (capability oracle) references.
  * Paired test FastVLM-8bit vs Idefics3-8bit (segment, paired by stream x qp) at
    equal memory: mean diff, 95% CI, Wilcoxon signed-rank p.

Usage:
    # 1. FastVLM isolation over the same 26-clip pool (Idefics3 manifest exists):
    python examples/phase2b_scaled.py <fastvlm-8bit> --isolate \
        --manifest examples/phase2b_isolation_fastvlm.json
    # 2. smoke (prove path + quantized load + byte-budget windowing end to end):
    python examples/phase2c_equal_memory.py --fastvlm <ckpt> --streams 2
    # 3. scaled decisive run:
    python examples/phase2c_equal_memory.py --fastvlm <ckpt> --streams 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import mlx.core as mx
import numpy as np

from phase2b_benchmark import (
    F,
    N,
    RECENT_Q,
    RECENT_TOK,
    configure_model,
    query_points,
    recall,
    run_segment,
    run_selection,
    sample_clip_frames,
)
from phase2b_scaled import (
    BASE_SEED,
    build_streams_seeded,
    mean_ci,
    wilcoxon_signed_rank,
)

from mlx_vlm import load
from mlx_vlm.generate.video import _ingest_frame, kv_bytes, open_stream
from mlx_vlm.models import cache as _cache

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(HERE, "videos_phase2b")
IDEFICS_MANIFEST = os.path.join(HERE, "phase2b_isolation_pass.json")
FASTVLM_MANIFEST = os.path.join(HERE, "phase2b_isolation_fastvlm.json")
RESULTS_PATH = os.path.join(HERE, "phase2c_equal_memory_results.json")
IDEFICS_ID = "mlx-community/Idefics3-8B-Llama3-8bit"


# ==========================================================================
# Isolation INTERSECTION — a subject must pass for BOTH models.
# ==========================================================================
def load_manifest_subjects(path: str) -> dict:
    """Return {label: synonyms} from an isolation-pass manifest (no decode)."""
    data = json.load(open(path))
    return {rec["label"]: rec["synonyms"] for rec in data["clips"]}


def load_manifest_files(path: str) -> dict:
    """Return {label: file} from an isolation-pass manifest."""
    data = json.load(open(path))
    return {rec["label"]: rec["file"] for rec in data["clips"]}


def build_intersection(idefics_manifest: str, fastvlm_manifest: str) -> dict:
    """Subjects BOTH models named alone ON THE SAME CLIP FILE; re-decode it.

    Returns kept = {label: (frames, synonyms)}. Rigorous file-level intersection:
    a subject enters only if BOTH manifests kept the IDENTICAL clip file for it,
    so the exact clip used in every stream is verified perceived-alone by BOTH
    models (subject-level agreement on different files is dropped — we can't prove
    both perceive the shared clip). Frame content is identical for both models;
    each model's processor handles its own resize.
    """
    idf_syn = load_manifest_subjects(idefics_manifest)
    fv_syn = load_manifest_subjects(fastvlm_manifest)
    idf_file = load_manifest_files(idefics_manifest)
    fv_file = load_manifest_files(fastvlm_manifest)
    subj_common = sorted(set(idf_syn) & set(fv_syn))
    common = sorted(l for l in subj_common if idf_file[l] == fv_file[l])
    dropped_diff = sorted(l for l in subj_common if idf_file[l] != fv_file[l])
    print(
        f"[intersect] Idefics3={len(idf_syn)} FastVLM={len(fv_syn)}  "
        f"subject-common={len(subj_common)}  SAME-FILE={len(common)}: {common}"
    )
    if dropped_diff:
        print(
            f"[intersect] dropped {len(dropped_diff)} diff-file subjects: {dropped_diff}"
        )
    kept: dict = {}
    missing = []
    for lab in common:
        fpath = os.path.join(CLIPS_DIR, idf_file[lab])
        if not os.path.exists(fpath):
            missing.append(idf_file[lab])
            continue
        frames = sample_clip_frames(fpath, F)
        if len(frames) < F:
            missing.append(idf_file[lab] + " (decode-fail)")
            continue
        kept[lab] = (frames, idf_syn[lab])
    if missing:
        print(f"[intersect][warn] {len(missing)} clips unavailable: {missing}")
    print(f"[intersect] decoded {len(kept)} usable intersected subjects")
    return kept


# ==========================================================================
# KV/frame measurement -> equal-BYTE window sizing.
# ==========================================================================
def measure_kv_per_frame(model, processor, frames, warmup: int = 2) -> float:
    """Mean steady-state KV-byte delta per ingested frame (fresh dense cache)."""
    cache = _cache.make_prompt_cache(model.language_model)
    open_stream(model, processor, RECENT_Q, cache)
    prev = kv_bytes(cache)
    deltas = []
    for fr in frames:
        st = _ingest_frame(model, processor, fr, cache)
        deltas.append(st["kv_bytes"] - prev)
        prev = st["kv_bytes"]
    ss = deltas[warmup:] or deltas
    return float(np.mean(ss))


def window_for_budget(kv_per_frame: float, budget_bytes: int) -> int:
    """frames that fit the byte budget (>=1)."""
    return max(1, int(budget_bytes // kv_per_frame))


# ==========================================================================
# Scored evaluation for ONE model at the equal-byte budget (incremental save).
# ==========================================================================
def evaluate_model(
    model,
    processor,
    tag,
    kept,
    streams,
    window_frames,
    budget_mib,
    rows,
    out_path,
    meta,
):
    """Run segment(window) + uniform(window) + full for one model; append rows."""
    syn = {lab: s for lab, (_f, s) in kept.items()}
    t0 = time.time()
    for s_idx, order in enumerate(streams):
        frames = []
        for lab in order:
            frames.extend(kept[lab][0])
        qps_clip = query_points(len(order))
        qp_frames = [c * F - 1 for c in qps_clip]
        print(
            f"\n[{tag} stream {s_idx}] M={len(order)} T={len(frames)} "
            f"win={window_frames}f ({window_frames // F}clips) qp_clips={qps_clip} "
            f"order={order}  ({time.time() - t0:.0f}s)"
        )
        full = run_selection(
            model, processor, frames, qp_frames, None, "full", RECENT_Q, RECENT_TOK
        )
        seg = run_segment(
            model, processor, frames, qp_frames, window_frames, RECENT_Q, RECENT_TOK
        )
        uni = run_selection(
            model,
            processor,
            frames,
            qp_frames,
            window_frames,
            "uniform",
            RECENT_Q,
            RECENT_TOK,
        )
        for c, qpf in enumerate(qp_frames):
            clip_idx = qps_clip[c]
            gt = order[clip_idx - N : clip_idx]  # last N clips, chronological
            for cond, res in (
                ("segment", seg[qpf]),
                ("uniform", uni[qpf]),
                ("full", full[qpf]),
            ):
                rows.append(
                    dict(
                        model=tag,
                        stream=s_idx,
                        clip_idx=clip_idx,
                        budget_mib=budget_mib,
                        window_frames=(None if cond == "full" else window_frames),
                        cond=cond,
                        gt="|".join(gt),
                        answer=res["text"],
                        recall=recall(res["text"], gt, syn),
                        kv_bytes=res["kv_bytes"],
                    )
                )
            print(
                f"   clip{clip_idx:>2} gt={'|'.join(gt):26s} "
                f"seg={recall(seg[qpf]['text'], gt, syn):.2f} "
                f"uni={recall(uni[qpf]['text'], gt, syn):.2f} "
                f"full={recall(full[qpf]['text'], gt, syn):.2f}"
            )
        payload = {**meta, "rows": rows}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        mx.clear_cache()
        gc.collect()
    return rows


# ==========================================================================
# Stats — per-cell mean +/- 95% CI + paired FastVLM-vs-Idefics3 (segment).
# ==========================================================================
def _cell(rows, model, cond):
    return [r["recall"] for r in rows if r["model"] == model and r["cond"] == cond]


def _paired_seg(rows, model_a, model_b):
    """seg recall paired by (stream, clip_idx): model_a - model_b."""
    a = {
        (r["stream"], r["clip_idx"]): r["recall"]
        for r in rows
        if r["model"] == model_a and r["cond"] == "segment"
    }
    b = {
        (r["stream"], r["clip_idx"]): r["recall"]
        for r in rows
        if r["model"] == model_b and r["cond"] == "segment"
    }
    keys = sorted(set(a) & set(b))
    return [a[k] - b[k] for k in keys], keys


def stats_report(rows, meta) -> str:
    fv, idf = meta["fastvlm_tag"], meta["idefics_tag"]
    L = ["=" * 82]
    L.append(
        f"PHASE-2c EQUAL-MEMORY   budget={meta['budget_mib']} MiB   "
        f"N(last)={N}  F={F}f/clip  pool={len(meta['pool_subjects'])} subj"
    )
    L.append(
        f"windows @ equal bytes:  {fv}={meta['fastvlm_window']}f "
        f"({meta['fastvlm_window'] // F} clips, {meta['fastvlm_kv_mib']:.2f} MiB/f)  |  "
        f"{idf}={meta['idefics_window']}f "
        f"({meta['idefics_window'] // F} clips, {meta['idefics_kv_mib']:.2f} MiB/f)"
    )
    L.append("=" * 82)
    L.append(f"\n-- mean recall of last-{N} +/- 95% CI --")
    L.append(f"{'model':<26}{'condition':<12}{'mean':>7}{'95% CI':>20}{'n':>5}")
    for model in (fv, idf):
        for cond in ("segment", "uniform", "full"):
            c = mean_ci(_cell(rows, model, cond))
            if c["n"]:
                L.append(
                    f"{model:<26}{cond:<12}{c['mean']:>7.3f}"
                    f"{'[%.3f, %.3f]' % (c['lo'], c['hi']):>20}{c['n']:>5}"
                )
    diffs, keys = _paired_seg(rows, fv, idf)
    if diffs:
        ci = mean_ci(diffs)
        w = wilcoxon_signed_rank(diffs)
        p = w["p"]
        p_s = "n/a" if (p != p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")
        L.append(
            f"\n-- PAIRED (segment, equal memory)  {fv} - {idf}  "
            f"paired by stream x query-point --"
        )
        L.append(
            f"  n_pair={ci['n']}  {fv}={np.mean(_seg_at(rows, fv, keys)):.3f}  "
            f"{idf}={np.mean(_seg_at(rows, idf, keys)):.3f}  "
            f"diff={ci['mean']:+.3f}  95%CI=[{ci['lo']:+.3f}, {ci['hi']:+.3f}]  "
            f"Wilcoxon p={p_s}"
        )
        L.append(
            "  (diff>0 & CI excludes 0 => at equal KV memory FastVLM's window beats "
            "Idefics3's LLM on recency recall.)"
        )
    return "\n".join(L)


def _seg_at(rows, model, keys):
    d = {
        (r["stream"], r["clip_idx"]): r["recall"]
        for r in rows
        if r["model"] == model and r["cond"] == "segment"
    }
    return [d[k] for k in keys]


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Phase-2c equal-KV-memory decisive test")
    ap.add_argument(
        "--fastvlm", required=True, help="FastVLM-1.5B 8-bit ckpt (path/id)"
    )
    ap.add_argument("--idefics", default=IDEFICS_ID)
    ap.add_argument("--fastvlm-tag", default="FastVLM-1.5B-8bit")
    ap.add_argument("--idefics-tag", default="Idefics3-8B-8bit")
    ap.add_argument(
        "--budget-mib",
        type=float,
        default=48.0,
        help="equal KV-byte budget (MiB); default 48 => FV 6f/3clip, Idf 2f/1clip",
    )
    ap.add_argument("--streams", type=int, default=10)
    ap.add_argument("--m-stream", type=int, default=8, help="clips per stream")
    ap.add_argument("--idefics-manifest", default=IDEFICS_MANIFEST)
    ap.add_argument("--fastvlm-manifest", default=FASTVLM_MANIFEST)
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args()

    budget_bytes = int(args.budget_mib * 1024 * 1024)

    # --- intersection pool ---
    kept = build_intersection(args.idefics_manifest, args.fastvlm_manifest)
    if len(kept) < N + 1:
        print(f"FATAL: intersection has {len(kept)} subjects; need >= {N + 1}. STOP.")
        return

    m_stream = min(len(kept), args.m_stream)
    streams = build_streams_seeded(list(kept), args.streams, m_stream, BASE_SEED)
    print(
        f"[run] streams={args.streams} m_stream={m_stream} budget={args.budget_mib} MiB"
    )

    rows: list = []
    meta = {
        "test": "phase2c_equal_memory",
        "budget_mib": args.budget_mib,
        "N": N,
        "F": F,
        "m_stream": m_stream,
        "n_streams": args.streams,
        "base_seed": BASE_SEED,
        "fastvlm_id": args.fastvlm,
        "idefics_id": args.idefics,
        "fastvlm_tag": args.fastvlm_tag,
        "idefics_tag": args.idefics_tag,
        "pool_subjects": sorted(kept),
    }

    # frames of stream 0 for the KV/frame measurement.
    m0 = []
    for lab in streams[0]:
        m0.extend(kept[lab][0])

    # --- run each model: measure KV/frame -> size window -> score ---
    for which, mid, tag in (
        ("fastvlm", args.fastvlm, args.fastvlm_tag),
        ("idefics", args.idefics, args.idefics_tag),
    ):
        print(f"\n{'=' * 70}\n[load] {tag}: {mid}\n{'=' * 70}")
        model, processor = load(mid)
        configure_model(
            model, processor
        )  # no-op for FastVLM (no image_processor knobs)
        kv_pf = measure_kv_per_frame(model, processor, m0)
        win = window_for_budget(kv_pf, budget_bytes)
        print(
            f"[{tag}] KV/frame={kv_pf / 1024**2:.3f} MiB  "
            f"-> window={win} frames ({win // F} clips) @ {args.budget_mib} MiB"
        )
        meta[f"{which}_kv_mib"] = kv_pf / 1024**2
        meta[f"{which}_window"] = win
        with open(args.out, "w") as f:
            json.dump({**meta, "rows": rows}, f, indent=2)
        evaluate_model(
            model,
            processor,
            tag,
            kept,
            streams,
            win,
            args.budget_mib,
            rows,
            args.out,
            meta,
        )
        del model, processor
        gc.collect()
        mx.clear_cache()

    print("\n" + stats_report(rows, meta))
    with open(args.out, "w") as f:
        json.dump({**meta, "rows": rows}, f, indent=2)
    print(f"\n[saved] {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
