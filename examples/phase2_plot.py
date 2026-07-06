#!/usr/bin/env python3
"""Render the Phase-2 money-table figure from phase2_results.json.

Three panels:
  (1) KV memory vs frames-in-stream  — segment-aware bounded (flat) vs
      full-context O(T) growth (the headline).
  (2) accuracy vs KV budget          — segment-aware >= uniform at equal budget.
  (3) query TTFT vs frames-in-stream — flat for bounded window, rising for full.
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "phase2_results.json")))
rows = DATA["rows"]
budgets = DATA["budgets"]


def series(cond, budget, key, xkey="m_frames"):
    pts = {}
    for r in rows:
        if r["cond"] != cond:
            continue
        if budget is not None and r["budget"] != budget:
            continue
        pts.setdefault(r[xkey], []).append(r[key])
    xs = sorted(pts)
    return xs, [float(np.mean(pts[x])) for x in xs]


fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# Panel 1 — KV memory vs frames.
for cond, budget, lab, style in [
    ("segment", 4, "segment-aware b=4", "-o"),
    ("segment", 8, "segment-aware b=8", "-s"),
    ("uniform", 4, "uniform b=4", "--o"),
    ("full", None, "full-context (O(T))", "-^"),
]:
    xs, ys = series(cond, budget, "kv_bytes")
    ax[0].plot(xs, [y / 1024 / 1024 for y in ys], style, label=lab)
ax[0].set_xlabel("frames ingested (stream length)")
ax[0].set_ylabel("KV cache (MB)")
ax[0].set_title("(1) Bounded KV: segment-aware flat, full-context grows")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

# Panel 2 — accuracy vs budget (recency-contrast subset: m>budget).
contrast = [
    r
    for r in rows
    if r["cond"] in ("segment", "uniform") and r["m_frames"] > r["budget"]
]


def acc(cond, budget):
    rs = [r for r in contrast if r["cond"] == cond and r["budget"] == budget]
    return sum(r["correct"] for r in rs) / max(1, len(rs))


x = np.arange(len(budgets))
w = 0.35
seg = [acc("segment", b) for b in budgets]
uni = [acc("uniform", b) for b in budgets]
ax[1].bar(x - w / 2, seg, w, label="segment-aware")
ax[1].bar(x + w / 2, uni, w, label="uniform")
ax[1].axhline(1.0, color="green", ls=":", lw=1, label="full-context = 1.00")
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"b={b} frames" for b in budgets])
ax[1].set_ylabel("accuracy (recency-contrast items)")
ax[1].set_ylim(0.0, 1.08)
ax[1].set_title("(2) Accuracy vs KV budget (equal budget)")
ax[1].legend(fontsize=8)
for i, (s, u) in enumerate(zip(seg, uni)):
    ax[1].text(i - w / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=8)
    ax[1].text(i + w / 2, u + 0.01, f"{u:.2f}", ha="center", fontsize=8)

# Panel 3 — TTFT vs frames.
for cond, budget, lab, style in [
    ("segment", 4, "segment-aware b=4", "-o"),
    ("uniform", 4, "uniform b=4", "--o"),
    ("full", None, "full-context", "-^"),
]:
    xs, ys = series(cond, budget, "ttft_ms")
    ax[2].plot(xs, ys, style, label=lab)
ax[2].set_xlabel("frames ingested (stream length)")
ax[2].set_ylabel("query TTFT (ms)")
ax[2].set_title("(3) Flat query latency vs full-context drift")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)

fig.suptitle(
    f"Streaming-video KV: {DATA['model'].split('/')[-1]}  "
    f"(K={DATA['K']} frames/scene, {DATA['tokens_per_frame']} tok/frame, real slideshow)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "phase2_money_table.png")
fig.savefig(out, dpi=130)
print("[saved]", out)
