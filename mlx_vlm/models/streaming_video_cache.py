"""Streaming-video KV cache (tracer-bullet scaffold).

A segment-aware sink + sliding-window KV cache for *infinite* video streams,
subclassing the existing ``BufferedRotatingKVCache``. It implements the
StreamingVLM (arXiv:2510.09608) retention recipe on top of MLX's existing
attention-sink + rotating-window machinery:

    * keep the first ``sink`` tokens forever (system prompt + first frame),
    * keep **all** recent *text* tokens (long text window),
    * evict the oldest *vision* spans once the retained vision-token count
      exceeds ``vision_window`` (short vision window).

This is the only genuinely new logic in the wedge (see
``/root/mlx-streaming-video-kv-plan.md`` §3): the parent already provides the
ring-buffer, compaction (``_compact``), and drop-planning (``_planned_drop``);
here we make the drop *segment-aware* so text is preserved while stale frames
are dropped.

Status: SCAFFOLD. The span bookkeeping is real; the segment-aware eviction and
RoPE position re-anchoring (plan §6 risk #1 — THE main risk) are stubbed and
currently fall back to the parent's contiguous drop. Fill these in for the
tracer bullet.
"""

from __future__ import annotations

from typing import List, Literal, Tuple

from .cache import BufferedRotatingKVCache

SpanKind = Literal["text", "vision"]


class StreamingVideoCache(BufferedRotatingKVCache):
    """Sink + asymmetric (short-vision / long-text) window cache.

    Args:
        max_size: hard ceiling on retained tokens (window upper bound).
        sink: number of leading tokens pinned permanently (``keep``).
        vision_window: max retained *vision* tokens before oldest frames evict.
        text_window: max retained *text* tokens (``max_size`` if ``None``).
        buffer_size: rollback slack passed through to the parent.
    """

    def __init__(
        self,
        max_size: int,
        sink: int = 64,
        vision_window: int = 2048,
        text_window: int | None = None,
        buffer_size: int = 64,
    ):
        super().__init__(max_size=max_size, keep=sink, buffer_size=buffer_size)
        self.sink = int(sink)
        self.vision_window = int(vision_window)
        self.text_window = int(text_window) if text_window is not None else max_size
        # Ordered record of appended spans: (start_offset, end_offset, kind).
        self._spans: List[Tuple[int, int, SpanKind]] = []

    # -- span bookkeeping (real) ------------------------------------------
    def mark_span(self, n_tokens: int, kind: SpanKind) -> None:
        """Record that the next ``n_tokens`` appended belong to ``kind``.

        Call this immediately before the forward pass that appends the frame's
        vision tokens (kind="vision") or the query/answer text (kind="text").
        """
        if n_tokens <= 0:
            return
        start = self.offset
        self._spans.append((start, start + n_tokens, kind))

    def _vision_token_count(self) -> int:
        return sum(e - s for (s, e, k) in self._spans if k == "vision")

    def _oldest_evictable_vision_span(self) -> Tuple[int, int] | None:
        for s, e, k in self._spans:
            if k == "vision" and s >= self.sink:
                return (s, e)
        return None

    # -- retention policy (STUB — the tracer-bullet work) -----------------
    def _planned_drop(self, incoming: int) -> int:
        """Override: evict oldest *vision* spans, never text.

        TODO(tracer-bullet): when ``_vision_token_count() > vision_window``,
        select the oldest vision span past the sink and drop exactly its token
        range (not a contiguous prefix), then re-anchor survivor RoPE positions
        to be contiguous (plan §6 risk #1). Until implemented we defer to the
        parent's contiguous rotating drop so the cache stays runnable.
        """
        # TODO: replace with segment-aware, text-preserving eviction.
        return super()._planned_drop(incoming)

    def reset(self) -> None:  # pragma: no cover - convenience for demos
        self._spans.clear()
        self.start_position = 0
        self._idx = 0
        self.offset = 0
        self.keys = None
        self.values = None
