"""Streaming video generation (tracer-bullet scaffold).

The missing sibling to ``image.py`` / ``ar.py`` / ``diffusion.py`` in this
package, and the direct answer to issue #492 ("Unify video generate into
generate.py"). Ingests a video *frame by frame* through one persistent,
memory-bounded ``StreamingVideoCache`` and answers questions about the recent
stream WITHOUT re-prefilling history — the eyes analog of mlx-audio's
``stream_generate``.

Status: SCAFFOLD. The control flow and cache wiring are laid out; the
per-frame vision-embed append and the query-decode call are marked TODO. See
``/root/mlx-streaming-video-kv-plan.md`` §3 for the definition of done.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..models.streaming_video_cache import StreamingVideoCache
from .ar import generate_step


def make_streaming_cache(
    model: nn.Module,
    *,
    sink: int = 64,
    vision_window: int = 2048,
    text_window: Optional[int] = None,
    max_size: int = 4096,
) -> List[StreamingVideoCache]:
    """One ``StreamingVideoCache`` per model layer (single-stream; see plan §6.5)."""
    n_layers = len(model.language_model.layers)
    return [
        StreamingVideoCache(
            max_size=max_size,
            sink=sink,
            vision_window=vision_window,
            text_window=text_window,
        )
        for _ in range(n_layers)
    ]


def stream_generate_video(
    model: nn.Module,
    processor: Any,
    frames: Iterable[Any],
    question: str,
    *,
    cache: Optional[List[StreamingVideoCache]] = None,
    answer_every: Optional[int] = None,
    max_tokens: int = 128,
    **kwargs,
) -> Iterator[dict]:
    """Stream frames into a persistent cache; answer over the retained window.

    Args:
        model: a loaded VLM (SmolVLM2 for the tracer bullet; Qwen2.5-VL is
            Phase 2 because of M-RoPE, see plan §6.2).
        processor: the model's processor / image processor.
        frames: an iterable/generator of PIL frames (see
            ``streaming.stream_video_frames``).
        question: the query to answer about the ongoing stream.
        cache: reuse an existing streaming cache, or a fresh one is made.
        answer_every: if set, emit an answer every N frames (rolling caption);
            otherwise answer once at end-of-stream.
        max_tokens: max answer tokens per query.

    Yields:
        dicts like ``{"frame": i, "text": <answer>, "kv_bytes": <int>}``.
    """
    if cache is None:
        cache = make_streaming_cache(model)

    for i, frame in enumerate(frames):
        # 1) Embed this frame's vision tokens (memoize via VisionFeatureCache).
        #    TODO(tracer-bullet): run vision_tower + projector on `frame`,
        #    mark_span(n_vision_tokens, "vision") on each layer cache, and do
        #    one forward pass appending those tokens to the persistent cache.
        _ingest_frame(model, processor, frame, cache)  # noqa: F821 (stub below)

        if answer_every and (i + 1) % answer_every == 0:
            yield _answer(model, processor, question, cache, max_tokens, **kwargs)

    if not answer_every:
        yield _answer(model, processor, question, cache, max_tokens, **kwargs)


# --------------------------------------------------------------------------
# Stubs — the two hot spots to implement for the tracer bullet.
# --------------------------------------------------------------------------
def _ingest_frame(model, processor, frame, cache) -> None:
    """TODO: embed one frame and append its vision KV to `cache` in place."""
    raise NotImplementedError(
        "stream_generate_video: per-frame vision ingest not yet implemented "
        "(tracer-bullet step 3.2 — see mlx-streaming-video-kv-plan.md)"
    )


def _answer(model, processor, question, cache, max_tokens, **kwargs) -> dict:
    """TODO: decode an answer over the retained window via generate_step,
    passing `prompt_cache=cache` so no history is re-prefilled."""
    raise NotImplementedError(
        "stream_generate_video: query decode not yet implemented "
        "(tracer-bullet step 3.3). Wire generate_step(..., prompt_cache=cache)."
    )
