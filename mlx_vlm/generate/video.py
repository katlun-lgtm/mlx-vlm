"""Streaming video generation (tracer-bullet).

The missing sibling to ``image.py`` / ``ar.py`` / ``diffusion.py`` in this
package, and the direct answer to issue #492 ("Unify video generate into
generate.py"). Ingests a video *frame by frame* through one persistent,
memory-bounded KV cache and answers questions about the recent stream WITHOUT
re-prefilling history — the eyes analog of mlx-audio's ``stream_generate``.

v1 (this file) proves the LOOP: frame-by-frame prefill into a bounded cache +
answer over the window. It uses a plain ``RotatingKVCache(max_size, keep)`` per
layer (StreamingLLM = attention-sink + sliding window). Segment-aware eviction
and RoPE re-anchoring (``StreamingVideoCache``) are Phase 1, not needed here.

The streamed context is laid out as ONE multi-image user turn::

    <bos>User:<frame_1><frame_2>...<frame_N>{question}<end_of_utterance>
    Assistant:

The opening (``User:`` prefix) is ingested once, each frame's vision tokens are
appended incrementally into the bounded cache, and the closing
(``{question} ... Assistant:``) is prefilled at answer time on top of the
retained window — so the frame history is never re-prefilled.

See ``docs/streaming-video-kv-plan.md`` §3 for the definition of done.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..models import cache as _cache
from ..models.streaming_video_cache import StreamingVideoCache
from .ar import generate_step


def make_streaming_cache(
    model: nn.Module,
    *,
    sink: int = 64,
    window: int = 2048,
    use_streaming_cache: bool = False,
    vision_window: Optional[int] = None,
    text_window: Optional[int] = None,
) -> List[Any]:
    """One bounded KV cache per model layer (single-stream; see plan §6.5).

    v1 default: a plain ``RotatingKVCache(max_size=window, keep=sink)`` per
    layer — attention-sink (``keep`` leading tokens pinned forever) plus a
    sliding window of the most recent ``window - sink`` tokens. This is
    StreamingLLM, and it is all the tracer bullet needs: KV memory is bounded by
    ``window`` no matter how long the stream runs.

    Set ``use_streaming_cache=True`` to use the (Phase-1) segment-aware
    ``StreamingVideoCache`` instead; for v1 it falls back to the parent's
    contiguous rotating drop, so behaviour matches the RotatingKVCache path.
    """
    n_layers = len(model.language_model.layers)
    if use_streaming_cache:
        return [
            StreamingVideoCache(
                max_size=window,
                sink=sink,
                vision_window=vision_window if vision_window is not None else window,
                text_window=text_window,
            )
            for _ in range(n_layers)
        ]
    return [_cache.RotatingKVCache(max_size=window, keep=sink) for _ in range(n_layers)]


def kv_bytes(cache: List[Any]) -> int:
    """Total bytes held across the per-layer caches (the headline metric)."""
    return int(sum(getattr(c, "nbytes", 0) for c in cache))


def _image_marker(processor: Any) -> str:
    return getattr(processor, "image_token", "<image>")


def _tokenizer(processor: Any):
    return getattr(processor, "tokenizer", processor)


def _stop_ids(model: nn.Module, processor: Any) -> set:
    """Collect the token ids that terminate an answer."""
    ids: set = set()
    cfg_eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(cfg_eos, int):
        ids.add(cfg_eos)
    elif isinstance(cfg_eos, (list, tuple)):
        ids.update(int(x) for x in cfg_eos)
    tok = _tokenizer(processor)
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None:
        ids.add(int(eos))
    # SmolVLM ends assistant turns with <end_of_utterance>.
    for name in ("<end_of_utterance>",):
        try:
            tid = tok.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0:
                ids.add(tid)
        except Exception:
            pass
    return ids


def _split_user_turn(
    model: nn.Module, processor: Any, question: str
) -> Tuple[str, str]:
    """Split a single-image user turn into (prefix, suffix) around the image.

    ``apply_chat_template`` for SmolVLM yields e.g.::

        <|im_start|>User:<image>{question}<end_of_utterance>\\nAssistant:

    We split on the ``<image>`` marker so the frames stream into the position
    the image would occupy — turning it into a multi-image user turn.
    """
    from ..prompt_utils import apply_chat_template

    config = model.config
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    full = apply_chat_template(
        processor, config, messages, num_images=1, add_generation_prompt=True
    )
    marker = _image_marker(processor)
    if marker in full:
        prefix, suffix = full.split(marker, 1)
    else:  # pragma: no cover - defensive
        prefix, suffix = "", full
    return prefix, suffix


def _prefill_text(model, processor, text, cache, *, add_special_tokens: bool) -> dict:
    """Tokenize ``text`` and prefill it into ``cache`` (no images)."""
    tok = _tokenizer(processor)
    enc = tok(text, add_special_tokens=add_special_tokens, return_tensors="mlx")
    input_ids = enc["input_ids"]
    t0 = time.perf_counter()
    emb = model.get_input_embeddings(input_ids, None)
    mx.eval(emb.inputs_embeds)
    model.language_model(inputs=input_ids, inputs_embeds=emb.inputs_embeds, cache=cache)
    mx.eval([c.state for c in cache])
    return {
        "n_tokens": int(input_ids.shape[1]),
        "ms": (time.perf_counter() - t0) * 1000,
    }


def open_stream(model, processor, question: str, cache) -> str:
    """Ingest the ``User:`` opening prefix; return the closing suffix.

    Call once before streaming frames. The returned suffix (question +
    generation prompt) is what ``_answer`` prefills at query time.
    """
    prefix, suffix = _split_user_turn(model, processor, question)
    if prefix:
        _prefill_text(model, processor, prefix, cache, add_special_tokens=True)
    return suffix


def stream_generate_video(
    model: nn.Module,
    processor: Any,
    frames: Iterable[Any],
    question: str,
    *,
    cache: Optional[List[Any]] = None,
    sink: int = 64,
    window: int = 2048,
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
        sink/window: attention-sink size and sliding-window size for a fresh
            cache (ignored if ``cache`` is passed).
        answer_every: if set, emit an answer every N frames (rolling caption);
            otherwise answer once at end-of-stream.
        max_tokens: max answer tokens per query.

    Yields:
        dicts like ``{"frame": i, "text": <answer>, "kv_bytes": <int>}``.
    """
    if cache is None:
        cache = make_streaming_cache(model, sink=sink, window=window)

    suffix = open_stream(model, processor, question, cache)

    for i, frame in enumerate(frames):
        # Embed this frame's vision tokens and append them to the persistent
        # cache via one forward pass (logits discarded).
        _ingest_frame(model, processor, frame, cache)

        if answer_every and (i + 1) % answer_every == 0:
            ans = _answer(model, processor, suffix, cache, max_tokens, **kwargs)
            ans["frame"] = i
            yield ans

    if not answer_every:
        ans = _answer(model, processor, suffix, cache, max_tokens, **kwargs)
        ans["frame"] = None
        yield ans


# --------------------------------------------------------------------------
# The two hot spots.
# --------------------------------------------------------------------------
def _ingest_frame(model, processor, frame, cache) -> dict:
    """Embed one frame and append its vision KV to ``cache`` in place.

    Builds a single-frame ``(input_ids, pixel_values)`` with the processor (just
    the expanded ``<image>`` block — no chat wrapper, so only vision/structural
    tokens are streamed), runs the vision tower + projector via
    ``get_input_embeddings``, then does ONE ``language_model`` forward with the
    persistent ``cache`` so the frame's keys/values are appended. Logits are
    discarded. Returns per-frame stats for the demo.
    """
    image_token = _image_marker(processor)

    # Add BOS only if nothing has been ingested yet (i.e. ``open_stream`` was
    # not called). Normally the prefix already carries BOS, so frames append raw
    # image tokens.
    first = int(getattr(cache[0], "offset", 0)) == 0

    inputs = processor(
        text=[image_token],
        images=[frame],
        add_special_tokens=first,
        return_tensors="mlx",
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None and pixel_values.ndim == 4:
        # (N, C, H, W) -> (1, N, C, H, W) for the idefics3/smolvlm embed path.
        pixel_values = pixel_values[None]

    embed_kwargs = {}
    pam = inputs.get("pixel_attention_mask")
    if pam is not None:
        embed_kwargs["pixel_attention_mask"] = pam

    t0 = time.perf_counter()
    emb = model.get_input_embeddings(input_ids, pixel_values, **embed_kwargs)
    inputs_embeds = emb.inputs_embeds
    mx.eval(inputs_embeds)
    t1 = time.perf_counter()

    # One forward appending this frame's KV to the persistent cache.
    model.language_model(
        inputs=input_ids,
        inputs_embeds=inputs_embeds,
        cache=cache,
    )
    mx.eval([c.state for c in cache])
    t2 = time.perf_counter()

    return {
        "n_tokens": int(input_ids.shape[1]),
        "encode_ms": (t1 - t0) * 1000.0,
        "lm_ms": (t2 - t1) * 1000.0,
        "kv_bytes": kv_bytes(cache),
    }


def _answer(model, processor, suffix, cache, max_tokens, **kwargs) -> dict:
    """Decode an answer over the retained window via ``generate_step``.

    ``suffix`` is the closing of the user turn (question + generation prompt)
    returned by ``open_stream``. If a raw question is passed instead, its
    post-image suffix is derived on the fly. Passing ``prompt_cache=cache``
    means the streamed frame history is NOT re-prefilled — only the short suffix
    is prefilled on top of the window, then the answer is decoded.
    """
    if "Assistant" not in suffix:
        # A raw question was passed; build the proper user-turn suffix.
        _, suffix = _split_user_turn(model, processor, suffix)

    tok = _tokenizer(processor)
    enc = tok(suffix, add_special_tokens=False, return_tensors="mlx")
    question_ids = enc["input_ids"]

    stop = _stop_ids(model, processor)

    gen_kwargs = {"temperature": 0.0}
    gen_kwargs.update(kwargs)

    tic = time.perf_counter()
    ttft = None
    out_ids: List[int] = []
    for token, _ in generate_step(
        question_ids,
        model,
        None,  # pixel_values
        None,  # mask
        prompt_cache=cache,
        max_tokens=max_tokens,
        **gen_kwargs,
    ):
        if ttft is None:
            ttft = (time.perf_counter() - tic) * 1000.0
        if token in stop:
            break
        out_ids.append(int(token))

    text = tok.decode(out_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "kv_bytes": kv_bytes(cache),
        "ttft_ms": ttft,
        "answer_tokens": len(out_ids),
    }
