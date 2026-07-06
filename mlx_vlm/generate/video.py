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
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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


def _is_llava_family(model: nn.Module, processor: Any) -> bool:
    """True for LLaVA-style image handling (a negative image-token placeholder).

    FastVLM (``model_type='llava_qwen2'``) inserts a SINGLE negative placeholder
    token (``image_token_index = -200``) into ``input_ids`` that
    ``get_input_embeddings`` then expands into H*W vision tokens — unlike
    SmolVLM/Idefics3, where the processor expands the ``<image>`` marker into
    placeholder tokens up front so ``input_ids`` already matches the vision-token
    count. The two conventions need different per-frame tensor plumbing (below),
    but both feed the same ``get_input_embeddings`` -> ``language_model(cache=)``
    mechanism, so the bounded-KV streaming loop is otherwise identical.
    """
    idx = getattr(processor, "image_token_index", None)
    if isinstance(idx, int) and idx < 0:
        return True
    mt = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    return mt in ("llava_qwen2", "llava-qwen2")


def _build_frame_inputs(
    model: nn.Module, processor: Any, frame: Any, *, first: bool
) -> Tuple[mx.array, Optional[mx.array], Dict[str, Any]]:
    """Build one frame's ``(input_ids, pixel_values, embed_kwargs)`` per family.

    Both families run the result through ``get_input_embeddings`` ->
    ``language_model(cache=)``; they differ ONLY in per-frame tensor layout:

    * SmolVLM/Idefics3 — the processor expands ``<image>`` into many placeholder
      tokens (``input_ids`` already matches the vision-token count), and
      ``pixel_values`` is lifted to ``(1, N, C, H, W)`` with an optional
      ``pixel_attention_mask``.
    * LLaVA/FastVLM — the processor emits a single ``-200`` placeholder and
      ``pixel_values`` stays ``(B, C, H, W)``; the placeholder is expanded into
      H*W vision tokens inside ``get_input_embeddings`` (via
      ``config.image_token_index``), so ``input_ids`` is length-1 while
      ``inputs_embeds`` is length H*W. ``Qwen2Model`` keys sequence length off
      the embeddings, so the length mismatch is expected and correct.
    """
    image_token = _image_marker(processor)
    inputs = processor(
        text=[image_token],
        images=[frame],
        add_special_tokens=first,
        return_tensors="mlx",
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    embed_kwargs: Dict[str, Any] = {}
    if _is_llava_family(model, processor):
        # FastVLM: keep pixel_values as (B, C, H, W); no pixel_attention_mask.
        # Its processor returns numpy pixel_values even with return_tensors="mlx"
        # (the non-tensor ``image_sizes`` field defeats BatchFeature's mlx cast),
        # so coerce to mx.array here — the vision tower expects an mlx tensor.
        if input_ids is not None and not isinstance(input_ids, mx.array):
            input_ids = mx.array(input_ids)
        if pixel_values is not None and not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)
        return input_ids, pixel_values, embed_kwargs
    if pixel_values is not None and pixel_values.ndim == 4:
        # (N, C, H, W) -> (1, N, C, H, W) for the idefics3/smolvlm embed path.
        pixel_values = pixel_values[None]
    pam = inputs.get("pixel_attention_mask")
    if pam is not None:
        embed_kwargs["pixel_attention_mask"] = pam
    return input_ids, pixel_values, embed_kwargs


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
    segment_aware: bool = False,
    vision_window: int = 8,
    keep_first_frame: bool = True,
    evict_slack: int = 0,
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
        cache: reuse an existing streaming cache, or a fresh one is made
            (v1 RotatingKVCache path only).
        sink/window: attention-sink size and sliding-window size for a fresh
            cache (ignored if ``cache`` is passed).
        answer_every: if set, emit an answer every N frames (rolling caption);
            otherwise answer once at end-of-stream.
        max_tokens: max answer tokens per query.
        segment_aware: use the Phase-1 segment-aware path — keep sink + ALL text
            + the ``vision_window`` most-recent frames, evict oldest vision only,
            and recompute-re-anchor on eviction (``SegmentAwareSession``).
        vision_window/keep_first_frame/evict_slack: segment-aware retention knobs.

    Yields:
        dicts like ``{"frame": i, "text": <answer>, "kv_bytes": <int>}``.
    """
    if segment_aware:
        yield from _stream_segment_aware(
            model,
            processor,
            frames,
            question,
            answer_every=answer_every,
            max_tokens=max_tokens,
            vision_window=vision_window,
            keep_first_frame=keep_first_frame,
            evict_slack=evict_slack,
            **kwargs,
        )
        return

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


def _stream_segment_aware(
    model: nn.Module,
    processor: Any,
    frames: Iterable[Any],
    question: str,
    *,
    answer_every: Optional[int],
    max_tokens: int,
    vision_window: int,
    keep_first_frame: bool,
    evict_slack: int,
    **kwargs,
) -> Iterator[dict]:
    """Segment-aware streaming loop over a ``SegmentAwareSession`` (Phase 1)."""
    session = SegmentAwareSession(
        model,
        processor,
        vision_window=vision_window,
        keep_first_frame=keep_first_frame,
        evict_slack=evict_slack,
    )
    session.begin(question)
    for i, frame in enumerate(frames):
        st = session.ingest_frame(frame)
        if answer_every and (i + 1) % answer_every == 0:
            ans = session.answer(max_tokens, **kwargs)
            ans["frame"] = i
            ans["kv_bytes"] = st["kv_bytes"]
            ans["evicted"] = st["evicted"]
            yield ans
    if not answer_every:
        ans = session.answer(max_tokens, **kwargs)
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
    # Add BOS only if nothing has been ingested yet (i.e. ``open_stream`` was
    # not called). Normally the prefix already carries BOS, so frames append raw
    # image tokens.
    first = int(getattr(cache[0], "offset", 0)) == 0

    input_ids, pixel_values, embed_kwargs = _build_frame_inputs(
        model, processor, frame, first=first
    )

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
        # Vision-token count = the KV positions actually appended. For FastVLM
        # ``input_ids`` is a length-1 placeholder, so read the embeddings length.
        "n_tokens": int(inputs_embeds.shape[1]),
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


# ==========================================================================
# Phase 1 — segment-aware KV eviction with recompute re-anchoring.
# ==========================================================================
#
# WHY a whole new path (see docs/streaming-video-kv-plan.md §6 risk #1):
#   RoPE is baked into cached keys at their TRUE positions and the query is
#   roped at ``cache.offset`` — so relative distances stay correct even after a
#   drop. BUT ``create_causal_mask`` (models/cache.py) builds its mask from a
#   DENSE ``arange(offset + N)``: it assumes a key exists at EVERY position.
#   Drop a *middle* vision span and keep the survivors at their true positions
#   and the mask can no longer describe the hole without model-file changes.
#
#   Phase-1 answer (correctness-first): after evicting the oldest vision frames
#   we RE-ANCHOR the survivors to CONTIGUOUS positions 0..M by REBUILDING the
#   per-layer KV cache — re-prefilling each retained "unit" in order into a
#   fresh dense cache. The model then always sees a normal, gap-free cache.
#   This is O(retained window) per eviction — bounded, not O(T). (In-place
#   rotary-shift re-anchoring is a later optimization, deliberately out of
#   scope here.)
#
# The retained context is an ordered list of UNITS:
#   * "prefix" — the opening ``User:`` block; pinned forever (the sink).
#   * "text"   — any streamed instruction/question text; ALL text is pinned.
#   * "vision" — one frame's vision tokens; evictable (oldest first), except an
#     optional pinned first frame that extends the sink.
# Each unit stores what is needed to re-prefill it: its ``input_ids`` and the
# already-computed ``inputs_embeds`` (so a rebuild never re-runs the vision
# tower — it only re-runs the language model, which is what re-anchors RoPE).


@dataclass
class _Unit:
    """One re-prefillable span of the streamed context."""

    kind: str  # "prefix" | "text" | "vision"
    input_ids: mx.array  # (1, n) token ids (image placeholders for vision)
    inputs_embeds: mx.array  # (1, n, d) precomputed embeddings (vision baked in)
    n_tokens: int
    permanent: bool  # pinned (sink prefix, all text, optional first frame)
    tag: Any = None  # debugging label (e.g. frame index / text preview)

    @property
    def is_evictable_vision(self) -> bool:
        return self.kind == "vision" and not self.permanent


class SegmentAwareSession:
    """Segment-aware streaming-video session (StreamingVLM retention recipe).

    Keeps the attention-sink (opening prefix + optional first frame) + ALL text
    forever, and only the ``vision_window`` most-recent frames of vision; the
    oldest vision frames evict. On eviction the whole retained cache is rebuilt
    by re-prefilling the surviving units in order (recompute re-anchor), which
    lands them at contiguous RoPE positions and presents a dense cache to the
    model.

    Args:
        model: a loaded VLM (SmolVLM2 for Phase 1; 1-D RoPE only — see plan §6.2).
        processor: the model's processor.
        vision_window: number of most-recent *evictable* frames to retain.
        keep_first_frame: also pin the very first frame into the sink.
        evict_slack: allow this many extra frames past ``vision_window`` before
            triggering a rebuild (amortizes rebuild cost; 0 = rebuild as soon as
            the window is exceeded, giving perfectly flat KV bytes).
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        *,
        vision_window: int = 8,
        keep_first_frame: bool = True,
        evict_slack: int = 0,
    ):
        self.model = model
        self.processor = processor
        self.vision_window = int(vision_window)
        self.keep_first_frame = bool(keep_first_frame)
        self.evict_slack = max(0, int(evict_slack))

        self.units: List[_Unit] = []
        self.cache: List[Any] = self._fresh_cache()
        self.suffix: Optional[str] = None

        self._n_vision_ingested = 0
        self._n_rebuilds = 0
        # Next-token logits at the last cache position — the handle TEST A
        # compares against a fresh contiguous reference.
        self.last_logits: Optional[mx.array] = None

    # -- cache plumbing ---------------------------------------------------
    def _fresh_cache(self) -> List[Any]:
        """A fresh dense (KVCache) per-layer cache: offset starts at 0."""
        return _cache.make_prompt_cache(self.model.language_model)

    def _forward_unit(self, unit: _Unit) -> mx.array:
        """Append ``unit`` to ``self.cache`` via one LM forward; return logits."""
        out = self.model.language_model(
            inputs=unit.input_ids,
            inputs_embeds=unit.inputs_embeds,
            cache=self.cache,
        )
        return out.logits

    # -- unit construction ------------------------------------------------
    def _make_text_unit(
        self, text: str, *, add_special_tokens: bool, kind: str, permanent: bool
    ) -> _Unit:
        tok = _tokenizer(self.processor)
        enc = tok(text, add_special_tokens=add_special_tokens, return_tensors="mlx")
        input_ids = enc["input_ids"]
        emb = self.model.get_input_embeddings(input_ids, None)
        inputs_embeds = emb.inputs_embeds
        mx.eval(inputs_embeds)
        return _Unit(
            kind=kind,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            n_tokens=int(input_ids.shape[1]),
            permanent=permanent,
            tag=(text[:24] + "…") if len(text) > 24 else text,
        )

    def _make_vision_unit(self, frame: Any, permanent: bool) -> _Unit:
        # BOS only if nothing has been ingested yet (normally the prefix carries
        # it, so frames append raw image tokens).
        first = int(getattr(self.cache[0], "offset", 0)) == 0
        input_ids, pixel_values, embed_kwargs = _build_frame_inputs(
            self.model, self.processor, frame, first=first
        )
        emb = self.model.get_input_embeddings(input_ids, pixel_values, **embed_kwargs)
        inputs_embeds = emb.inputs_embeds
        mx.eval(inputs_embeds)
        return _Unit(
            kind="vision",
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            # KV positions appended = embeddings length (FastVLM's input_ids is a
            # length-1 placeholder; the vision tokens live in inputs_embeds).
            n_tokens=int(inputs_embeds.shape[1]),
            permanent=permanent,
            tag=f"frame{self._n_vision_ingested}",
        )

    # -- public streaming API --------------------------------------------
    def begin(self, question: str) -> str:
        """Ingest the opening ``User:`` prefix (the sink); store the suffix."""
        prefix, suffix = _split_user_turn(self.model, self.processor, question)
        self.suffix = suffix
        if prefix:
            unit = self._make_text_unit(
                prefix, add_special_tokens=True, kind="prefix", permanent=True
            )
            self.last_logits = self._forward_unit(unit)[:, -1, :]
            self.units.append(unit)
            mx.eval([c.state for c in self.cache])
        return suffix

    def ingest_text(self, text: str) -> Dict[str, Any]:
        """Ingest a text instruction mid-stream; text is pinned forever."""
        unit = self._make_text_unit(
            text, add_special_tokens=False, kind="text", permanent=True
        )
        self.last_logits = self._forward_unit(unit)[:, -1, :]
        self.units.append(unit)
        mx.eval([c.state for c in self.cache])
        return {
            "kind": "text",
            "n_tokens": unit.n_tokens,
            "kv_bytes": kv_bytes(self.cache),
        }

    def ingest_frame(self, frame: Any) -> Dict[str, Any]:
        """Ingest one frame's vision tokens; evict + re-anchor if past window."""
        t0 = time.perf_counter()
        permanent = self.keep_first_frame and self._n_vision_ingested == 0
        unit = self._make_vision_unit(frame, permanent)
        t1 = time.perf_counter()
        self.last_logits = self._forward_unit(unit)[:, -1, :]
        self.units.append(unit)
        self._n_vision_ingested += 1
        mx.eval([c.state for c in self.cache])
        t2 = time.perf_counter()

        evicted = self._maybe_evict()
        t3 = time.perf_counter()
        return {
            "n_tokens": unit.n_tokens,
            "encode_ms": (t1 - t0) * 1000.0,
            "lm_ms": (t2 - t1) * 1000.0,
            "evict_ms": (t3 - t2) * 1000.0,
            "kv_bytes": kv_bytes(self.cache),
            "evicted": [u.tag for u in evicted],
            "retained_frames": self._retained_frame_count(),
            "n_rebuilds": self._n_rebuilds,
        }

    def answer(self, max_tokens: int = 64, **kwargs) -> Dict[str, Any]:
        """Decode an answer over the retained window, then restore the cache.

        ``_answer`` prefills the suffix onto ``self.cache`` and decodes — which
        appends transient KV. We rebuild afterward so the streaming cache is
        left holding only the retained units (safe for rolling-caption use).
        """
        if self.suffix is None:
            raise RuntimeError("call begin(question) before answer()")
        ans = _answer(
            self.model, self.processor, self.suffix, self.cache, max_tokens, **kwargs
        )
        self._rebuild()  # drop the transient suffix + answer KV
        return ans

    # -- retention policy -------------------------------------------------
    def _retained_frame_count(self) -> int:
        return sum(1 for u in self.units if u.kind == "vision")

    def _maybe_evict(self) -> List[_Unit]:
        """Drop oldest evictable vision frames past the window; rebuild if any."""
        evictable = [u for u in self.units if u.is_evictable_vision]
        if len(evictable) <= self.vision_window + self.evict_slack:
            return []
        n_drop = len(evictable) - self.vision_window  # back down to the window
        drop_ids = {id(u) for u in evictable[:n_drop]}
        dropped = [u for u in self.units if id(u) in drop_ids]
        self.units = [u for u in self.units if id(u) not in drop_ids]
        self._rebuild()
        return dropped

    def _rebuild(self) -> None:
        """Recompute re-anchor: re-prefill retained units into a fresh cache.

        A fresh cache resets ``offset`` to 0, so re-forwarding the surviving
        units in order lands them at CONTIGUOUS positions 0..M — presenting a
        dense, gap-free cache to the model (the whole point of Phase 1).
        """
        self.cache = self._fresh_cache()
        last = None
        for unit in self.units:
            last = self._forward_unit(unit)
        mx.eval([c.state for c in self.cache])
        if last is not None:
            self.last_logits = last[:, -1, :]
            mx.eval(self.last_logits)
        self._n_rebuilds += 1

    # -- correctness reference (TEST A) ----------------------------------
    def retained_tags(self) -> List[str]:
        """Ordered tags of the currently-retained units (for verification)."""
        return [str(u.tag) for u in self.units]

    def reference_logits(self, mode: str = "mono") -> mx.array:
        """Next-token logits from a FRESH prefill of the retained units.

        Two ways to feed "the exact retained unit sequence fresh, no eviction":

        * ``mode="chunked"`` — re-prefill the units one-by-one into a fresh cache
          (exactly how a clean stream would ingest them). This is the apples-to-
          apples reference for the re-anchor: it isolates eviction/re-anchor
          correctness from prefill-chunking, so the streaming cache should match
          it to within deterministic bit-noise.
        * ``mode="mono"`` — concatenate every unit's ids/embeds and run ONE
          contiguous forward. This is the strictest "contiguous, no eviction"
          reference, but it differs from ANY chunked prefill (evicted or not) by
          fp16 chunk-boundary accumulation in attention (argmax is unaffected).
        """
        if not self.units:
            raise RuntimeError("no retained units to reference")
        fresh = self._fresh_cache()
        if mode == "chunked":
            last = None
            for unit in self.units:
                out = self.model.language_model(
                    inputs=unit.input_ids,
                    inputs_embeds=unit.inputs_embeds,
                    cache=fresh,
                )
                last = out.logits
            logits = last[:, -1, :]
        else:
            ids = mx.concatenate([u.input_ids for u in self.units], axis=1)
            embeds = mx.concatenate([u.inputs_embeds for u in self.units], axis=1)
            out = self.model.language_model(
                inputs=ids, inputs_embeds=embeds, cache=fresh
            )
            logits = out.logits[:, -1, :]
        mx.eval(logits)
        return logits
