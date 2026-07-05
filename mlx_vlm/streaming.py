"""Incremental video frame ingestion for streaming VLM inference.

The incremental counterpart to ``utils.load_video`` (``utils.py:1418``), which
pre-samples *all* frames up front. ``stream_video_frames`` instead yields one
PIL frame at a time from a file or a live camera, so a streaming generator can
ingest an open-ended stream through a bounded KV cache.

Status: SCAFFOLD but functional for file/webcam decode. Preprocessing into
model tensors is left to the processor in ``generate/video.py``.
"""

from __future__ import annotations

from typing import Iterator, Union

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is a declared dependency
    cv2 = None

from PIL import Image


def stream_video_frames(
    source: Union[str, int],
    fps: float = 1.0,
    max_frames: int | None = None,
) -> Iterator[Image.Image]:
    """Yield frames one at a time from a video file or camera.

    Args:
        source: path to a video file, or an int camera index (e.g. ``0``) for a
            live webcam stream.
        fps: target sampling rate. Frames are subsampled to approximate this
            rate from the source's native fps.
        max_frames: optional cap on the number of frames yielded (``None`` =
            unbounded — the point of streaming).

    Yields:
        ``PIL.Image`` RGB frames, in temporal order.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for stream_video_frames")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"could not open video source: {source!r}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(native_fps / max(fps, 1e-6))))

    emitted = 0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(rgb)
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
