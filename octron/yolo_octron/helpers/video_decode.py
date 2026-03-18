"""
Sequential video decoder with optional CUDA hardware acceleration.

Replaces seek-based (random-access) frame reading in the prediction pipeline
with a single sequential pass through the video container.  Two benefits:

1. **No seek overhead** — H.264/HEVC uses inter-frame compression; seeking to a
   non-keyframe forces decoding from the previous keyframe.  Sequential decode
   eliminates this entirely.

2. **Optional NVDEC** — when ``device='cuda'`` the decoder attempts to offload
   H.264/HEVC decompression to the GPU's dedicated video-decode engine (NVDEC),
   which runs independently of the CUDA cores used for inference.  Falls back
   to software decode silently if unavailable.
"""

import numpy as np


def _try_hw_container(video_path):
    """
    Try to open *video_path* with CUDA hwaccel.

    Returns ``(container, True)`` on success, ``(None, False)`` on failure.
    The caller is responsible for closing the container.
    """
    import av
    try:
        container = av.open(
            str(video_path),
            options={"hwaccel": "cuda", "hwaccel_output_format": "nv12"},
        )
        # Verify by decoding the very first frame — failure raises here.
        stream = container.streams.video[0]
        gen = container.decode(stream)
        frame = next(gen)
        frame.to_ndarray(format="rgb24")
        # Rewind for real use.
        container.seek(0, any_frame=False, backward=True)
        return container, True
    except Exception:
        try:
            container.close()
        except Exception:
            pass
        return None, False


def iter_frames_sequential(video_path, frame_iterator, device="cpu"):
    """
    Yield decoded video frames in order for the frames listed in
    *frame_iterator*.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file.
    frame_iterator : iterable of int
        Sorted frame indices to yield (e.g. ``range(0, n, skip+1)``).
        Must be in ascending order.
    device : str
        ``'cuda'`` to attempt NVDEC hardware decode; anything else uses
        software decode.

    Yields
    ------
    tuple[int, int, numpy.ndarray]
        ``(frame_no, frame_idx, rgb_array)`` where *frame_no* is the
        sequential batch index (0-based), *frame_idx* is the video frame
        index, and *rgb_array* is a ``(H, W, 3)`` uint8 RGB array.
    """
    import av

    wanted = {idx: no for no, idx in enumerate(frame_iterator)}
    if not wanted:
        return

    max_wanted = max(wanted)

    # --- open container -------------------------------------------------------
    hw_active = False
    container = None
    if device == "cuda":
        container, hw_active = _try_hw_container(video_path)
        if hw_active:
            print(f"  Video decode: NVDEC (CUDA hardware)")

    if container is None:
        container = av.open(str(video_path))
        if not hw_active:
            # Multi-threaded software decode — fastest CPU option.
            stream = container.streams.video[0]
            stream.codec_context.thread_type = "AUTO"
            stream.codec_context.thread_count = 0

    # --------------------------------------------------------------------------
    try:
        stream = container.streams.video[0]
        for frame_idx, av_frame in enumerate(container.decode(stream)):
            if frame_idx > max_wanted:
                break
            if frame_idx in wanted:
                frame_no = wanted[frame_idx]
                rgb = av_frame.to_ndarray(format="rgb24")
                yield frame_no, frame_idx, rgb
    finally:
        container.close()
