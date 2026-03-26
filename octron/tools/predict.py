"""
OCTRON prediction pipeline.

Wraps YOLO_octron.predict_batch() into a single callable function
that can be used from the CLI or called programmatically.
"""

import atexit
import os
import shutil
import tempfile
from collections import deque
from pathlib import Path
import time


def _is_network_path(path) -> bool:
    """Return True when *path* is a UNC / network-share path (\\\\server\\share or //server/share)."""
    s = str(os.fspath(path)).replace("\\", "/")
    return s.startswith("//")


def _transfer_results(temp_root: Path, final_root: Path) -> None:
    """Move completed octron_predictions/ sub-folders from *temp_root* to *final_root*.

    Each video writes one sub-folder inside octron_predictions/.  We move them
    one at a time so that a partial transfer (e.g. after interrupt) leaves as
    many complete video results as possible on the destination.
    """
    src = temp_root / "octron_predictions"
    if not src.exists():
        return
    dst_parent = final_root / "octron_predictions"
    dst_parent.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        dst = dst_parent / item.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(item), str(dst))


def run_predict(
    videos,
    model_path,
    tracker_name=None,
    tracker_cfg_path=None,
    tracker_params=None,
    device="auto",
    conf_thresh=0.5,
    iou_thresh=0.7,
    skip_frames=0,
    one_object_per_label=False,
    opening_radius=0,
    overwrite=False,
    buffer_size=200,
    region_properties=None,
    infer_batch_size=8,
    output_dir=None,
    debug=False,
    zarr_cache=None,
    video_cache=None,
):
    """
    Run YOLO prediction and tracking on one or more videos.

    Parameters
    ----------
    videos : str, Path, or list
        Single video path, or a list of video paths.
    model_path : str or Path
        Path to the trained YOLO model (.pt file).
    device : str
        Device to run inference on ('auto', 'cpu', 'cuda', 'mps'). 'auto'
        selects CUDA if available, then MPS, then CPU.
    tracker_name : str, optional
        Tracker to use (e.g. 'ByteTrack', 'BotSort'). Either this or
        tracker_cfg_path must be provided.
    tracker_cfg_path : str or Path, optional
        Path to a custom boxmot tracker config YAML. Takes priority over
        tracker_name when provided.
    tracker_params : dict, optional
        Parameter overrides applied on top of the resolved tracker config
        (e.g. {'det_thresh': 0.5, 'max_age': 100}).
    skip_frames : int
        Number of frames to skip between predictions.
    one_object_per_label : bool
        Track only the highest-confidence detection per label per frame.
    iou_thresh : float
        IOU threshold for detection.
    conf_thresh : float
        Confidence threshold for detection.
    opening_radius : int
        Morphological opening radius applied to masks to remove noise.
    overwrite : bool
        Overwrite existing prediction results.
    buffer_size : int
        Number of frames buffered before writing to zarr.
    region_properties : tuple or None
        Region property names to extract via skimage.measure.regionprops_table.
        Pass DEFAULT_REGION_PROPERTIES for the standard set, or None to skip.
    output_dir : str or Path, optional
        Root directory for octron_predictions output. Defaults to alongside
        each video file.
    debug : bool
        Enable DEBUG-level logging (per-stage timing diagnostics).
    zarr_cache : bool or None
        Cache zarr output to a local temp dir, then move to *output_dir* when
        each video finishes.  Avoids SMB atomic-write (os.replace) failures on
        network shares.  ``None`` (default) auto-enables when *output_dir* is a
        UNC / network path.  ``False`` suppresses auto-detection.
    video_cache : bool or None
        Copy each video to a local temp dir before decoding.  Eliminates
        network read bottlenecks for high-resolution or high-bitrate videos.
        ``None`` (default) auto-enables when the input video path is a UNC /
        network path.  ``False`` suppresses auto-detection.
    """
    from loguru import logger
    from octron.yolo_octron.yolo_octron import YOLO_octron
    from octron.test_gpu import auto_device

    # Silence boxmot's verbose INFO chatter (tracker init parameter dumps).
    logger.disable("boxmot")

    if device == "auto":
        device = auto_device()

    yolo = YOLO_octron()

    # --- local-cache setup ---------------------------------------------------
    # zarr_cache: write zarr output to a local temp dir, then move to output_dir.
    #   Required on SMB/network shares because zarr uses os.replace() which
    #   fails on Windows SMB.  Auto-enabled when output_dir is a network path.
    # video_cache: copy the video file locally before the decode thread opens
    #   it.  Eliminates network read bottlenecks for large/high-res videos.
    #   Auto-enabled when the input video path is a network path.
    # Both share a single temp dir to keep cleanup simple.
    _temp_dir: Path | None = None
    _final_output_dir = Path(output_dir) if output_dir is not None else None

    # Normalise videos to a flat list of Paths for network-path detection.
    if isinstance(videos, (str, Path)):
        _video_list: list[Path] = [Path(videos)]
    elif isinstance(videos, list):
        _video_list = [Path(v) for v in videos]
    else:
        _video_list = []  # dict format (GUI): skip auto-detection on video paths

    # Resolve each cache flag: explicit True/False overrides auto-detection;
    # None means "decide based on path".
    _do_zarr_cache: bool = (
        zarr_cache
        if zarr_cache is not None
        else (
            (_final_output_dir is not None and _is_network_path(_final_output_dir))
            or (_final_output_dir is None and any(_is_network_path(v) for v in _video_list))
        )
    )
    _do_video_cache: bool = (
        video_cache
        if video_cache is not None
        else any(_is_network_path(v) for v in _video_list)
    )

    if _do_zarr_cache or _do_video_cache:
        _temp_dir = Path(tempfile.mkdtemp(prefix="octron_cache_"))

        # Belt-and-suspenders: also register an atexit handler so the temp dir
        # is cleaned up even if the process is killed before the finally block
        # runs (e.g. SIGKILL, os._exit).
        def _atexit_cleanup():
            if _temp_dir.exists():
                shutil.rmtree(_temp_dir, ignore_errors=True)

        atexit.register(_atexit_cleanup)

        if _do_zarr_cache:
            effective_output_dir = str(_temp_dir)
            _dest_label = str(_final_output_dir) if _final_output_dir else "(alongside videos)"
            logger.info(f"Zarr cache:   {_temp_dir}  →  {_dest_label}")
        else:
            effective_output_dir = output_dir

        if _do_video_cache:
            logger.info(f"Video cache:  {_temp_dir / 'videos'}")
    else:
        effective_output_dir = output_dir
    # -------------------------------------------------------------------------

    _wall_start = time.time()
    # Consumer-side fps: timestamps of the last _FPS_WINDOW seconds of frames.
    # Time-based eviction means stall outliers fall out of the window immediately
    # after the stall ends, avoiding both mean-poisoning and median bimodal issues.
    _FPS_WINDOW = 5.0  # seconds
    _fps_ts: deque[float] = deque()
    _current_video_frames = 0
    _last_print_t = 0.0
    # In debug mode slow progress updates to 2 Hz so they don't fight with
    # timing diagnostic lines. In normal mode cap at 30 Hz to avoid
    # Windows CONPTY overhead.
    _PRINT_INTERVAL = 0.5 if debug else 1 / 30
    # Track whether a \r progress line is currently open so we can close it
    # (with a plain newline) before emitting any logger output.
    _progress_line_open = False
    # Set to True once the first processing frame of a video is seen so we
    # know there is a completed video to transfer before the next video_init.
    _video_has_results = False

    def _close_progress():
        nonlocal _progress_line_open
        if _progress_line_open:
            print()
            _progress_line_open = False

    def _flush_cache(label: str = "") -> None:
        """Transfer completed results from the local cache to the destination."""
        if not _do_zarr_cache or _temp_dir is None or _final_output_dir is None:
            return
        _close_progress()
        _t0 = time.time()
        suffix = f" ({label})" if label else ""
        logger.info(f"Transferring results to destination{suffix}...")
        _transfer_results(_temp_dir, _final_output_dir)
        logger.info(f"Transfer complete ({time.time() - _t0:.1f}s)")

    try:
        for progress in yolo.predict_batch(
            videos=videos,
            model_path=model_path,
            device=device,
            tracker_name=tracker_name,
            tracker_cfg_path=tracker_cfg_path,
            tracker_params=tracker_params,
            skip_frames=skip_frames,
            one_object_per_label=one_object_per_label,
            iou_thresh=iou_thresh,
            conf_thresh=conf_thresh,
            opening_radius=opening_radius,
            overwrite=overwrite,
            buffer_size=buffer_size,
            region_properties=region_properties,
            infer_batch_size=infer_batch_size,
            output_dir=effective_output_dir,
            video_cache_dir=str(_temp_dir / "videos") if _do_video_cache and _temp_dir is not None else None,
        ):
            stage = progress.get("stage", "")

            if stage == "predict_init":
                task     = progress.get("model_task", "?")
                imgsz    = progress.get("imgsz", "?")
                tracker  = progress.get("tracker", "?")
                n_videos = progress.get("total_videos", "?")
                skip     = progress.get("skip_frames", 0)
                model_p  = Path(progress.get("model_path", str(model_path)))
                flags = []
                if skip > 0:
                    flags.append(f"skip={skip}")
                if progress.get("one_object_per_label"):
                    flags.append("one-per-label")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                logger.info(f"Model:    {model_p.name}  [{task} · imgsz={imgsz}]")
                logger.info(f"Tracker:  {tracker}")
                logger.info(f"Device:   {device}")
                logger.info(f"Videos:   {n_videos}{flag_str}")
                continue

            if stage == "video_init":
                # Previous video is fully flushed to temp by the time predict_batch
                # yields the next video_init — safe to transfer now.
                if _video_has_results:
                    _flush_cache()
                    _video_has_results = False
                video_name = progress.get("video_name", "")
                vidx       = progress.get("video_index", 0) + 1
                total_v    = progress.get("total_videos", "?")
                num_frames = progress.get("num_frames", 0)
                save_dir   = progress.get("save_dir", "")
                _current_video_frames = num_frames
                _fps_ts.clear()  # reset fps window per video
                _close_progress()
                logger.info(f"Video {vidx}/{total_v}: {video_name}")
                logger.info(f"Frames:   {num_frames:,}")
                logger.info(f"Output:   {save_dir if not _temp_dir else str(_final_output_dir / 'octron_predictions' / Path(save_dir).name)}")
                continue

            if stage == "skipped_video":
                video_name = progress.get("video_name", "")
                save_dir   = progress.get("save_dir", "")
                _close_progress()
                logger.info(f"Video: {video_name}")
                logger.warning(f"Skipped — predictions already exist at {save_dir}. Use --overwrite to replace.")
                continue

            if stage != "processing":
                continue

            _video_has_results = True
            frame   = progress.get("frame", 0)
            total_f = _current_video_frames or progress.get("total_frames", 0)

            # Consumer-side fps: record arrival timestamp, evict entries older
            # than _FPS_WINDOW seconds, then compute fps from the window.
            now_t = time.time()
            _fps_ts.append(now_t)
            while _fps_ts[0] < now_t - _FPS_WINDOW:
                _fps_ts.popleft()
            if len(_fps_ts) >= 2:
                fps = (len(_fps_ts) - 1) / (_fps_ts[-1] - _fps_ts[0])
            else:
                fps = 0.0

            pct = 100.0 * frame / total_f if total_f > 0 else 0.0
            eta_s = int((total_f - frame) / fps) if fps > 0 else 0
            eta_str = f"{eta_s // 3600:02d}:{(eta_s % 3600) // 60:02d}:{eta_s % 60:02d}"

            if now_t - _last_print_t >= _PRINT_INTERVAL:
                if debug:
                    logger.info(
                        f"[progress] {frame:,}/{total_f:,} | {pct:.1f}% | {fps:.1f} fps | ETA: {eta_str}"
                    )
                else:
                    print(
                        f"\r\033[K  {frame:,}/{total_f:,} | {pct:.1f}% | {fps:.1f} fps | ETA: {eta_str}",
                        end="",
                        flush=True,
                    )
                    _progress_line_open = True
                _last_print_t = now_t

    except KeyboardInterrupt:
        _close_progress()
        if _video_has_results:
            _flush_cache("partial results")
        logger.warning("Interrupted — stopping prediction.")
        return

    else:
        # Transfer the last video's results (loop ended normally).
        if _video_has_results:
            _flush_cache()

    finally:
        # Always remove the local cache, whether we finished, interrupted, or
        # crashed.  _flush_cache() already moved completed results out, so
        # anything remaining here is genuinely incomplete.
        if _temp_dir is not None and _temp_dir.exists():
            shutil.rmtree(_temp_dir, ignore_errors=True)

    elapsed = time.time() - _wall_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Done. Total time: {h:02d}:{m:02d}:{s:02d}")
