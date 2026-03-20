"""
Pipelined inference + tracking workers for OCTRON prediction.

Separates GPU inference (Stage 2) and CPU tracking (Stage 3) so all three
stages run concurrently:

    Stage 1  decode thread   → frame_queue      (CPU, video decode)
    Stage 2  infer  thread   → results_queue    (GPU, YOLO inference)
    Stage 3  tracker thread  → tracking_queue   (CPU, boxmot tracker)
    Stage 4  main generator  ← tracking_queue   (CPU, data accumulation)

The GPU no longer idles while tracker.update() runs on CPU, and the tracker
no longer waits for the main thread to finish accumulating data.

For non-ReID trackers (ByteTrack, etc.) the raw frame is dropped after
inference to keep results_queue memory bounded.  For ReID trackers
(BotSort, StrongSort, …) the frame is forwarded so the tracker can
compute appearance embeddings.  ReID backbones are forced to CPU when
threading to avoid concurrent CUDA operations with the inference thread.
"""

import threading
import time
from collections import deque

import numpy as np
from loguru import logger


# Sentinels consumed by workers to signal completion
INFER_DONE = object()
TRACK_DONE = object()


def run_inference_worker(
    frame_queue,
    results_queue,
    decode_done_sentinel,
    model,
    model_task,
    save_dir,
    rect,
    imgsz,
    conf_thresh,
    iou_thresh,
    device,
    retina_masks,
    infer_batch_size,
    is_reid,
):
    """
    Pull decoded frames from *frame_queue*, run batched YOLO inference,
    and push results onto *results_queue*.

    Each item pushed is a tuple::

        (frame_no, frame_idx, frame_or_none, result)

    where *frame_or_none* is the raw RGB numpy array for ReID trackers,
    or ``None`` for IoU-only trackers (saves ~6 MB per frame in the queue).

    When the decode thread signals completion via *decode_done_sentinel*,
    the worker drains any remaining pending frames, then pushes
    ``INFER_DONE`` and returns.

    Parameters
    ----------
    frame_queue : queue.Queue
        Input queue fed by the decode thread.
    results_queue : queue.Queue
        Output queue consumed by the tracker thread.
    decode_done_sentinel : object
        The sentinel object the decode thread places in *frame_queue* when
        it has no more frames to produce (``_DECODE_DONE``).
    model : ultralytics.YOLO
        Loaded YOLO model.
    model_task : str
        ``'segment'`` or ``'detect'``.
    save_dir : Path
        YOLO project/name for result saving (used only internally by YOLO).
    rect : bool
        Whether to use rectangular inference (non-square images).
    imgsz : int
        Inference image size.
    conf_thresh : float
        Confidence threshold.
    iou_thresh : float
        IOU threshold.
    device : str
        CUDA device string (e.g. ``'cuda'``, ``'cpu'``).
    retina_masks : bool
        Whether to use retina (full-resolution) masks.
    infer_batch_size : int
        Maximum number of frames per GPU inference call.
    is_reid : bool
        If True, forward the raw frame in the results tuple so the tracker
        can compute ReID embeddings.  If False, forward ``None`` instead.
    """
    _infer_times: deque = deque(maxlen=50)
    try:
        while True:
            # --- drain up to infer_batch_size frames from the decode queue ---
            pending = []
            decode_finished = False
            for _ in range(infer_batch_size):
                item = frame_queue.get()
                if item is decode_done_sentinel:
                    decode_finished = True
                    break
                pending.append(item)

            if not pending:
                break

            # --- GPU inference ------------------------------------------------
            _t_infer = time.perf_counter()
            results_list = model.predict(
                source=[p[2] for p in pending],
                task=model_task,
                project=save_dir.parent.as_posix(),
                name=save_dir.name,
                show=False,
                rect=rect,
                save=False,
                verbose=False,
                imgsz=imgsz,
                max_det=100,
                conf=conf_thresh,
                iou=iou_thresh,
                device=device,
                retina_masks=retina_masks,
                save_txt=False,
                save_conf=False,
            )

            # --- record inference timing -----------------------------------
            _dt_infer = time.perf_counter() - _t_infer
            _n = len(pending)
            for _ in range(_n):
                _infer_times.append(_dt_infer / _n)
            if len(_infer_times) == _infer_times.maxlen:
                avg_ms = 1000 * sum(_infer_times) / len(_infer_times)
                logger.debug(
                    f"[infer]  {avg_ms:.1f} ms/frame  ({1000/avg_ms:.0f} fps)"
                    f"  batch={_n}  frame_q={frame_queue.qsize()}"
                    f"  results_q={results_queue.qsize()}"
                )
                _infer_times.clear()

            # --- push results to tracker stage --------------------------------
            for (frame_no, frame_idx, frame), result in zip(pending, results_list):
                frame_payload = frame if is_reid else None
                results_queue.put((frame_no, frame_idx, frame_payload, result))

            del results_list, pending

            if decode_finished:
                break

    except Exception as e:
        # Propagate the exception via the queue so the tracker thread can raise it
        results_queue.put(e)

    results_queue.put(INFER_DONE)


def run_tracker_worker(
    results_queue,
    tracking_queue,
    infer_done_sentinel,
    tracker,
    is_segment,
    per_class,
    map_detection_index_fn,
):
    """
    Pull inference results from *results_queue*, run the boxmot tracker
    sequentially, and push per-frame tracking summaries onto *tracking_queue*.

    The tracker must be called in strict frame order (it is stateful), so
    this worker processes one frame at a time.  GPU→CPU data transfers also
    happen here so the main thread only handles plain numpy arrays.

    Each item pushed to *tracking_queue* is a dict with keys:

    ``frame_no``, ``frame_idx``, ``frame``
        Frame counter, zarr index, and raw RGB array (may be None for
        non-ReID trackers when region_details is False).

    ``status``
        ``'ok'`` — at least one confirmed track this frame.
        ``'no_result'`` — YOLO produced no output (AttributeError on boxes).
        ``'no_track'`` — tracker returned zero confirmed tracks.

    When ``status == 'ok'`` the dict additionally contains:

    ``tracked_ids``, ``tracked_label_names``, ``tracked_confidences``,
    ``tracked_boxes``, ``tracked_masks``, ``result_names``
        Filtered track data ready for data accumulation.

    Parameters
    ----------
    results_queue : queue.Queue
        Input queue fed by the inference worker.
    tracking_queue : queue.Queue
        Output queue consumed by the main generator.
    infer_done_sentinel : object
        The sentinel the inference worker places in *results_queue* when
        it is finished (``INFER_DONE``).
    tracker : boxmot tracker
        Stateful boxmot tracker instance.  Must be called sequentially.
    is_segment : bool
        True for segmentation models (masks are extracted); False for detect.
    per_class : bool
        Passed to map_detection_index_fn.
    map_detection_index_fn : callable
        Bound method ``YOLO_octron.map_detection_index``.
    """
    _track_times: deque = deque(maxlen=100)
    try:
        while True:
            item = results_queue.get()
            if isinstance(item, Exception):
                tracking_queue.put(item)
                break
            if item is infer_done_sentinel:
                break

            frame_no, frame_idx, frame, result = item

            # --- GPU → CPU transfers -----------------------------------------
            try:
                confidences  = result.boxes.conf.cpu().numpy()
                classes      = result.boxes.cls.cpu().numpy()
                label_names  = tuple(result.names[int(c)] for c in classes)
                boxes        = result.boxes.xyxy.cpu().numpy()
                masks        = result.masks.data.cpu().numpy() if is_segment else None
                result_names = result.names
            except AttributeError:
                del result
                tracking_queue.put({
                    'frame_no': frame_no,
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'status': 'no_result',
                })
                continue
            finally:
                # Release the YOLO result object immediately — it holds references to
                # GPU tensors that won't be freed until the object is GC'd otherwise.
                del result

            # --- tracker update ----------------------------------------------
            tracker_input = np.hstack([
                boxes,
                confidences[:, np.newaxis],
                classes[:, np.newaxis],
            ])
            # boxmot validates that the image argument is an ndarray even for
            # IoU-only trackers that don't actually use it.  When the full frame
            # was not forwarded (non-ReID, no region_details) pass a 1×1 dummy
            # so the validation passes without the 6 MB memory traffic.
            tracker_frame = frame if frame is not None else np.zeros((1, 1, 3), dtype=np.uint8)
            _t_track = time.perf_counter()
            tracking_result = tracker.update(tracker_input, tracker_frame)

            if tracking_result.shape[0] == 0:
                _track_times.append(time.perf_counter() - _t_track)
                tracking_queue.put({
                    'frame_no': frame_no,
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'status': 'no_track',
                })
                continue

            # --- map detection indices and filter arrays ----------------------
            tracked_ids, tracked_idxs = map_detection_index_fn(
                tracker_input, tracking_result, per_class=per_class, verbose=False,
            )
            if not tracked_idxs:
                _track_times.append(time.perf_counter() - _t_track)
                tracking_queue.put({
                    'frame_no': frame_no,
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'status': 'no_track',
                })
                continue

            # Convert only the tracked masks to int8 here (tracker thread) so
            # the main thread's chunk-buffer write becomes a plain int8→int8
            # memcpy instead of a float32→int8 type-converting copy.
            #
            # We write each mask directly into a pre-allocated int8 output via
            # np.copyto(casting='unsafe') rather than masks[tracked_idxs].astype()
            # — the latter first fancy-indexes a new float32 array (allocating
            # ~50 MB) and then type-converts it, doubling the memory traffic.
            # The loop below does a single type-converting write per mask (~10 MB
            # each) with no intermediate allocation.
            if is_segment:
                n = len(tracked_idxs)
                _, H, W = masks.shape
                tracked_masks = np.empty((n, H, W), dtype='int8')
                for i, idx in enumerate(tracked_idxs):
                    np.copyto(tracked_masks[i], masks[idx], casting='unsafe')
            else:
                tracked_masks = [None] * len(tracked_idxs)

            _track_times.append(time.perf_counter() - _t_track)
            if len(_track_times) == _track_times.maxlen:
                avg_ms = 1000 * sum(_track_times) / len(_track_times)
                logger.debug(
                    f"[track]  {avg_ms:.1f} ms/frame  ({1000/avg_ms:.0f} fps)"
                    f"  results_q={results_queue.qsize()}"
                    f"  track_q={tracking_queue.qsize()}"
                )
                _track_times.clear()

            tracking_queue.put({
                'frame_no':              frame_no,
                'frame_idx':             frame_idx,
                'frame':                 frame,
                'status':                'ok',
                'tracked_ids':           tracked_ids,
                'tracked_label_names':   [label_names[i] for i in tracked_idxs],
                'tracked_confidences':   confidences[tracked_idxs],
                'tracked_boxes':         boxes[tracked_idxs],
                'tracked_masks':         tracked_masks,
                'result_names':          result_names,
            })

    except Exception as e:
        tracking_queue.put(e)

    tracking_queue.put(TRACK_DONE)
