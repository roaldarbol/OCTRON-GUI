"""
Pipelined inference worker for OCTRON prediction.

Separates GPU inference (Stage 2) from CPU tracking (Stage 3) so they
run concurrently:

    Stage 1  decode thread   → frame_queue
    Stage 2  THIS worker     → results_queue      (GPU, background thread)
    Stage 3  main generator  ← results_queue      (CPU tracking + data)

The GPU no longer idles while tracker.update() runs on CPU.

For non-ReID trackers (ByteTrack, etc.) the raw frame is dropped after
inference to keep results_queue memory bounded.  For ReID trackers
(BotSort, StrongSort, …) the frame is forwarded so the tracker can
compute appearance embeddings.
"""

import threading


# Sentinel consumed by the inference worker to signal it is finished
INFER_DONE = object()


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
        Output queue consumed by the tracking stage (main thread).
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

            # --- push results to tracking stage --------------------------------
            for (frame_no, frame_idx, frame), result in zip(pending, results_list):
                frame_payload = frame if is_reid else None
                results_queue.put((frame_no, frame_idx, frame_payload, result))

            del results_list, pending

            if decode_finished:
                break

    except Exception as e:
        # Propagate the exception via the queue so the main thread can raise it
        results_queue.put(e)

    results_queue.put(INFER_DONE)
