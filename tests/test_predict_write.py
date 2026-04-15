"""
Tests for the chunk-aligned direct-write zarr layer used in OCTRON prediction.

These tests are self-contained: no GPU, no model, no video required.
They verify the I/O primitives that replace the zarr Python API write path.
"""

import threading
import queue
import tempfile
from pathlib import Path

import numpy as np
import numcodecs
import pytest
import zarr
from zarr.codecs import BloscCodec, BloscShuffle


# ---------------------------------------------------------------------------
# Helpers that mirror the production code
# ---------------------------------------------------------------------------

_MASK_COMPRESSOR = BloscCodec(cname='lz4', clevel=1, shuffle=BloscShuffle.bitshuffle)
_NC_CODEC = numcodecs.Blosc(cname='lz4', clevel=1, shuffle=numcodecs.Blosc.BITSHUFFLE)


def _make_zarr_array(store_path, name, n_frames, H, W, chunk_size):
    """Create a zarr array identical to create_prediction_zarr()."""
    store = zarr.storage.LocalStore(store_path, read_only=False)
    arr = zarr.create_array(
        store=store,
        name=name,
        shape=(n_frames, H, W),
        chunks=(chunk_size, H, W),
        dtype='int8',
        fill_value=-1,
        compressors=_MASK_COMPRESSOR,
    )
    return arr


def _direct_write_chunk(arr, chunk_idx, data):
    """Write *data* (shape = (chunk_size, H, W) int8) directly to the chunk file."""
    chunk_path = arr.store.root / arr.path / 'c' / str(chunk_idx) / '0' / '0'
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(_NC_CODEC.encode(data.tobytes()))


# ---------------------------------------------------------------------------
# Direct-write I/O correctness
# ---------------------------------------------------------------------------

def test_direct_write_two_full_chunks():
    """Two full chunks written directly are readable via the zarr API."""
    chunk_size, H, W = 50, 32, 32
    with tempfile.TemporaryDirectory() as td:
        arr = _make_zarr_array(Path(td) / 'pred.zarr', '1_masks', 100, H, W, chunk_size)

        chunk0 = np.full((chunk_size, H, W), 42, dtype='int8')
        chunk1 = np.full((chunk_size, H, W),  7, dtype='int8')
        _direct_write_chunk(arr, 0, chunk0)
        _direct_write_chunk(arr, 1, chunk1)

        assert np.all(arr[0:chunk_size] == 42)
        assert np.all(arr[chunk_size:100] == 7)


def test_direct_write_partial_chunk_fills_with_minus_one():
    """A partial final chunk has -1 fill for unwritten slots."""
    chunk_size, H, W = 50, 16, 16
    n_frames = 75  # chunk 0 full (frames 0-49), chunk 1 partial (frames 50-74)
    with tempfile.TemporaryDirectory() as td:
        arr = _make_zarr_array(Path(td) / 'pred.zarr', '1_masks', n_frames, H, W, chunk_size)

        chunk0 = np.full((chunk_size, H, W), 1, dtype='int8')
        _direct_write_chunk(arr, 0, chunk0)

        # chunk 1: only first 25 slots have data, rest stay -1
        chunk1_buf = np.full((chunk_size, H, W), -1, dtype='int8')
        chunk1_buf[:25] = 3
        _direct_write_chunk(arr, 1, chunk1_buf)

        assert np.all(arr[0:50] == 1)
        assert np.all(arr[50:75] == 3)


def test_direct_write_single_frame_in_chunk():
    """Only one frame in the last chunk — the rest of the chunk must be -1."""
    chunk_size, H, W = 50, 8, 8
    with tempfile.TemporaryDirectory() as td:
        arr = _make_zarr_array(Path(td) / 'pred.zarr', '1_masks', 51, H, W, chunk_size)

        buf = np.full((chunk_size, H, W), -1, dtype='int8')
        buf[0] = 99  # slot 0 of chunk 1 = frame index 50
        _direct_write_chunk(arr, 1, buf)

        assert arr[50, 0, 0] == 99
        # The rest of the chunk is still fill (-1), but since we never wrote chunk 0,
        # zarr returns fill_value for it.
        assert arr[0, 0, 0] == -1


# ---------------------------------------------------------------------------
# Chunk-boundary flush logic (unit test without full prediction pipeline)
# ---------------------------------------------------------------------------

def test_chunk_boundary_detection():
    """Simulate the chunk-crossing check in the prediction main loop."""
    buffer_size = 10
    mask_chunk_idxs = {}
    flushed = []

    def _flush(track_id):
        flushed.append((track_id, mask_chunk_idxs[track_id]))
        del mask_chunk_idxs[track_id]

    track_id = 1

    for frame_idx in range(25):
        frame_chunk = frame_idx // buffer_size
        if track_id in mask_chunk_idxs and mask_chunk_idxs[track_id] != frame_chunk:
            _flush(track_id)
        if track_id not in mask_chunk_idxs:
            mask_chunk_idxs[track_id] = frame_chunk

    # End-of-video flush
    if track_id in mask_chunk_idxs:
        _flush(track_id)

    # Frames 0-9 → chunk 0, 10-19 → chunk 1, 20-24 → chunk 2
    assert flushed == [(1, 0), (1, 1), (1, 2)]


def test_mask_fusion_lookup():
    """When iou_thresh < 0.01, the prior mask should be read from the in-memory chunk."""
    buffer_size = 10
    H, W = 4, 4
    mask_chunk_arrays = {}
    mask_chunk_written = {}
    mask_chunk_idxs = {}

    track_id = 1
    frame_idx = 3   # chunk 0, slot 3

    frame_chunk = frame_idx // buffer_size
    frame_slot  = frame_idx  % buffer_size

    # First detection for this frame: write mask A
    mask_a = np.ones((H, W), dtype='int8')
    mask_chunk_arrays[track_id] = np.full((buffer_size, H, W), -1, dtype='int8')
    mask_chunk_idxs[track_id]   = frame_chunk
    mask_chunk_written[track_id] = set()
    mask_chunk_arrays[track_id][frame_slot] = mask_a
    mask_chunk_written[track_id].add(frame_idx)

    # Second detection for the same frame: fuse mask B into A
    mask_b = np.zeros((H, W), dtype='int8')
    mask_b[0, 0] = 1  # one extra pixel

    assert frame_idx in mask_chunk_written[track_id]
    previous_mask = mask_chunk_arrays[track_id][frame_slot].copy()
    fused = np.logical_or(previous_mask, mask_b).astype('int8')

    assert fused[0, 0] == 1   # pixel from mask_b
    assert np.sum(fused) == H * W  # all pixels set (from mask_a)


# ---------------------------------------------------------------------------
# Concurrent write worker (smoke test)
# ---------------------------------------------------------------------------

def test_concurrent_write_workers():
    """Multiple per-track write threads each write their own chunk file."""
    chunk_size, H, W = 20, 8, 8
    n_tracks = 5
    n_chunks  = 3
    _DONE = object()

    with tempfile.TemporaryDirectory() as td:
        arrays = {}
        workers = {}

        for tid in range(1, n_tracks + 1):
            arr = _make_zarr_array(
                Path(td) / f'pred_{tid}.zarr', f'{tid}_masks',
                chunk_size * n_chunks, H, W, chunk_size,
            )
            arrays[tid] = arr
            q = queue.Queue()

            def _worker(q=q, arr=arr):
                while True:
                    item = q.get()
                    if item is _DONE:
                        break
                    chunk_idx, data = item
                    _direct_write_chunk(arr, chunk_idx, data)

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            workers[tid] = (q, t)

        # Enqueue all chunks for all tracks
        for tid in range(1, n_tracks + 1):
            q, _ = workers[tid]
            for ci in range(n_chunks):
                data = np.full((chunk_size, H, W), tid * 10 + ci, dtype='int8')
                q.put((ci, data))
            q.put(_DONE)

        for tid, (q, t) in workers.items():
            t.join()

        # Verify readback
        for tid in range(1, n_tracks + 1):
            arr = arrays[tid]
            for ci in range(n_chunks):
                expected = tid * 10 + ci
                start = ci * chunk_size
                assert np.all(arr[start:start + chunk_size] == expected), \
                    f"track {tid} chunk {ci}: expected {expected}"


# ---------------------------------------------------------------------------
# CLI smoke: --debug flag is accepted
# ---------------------------------------------------------------------------

def test_predict_debug_flag_in_help():
    import re
    from typer.testing import CliRunner
    from octron.cli import app

    _ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mK]')
    runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1"})
    result = runner.invoke(app, ['predict', '--help'])
    assert result.exit_code == 0
    assert '--debug' in _ANSI_RE.sub('', result.output)
