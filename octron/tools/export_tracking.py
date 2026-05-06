"""
OCTRON tracking export pipeline.

Converts zarr mask archives and per-track CSVs produced by ``octron predict``
into analysis-ready CSV (or parquet) files.

When multiple disconnected segments are detected in a single frame, every
per-segment column (positions, areas, shape descriptors, intensity properties)
is stored as a tuple-string — e.g. ``"(120.5, 85.3)"``.  The ``method``
parameter controls how those tuple-valued rows are handled at export time.

Resolution methods
------------------
raw       : default.  Tuple-strings pass through unchanged; single-segment
            rows stay scalar.  Matches the original predict output (no info
            loss).  CSV columns end up as ``object`` dtype on read.  When
            writing parquet, mixed columns are stored as strings — including
            the scalar values — which loses numeric typing on disk; a warning
            is emitted in that case.
largest   : use the value from the single largest segment per frame.  Fully
            numeric output.  Fast, CSV-only, no zarr required.
weighted  : area-weighted mean across all segments per frame.  Fully numeric.
            ``area`` becomes the *sum* of segment areas.

Special cases (largest / weighted only)
---------------------------------------
area      : ``largest`` → area of the largest segment.
            ``weighted`` → sum of all segment areas.
orientation : always resolved via ``largest`` (circular quantity; weighted mean
              is undefined for angles).

Public functions
----------------
export_tracking            : Export CSVs from an existing predictions directory.
compute_weighted_centroids : Per-frame area-weighted (cx, cy) read from CSVs.
"""

import ast
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Region-property constants and per-batch worker (module-level for pickling)
# ---------------------------------------------------------------------------

# Properties that require the original video frame (intensity image) — cannot
# be computed from masks alone at export time.
# When a video is provided, these are computed on a 4-channel image
# (R, G, B, luminance), so each property expands to four columns:
#   intensity_mean_r, intensity_mean_g, intensity_mean_b, intensity_mean_lum
_INTENSITY_PROPS = frozenset({
    "intensity_max", "intensity_mean", "intensity_min", "intensity_std",
    "weighted_centroid",
    # Intensity-weighted spatial spread of the per-channel pixel intensities
    # (variance and covariance about the weighted centroid, per channel).
    "weighted_var_y", "weighted_var_x", "weighted_cov_yx",
})

# Pseudo-properties derived from skimage's weighted_moments_central (μ_pq):
#   weighted_var_y    = μ_20 / μ_00   (intensity-weighted variance along y)
#   weighted_var_x    = μ_02 / μ_00   (intensity-weighted variance along x)
#   weighted_cov_yx   = μ_11 / μ_00   (intensity-weighted yx covariance)
# Each is emitted per channel: e.g. weighted_var_y_r … _lum.
# These are the "spread" companions to weighted_centroid: small variance →
# pixels of that colour cluster tightly around the weighted centroid; large
# variance → they are spread out.  All three are translation-invariant, so
# they need no bbox-offset correction.
_WEIGHTED_SPREAD_PROPS = frozenset({
    "weighted_var_y", "weighted_var_x", "weighted_cov_yx",
})

# (i, j) suffix on weighted_moments_central → derived spread base name.
_SPREAD_FROM_MOMENT = {
    ("2", "0"): "weighted_var_y",
    ("0", "2"): "weighted_var_x",
    ("1", "1"): "weighted_cov_yx",
}

# Properties whose values are NOT translation-invariant: they depend on the
# absolute pixel coordinates, so cropping the input shifts the output by the
# bbox origin and the result must be offset back into full-frame coordinates.
# (Plain `bbox` columns are emitted from the bbox-detection step, not from
# regionprops, so they are not listed here.)
_TRANSLATION_VARIANT_PROPS = frozenset({
    "centroid",
    "weighted_centroid",
})

# Properties whose multichannel-intensity output gains an extra trailing
# channel suffix on a 4-channel intensity image (R, G, B, luminance):
#   intensity_mean-0..3  →  intensity_mean_{r,g,b,lum}
#   weighted_centroid-{axis}-{channel}  →  weighted_centroid_{y,x}_{r,g,b,lum}
#   weighted_var_y-{channel}  →  weighted_var_y_{r,g,b,lum}
_MULTICHANNEL_INTENSITY_PROPS = frozenset({
    "intensity_max", "intensity_mean", "intensity_min", "intensity_std",
    "weighted_centroid",
    "weighted_var_y", "weighted_var_x", "weighted_cov_yx",
})

# Properties whose first numeric suffix is a spatial axis (y, x).
_POSITION_PROPS = frozenset({"centroid", "weighted_centroid"})

_AXIS_LABELS = ("y", "x")
_CHANNEL_LABELS = ("r", "g", "b", "lum")

_PROPS_BATCH = 500     # zarr frames per contiguous read when computing props
_OPENCL_WORKERS = 4    # OpenCL contexts on one GPU; bounded to avoid VRAM blow-up

# Properties that pyclesperanto's statistics_of_labelled_pixels can produce
# directly (or that we derive cheaply from those primitives in numpy on host).
# Anything outside this set falls back to skimage on host using the SAME
# labelled image — no double labelling.
_OPENCL_NATIVE = frozenset({
    "area",  # not user-facing (already in _BASE_COLS) but needed as scratch
    "intensity_max", "intensity_mean", "intensity_min", "intensity_std",
    "weighted_centroid",
    "equivalent_diameter_area",  # = 2·sqrt(area/π), derived in numpy
    "extent",                    # = area / (bbox_w·bbox_h), derived in numpy
})


def _pyclesperanto_available() -> bool:
    """True when pyclesperanto + a usable device (any backend) are visible."""
    try:
        import pyclesperanto as _cle
        return bool(_cle.list_available_devices())
    except Exception:
        return False


def _resolve_device(device: str) -> str:
    """Reduce 'auto'/'cpu'/'cuda'/'mps' to either 'opencl' (pyclesperanto path) or 'cpu'.

    The CLI accepts 'cuda' for cross-subcommand consistency with `train` /
    `predict`; the GPU regionprops path itself is OpenCL via pyclesperanto.
    """
    from loguru import logger as _log

    d = (device or "auto").lower()
    if d in ("cuda", "opencl"):
        if not _pyclesperanto_available():
            raise RuntimeError(
                f"device='{d}' requested but pyclesperanto is not available "
                f"or no device is visible. Install with `uv pip install pyclesperanto`."
            )
        return "opencl"
    if d == "auto":
        return "opencl" if _pyclesperanto_available() else "cpu"
    if d == "mps":
        _log.warning(
            "device='mps' has no GPU regionprops backend; falling back to CPU."
        )
        return "cpu"
    return "cpu"


def _rename_prop_column(col_key, has_intensity):
    """Rename skimage's numeric '-N' suffixes to readable labels.

    Skimage's regionprops_table flattens multi-dimensional outputs by suffixing
    each numeric index, with the channel axis appended last when the intensity
    image is multi-channel.  For our 4-channel (R, G, B, luminance) layout:

      intensity_mean-0           →  intensity_mean_r
      weighted_centroid-0-0      →  weighted_centroid_y_r
      weighted_centroid-1-3      →  weighted_centroid_x_lum
      centroid-0                 →  centroid_y
      moments_hu-0               →  moments_hu-0           (unchanged)

    The ``has_intensity`` flag tells us whether to interpret the trailing
    index as a channel — without an intensity image, even multichannel-aware
    props like ``weighted_centroid`` are not present.
    """
    parts = col_key.split("-")
    if len(parts) == 1:
        return col_key

    base = parts[0]
    nums = parts[1:]

    is_pos = base in _POSITION_PROPS
    is_multich_intensity = has_intensity and base in _MULTICHANNEL_INTENSITY_PROPS

    if is_multich_intensity:
        try:
            ch_idx = int(nums[-1])
        except ValueError:
            return col_key
        ch_label = (
            _CHANNEL_LABELS[ch_idx]
            if 0 <= ch_idx < len(_CHANNEL_LABELS) else nums[-1]
        )
        spatial = nums[:-1]
        if is_pos and len(spatial) == 1:
            try:
                ax_idx = int(spatial[0])
            except ValueError:
                return col_key
            ax_label = (
                _AXIS_LABELS[ax_idx]
                if 0 <= ax_idx < len(_AXIS_LABELS) else spatial[0]
            )
            return f"{base}_{ax_label}_{ch_label}"
        if not spatial:
            return f"{base}_{ch_label}"
        # Unexpected spatial-suffix shape — keep as-is to avoid silent loss.
        return col_key

    if is_pos and len(nums) == 1:
        try:
            ax_idx = int(nums[0])
        except ValueError:
            return col_key
        ax_label = (
            _AXIS_LABELS[ax_idx]
            if 0 <= ax_idx < len(_AXIS_LABELS) else nums[0]
        )
        return f"{base}_{ax_label}"

    # Default: keep multi-value expansions like moments_hu-0..6 unchanged.
    return col_key


def _postprocess_region_props(props, computable, wanted_spread, n_channels,
                               r0, c0, has_intensity):
    """Convert raw regionprops_table output (dict of host-side numpy arrays)
    into a per-frame {col_name: value} dict applying the OCTRON column contract.

    Steps:
      1. Derive weighted_var_y/x and weighted_cov_yx from weighted_moments_central
         (only when those pseudo-props were requested).
      2. Shift translation-variant columns (centroid, weighted_centroid) from
         cropped-bbox coordinates back into full-frame coordinates by adding
         (r0, c0).
      3. Drop columns the caller didn't request.
      4. Rename skimage's '-N' suffixes to readable labels via _rename_prop_column.
      5. Format single-region values as floats and multi-region values as
         tuple-strings (matching the legacy column contract).

    Both the CPU (skimage) and GPU (pyclesperanto + skimage) workers call
    this once per frame.
    """
    import numpy as _np

    if wanted_spread and n_channels > 0:
        for ch in range(n_channels):
            m00 = props.get(f"weighted_moments_central-0-0-{ch}")
            if m00 is None:
                continue
            for (i, j), base in _SPREAD_FROM_MOMENT.items():
                if base not in wanted_spread:
                    continue
                m_ij = props.get(f"weighted_moments_central-{i}-{j}-{ch}")
                if m_ij is None:
                    continue
                with _np.errstate(divide="ignore", invalid="ignore"):
                    derived = _np.where(m00 != 0, m_ij / m00, _np.nan)
                props[f"{base}-{ch}"] = derived

    for col_key, vals in list(props.items()):
        base = col_key.split("-")[0]
        if base not in _TRANSLATION_VARIANT_PROPS:
            continue
        parts = col_key.split("-")
        if len(parts) < 2:
            continue
        try:
            ax_idx = int(parts[1])
        except ValueError:
            continue
        offset = r0 if ax_idx == 0 else c0 if ax_idx == 1 else 0
        if offset:
            props[col_key] = vals + offset

    frame_result = {}
    for col_key, vals in props.items():
        base = col_key.split("-")[0]
        if base not in computable and col_key not in computable:
            continue
        out_key = _rename_prop_column(col_key, has_intensity)
        frame_result[out_key] = (
            float(vals[0]) if len(vals) == 1
            else str(tuple(float(v) for v in vals))
        )
    return frame_result


def _zarr_batch_worker(args):
    """Process-pool worker: opens zarr independently and computes regionprops.

    Must be a module-level function so it can be pickled by ProcessPoolExecutor
    on Windows (spawn start method).  All imports are local so the worker
    process only loads what it needs.
    """
    zarr_store_path, arr_key, batch_fi, fi_positions, computable, video_path, mask_h, mask_w = args
    import zarr as _zarr
    import numpy as _np
    from skimage import measure as _measure
    from time import perf_counter as _pc
    from pathlib import Path as _Path

    # Translate pseudo-properties to the underlying skimage names.  We expose
    # weighted_var_y/x and weighted_cov_yx as user-friendly columns derived
    # from weighted_moments_central; skimage doesn't know those names, so we
    # ask for the moments and post-process below.
    wanted_spread = [p for p in computable if p in _WEIGHTED_SPREAD_PROPS]
    sk_properties = [p for p in computable if p not in _WEIGHTED_SPREAD_PROPS]
    if wanted_spread and "weighted_moments_central" not in sk_properties:
        sk_properties.append("weighted_moments_central")

    store = _zarr.storage.LocalStore(_Path(zarr_store_path), read_only=True)
    root = _zarr.open_group(store=store, mode="r")
    zarr_arr = root[arr_key]

    fi_lo, fi_hi = batch_fi[0], batch_fi[-1]
    t0 = _pc()
    raw_batch = _np.asarray(zarr_arr[fi_lo : fi_hi + 1])
    t1 = _pc()

    cap = None
    current_video_pos = -1
    if video_path is not None:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(video_path)

    results = {}
    t_binary = t_bbox = t_label = t_props = 0.0
    for fi, pos in zip(batch_fi, fi_positions):
        raw = raw_batch[fi - fi_lo]

        ta = _pc()
        binary = raw == 1  # bool view, no copy
        tb = _pc()
        rows_any = binary.any(axis=1)
        if not rows_any.any():
            t_binary += tb - ta
            t_bbox += _pc() - tb
            continue
        cols_any = binary.any(axis=0)
        r0 = int(rows_any.argmax())
        r1 = int(len(rows_any) - rows_any[::-1].argmax())
        c0 = int(cols_any.argmax())
        c1 = int(len(cols_any) - cols_any[::-1].argmax())
        tc = _pc()

        intensity_crop = None
        if cap is not None:
            if fi != current_video_pos:
                cap.set(_cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            current_video_pos = fi + 1 if ok else -1
            if ok:
                # Resize to mask resolution if needed
                fh, fw = frame.shape[:2]
                if fh != mask_h or fw != mask_w:
                    frame = _cv2.resize(frame, (mask_w, mask_h), interpolation=_cv2.INTER_LINEAR)
                # Channels: 0=R, 1=G, 2=B, 3=luminance
                rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                lum = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
                intensity_crop = _np.dstack([rgb, lum])[r0:r1, c0:c1]

        labeled = _measure.label(binary[r0:r1, c0:c1], background=0, connectivity=2)
        td = _pc()
        # Suppress numpy's 0/0 RuntimeWarning from weighted_centroid (M/M0
        # divide).  Regions with zero total intensity in a channel are
        # legitimate in biological videos — we report them once per track
        # in compute_region_props_from_zarr instead of as a noisy warning.
        with _np.errstate(divide="ignore", invalid="ignore"):
            props = _measure.regionprops_table(
                labeled, intensity_image=intensity_crop, properties=sk_properties
            )
        te = _pc()

        t_binary += tb - ta
        t_bbox += tc - tb
        t_label += td - tc
        t_props += te - td

        n_channels = intensity_crop.shape[2] if intensity_crop is not None else 0
        frame_result = _postprocess_region_props(
            props, computable, wanted_spread, n_channels,
            r0=r0, c0=c0, has_intensity=intensity_crop is not None,
        )
        if frame_result:
            results[pos] = frame_result

    if cap is not None:
        cap.release()

    return results, fi_lo, fi_hi, t1 - t0, _pc() - t1, t_binary, t_bbox, t_label, t_props


def _zarr_batch_worker_opencl(args):
    """Process-pool worker for the pyclesperanto/OpenCL GPU path.

    Same args/returns as ``_zarr_batch_worker``.  Each worker process
    imports pyclesperanto independently and creates its own OpenCL context;
    OpenCL contexts are lightweight enough on NVIDIA Windows that a small
    pool (typically 4) on one physical device is fine and gives us the host
    I/O parallelism that a single-context serial design would give up.

    Routes connected-components labelling and intensity statistics through
    the GPU; falls back to skimage on host (against the SAME labelled image,
    no double work) for any property pyclesperanto doesn't expose — e.g.
    eccentricity, perimeter, moments_hu, weighted_moments_central (the basis
    for our weighted_var/cov spread features).
    """
    zarr_store_path, arr_key, batch_fi, fi_positions, computable, video_path, mask_h, mask_w = args
    import zarr as _zarr
    import numpy as _np
    import pyclesperanto as _cle
    from skimage import measure as _measure
    from time import perf_counter as _pc
    from pathlib import Path as _Path

    # Partition computable into GPU-handled vs CPU-handled.
    cpu_residual = [p for p in computable if p not in _OPENCL_NATIVE]
    wanted_spread = [p for p in cpu_residual if p in _WEIGHTED_SPREAD_PROPS]
    cpu_sk_properties = [p for p in cpu_residual if p not in _WEIGHTED_SPREAD_PROPS]
    if wanted_spread and "weighted_moments_central" not in cpu_sk_properties:
        cpu_sk_properties.append("weighted_moments_central")

    store = _zarr.storage.LocalStore(_Path(zarr_store_path), read_only=True)
    root = _zarr.open_group(store=store, mode="r")
    zarr_arr = root[arr_key]

    fi_lo, fi_hi = batch_fi[0], batch_fi[-1]
    t0 = _pc()
    raw_batch = _np.asarray(zarr_arr[fi_lo : fi_hi + 1])
    t1 = _pc()

    cap = None
    current_video_pos = -1
    if video_path is not None:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(video_path)

    results = {}
    t_binary = t_bbox = t_label = t_props = 0.0
    for fi, pos in zip(batch_fi, fi_positions):
        raw = raw_batch[fi - fi_lo]

        ta = _pc()
        binary = raw == 1
        tb = _pc()
        rows_any = binary.any(axis=1)
        if not rows_any.any():
            t_binary += tb - ta
            t_bbox += _pc() - tb
            continue
        cols_any = binary.any(axis=0)
        r0 = int(rows_any.argmax())
        r1 = int(len(rows_any) - rows_any[::-1].argmax())
        c0 = int(cols_any.argmax())
        c1 = int(len(cols_any) - cols_any[::-1].argmax())
        tc = _pc()

        intensity_crop = None
        if cap is not None:
            if fi != current_video_pos:
                cap.set(_cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            current_video_pos = fi + 1 if ok else -1
            if ok:
                fh, fw = frame.shape[:2]
                if fh != mask_h or fw != mask_w:
                    frame = _cv2.resize(frame, (mask_w, mask_h), interpolation=_cv2.INTER_LINEAR)
                rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                lum = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
                intensity_crop = _np.dstack([rgb, lum])[r0:r1, c0:c1]

        # GPU: connected-components on the cropped binary mask.
        binary_crop = binary[r0:r1, c0:c1].astype(_np.uint8)
        bin_d = _cle.push(binary_crop)
        labeled_d = _cle.connected_components_labeling(bin_d, connectivity="box")
        td = _pc()

        props = {}
        n_channels = intensity_crop.shape[2] if intensity_crop is not None else 0

        # GPU: per-channel statistics.  Geometry fields are channel-invariant
        # so we only keep them from the first call.
        bbox_w = bbox_h = None
        if intensity_crop is not None and n_channels > 0:
            for ch in range(n_channels):
                ch_d = _cle.push(intensity_crop[:, :, ch].astype(_np.float32))
                stats = _cle.statistics_of_labelled_pixels(
                    intensity=ch_d, label=labeled_d
                )
                if ch == 0:
                    props["area"] = _np.asarray(stats["area"])
                    bbox_w = _np.asarray(stats["bbox_width"])
                    bbox_h = _np.asarray(stats["bbox_height"])
                # Always populate; the post-process filter drops what wasn't
                # requested.
                props[f"intensity_mean-{ch}"] = _np.asarray(stats["mean_intensity"])
                props[f"intensity_min-{ch}"]  = _np.asarray(stats["min_intensity"])
                props[f"intensity_max-{ch}"]  = _np.asarray(stats["max_intensity"])
                props[f"intensity_std-{ch}"]  = _np.asarray(stats["standard_deviation_intensity"])
                # mass_center_x/y in cropped-image coords; bbox-offset added
                # back to full-frame in the post-process step.
                props[f"weighted_centroid-0-{ch}"] = _np.asarray(stats["mass_center_y"])
                props[f"weighted_centroid-1-{ch}"] = _np.asarray(stats["mass_center_x"])
        else:
            stats = _cle.statistics_of_labelled_pixels(intensity=None, label=labeled_d)
            props["area"] = _np.asarray(stats["area"])
            bbox_w = _np.asarray(stats["bbox_width"])
            bbox_h = _np.asarray(stats["bbox_height"])

        # Cheap derived fields (translation-invariant; no offset correction).
        if "equivalent_diameter_area" in computable:
            with _np.errstate(invalid="ignore"):
                props["equivalent_diameter_area"] = 2.0 * _np.sqrt(props["area"] / _np.pi)
        if "extent" in computable:
            with _np.errstate(divide="ignore", invalid="ignore"):
                bbox_area = bbox_w * bbox_h
                props["extent"] = _np.where(
                    bbox_area != 0, props["area"] / bbox_area, _np.nan
                )

        te = _pc()

        # CPU fallback for properties pyclesperanto doesn't expose.  Reuse
        # the SAME labelled image (pulled to host once) so we don't pay the
        # connected-components cost twice.
        if cpu_sk_properties:
            labeled_host = _cle.pull(labeled_d).astype(_np.int32)
            with _np.errstate(divide="ignore", invalid="ignore"):
                cpu_props = _measure.regionprops_table(
                    labeled_host,
                    intensity_image=intensity_crop,
                    properties=cpu_sk_properties,
                )
            props.update(cpu_props)

        t_binary += tb - ta
        t_bbox += tc - tb
        t_label += td - tc
        t_props += te - td

        frame_result = _postprocess_region_props(
            props, computable, wanted_spread, n_channels,
            r0=r0, c0=c0, has_intensity=intensity_crop is not None,
        )
        if frame_result:
            results[pos] = frame_result

    if cap is not None:
        cap.release()

    return results, fi_lo, fi_hi, t1 - t0, _pc() - t1, t_binary, t_bbox, t_label, t_props


# ---------------------------------------------------------------------------
# Tuple-resolution helpers
# ---------------------------------------------------------------------------

def _parse_tuple_or_scalar(val):
    """Return a tuple of floats from a scalar value or a tuple-string '(1.0, 2.0)'."""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return (float(val),)
    if isinstance(val, str) and val.startswith('('):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, tuple):
                return tuple(float(v) for v in parsed)
        except (ValueError, SyntaxError):
            pass
    try:
        return (float(val),)
    except (TypeError, ValueError):
        return (float('nan'),)


def _largest_idx(area_vals):
    """Return the index of the largest segment given a tuple of area values."""
    if len(area_vals) == 1:
        return 0
    return int(np.argmax(area_vals))


def _resolve_xy_largest(xs, ys, areas):
    """Pick the position of the largest segment for each row."""
    cx = np.empty(len(xs), dtype=float)
    cy = np.empty(len(ys), dtype=float)
    for i, (x, y, a) in enumerate(zip(xs, ys, areas)):
        xv = _parse_tuple_or_scalar(x)
        yv = _parse_tuple_or_scalar(y)
        av = _parse_tuple_or_scalar(a)
        idx = _largest_idx(av)
        cx[i] = xv[idx] if idx < len(xv) else xv[0]
        cy[i] = yv[idx] if idx < len(yv) else yv[0]
    return cx, cy


def _resolve_xy_weighted(xs, ys, areas):
    """Compute the area-weighted mean position across all segments per row."""
    cx = np.empty(len(xs), dtype=float)
    cy = np.empty(len(ys), dtype=float)
    for i, (x, y, a) in enumerate(zip(xs, ys, areas)):
        xv = np.array(_parse_tuple_or_scalar(x))
        yv = np.array(_parse_tuple_or_scalar(y))
        av = np.array(_parse_tuple_or_scalar(a))
        total = av.sum()
        if total > 0:
            cx[i] = float((xv * av).sum() / total)
            cy[i] = float((yv * av).sum() / total)
        else:
            cx[i] = float(xv.mean())
            cy[i] = float(yv.mean())
    return cx, cy


def _resolve_prop_largest(values, areas):
    """Pick the regionprop value for the largest segment per row."""
    out = np.empty(len(values), dtype=float)
    for i, (v, a) in enumerate(zip(values, areas)):
        vv = _parse_tuple_or_scalar(v)
        av = _parse_tuple_or_scalar(a)
        idx = _largest_idx(av)
        out[i] = vv[idx] if idx < len(vv) else vv[0]
    return out


def _resolve_prop_weighted(values, areas):
    """Compute the area-weighted mean of a regionprop across all segments per row."""
    out = np.empty(len(values), dtype=float)
    for i, (v, a) in enumerate(zip(values, areas)):
        vv = np.array(_parse_tuple_or_scalar(v))
        av = np.array(_parse_tuple_or_scalar(a))
        total = av.sum()
        out[i] = float((vv * av).sum() / total) if total > 0 else float(vv.mean())
    return out


def _sum_segments_area(areas):
    """Sum all segment areas per row to give total mask area."""
    out = np.empty(len(areas), dtype=float)
    for i, a in enumerate(areas):
        out[i] = float(sum(_parse_tuple_or_scalar(a)))
    return out


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def _build_header(video_metadata):
    lines = [
        f"video_name: {video_metadata.get('video_name', 'unknown')}",
        f"frame_count: {video_metadata.get('frame_count', '')}",
        f"frame_count_analyzed: {video_metadata.get('frame_count_analyzed', '')}",
        f"video_height: {video_metadata.get('video_height', '')}",
        f"video_width: {video_metadata.get('video_width', '')}",
        f"created_at: {video_metadata.get('created_at', str(datetime.now()))}",
        "",  # empty separator line
    ]
    return '\n'.join(lines) + '\n'


def _read_csv_metadata(csv_path, n_header_lines=7):
    """Parse the key: value metadata from the top of a prediction CSV."""
    meta = {}
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i >= n_header_lines - 1:
                break
            line = line.strip()
            if ':' in line:
                key, _, val = line.partition(':')
                meta[key.strip()] = val.strip()
    return meta


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------

_COL_BASE = [
    "label", "confidence", "pos_x", "pos_y",
    "bbox_x_min", "bbox_x_max", "bbox_y_min", "bbox_y_max",
    "bbox_area", "bbox_aspect_ratio", "area",
]
_CHANNEL_SUFFIXES = ("_r", "_g", "_b", "_lum")


def _ordered_columns(df_columns):
    """Return df_columns in canonical order.

    Order: base cols → shape props → intensity props (ALL_REGION_PROPERTIES
    order).  Multi-value expansions (moments_hu-0 … -6), channel expansions
    (intensity_mean_r/g/b/lum), axis expansions (centroid_y/x), and
    axis-channel expansions (weighted_centroid_y_r … _x_lum) are all grouped
    immediately after their base name.  Unknown columns follow at the end.
    """
    from octron.yolo_octron.constants import ALL_REGION_PROPERTIES

    known_props = [
        p for group in ALL_REGION_PROPERTIES.values() for p in group
        if p != "area"  # area is already in _COL_BASE
    ]

    cols = set(df_columns)
    result = []
    seen = set()

    def _add(c):
        if c in cols and c not in seen:
            result.append(c)
            seen.add(c)

    def _add_with_expansions(prop):
        _add(prop)
        for c in sorted(c for c in cols if c.startswith(prop + "-")):
            _add(c)
        for suf in _CHANNEL_SUFFIXES:
            _add(prop + suf)
        for ax in _AXIS_LABELS:
            _add(f"{prop}_{ax}")
            for suf in _CHANNEL_SUFFIXES:
                _add(f"{prop}_{ax}{suf}")

    for col in _COL_BASE:
        _add(col)

    for prop in known_props:
        _add_with_expansions(prop)

    for c in df_columns:  # anything not yet placed
        _add(c)

    return result


# ---------------------------------------------------------------------------
# Private core function
# ---------------------------------------------------------------------------

def _export_tracking_from_data(
    output_dir,
    track_ids,
    labels,
    frame_counters,
    frame_indices,
    confidence,
    segments_x,
    segments_y,
    segments_area,
    bbox,
    region_props,
    video_metadata,
    method="raw",
    zarr_root=None,
    combined=False,
    fmt="csv",
):
    """
    Build and write tracking CSVs from raw per-track arrays.

    This is the single place where prediction CSVs are written — called by
    both ``predict_batch`` (in-memory path) and ``export_tracking`` (disk path).

    Method ``"raw"`` (default) preserves multi-segment values as tuple-strings,
    matching the original predict output.  The other methods resolve tuples to
    scalars:

    - ``area`` uses the *largest segment's area* for ``"largest"``, and the
      *sum of all segment areas* for ``"weighted"``.
    - ``orientation`` always uses the largest segment (circular quantity).

    Parameters
    ----------
    output_dir : Path or str
    track_ids : list[int]
    labels : dict[int, str]
    frame_counters : dict[int, array-like]
        Sequential counter within the analyzed frames (frame_no level).
    frame_indices : dict[int, array-like]
        Actual video frame indices (frame_idx level).
    confidence : dict[int, array-like]
    segments_x : dict[int, array-like]
        Raw pos_x values — scalars or tuple-strings when multi-segment.
    segments_y : dict[int, array-like]
    segments_area : dict[int, array-like]
        Raw area values — scalars or tuple-strings when multi-segment.
    bbox : dict[int, dict[str, array-like]]
        Bounding-box columns keyed by column name.
    region_props : dict[int, dict[str, array-like]]
        Extra regionprop columns (all columns except the above).
    video_metadata : dict
        Keys: video_name, frame_count, frame_count_analyzed,
              video_height, video_width, created_at.
    method : {"raw", "largest", "weighted"}
    zarr_root : zarr.Group or None
        Unused (kept for backwards-compatible call signature).
    combined : bool
        True → single all_tracks.csv; False → one file per track.
    fmt : {"csv", "parquet"}
        Output format.  Parquet stores video metadata in the file schema.
        ``method="raw"`` + ``fmt="parquet"`` produces string-typed columns
        for any field that contains tuple-strings; a warning is emitted.
    """
    import pandas as pd
    from loguru import logger

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "raw" and fmt == "parquet":
        logger.warning(
            "method='raw' + fmt='parquet': any column containing tuple-strings "
            "will be stored as a string column (including the scalar values in "
            "single-segment rows), losing numeric type on disk. Use a "
            "resolution method (largest/weighted) for fully numeric parquet output."
        )

    _use_weighted = method == "weighted"

    from tqdm import tqdm

    header = _build_header(video_metadata)
    dfs = {}

    for tid in tqdm(track_ids, desc="Processing tracks", unit="track", total=len(track_ids)):
        if tid not in frame_indices or len(frame_indices[tid]) == 0:
            continue

        label = labels.get(tid, "unknown")
        n = len(frame_indices[tid])

        sx = np.asarray(segments_x.get(tid, np.full(n, np.nan)), dtype=object)
        sy = np.asarray(segments_y.get(tid, np.full(n, np.nan)), dtype=object)
        sa = np.asarray(segments_area.get(tid, np.ones(n)), dtype=object)

        # --- pos_x / pos_y ---
        if method == "raw":
            px, py = sx, sy
        elif method == "weighted":
            px, py = _resolve_xy_weighted(sx, sy, sa)
        else:  # "largest"
            px, py = _resolve_xy_largest(sx, sy, sa)

        # --- area ---
        # "raw"      → tuple-strings pass through unchanged.
        # "largest"  → area of the largest segment.
        # "weighted" → sum of all segment areas (total mask coverage).
        if method == "raw":
            area = sa
        elif _use_weighted:
            area = _sum_segments_area(sa)
        else:
            area = _resolve_prop_largest(sa, sa)

        # --- all other regionprops ---
        resolved_props = {}
        prop_items = list((region_props.get(tid) or {}).items())
        for prop_name, prop_values in tqdm(
            prop_items, desc=f"  resolving props", unit="prop", leave=False,
        ):
            pv = np.asarray(prop_values, dtype=object)
            if method == "raw":
                resolved_props[prop_name] = pv
            elif prop_name == "orientation" or not _use_weighted:
                # orientation is a circular quantity — always use largest segment.
                resolved_props[prop_name] = _resolve_prop_largest(pv, sa)
            else:
                resolved_props[prop_name] = _resolve_prop_weighted(pv, sa)

        # --- Build DataFrame ---
        data = {
            "label":      np.full(n, label),
            "confidence": np.asarray(confidence.get(tid, np.full(n, np.nan)), dtype=float),
            "pos_x":      px,
            "pos_y":      py,
        }

        for col, arr in (bbox.get(tid) or {}).items():
            data[col] = np.asarray(arr)

        data["area"] = area
        data.update(resolved_props)

        idx = pd.MultiIndex.from_arrays(
            [
                np.asarray(frame_counters[tid]),
                np.asarray(frame_indices[tid]),
                np.full(n, tid, dtype=int),
            ],
            names=["frame_counter", "frame_idx", "track_id"],
        )
        df = pd.DataFrame(data, index=idx)
        df = df[_ordered_columns(df.columns)]
        dfs[tid] = (label, df)

    if not dfs:
        logger.warning("No tracking data to export.")
        return

    ext = ".parquet" if fmt == "parquet" else ".csv"

    def _write(path, df):
        tmp = path.with_suffix(".tmp")
        try:
            if fmt == "parquet":
                import json
                import pyarrow as _pa
                import pyarrow.parquet as _pq
                table = _pa.Table.from_pandas(df)
                meta = {**table.schema.metadata,
                        b"octron_metadata": json.dumps(video_metadata).encode()}
                _pq.write_table(table.replace_schema_metadata(meta), tmp)
            else:
                _WRITE_CHUNK = 5000
                with open(tmp, "w") as f:
                    f.write(header)
                    chunks = range(0, len(df), _WRITE_CHUNK)
                    for i in tqdm(chunks, desc=f"  {path.stem}", unit="chunk", leave=False):
                        df.iloc[i : i + _WRITE_CHUNK].to_csv(
                            f, header=(i == 0), na_rep="NaN", lineterminator="\n"
                        )
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    if combined:
        out_path = output_dir / f"all_tracks{ext}"
        tmp = out_path.with_suffix(".tmp")
        if fmt == "parquet":
            # Write each track as its own row group — keeps track data contiguous
            # for better compression and selective reads without full-file scans.
            import json
            import pyarrow as _pa
            import pyarrow.parquet as _pq
            writer = None
            try:
                for tid, (label, df) in tqdm(
                    dfs.items(), desc="Writing parquet", unit="track", total=len(dfs)
                ):
                    table = _pa.Table.from_pandas(df)
                    if writer is None:
                        octron_meta = {
                            **table.schema.metadata,
                            b"octron_metadata": json.dumps(video_metadata).encode(),
                        }
                        writer = _pq.ParquetWriter(tmp, table.schema.with_metadata(octron_meta))
                    writer.write_table(table)
                if writer:
                    writer.close()
                os.replace(tmp, out_path)
            except BaseException:
                if writer:
                    writer.close()
                tmp.unlink(missing_ok=True)
                raise
        else:
            all_df = pd.concat([df for _, df in dfs.values()]).sort_index()
            tqdm.write(f"Writing combined CSV → {out_path.name}")
            _write(out_path, all_df)
        logger.debug(f"Saved combined tracking data ({len(dfs)} tracks) to {out_path.name}")
    else:
        desc = "Writing parquet" if fmt == "parquet" else "Writing CSVs"
        for tid, (label, df) in tqdm(dfs.items(), desc=desc, unit="track", total=len(dfs)):
            out_path = output_dir / f"{label}_track_{tid}{ext}"
            _write(out_path, df)
            logger.debug(f"Saved tracking data for '{label}' (track ID: {tid}) to {out_path.name}")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

_CSV_HEADER_LINES = 7
_BBOX_COLS = frozenset({"bbox_x_min", "bbox_x_max", "bbox_y_min", "bbox_y_max",
                        "bbox_area", "bbox_aspect_ratio"})
_BASE_COLS = frozenset({"frame_counter", "frame_idx", "track_id", "label",
                        "confidence", "pos_x", "pos_y", "area"}) | _BBOX_COLS


def compute_region_props_from_zarr(zarr_root, track_ids, frame_indices_dict, properties,
                                   zarr_path=None, video_path=None, device="cpu"):
    """
    Compute per-frame regionprops directly from zarr segmentation masks.

    Used by :func:`export_tracking` to add shape metrics (and optionally
    intensity metrics) at export time.

    Parameters
    ----------
    zarr_root : zarr.Group
    track_ids : list[int]
    frame_indices_dict : dict[int, array-like]
        ``{track_id: array of video frame indices}`` matching the CSV rows.
    properties : list[str]
        skimage regionprop base names to compute.
    zarr_path : Path or None
        Filesystem path to the zarr store — enables ProcessPoolExecutor.
    video_path : Path or str or None
        Path to the original video file.  Required to compute intensity
        properties (intensity_mean, intensity_max, intensity_min,
        intensity_std).  Frames are resized to mask resolution if needed.
    device : {"auto", "cpu", "cuda", "mps"}
        ``"auto"`` (default) picks the pyclesperanto/OpenCL GPU path when a
        device is visible, else CPU.  ``"cuda"`` is a hard request for the
        GPU path (kept for naming consistency with `train` / `predict`; the
        backend itself is OpenCL).  ``"mps"`` has no GPU regionprops backend
        and silently falls back to CPU.

    Returns
    -------
    dict[int, dict[str, np.ndarray]]
        ``{track_id: {col_name: per-frame array}}``.
        Multi-segment frames are encoded as tuple-strings ``"(v1, v2, ...)"``.
        Empty-mask frames are ``NaN``.
    """
    from tqdm import tqdm
    from loguru import logger as _log

    has_video = video_path is not None
    computable = [
        p for p in properties
        if p not in _BASE_COLS and (has_video or p not in _INTENSITY_PROPS)
    ]
    skipped_intensity = [p for p in properties if p in _INTENSITY_PROPS] if not has_video else []
    if skipped_intensity:
        _log.warning(
            f"Intensity properties {skipped_intensity} require the original video. "
            f"Pass --video (or video_path=) to compute them."
        )
    if not computable:
        return {}

    import os
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    resolved_device = _resolve_device(device)
    on_gpu = resolved_device == "opencl"
    batch_size = _PROPS_BATCH

    if on_gpu:
        gpu_native = _OPENCL_NATIVE & set(computable)
        cpu_residual = sorted(set(computable) - _OPENCL_NATIVE)
        n_workers = min(_OPENCL_WORKERS, os.cpu_count() or 1)
        _log.info(
            f"Using pyclesperanto (OpenCL GPU) for regionprops with "
            f"{n_workers} worker(s). GPU-native: {sorted(gpu_native) or '[]'}; "
            f"CPU fallback: {cpu_residual or '[]'}."
        )
        worker_fn = _zarr_batch_worker_opencl
    else:
        n_workers = min(8, os.cpu_count() or 4)
        worker_fn = _zarr_batch_worker
        _log.debug(f"Using ProcessPoolExecutor with {n_workers} worker(s) (CPU)")

    # ProcessPoolExecutor bypasses the GIL and parallelises the host-side
    # work in both paths.  For the GPU path: each worker process imports
    # pyclesperanto and creates its own OpenCL context — these are
    # lightweight enough on NVIDIA OpenCL (a few hundred MB each) that
    # 4 of them on one device is fine, and the host I/O parallelism is
    # exactly what we'd give up by going serial.  ThreadPoolExecutor is
    # only used as a fallback when zarr_path is None (then workers can't
    # re-open the store independently).
    use_processes = zarr_path is not None
    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    result = {}

    for tid in tqdm(track_ids, desc="Computing region props from masks", unit="track"):
        arr_key = f"{tid}_masks"
        if arr_key not in zarr_root:
            _log.warning(f"No mask array found in zarr for track {tid} — skipping.")
            continue

        zarr_arr = zarr_root[arr_key]
        n_zarr_frames = zarr_arr.shape[0]
        mask_h, mask_w = int(zarr_arr.shape[1]), int(zarr_arr.shape[2])
        frame_idxs = np.asarray(frame_indices_dict[tid], dtype=int)
        n = len(frame_idxs)

        fi_to_pos = {int(fi): i for i, fi in enumerate(frame_idxs)}
        valid_fi = sorted(fi for fi in fi_to_pos if fi < n_zarr_frames)

        # per-frame result as list-of-dicts — handles dynamically-expanded
        # column names (e.g. moments_hu → moments_hu-0 … moments_hu-6)
        rows = [{} for _ in range(n)]

        _video_path_str = str(video_path) if video_path is not None else None

        # Build picklable args for the worker function.
        # Each entry contains everything the worker needs independently.
        batch_args = [
            (
                str(zarr_path),
                arr_key,
                valid_fi[s : s + batch_size],
                [fi_to_pos[fi] for fi in valid_fi[s : s + batch_size]],
                computable,
                _video_path_str,
                mask_h,
                mask_w,
            )
            for s in range(0, len(valid_fi), batch_size)
        ]

        def _consume_batch_results(batch_iter):
            for batch_results, fi_lo, fi_hi, t_io, t_cpu, t_bin, t_bbox, t_lbl, t_rpt in tqdm(
                batch_iter,
                total=len(batch_args),
                desc=f"  track {tid}", unit="batch", leave=False,
            ):
                _log.debug(
                    f"batch {fi_lo}-{fi_hi}: zarr_read={t_io:.3f}s  "
                    f"regionprops={t_cpu:.3f}s "
                    f"(binary={t_bin:.3f}s  bbox={t_bbox:.3f}s  "
                    f"label={t_lbl:.3f}s  props_table={t_rpt:.3f}s)"
                )
                for pos, frame_result in batch_results.items():
                    rows[pos].update(frame_result)

        with ExecutorClass(max_workers=n_workers) as executor:
            _consume_batch_results(executor.map(worker_fn, batch_args))

        # Collect all column names that actually appeared (handles expansion)
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())

        # Detect region-frames where a channel had zero total intensity.
        # weighted_* columns come back NaN for those — common in biological
        # videos when one or more channels are dim or absent.  We surface
        # this as one INFO line per track instead of skimage's per-call
        # divide-by-zero RuntimeWarning.  Distinguish from "frame had no
        # mask" NaN by requiring the column to be PRESENT in the row dict
        # (rows[i] only has keys for frames where regionprops actually ran).
        zero_intensity_counts = {}
        for ch in _CHANNEL_LABELS:
            suf = f"_{ch}"
            ch_cols = [c for c in all_keys
                       if c.startswith("weighted_") and c.endswith(suf)]
            if not ch_cols:
                continue
            probe = ch_cols[0]  # all weighted_*_<ch> share NaN per (frame, ch)
            n_bad = 0
            for row in rows:
                if probe not in row:
                    continue
                v = row[probe]
                if isinstance(v, float) and v != v:
                    n_bad += 1
                elif isinstance(v, str) and "nan" in v.lower():
                    n_bad += 1
            if n_bad:
                zero_intensity_counts[ch] = n_bad

        if zero_intensity_counts:
            summary = ", ".join(
                f"{ch}={n}" for ch, n in zero_intensity_counts.items()
            )
            _log.info(
                f"track {tid}: zero total intensity in some region-frames "
                f"({summary}) — weighted_* columns are NaN for those entries. "
                f"This is normal when a channel is dim or absent in the video."
            )

        tid_result = {}
        for col_key in all_keys:
            arr = np.empty(n, dtype=object)
            arr[:] = np.nan
            for i, row in enumerate(rows):
                if col_key in row:
                    arr[i] = row[col_key]
            tid_result[col_key] = arr

        result[tid] = tid_result

    return result


_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".mts", ".m4v")


def _autodetect_video(predictions_path: Path):
    """Replicate YOLO_results auto-detection: strip tracker suffix, look two levels up."""
    stem = "_".join(predictions_path.name.split("_")[:-1])
    search_dir = predictions_path.parent.parent
    for ext in _VIDEO_EXTENSIONS:
        candidate = search_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def compute_weighted_centroids(predictions_path, track_ids=None):
    """
    Compute per-frame area-weighted centroids by re-reading the raw
    ``pos_x``/``pos_y``/``area`` columns from per-track prediction CSVs.

    Multi-segment frames (where the columns hold tuple-strings) are reduced
    to a single (cx, cy) via ``Σ(aᵢ·xᵢ) / Σ(aᵢ)``.  Single-segment scalars
    pass through unchanged.

    Parameters
    ----------
    predictions_path : str or Path
        Path to an ``octron_predictions/<video>/`` output directory.
    track_ids : iterable[int] or None
        If given, only compute centroids for these track IDs.

    Returns
    -------
    dict[int, dict[int, tuple[float, float]]]
        ``{track_id: {frame_idx: (cx, cy)}}``.
    """
    import pandas as pd
    from natsort import natsorted

    predictions_path = Path(predictions_path)
    csvs = natsorted(predictions_path.glob("*track_*.csv"))
    if not csvs:
        return {}

    wanted = set(track_ids) if track_ids is not None else None
    out = {}
    for csv_file in csvs:
        df = pd.read_csv(csv_file, skiprows=_CSV_HEADER_LINES)
        if df.empty:
            continue
        tid = int(df["track_id"].iloc[0])
        if wanted is not None and tid not in wanted:
            continue
        sx = df["pos_x"].to_numpy(dtype=object)
        sy = df["pos_y"].to_numpy(dtype=object)
        sa = (df["area"].to_numpy(dtype=object)
              if "area" in df.columns else np.ones(len(df), dtype=object))
        cx, cy = _resolve_xy_weighted(sx, sy, sa)
        fi = df["frame_idx"].to_numpy()
        out[tid] = {int(fi[i]): (float(cx[i]), float(cy[i])) for i in range(len(fi))}
    return out


def list_region_properties(*, print_output: bool = True) -> dict:
    """Return (and optionally print) all available regionprop names grouped by category.

    Examples
    --------
    >>> from octron.tools.export_tracking import list_region_properties
    >>> props = list_region_properties()
    """
    from octron.yolo_octron.constants import ALL_REGION_PROPERTIES

    if print_output:
        for category, names in ALL_REGION_PROPERTIES.items():
            print(f"\n{category}:")
            for name in names:
                print(f"  {name}")
        print()
    return ALL_REGION_PROPERTIES


def export_tracking(
    predictions_path,
    output_dir=None,
    method: Literal["raw", "largest", "weighted"] = "raw",
    region_properties=None,
    video_path=None,
    fmt: Literal["csv", "parquet"] = "csv",
    combined=False,
    overwrite=False,
    device: Literal["auto", "cpu", "cuda", "mps"] = "cpu",
):
    """
    Export tracking CSVs from an existing OCTRON predictions directory.

    Reads the per-track CSVs and (optionally) zarr masks produced by
    ``octron predict`` and writes new CSVs.  By default (``method="raw"``)
    multi-segment values pass through as tuple-strings; the other methods
    resolve those to scalars.

    Parameters
    ----------
    predictions_path : str or Path
        Path to an ``octron_predictions/<video>/`` output directory.
    output_dir : str or Path or None
        Where to write the output CSVs.  Defaults to *predictions_path*.
    method : {"raw", "largest", "weighted"}
        How to handle multi-segment rows (default: ``"raw"`` — pass through
        tuple-strings unchanged).  See module docstring.
    region_properties : None, "all", "none", "shape", "intensity", or list[str]
        Which regionprop columns to include in the output.

        ``None``
            Keep every regionprop column already present in the CSV (default).
            No zarr computation is performed.
        ``"all"``
            Compute every property from all groups in ``ALL_REGION_PROPERTIES``
            (equivalent to ``"shape"`` + ``"intensity"``).
        ``"none"``
            Strip all regionprop columns; output contains only the base
            columns (frame counters, track id, label, confidence, pos_x/y,
            area, bbox).
        ``"shape"``
            Include only the size-and-shape group (area, eccentricity,
            solidity, etc.).
        ``"intensity"``
            Include only the intensity group (intensity_mean, intensity_max,
            intensity_min, intensity_std).
        list of strings
            Include only the named columns that are present in the CSV
            (e.g. ``["eccentricity", "solidity"]``).  Use
            :func:`list_region_properties` to see what is available.
    video_path : str or Path or None
        Path to the original video file.  Required to compute intensity
        properties (intensity_mean, intensity_max, intensity_min,
        intensity_std).  Auto-detected from the predictions directory name
        if not provided.
    fmt : {"csv", "parquet"}
        Output format.  ``"parquet"`` writes a Parquet file per track (or
        ``all_tracks.parquet`` when *combined* is True); video metadata is
        stored in the Parquet schema.  Requires ``pyarrow``.
    combined : bool
        If True write a single ``all_tracks.<ext>``; otherwise one file per track.
    overwrite : bool
        If False (default) raise FileExistsError when any output file already
        exists.  Set True to silently overwrite.
    device : {"auto", "cpu", "cuda", "mps"}
        Compute backend for the regionprops step.  ``"auto"`` (default) uses
        the pyclesperanto/OpenCL GPU path when a device is visible, else CPU.
        ``"cuda"`` is a hard request for the GPU path (kept for naming
        consistency with `train` / `predict`; the backend itself is OpenCL,
        which works on NVIDIA, AMD, and Intel GPUs).  ``"mps"`` is unsupported
        and falls back to CPU.
    """
    t0 = perf_counter()
    from loguru import logger
    logger.debug(f"loguru import: {perf_counter()-t0:.3f}s")

    t1 = perf_counter()
    import zarr
    logger.debug(f"zarr import: {perf_counter()-t1:.3f}s")

    t2 = perf_counter()
    from natsort import natsorted
    import pandas as pd
    logger.debug(f"natsort+pandas import: {perf_counter()-t2:.3f}s")

    predictions_path = Path(predictions_path)

    # --- Discover CSV files ---
    t3 = perf_counter()
    csvs = natsorted(predictions_path.glob("*track_*.csv"))
    logger.debug(f"CSV discovery: {perf_counter()-t3:.3f}s")
    if not csvs:
        raise FileNotFoundError(f"No tracking CSV files found in {predictions_path}")
    logger.info(f"Found {len(csvs)} tracking CSV(s) in {predictions_path.name}")

    # --- Discover zarr (optional; needed only when computing region_properties) ---
    t4 = perf_counter()
    zarr_root = None
    zarr_candidate = predictions_path / "predictions.zarr"
    logger.debug(f"zarr discovery: {perf_counter()-t4:.3f}s")
    if zarr_candidate.exists():
        store = zarr.storage.LocalStore(zarr_candidate, read_only=True)
        zarr_root = zarr.open_group(store=store, mode="r")
        logger.info("Found zarr archive")

    # --- Resolve region_properties group aliases ---
    _GROUP_ALIASES = {"shape": "Size and Shape", "intensity": "Intensity"}
    if region_properties == "all":
        from octron.yolo_octron.constants import ALL_REGION_PROPERTIES
        region_properties = [p for props in ALL_REGION_PROPERTIES.values() for p in props]
    elif isinstance(region_properties, str) and region_properties in _GROUP_ALIASES:
        from octron.yolo_octron.constants import ALL_REGION_PROPERTIES
        region_properties = list(ALL_REGION_PROPERTIES[_GROUP_ALIASES[region_properties]])

    # --- Read each CSV ---
    t5 = perf_counter()
    track_ids      = []
    labels         = {}
    frame_counters = {}
    frame_indices  = {}
    confidence_d   = {}
    segments_x_d   = {}
    segments_y_d   = {}
    segments_area_d = {}
    bbox_d         = {}
    region_props_d = {}
    video_metadata = {}

    for csv_file in csvs:
        meta = _read_csv_metadata(csv_file, _CSV_HEADER_LINES)
        if not video_metadata:
            video_metadata = meta

        df = pd.read_csv(csv_file, skiprows=_CSV_HEADER_LINES)
        if df.empty:
            continue

        tid = int(df["track_id"].iloc[0])
        track_ids.append(tid)
        labels[tid] = str(df["label"].iloc[0])

        # frame_counter is the first column (MultiIndex level written by to_csv)
        frame_counters[tid] = df.iloc[:, 0].to_numpy()
        frame_indices[tid]  = df["frame_idx"].to_numpy()
        confidence_d[tid]   = df["confidence"].to_numpy()
        segments_x_d[tid]   = df["pos_x"].to_numpy()
        segments_y_d[tid]   = df["pos_y"].to_numpy()

        area_col = "area" if "area" in df.columns else None
        segments_area_d[tid] = (
            df[area_col].to_numpy() if area_col else np.ones(len(df))
        )

        bbox_d[tid] = {c: df[c].to_numpy() for c in _BBOX_COLS if c in df.columns}

        # Extra regionprop columns
        extra_cols = [c for c in df.columns if c not in _BASE_COLS]
        if region_properties == "none":
            extra_cols = []
        elif isinstance(region_properties, list):
            extra_cols = [c for c in extra_cols if c in region_properties]
        # region_properties=None → keep all extra columns already in the CSV
        region_props_d[tid] = {c: df[c].to_numpy() for c in extra_cols}

    logger.debug(f"CSV reading ({len(track_ids)} track(s)): {perf_counter()-t5:.3f}s")

    # --- Auto-detect video if not provided ---
    if video_path is None:
        video_path = _autodetect_video(predictions_path)
        if video_path is not None:
            logger.info(f"Auto-detected video: {video_path.name}")
        else:
            logger.debug("No video file found alongside predictions directory.")
    else:
        video_path = Path(video_path)

    # --- Compute region props from zarr (always, when zarr is available) ---
    # Zarr masks are the ground truth — always recompute requested props from
    # them rather than relying on CSV values, which may have been produced with
    # a different method or an older version of the pipeline.
    if isinstance(region_properties, list) and zarr_root is not None:
        props_to_compute = [p for p in region_properties if p not in _BASE_COLS]
        if props_to_compute:
            logger.info(f"Computing {props_to_compute} from zarr masks...")
            t_zarr = perf_counter()
            zarr_props = compute_region_props_from_zarr(
                zarr_root, track_ids, frame_indices, props_to_compute,
                zarr_path=zarr_candidate,
                video_path=video_path,
                device=device,
            )
            logger.debug(f"zarr regionprops: {perf_counter()-t_zarr:.3f}s")
            for tid in track_ids:
                region_props_d.setdefault(tid, {}).update(zarr_props.get(tid, {}))

    output_dir = Path(output_dir) if output_dir is not None else predictions_path

    # --- Overwrite guard ---
    ext = ".parquet" if fmt == "parquet" else ".csv"
    if not overwrite:
        if combined:
            candidate = output_dir / f"all_tracks{ext}"
            if candidate.exists():
                raise FileExistsError(
                    f"{candidate} already exists. Pass overwrite=True to replace it."
                )
        else:
            existing = [
                output_dir / f"{labels[tid]}_track_{tid}{ext}"
                for tid in track_ids
                if (output_dir / f"{labels[tid]}_track_{tid}{ext}").exists()
            ]
            if existing:
                names = ", ".join(p.name for p in existing)
                raise FileExistsError(
                    f"Output file(s) already exist: {names}. "
                    "Pass overwrite=True to replace them."
                )

    t6 = perf_counter()
    _export_tracking_from_data(
        output_dir=output_dir,
        track_ids=track_ids,
        labels=labels,
        frame_counters=frame_counters,
        frame_indices=frame_indices,
        confidence=confidence_d,
        segments_x=segments_x_d,
        segments_y=segments_y_d,
        segments_area=segments_area_d,
        bbox=bbox_d,
        region_props=region_props_d,
        video_metadata=video_metadata,
        method=method,
        zarr_root=zarr_root,
        combined=combined,
        fmt=fmt,
    )
    logger.debug(f"export write: {perf_counter()-t6:.3f}s")
    logger.debug(f"total: {perf_counter()-t0:.3f}s")
