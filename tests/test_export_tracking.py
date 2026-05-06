"""
Tests for octron/tools/export_tracking.py.

No video files, zarr archives, or GPU are required — all tests use synthetic
in-memory data or temporary CSV files.

Column contract
---------------
All tuple-valued columns are resolved to scalars in the output.  Original
column names (pos_x, pos_y, area, …) are kept — no renaming.

  pos_x / pos_y : scalar centroid per chosen method
  area          : largest segment's area ("largest") OR sum of all segments
                  ("weighted")
  orientation   : always from the largest segment (circular quantity)
  all other regionprops : resolved by chosen method

Covered
-------
_parse_tuple_or_scalar
    scalar int/float returned as single-element tuple
    tuple-string parsed correctly
    NaN / unparseable value returned as (nan,)

_resolve_xy_largest
    single-segment rows: returns the scalar position unchanged
    multi-segment rows: returns position of the largest segment

_resolve_xy_weighted
    single-segment: result equals the scalar position
    multi-segment with equal areas: result is the simple mean
    multi-segment: result is between the two endpoints
    zero-area segments: no divide-by-zero

_resolve_prop_largest / _resolve_prop_weighted
    scalar passthrough
    multi-segment: correct value selected / weighted

_sum_segments_area
    scalar rows: result equals the scalar
    tuple rows: result equals the arithmetic sum

_build_header / _read_csv_metadata
    round-trip: written header parsed back to same dict

_export_tracking_from_data
    output file(s) created
    pos_x / pos_y columns present and numeric
    area column present and numeric
    no centroid_x / centroid_y / segments_x / segments_y columns in output
    "largest": single-segment pos_x == raw input
    "largest": multi-segment pos_x == x of biggest segment
    "largest": area == area of biggest segment
    "weighted": pos_x between segment endpoints
    "weighted": area == sum of segment areas
    orientation always resolved via largest regardless of method
    other regionprops resolved via chosen method
    combined=False → one file per track
    combined=True  → single all_tracks.csv
    header metadata preserved in output

export_tracking (public)
    reads CSV, writes new CSV with correct column names
    "largest" / "weighted" methods forwarded correctly
    combined flag forwarded correctly
    raises FileNotFoundError on missing CSVs
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from octron.tools.export_tracking import (
    _build_header,
    _export_tracking_from_data,
    _parse_tuple_or_scalar,
    _read_csv_metadata,
    _resolve_prop_largest,
    _resolve_prop_weighted,
    _resolve_xy_largest,
    _resolve_xy_weighted,
    _sum_segments_area,
    export_tracking,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _scalar_arr(*vals):
    return np.array(vals, dtype=object)


def _tup(t):
    """Produce a tuple-string exactly as predict_batch writes it."""
    return str(t)


def _minimal_data(n=4, n_tracks=1):
    """
    Return the positional args needed by _export_tracking_from_data
    (all single-segment scalar rows, no tuples).
    """
    tids   = list(range(n_tracks))
    labels = {tid: "animal" for tid in tids}
    fc     = {tid: np.arange(n) for tid in tids}
    fi     = {tid: np.arange(n) for tid in tids}
    conf   = {tid: np.full(n, 0.9) for tid in tids}
    sx     = {tid: np.arange(n, dtype=float) for tid in tids}
    sy     = {tid: np.arange(n, dtype=float) for tid in tids}
    sa     = {tid: np.full(n, 100.0) for tid in tids}
    bbox   = {tid: {
        "bbox_x_min": np.zeros(n), "bbox_x_max": np.ones(n),
        "bbox_y_min": np.zeros(n), "bbox_y_max": np.ones(n),
        "bbox_area":  np.ones(n),  "bbox_aspect_ratio": np.ones(n),
    } for tid in tids}
    rp     = {tid: {} for tid in tids}
    meta   = {
        "video_name": "test.mp4", "frame_count": "100",
        "frame_count_analyzed": str(n), "video_height": "480",
        "video_width": "640",   "created_at": "2026-01-01 00:00:00",
    }
    return tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta


def _read_output(tmp_path, filename="animal_track_0.csv"):
    return pd.read_csv(tmp_path / filename, skiprows=7)


# ===========================================================================
# _parse_tuple_or_scalar
# ===========================================================================

def test_parse_scalar_int():
    assert _parse_tuple_or_scalar(5) == (5.0,)

def test_parse_scalar_float():
    assert _parse_tuple_or_scalar(3.14) == pytest.approx((3.14,))

def test_parse_two_element_tuple_string():
    result = _parse_tuple_or_scalar("(10.5, 20.3)")
    assert len(result) == 2
    assert result[0] == pytest.approx(10.5)
    assert result[1] == pytest.approx(20.3)

def test_parse_single_element_tuple_string():
    assert _parse_tuple_or_scalar("(42.0,)") == pytest.approx((42.0,))

def test_parse_nan_string():
    result = _parse_tuple_or_scalar("NaN")
    assert len(result) == 1 and math.isnan(result[0])

def test_parse_unparseable_string():
    result = _parse_tuple_or_scalar("garbage")
    assert len(result) == 1 and math.isnan(result[0])


# ===========================================================================
# _resolve_xy_largest
# ===========================================================================

def test_largest_single_segment_passthrough():
    xs = _scalar_arr(10.0, 20.0)
    ys = _scalar_arr( 5.0, 15.0)
    sa = _scalar_arr(100.0, 200.0)
    cx, cy = _resolve_xy_largest(xs, ys, sa)
    assert cx == pytest.approx([10.0, 20.0])
    assert cy == pytest.approx([ 5.0, 15.0])

def test_largest_picks_first_when_biggest():
    cx, _ = _resolve_xy_largest(
        _scalar_arr(_tup((10.0, 90.0))),
        _scalar_arr(_tup(( 5.0, 50.0))),
        _scalar_arr(_tup((200.0, 50.0))),
    )
    assert cx[0] == pytest.approx(10.0)

def test_largest_picks_second_when_biggest():
    cx, cy = _resolve_xy_largest(
        _scalar_arr(_tup((10.0, 90.0))),
        _scalar_arr(_tup(( 5.0, 50.0))),
        _scalar_arr(_tup(( 50.0, 300.0))),
    )
    assert cx[0] == pytest.approx(90.0)
    assert cy[0] == pytest.approx(50.0)


# ===========================================================================
# _resolve_xy_weighted
# ===========================================================================

def test_weighted_single_segment_passthrough():
    cx, cy = _resolve_xy_weighted(
        _scalar_arr(10.0), _scalar_arr(5.0), _scalar_arr(100.0)
    )
    assert cx[0] == pytest.approx(10.0)
    assert cy[0] == pytest.approx(5.0)

def test_weighted_equal_areas_gives_mean():
    cx, cy = _resolve_xy_weighted(
        _scalar_arr(_tup((0.0, 100.0))),
        _scalar_arr(_tup((0.0, 200.0))),
        _scalar_arr(_tup((50.0, 50.0))),
    )
    assert cx[0] == pytest.approx(50.0)
    assert cy[0] == pytest.approx(100.0)

def test_weighted_result_between_endpoints():
    cx, _ = _resolve_xy_weighted(
        _scalar_arr(_tup((0.0, 100.0))),
        _scalar_arr(_tup((0.0,   0.0))),
        _scalar_arr(_tup((75.0, 25.0))),
    )
    assert 0.0 < cx[0] < 100.0

def test_weighted_zero_area_no_error():
    cx, _ = _resolve_xy_weighted(
        _scalar_arr(_tup((0.0, 100.0))),
        _scalar_arr(_tup((0.0,   0.0))),
        _scalar_arr(_tup((0.0,   0.0))),
    )
    assert not math.isnan(cx[0])


# ===========================================================================
# _resolve_prop_largest / _resolve_prop_weighted
# ===========================================================================

def test_resolve_prop_largest_scalar_passthrough():
    out = _resolve_prop_largest(_scalar_arr(0.8, 0.5), _scalar_arr(100.0, 200.0))
    assert out == pytest.approx([0.8, 0.5])

def test_resolve_prop_largest_picks_correct_index():
    # areas (50, 300) → index 1 is largest
    out = _resolve_prop_largest(
        _scalar_arr(_tup((0.8, 0.4))),
        _scalar_arr(_tup((50.0, 300.0))),
    )
    assert out[0] == pytest.approx(0.4)

def test_resolve_prop_weighted_scalar_passthrough():
    out = _resolve_prop_weighted(_scalar_arr(0.6, 0.9), _scalar_arr(100.0, 200.0))
    assert out == pytest.approx([0.6, 0.9])

def test_resolve_prop_weighted_equal_areas_is_mean():
    out = _resolve_prop_weighted(
        _scalar_arr(_tup((0.0, 1.0))),
        _scalar_arr(_tup((50.0, 50.0))),
    )
    assert out[0] == pytest.approx(0.5)


# ===========================================================================
# _sum_segments_area
# ===========================================================================

def test_sum_area_scalar():
    out = _sum_segments_area(_scalar_arr(100.0, 200.0))
    assert out == pytest.approx([100.0, 200.0])

def test_sum_area_tuples():
    out = _sum_segments_area(_scalar_arr(_tup((200.0, 50.0)), _tup((10.0, 10.0, 5.0))))
    assert out[0] == pytest.approx(250.0)
    assert out[1] == pytest.approx(25.0)


# ===========================================================================
# _build_header / _read_csv_metadata  round-trip
# ===========================================================================

def test_header_round_trip(tmp_path):
    meta = {
        "video_name": "myvideo.mp4", "frame_count": "500",
        "frame_count_analyzed": "250", "video_height": "720",
        "video_width": "1280", "created_at": "2026-01-01 12:00:00",
    }
    p = tmp_path / "track.csv"
    p.write_text(_build_header(meta) + "col1,col2\n1,2\n")
    recovered = _read_csv_metadata(p, n_header_lines=7)
    for key in meta:
        assert recovered[key] == meta[key]


# ===========================================================================
# _export_tracking_from_data — column contract
# ===========================================================================

def test_export_output_has_pos_x_pos_y_area(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(), method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    for col in ("pos_x", "pos_y", "area"):
        assert col in df.columns, f"Missing: {col}"

def test_export_no_centroid_or_segments_columns(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(), method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    for col in ("centroid_x", "centroid_y", "segments_x", "segments_y", "segments_area"):
        assert col not in df.columns, f"Unexpected column present: {col}"

def test_export_pos_columns_are_numeric(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(), method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert pd.api.types.is_float_dtype(df["pos_x"])
    assert pd.api.types.is_float_dtype(df["pos_y"])
    assert pd.api.types.is_float_dtype(df["area"])

def test_export_area_is_numeric(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(), method="weighted",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert pd.api.types.is_float_dtype(df["area"])


# ===========================================================================
# _export_tracking_from_data — method="largest"
# ===========================================================================

def test_largest_single_segment_pos_equals_input(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=4)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    np.testing.assert_allclose(df["pos_x"].values, sx[0].astype(float))

def test_largest_multi_segment_pos_x_picks_biggest(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    # Second segment has larger area (300 vs 50) → pos_x should be 90
    sx[0] = np.array([_tup((10.0, 90.0))], dtype=object)
    sy[0] = np.array([_tup(( 5.0, 50.0))], dtype=object)
    sa[0] = np.array([_tup((50.0, 300.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["pos_x"].iloc[0] == pytest.approx(90.0)

def test_largest_area_is_area_of_biggest_segment(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    sa[0] = np.array([_tup((50.0, 300.0))], dtype=object)
    sx[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    sy[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["area"].iloc[0] == pytest.approx(300.0)


# ===========================================================================
# _export_tracking_from_data — method="weighted"
# ===========================================================================

def test_weighted_pos_x_between_segments(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    sx[0] = np.array([_tup((0.0, 100.0))], dtype=object)
    sy[0] = np.array([_tup((0.0,   0.0))], dtype=object)
    sa[0] = np.array([_tup((75.0, 25.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="weighted",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert 0.0 < df["pos_x"].iloc[0] < 100.0

def test_weighted_area_is_sum_of_segments(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    sa[0] = np.array([_tup((200.0, 50.0))], dtype=object)
    sx[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    sy[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="weighted",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["area"].iloc[0] == pytest.approx(250.0)


# ===========================================================================
# _export_tracking_from_data — regionprops resolution
# ===========================================================================

def test_regionprop_tuples_resolved_to_scalar(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=2)
    sa[0] = np.array([_tup((200.0, 50.0)), 100.0], dtype=object)
    rp[0] = {"solidity": np.array([_tup((0.9, 0.4)), 0.8], dtype=object)}
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert "solidity" in df.columns
    assert pd.api.types.is_float_dtype(df["solidity"])
    # largest segment has area 200 (index 0) → solidity should be 0.9
    assert df["solidity"].iloc[0] == pytest.approx(0.9)

def test_orientation_always_uses_largest_regardless_of_method(tmp_path):
    """orientation is circular — must always come from the largest segment."""
    for method in ("largest", "weighted"):
        out = tmp_path / method
        out.mkdir()
        tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
        sa[0] = np.array([_tup((50.0, 300.0))], dtype=object)
        sx[0] = np.array([_tup((0.0, 0.0))], dtype=object)
        sy[0] = np.array([_tup((0.0, 0.0))], dtype=object)
        # orientation values differ per segment
        rp[0] = {"orientation": np.array([_tup((1.0, 2.0))], dtype=object)}
        _export_tracking_from_data(out, tids, labels, fc, fi, conf, sx, sy, sa,
                                    bbox, rp, meta, method=method,
                                    zarr_root=None, combined=False)
        df = pd.read_csv(out / "animal_track_0.csv", skiprows=7)
        # second segment is largest (area 300 → index 1 → orientation 2.0)
        assert df["orientation"].iloc[0] == pytest.approx(2.0), \
            f"orientation wrong for method={method}"


# ===========================================================================
# _export_tracking_from_data — file-writing
# ===========================================================================

def test_export_per_track_files(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(n_tracks=2),
                                method="largest", zarr_root=None, combined=False)
    assert (tmp_path / "animal_track_0.csv").exists()
    assert (tmp_path / "animal_track_1.csv").exists()

def test_export_combined_file(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(n_tracks=2),
                                method="largest", zarr_root=None, combined=True)
    assert (tmp_path / "all_tracks.csv").exists()
    assert not (tmp_path / "animal_track_0.csv").exists()

def test_export_combined_contains_all_tracks(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(n_tracks=3),
                                method="largest", zarr_root=None, combined=True)
    df = pd.read_csv(tmp_path / "all_tracks.csv", skiprows=7)
    assert set(df["track_id"].unique()) == {0, 1, 2}

def test_export_header_preserved(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(),
                                method="largest", zarr_root=None, combined=False)
    meta = _read_csv_metadata(tmp_path / "animal_track_0.csv")
    assert meta["video_name"] == "test.mp4"
    assert meta["video_height"] == "480"


# ===========================================================================
# export_tracking  (public)
# ===========================================================================

def _write_prediction_csv(path, track_id, label="animal", n=4):
    """Write a CSV in the format produced by _export_tracking_from_data."""
    from octron.tools.export_tracking import _build_header
    meta = {
        "video_name": "vid.mp4", "frame_count": "100",
        "frame_count_analyzed": str(n), "video_height": "480",
        "video_width": "640",   "created_at": "2026-01-01 00:00:00",
    }
    df = pd.DataFrame({
        "frame_counter":    np.arange(n),
        "frame_idx":        np.arange(n),
        "track_id":         track_id,
        "label":            label,
        "confidence":       0.9,
        "pos_x":            np.arange(n, dtype=float),
        "pos_y":            np.arange(n, dtype=float),
        "bbox_x_min":       0.0, "bbox_x_max": 1.0,
        "bbox_y_min":       0.0, "bbox_y_max": 1.0,
        "bbox_area":        1.0, "bbox_aspect_ratio": 1.0,
        "area":             100.0,
    })
    with open(path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")


def test_public_export_creates_output(tmp_path):
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    assert (tmp_path / "animal_track_1.csv").exists()

def test_public_export_output_has_pos_and_area(tmp_path):
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    df = pd.read_csv(tmp_path / "animal_track_1.csv", skiprows=7)
    assert "pos_x" in df.columns
    assert "area" in df.columns

def test_public_export_no_spurious_columns(tmp_path):
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    df = pd.read_csv(tmp_path / "animal_track_1.csv", skiprows=7)
    for col in ("centroid_x", "centroid_y", "segments_x", "segments_y", "segments_area"):
        assert col not in df.columns

def test_public_export_combined(tmp_path):
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    _write_prediction_csv(tmp_path / "animal_track_2.csv", track_id=2)
    export_tracking(tmp_path, output_dir=tmp_path, combined=True)
    assert (tmp_path / "all_tracks.csv").exists()

def test_public_export_region_properties_filter(tmp_path):
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1)
    # Append extra regionprop columns
    df = pd.read_csv(csv_path, skiprows=7)
    df["eccentricity"] = 0.5
    df["solidity"]     = 0.8
    meta = _read_csv_metadata(csv_path)
    from octron.tools.export_tracking import _build_header
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")

    export_tracking(tmp_path, output_dir=tmp_path, region_properties=["eccentricity"], overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "eccentricity" in out.columns
    assert "solidity" not in out.columns

def test_public_export_region_properties_none_strips_extras(tmp_path):
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1)
    df = pd.read_csv(csv_path, skiprows=7)
    df["eccentricity"] = 0.5
    meta = _read_csv_metadata(csv_path)
    from octron.tools.export_tracking import _build_header
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")

    export_tracking(tmp_path, output_dir=tmp_path, region_properties="none", overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "eccentricity" not in out.columns

def test_public_export_region_properties_all_computes_all(tmp_path):
    # "all" expands to every known property; without zarr, only columns already
    # in the CSV (that are in ALL_REGION_PROPERTIES) survive the filter.
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1)
    df = pd.read_csv(csv_path, skiprows=7)
    df["eccentricity"] = 0.5
    meta = _read_csv_metadata(csv_path)
    from octron.tools.export_tracking import _build_header
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")

    export_tracking(tmp_path, output_dir=tmp_path, region_properties="all", overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "eccentricity" in out.columns

def _write_csv_with_extras(tmp_path, extras: dict):
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1)
    df = pd.read_csv(csv_path, skiprows=7)
    for col, val in extras.items():
        df[col] = val
    meta = _read_csv_metadata(csv_path)
    from octron.tools.export_tracking import _build_header
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")
    return csv_path

def test_public_export_region_properties_shape_alias(tmp_path):
    csv_path = _write_csv_with_extras(tmp_path, {"eccentricity": 0.5, "intensity_mean": 100.0})
    export_tracking(tmp_path, output_dir=tmp_path, region_properties="shape", overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "eccentricity" in out.columns
    assert "intensity_mean" not in out.columns

def test_public_export_region_properties_intensity_alias(tmp_path):
    csv_path = _write_csv_with_extras(tmp_path, {"eccentricity": 0.5, "intensity_mean": 100.0})
    export_tracking(tmp_path, output_dir=tmp_path, region_properties="intensity", overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "intensity_mean" in out.columns
    assert "eccentricity" not in out.columns

def test_zarr_regionprops_computed_when_missing_from_csv(tmp_path):
    """Properties not in the CSV are computed from zarr masks at export time."""
    import zarr, numpy as np

    # Write a CSV with no extra regionprop columns
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1)
    df = pd.read_csv(csv_path, skiprows=7)
    n = len(df)

    # Build a minimal zarr archive with a small binary mask per frame
    store_path = tmp_path / "predictions.zarr"
    store = zarr.storage.LocalStore(str(store_path))
    root = zarr.open_group(store=store, mode="w")
    H, W = 64, 64
    masks = np.zeros((n, H, W), dtype=np.int8)
    for i in range(n):
        masks[i, 10:30, 10:30] = 1  # 20×20 square object
    arr = root.require_array("1_masks", shape=(n, H, W), dtype=np.int8)
    arr[:] = masks
    arr.attrs["video_height"] = H
    arr.attrs["video_width"] = W

    export_tracking(tmp_path, output_dir=tmp_path, region_properties=["eccentricity"], overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "eccentricity" in out.columns
    assert out["eccentricity"].notna().all()

def test_public_export_raises_on_missing_csvs(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_tracking(tmp_path / "nonexistent")


# ===========================================================================
# _export_tracking_from_data — method="raw"
# ===========================================================================

def test_raw_single_segment_pos_passthrough(tmp_path):
    """Scalar values pass through unchanged."""
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=4)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="raw",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    np.testing.assert_allclose(df["pos_x"].astype(float).values, sx[0].astype(float))


def test_raw_multi_segment_pos_x_keeps_tuple_string(tmp_path):
    """Multi-segment rows keep their tuple-string verbatim."""
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    sx[0] = np.array([_tup((10.0, 90.0))], dtype=object)
    sy[0] = np.array([_tup(( 5.0, 50.0))], dtype=object)
    sa[0] = np.array([_tup((50.0, 300.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="raw",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["pos_x"].iloc[0] == _tup((10.0, 90.0))
    assert df["pos_y"].iloc[0] == _tup(( 5.0, 50.0))


def test_raw_area_keeps_tuple_string(tmp_path):
    """area is NOT summed in raw mode — tuple-string preserved."""
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    sa[0] = np.array([_tup((200.0, 50.0))], dtype=object)
    sx[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    sy[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="raw",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["area"].iloc[0] == _tup((200.0, 50.0))


def test_raw_orientation_no_special_case(tmp_path):
    """orientation is NOT forced to 'largest' in raw mode."""
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    sa[0] = np.array([_tup((50.0, 300.0))], dtype=object)
    sx[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    sy[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    rp[0] = {"orientation": np.array([_tup((1.0, 2.0))], dtype=object)}
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="raw",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["orientation"].iloc[0] == _tup((1.0, 2.0))


def test_raw_regionprops_passthrough(tmp_path):
    """Mixed scalar + tuple regionprops survive verbatim in raw mode."""
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=2)
    sa[0] = np.array([_tup((200.0, 50.0)), 100.0], dtype=object)
    rp[0] = {"solidity": np.array([_tup((0.9, 0.4)), 0.8], dtype=object)}
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, method="raw",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["solidity"].iloc[0] == _tup((0.9, 0.4))
    assert float(df["solidity"].iloc[1]) == pytest.approx(0.8)


def test_raw_is_default_method_for_public_api(tmp_path):
    """export_tracking() with no method= argument should preserve tuple-strings."""
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1, n=2)
    df = pd.read_csv(csv_path, skiprows=7)
    df["pos_x"] = df["pos_x"].astype(object)
    df["area"] = df["area"].astype(object)
    df.loc[0, "pos_x"] = _tup((10.0, 90.0))
    df.loc[0, "area"] = _tup((50.0, 300.0))
    meta = _read_csv_metadata(csv_path)
    from octron.tools.export_tracking import _build_header
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    export_tracking(tmp_path, output_dir=out_dir)  # no method= → defaults to raw
    out = pd.read_csv(out_dir / "animal_track_1.csv", skiprows=7)
    assert out["pos_x"].iloc[0] == _tup((10.0, 90.0))
    assert out["area"].iloc[0] == _tup((50.0, 300.0))


def test_raw_parquet_emits_warning(tmp_path, caplog):
    """method='raw' + fmt='parquet' should warn about string-typed columns."""
    pytest.importorskip("pyarrow")
    import logging
    from loguru import logger as _logger

    # Loguru → stdlib logging bridge so caplog captures the warning
    class _PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = _logger.add(_PropagateHandler(), format="{message}", level="WARNING")
    try:
        with caplog.at_level("WARNING"):
            _export_tracking_from_data(tmp_path, *_minimal_data(),
                                        method="raw", zarr_root=None,
                                        combined=False, fmt="parquet")
    finally:
        _logger.remove(handler_id)

    assert any("raw" in rec.message and "parquet" in rec.message
               for rec in caplog.records), \
        "Expected a warning mentioning raw + parquet"


# ===========================================================================
# compute_weighted_centroids
# ===========================================================================

def test_weighted_centroids_scalar_passthrough(tmp_path):
    """Scalar pos_x/pos_y in the CSV passes through unchanged."""
    from octron.tools.export_tracking import compute_weighted_centroids
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1, n=3)
    out = compute_weighted_centroids(tmp_path)
    assert 1 in out
    # _write_prediction_csv writes pos_x = arange(n) and pos_y = arange(n)
    assert out[1][0] == pytest.approx((0.0, 0.0))
    assert out[1][2] == pytest.approx((2.0, 2.0))


def test_weighted_centroids_multi_segment(tmp_path):
    """Tuple-string rows resolve to area-weighted means."""
    from octron.tools.export_tracking import compute_weighted_centroids
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1, n=2)
    df = pd.read_csv(csv_path, skiprows=7)
    df["pos_x"] = df["pos_x"].astype(object)
    df["pos_y"] = df["pos_y"].astype(object)
    df["area"] = df["area"].astype(object)
    # Frame 0: equal areas → mean is 50, frame 1: 3:1 weight on x=0
    df.loc[0, "pos_x"] = _tup((0.0, 100.0))
    df.loc[0, "pos_y"] = _tup((0.0, 100.0))
    df.loc[0, "area"]  = _tup((50.0, 50.0))
    df.loc[1, "pos_x"] = _tup((0.0, 100.0))
    df.loc[1, "pos_y"] = _tup((0.0,   0.0))
    df.loc[1, "area"]  = _tup((75.0, 25.0))
    meta = _read_csv_metadata(csv_path)
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")

    out = compute_weighted_centroids(tmp_path)
    assert out[1][0] == pytest.approx((50.0, 50.0))
    cx1, cy1 = out[1][1]
    assert 0.0 < cx1 < 100.0   # weighted between segment positions
    assert cy1 == pytest.approx(0.0)


def test_weighted_centroids_track_id_filter(tmp_path):
    """Only the requested track_ids appear in the result."""
    from octron.tools.export_tracking import compute_weighted_centroids
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    _write_prediction_csv(tmp_path / "animal_track_2.csv", track_id=2)
    out = compute_weighted_centroids(tmp_path, track_ids=[1])
    assert set(out.keys()) == {1}


def test_weighted_centroids_empty_dir(tmp_path):
    """No CSVs → empty dict, no exception."""
    from octron.tools.export_tracking import compute_weighted_centroids
    assert compute_weighted_centroids(tmp_path) == {}


# ===========================================================================
# export_tracking — overwrite guard
# ===========================================================================

def test_overwrite_false_raises_when_file_exists(tmp_path):
    """Default overwrite=False should raise FileExistsError if output exists."""
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    # First export creates the file
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    # Second export without overwrite should fail
    with pytest.raises(FileExistsError):
        export_tracking(tmp_path, output_dir=tmp_path, overwrite=False)


def test_overwrite_true_replaces_existing_file(tmp_path):
    """overwrite=True should silently replace an existing output file."""
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)  # no error
    assert (tmp_path / "animal_track_1.csv").exists()


def test_overwrite_false_raises_for_combined(tmp_path):
    """overwrite=False should raise when all_tracks.csv already exists."""
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, combined=True, overwrite=True)
    with pytest.raises(FileExistsError):
        export_tracking(tmp_path, output_dir=tmp_path, combined=True, overwrite=False)


def test_overwrite_false_no_error_when_output_dir_is_different(tmp_path):
    """No error when writing to a fresh directory even with overwrite=False."""
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    out = tmp_path / "fresh"
    out.mkdir()
    export_tracking(tmp_path, output_dir=out, overwrite=False)  # should not raise
    assert (out / "animal_track_1.csv").exists()


# ===========================================================================
# Column renaming  (skimage '-N' suffixes → readable labels)
# ===========================================================================

def test_rename_intensity_channel_suffix():
    from octron.tools.export_tracking import _rename_prop_column
    assert _rename_prop_column("intensity_mean-0", has_intensity=True) == "intensity_mean_r"
    assert _rename_prop_column("intensity_mean-1", has_intensity=True) == "intensity_mean_g"
    assert _rename_prop_column("intensity_mean-2", has_intensity=True) == "intensity_mean_b"
    assert _rename_prop_column("intensity_mean-3", has_intensity=True) == "intensity_mean_lum"


def test_rename_weighted_centroid_axis_channel():
    from octron.tools.export_tracking import _rename_prop_column
    assert _rename_prop_column("weighted_centroid-0-0", True) == "weighted_centroid_y_r"
    assert _rename_prop_column("weighted_centroid-0-1", True) == "weighted_centroid_y_g"
    assert _rename_prop_column("weighted_centroid-0-2", True) == "weighted_centroid_y_b"
    assert _rename_prop_column("weighted_centroid-0-3", True) == "weighted_centroid_y_lum"
    assert _rename_prop_column("weighted_centroid-1-0", True) == "weighted_centroid_x_r"
    assert _rename_prop_column("weighted_centroid-1-3", True) == "weighted_centroid_x_lum"


def test_rename_centroid_axis_only():
    from octron.tools.export_tracking import _rename_prop_column
    assert _rename_prop_column("centroid-0", has_intensity=False) == "centroid_y"
    assert _rename_prop_column("centroid-1", has_intensity=False) == "centroid_x"


def test_rename_moments_hu_unchanged():
    from octron.tools.export_tracking import _rename_prop_column
    assert _rename_prop_column("moments_hu-0", has_intensity=False) == "moments_hu-0"
    assert _rename_prop_column("moments_hu-6", has_intensity=False) == "moments_hu-6"


def test_rename_bare_property_unchanged():
    from octron.tools.export_tracking import _rename_prop_column
    assert _rename_prop_column("eccentricity", has_intensity=False) == "eccentricity"
    assert _rename_prop_column("solidity", has_intensity=True) == "solidity"


# ===========================================================================
# Bbox-offset correction for translation-variant regionprops
# ===========================================================================

def _build_synthetic_mask_and_intensity(H=64, W=64, y0=10, x0=12, h=20, w=24):
    """Build a binary mask with a single rectangular region + 4-channel intensity."""
    mask = np.zeros((H, W), dtype=np.int8)
    mask[y0:y0+h, x0:x0+w] = 1
    rng = np.random.default_rng(42)
    # 4 channels (R, G, B, lum) — random integers in 0..255
    intensity = rng.integers(0, 256, size=(H, W, 4), dtype=np.int32).astype(np.uint8)
    return mask, intensity


def test_weighted_centroid_bbox_offset_matches_full_image():
    """Cropping then offsetting must match running on the full image."""
    from skimage import measure
    from octron.tools.export_tracking import _TRANSLATION_VARIANT_PROPS

    H, W, y0, x0, h, w = 64, 64, 10, 12, 20, 24
    mask, intensity = _build_synthetic_mask_and_intensity(H, W, y0, x0, h, w)

    properties = ["weighted_centroid", "centroid", "area", "intensity_mean"]

    # Full-image (correct reference)
    full_labeled = measure.label(mask, background=0, connectivity=2)
    full = measure.regionprops_table(
        full_labeled, intensity_image=intensity, properties=properties
    )

    # Cropped (would silently shift values for variant props)
    rows_any = mask.any(axis=1); cols_any = mask.any(axis=0)
    r0 = int(rows_any.argmax())
    r1 = int(len(rows_any) - rows_any[::-1].argmax())
    c0 = int(cols_any.argmax())
    c1 = int(len(cols_any) - cols_any[::-1].argmax())

    crop_labeled = measure.label(mask[r0:r1, c0:c1], background=0, connectivity=2)
    cropped = measure.regionprops_table(
        crop_labeled, intensity_image=intensity[r0:r1, c0:c1],
        properties=properties,
    )

    # Apply the same offset correction the worker does.
    for col_key, vals in list(cropped.items()):
        base = col_key.split("-")[0]
        if base not in _TRANSLATION_VARIANT_PROPS:
            continue
        ax_idx = int(col_key.split("-")[1])
        offset = r0 if ax_idx == 0 else c0
        cropped[col_key] = vals + offset

    for col_key in full:
        np.testing.assert_allclose(
            cropped[col_key], full[col_key], rtol=1e-6, atol=1e-6,
            err_msg=f"Mismatch in column {col_key} after offset correction",
        )


def test_weighted_centroid_displacement_signal():
    """R-channel intensity on the LEFT, B-channel on the RIGHT —
    weighted_centroid_x_r must lie left of weighted_centroid_x_b."""
    from skimage import measure

    H, W = 32, 64
    mask = np.zeros((H, W), dtype=np.int8)
    mask[8:24, 8:56] = 1
    intensity = np.zeros((H, W, 4), dtype=np.uint8)
    intensity[:, :W // 2, 0] = 200    # R bright on the left
    intensity[:, W // 2:, 2] = 200    # B bright on the right

    labeled = measure.label(mask, background=0, connectivity=2)
    out = measure.regionprops_table(
        labeled, intensity_image=intensity, properties=["weighted_centroid"]
    )
    # axis-1 (x) channel-0 (R) and channel-2 (B)
    x_r = out["weighted_centroid-1-0"][0]
    x_b = out["weighted_centroid-1-2"][0]
    assert x_r < x_b, f"Expected R-weighted x ({x_r}) < B-weighted x ({x_b})"


# ===========================================================================
# _ordered_columns expansion
# ===========================================================================

def test_ordered_columns_includes_axis_channel_expansions():
    from octron.tools.export_tracking import _ordered_columns

    cols = [
        "label", "confidence", "pos_x", "pos_y", "area",
        "weighted_centroid_y_r", "weighted_centroid_x_r",
        "weighted_centroid_y_lum", "weighted_centroid_x_lum",
        "intensity_mean_r", "intensity_mean_lum",
        "eccentricity",
    ]
    ordered = _ordered_columns(cols)
    # Every input column survives the ordering step.
    assert set(ordered) == set(cols)
    # Channel-only intensity expansion comes before axis-channel expansion
    # (intensity_mean before weighted_centroid in ALL_REGION_PROPERTIES order).
    assert ordered.index("intensity_mean_r") < ordered.index("weighted_centroid_y_r")
    # Within weighted_centroid, axis-y comes before axis-x.
    assert ordered.index("weighted_centroid_y_r") < ordered.index("weighted_centroid_x_r")


# ===========================================================================
# Intensity-weighted spread (weighted_var_y / weighted_var_x / weighted_cov_yx)
# ===========================================================================

def _weighted_var_xy_reference(intensity_2d):
    """Reference μ_20/μ_00 and μ_02/μ_00 over the full image, single channel."""
    H, W = intensity_2d.shape
    yy, xx = np.mgrid[:H, :W].astype(float)
    I = intensity_2d.astype(float)
    m00 = I.sum()
    cy = (yy * I).sum() / m00
    cx = (xx * I).sum() / m00
    var_y = ((yy - cy) ** 2 * I).sum() / m00
    var_x = ((xx - cx) ** 2 * I).sum() / m00
    cov_yx = ((yy - cy) * (xx - cx) * I).sum() / m00
    return var_y, var_x, cov_yx


def test_weighted_var_matches_manual_reference():
    """Pseudo-prop derivation in the worker post-process must match a hand-rolled
    intensity-weighted variance/covariance computation."""
    from skimage import measure
    from octron.tools.export_tracking import _SPREAD_FROM_MOMENT

    H, W = 40, 60
    mask = np.zeros((H, W), dtype=np.int8)
    mask[5:35, 8:52] = 1

    rng = np.random.default_rng(0)
    intensity_1ch = rng.integers(50, 200, size=(H, W), dtype=np.int32).astype(np.uint8)

    # Multi-channel image with the same data on the R channel; other channels
    # left at zero so we focus on a single channel comparison.
    intensity = np.zeros((H, W, 4), dtype=np.uint8)
    intensity[..., 0] = intensity_1ch  # R

    var_y_ref, var_x_ref, cov_yx_ref = _weighted_var_xy_reference(
        intensity_1ch * mask
    )

    labeled = measure.label(mask, background=0, connectivity=2)
    raw = measure.regionprops_table(
        labeled, intensity_image=intensity,
        properties=["weighted_moments_central"],
    )
    # Replicate the worker's derivation for channel 0.
    m00 = raw["weighted_moments_central-0-0-0"][0]
    derived = {}
    for (i, j), base in _SPREAD_FROM_MOMENT.items():
        derived[base] = raw[f"weighted_moments_central-{i}-{j}-0"][0] / m00

    assert derived["weighted_var_y"] == pytest.approx(var_y_ref, rel=1e-6)
    assert derived["weighted_var_x"] == pytest.approx(var_x_ref, rel=1e-6)
    assert derived["weighted_cov_yx"] == pytest.approx(cov_yx_ref, rel=1e-6)


def test_weighted_var_x_signals_horizontal_spread():
    """Wider region should produce larger weighted_var_x than a narrower one."""
    from skimage import measure
    from octron.tools.export_tracking import _SPREAD_FROM_MOMENT

    def _var_x_for(width):
        H, W = 32, 96
        mask = np.zeros((H, W), dtype=np.int8)
        x_lo = (W - width) // 2
        mask[8:24, x_lo : x_lo + width] = 1
        intensity = np.zeros((H, W, 4), dtype=np.uint8)
        # Uniform R intensity inside the mask
        intensity[..., 0] = (mask.astype(np.int32) * 200).astype(np.uint8)
        labeled = measure.label(mask, background=0, connectivity=2)
        out = measure.regionprops_table(
            labeled, intensity_image=intensity,
            properties=["weighted_moments_central"],
        )
        m00 = out["weighted_moments_central-0-0-0"][0]
        m02 = out["weighted_moments_central-0-2-0"][0]
        return m02 / m00

    narrow = _var_x_for(width=10)
    wide   = _var_x_for(width=60)
    assert wide > narrow


def test_weighted_spread_columns_via_export(tmp_path):
    """End-to-end: requesting weighted_var_y from the public export should
    produce a numeric column populated from zarr masks + video pixels."""
    import zarr
    import cv2

    pytest.importorskip("cv2")

    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1, n=3)
    df = pd.read_csv(csv_path, skiprows=7)
    n = len(df)

    # Tiny zarr store with a centred square mask each frame.
    store_path = tmp_path / "predictions.zarr"
    store = zarr.storage.LocalStore(str(store_path))
    root = zarr.open_group(store=store, mode="w")
    H, W = 64, 64
    masks = np.zeros((n, H, W), dtype=np.int8)
    masks[:, 16:48, 16:48] = 1
    arr = root.require_array("1_masks", shape=(n, H, W), dtype=np.int8)
    arr[:] = masks

    # Tiny matching video — uniform red so the R-channel weighted variance
    # is finite and equal to the geometric variance of the mask.
    video_path = tmp_path / "vid.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5, (W, H))
    if not writer.isOpened():
        pytest.skip("cv2 VideoWriter unavailable in this environment")
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[..., 2] = 255  # BGR red
    for _ in range(n):
        writer.write(frame)
    writer.release()

    export_tracking(
        tmp_path, output_dir=tmp_path,
        region_properties=["weighted_var_y", "weighted_var_x", "weighted_cov_yx"],
        video_path=video_path, overwrite=True,
    )
    out = pd.read_csv(csv_path, skiprows=7)
    for col in ("weighted_var_y_r", "weighted_var_x_r", "weighted_cov_yx_r"):
        assert col in out.columns, f"Missing derived column: {col}"
        assert out[col].notna().all(), f"Column {col} has NaNs"
    # Symmetric mask centred in the frame → cov_yx ≈ 0
    assert abs(out["weighted_cov_yx_r"].iloc[0]) < 1e-6
    # Square mask → var_y ≈ var_x
    np.testing.assert_allclose(
        out["weighted_var_y_r"].values, out["weighted_var_x_r"].values, atol=1e-6,
    )


def test_weighted_spread_in_intensity_props_set():
    """Pseudo-props must be flagged intensity-requiring so they're skipped
    when no video is provided (rather than silently producing NaN)."""
    from octron.tools.export_tracking import _INTENSITY_PROPS, _WEIGHTED_SPREAD_PROPS
    assert _WEIGHTED_SPREAD_PROPS <= _INTENSITY_PROPS


def test_translation_variant_set_classification():
    """Defensive: shape props should NOT be in the variant set; the variant
    set must contain centroid and weighted_centroid."""
    from octron.tools.export_tracking import _TRANSLATION_VARIANT_PROPS, _INTENSITY_PROPS
    from octron.yolo_octron.constants import ALL_REGION_PROPERTIES

    assert "centroid" in _TRANSLATION_VARIANT_PROPS
    assert "weighted_centroid" in _TRANSLATION_VARIANT_PROPS

    shape_props = set(ALL_REGION_PROPERTIES["Size and Shape"])
    # None of the size/shape props should be flagged variant — they are all
    # translation-invariant (eccentricity, perimeter, moments_hu, …).
    assert shape_props.isdisjoint(_TRANSLATION_VARIANT_PROPS)

    # weighted_centroid requires intensity_image — ensure it's flagged so
    # the no-video early-skip filter excludes it correctly.
    assert "weighted_centroid" in _INTENSITY_PROPS


# ===========================================================================
# device='auto' / 'cuda' / 'mps' resolution (pyclesperanto/OpenCL backend)
# ===========================================================================

def test_resolve_device_cpu_passes_through():
    from octron.tools import export_tracking as et
    assert et._resolve_device("cpu") == "cpu"


def test_resolve_device_mps_falls_back_to_cpu():
    from octron.tools import export_tracking as et
    assert et._resolve_device("mps") == "cpu"


def test_resolve_device_auto_falls_back_to_cpu_when_no_gpu(monkeypatch):
    """auto + no pyclesperanto/no device → cpu, no errors raised."""
    from octron.tools import export_tracking as et
    monkeypatch.setattr(et, "_pyclesperanto_available", lambda: False)
    assert et._resolve_device("auto") == "cpu"


def test_resolve_device_cuda_raises_when_unavailable(monkeypatch):
    """device='cuda' is a hard request — fall-through to CPU would be silent
    correctness ambiguity, so it must raise."""
    from octron.tools import export_tracking as et
    monkeypatch.setattr(et, "_pyclesperanto_available", lambda: False)
    with pytest.raises(RuntimeError, match="pyclesperanto"):
        et._resolve_device("cuda")


def _has_pyclesperanto() -> bool:
    """Module-level (not a fixture) so it can drive @pytest.mark.skipif."""
    try:
        import pyclesperanto as _cle
        return bool(_cle.list_available_devices())
    except Exception:
        return False


@pytest.mark.skipif(not _has_pyclesperanto(),
                    reason="pyclesperanto not available")
def test_gpu_path_matches_cpu_path(tmp_path):
    """GPU and CPU paths must agree on a small synthetic mask dataset.

    Covers a mix of GPU-native (extent, equivalent_diameter_area), CPU
    fallback (eccentricity, solidity), and cross-path consistency.
    """
    import zarr
    from octron.tools.export_tracking import compute_region_props_from_zarr

    n, H, W = 3, 64, 64
    store_path = tmp_path / "predictions.zarr"
    store = zarr.storage.LocalStore(str(store_path))
    root = zarr.open_group(store=store, mode="w")
    arr = root.require_array("1_masks", shape=(n, H, W), dtype=np.int8)
    arr[:] = 0
    arr[:, 16:48, 16:48] = 1  # 32×32 square

    properties = ["eccentricity", "solidity", "extent", "equivalent_diameter_area"]

    cpu = compute_region_props_from_zarr(
        zarr_root=root, track_ids=[1],
        frame_indices_dict={1: np.arange(n)},
        properties=properties,
        zarr_path=store_path, video_path=None, device="cpu",
    )
    gpu = compute_region_props_from_zarr(
        zarr_root=root, track_ids=[1],
        frame_indices_dict={1: np.arange(n)},
        properties=properties,
        zarr_path=store_path, video_path=None, device="cuda",
    )

    assert set(cpu[1].keys()) == set(gpu[1].keys())
    for col in cpu[1]:
        cf = np.array([float(x) for x in cpu[1][col]])
        gf = np.array([float(x) for x in gpu[1][col]])
        np.testing.assert_allclose(
            gf, cf, rtol=1e-5, atol=1e-5,
            err_msg=f"GPU/CPU mismatch in {col}",
        )


def test_gpu_dispatch_does_not_use_processpool(monkeypatch):
    """When device='cuda', ProcessPoolExecutor must NOT be instantiated.

    Each spawned child process would create its own GPU context, serialise
    on the same physical device, and (on Windows + OpenCL) often fail to
    initialise at all. Lock down the contract so a future refactor can't
    silently regress.
    """
    from octron.tools import export_tracking as et
    import concurrent.futures as cf

    monkeypatch.setattr(et, "_pyclesperanto_available", lambda: True)

    def _fake_gpu_worker(args):
        _, _, batch_fi, fi_positions, *_ = args
        return ({}, batch_fi[0], batch_fi[-1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(et, "_zarr_batch_worker_opencl", _fake_gpu_worker)

    instantiations = []
    real_pool = cf.ProcessPoolExecutor

    def _spy(*args, **kwargs):
        instantiations.append((args, kwargs))
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(cf, "ProcessPoolExecutor", _spy)

    import zarr
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "p.zarr"
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.open_group(store=store, mode="w")
        H, W, n = 16, 16, 3
        arr = root.require_array("1_masks", shape=(n, H, W), dtype=np.int8)
        arr[:] = 0
        arr[:, 4:12, 4:12] = 1

        et.compute_region_props_from_zarr(
            zarr_root=root, track_ids=[1],
            frame_indices_dict={1: np.arange(n)},
            properties=["eccentricity"],
            zarr_path=store_path, video_path=None, device="cuda",
        )

    assert instantiations == [], (
        f"ProcessPoolExecutor was instantiated under device='cuda': "
        f"{instantiations}"
    )
