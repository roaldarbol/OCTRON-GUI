"""
CLI smoke tests — verifies every subcommand and flag via --help.

No heavy dependencies (torch, cv2, ultralytics) are imported; all tests run
quickly in CI with no GPU and no model files required.

Covered subcommands and flags
------------------------------
gui         --help
gpu-test    --help; gpu-test runs (skipped if torch DLLs unavailable)
split       --help: --mode, --train, --val, --seed, --dry-run
train       --help: --model, --mode, --device, --epochs, --imagesz,
                    --save-period, --overwrite, --resume, --no-split,
                    --train, --val, --seed
predict     --help: --model, --tracker, --tracker-config, --device,
                    --conf-thresh, --iou-thresh, --skip-frames,
                    --one-object-per-label, --opening-radius, --overwrite,
                    --detailed, --buffer-size, --output-dir, --local-cache
render      --help: --video, --output, --preset, --start, --end, --alpha,
                    --masks/--no-masks, --boxes/--no-boxes,
                    --labels/--no-labels, --tracklets, --tracklet-overlay,
                    --tracklet-size, --tracklet-mask-centroids,
                    --tracklet-smooth-cutoff, --tracklet-smooth-order,
                    --tracklet-interpolate, --track-ids, --min-observations,
                    --bbox-sizes
transcode   --help: --output, --crf, --overwrite

auto_device returns 'cuda', 'mps', or 'cpu' (skipped if torch unavailable)
"""

import re
import pytest
from typer.testing import CliRunner
from octron.cli import app

# Wide terminal so Rich/typer never truncates long flag names in help output
runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1"})

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mK]')


def _clean(text: str) -> str:
    """Strip ANSI escape codes so flag-name assertions work regardless of
    terminal colour settings on CI (Rich wraps dashes and names separately,
    producing e.g. ESC[2m--ESC[0mESC[36mmode that breaks plain-string search)."""
    return _ANSI_RE.sub('', text)


# ---------------------------------------------------------------------------
# Root help
# ---------------------------------------------------------------------------

def test_root_help():
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'OCTRON' in result.output


# ---------------------------------------------------------------------------
# gui
# ---------------------------------------------------------------------------

def test_gui_help():
    result = runner.invoke(app, ['gui', '--help'])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# gpu-test
# ---------------------------------------------------------------------------

def test_gpu_test_help():
    result = runner.invoke(app, ['gpu-test', '--help'])
    assert result.exit_code == 0


def test_gpu_test_runs():
    try:
        import torch  # noqa: F401
    except OSError:
        pytest.skip("torch DLLs could not be loaded in this environment")
    result = runner.invoke(app, ['gpu-test'])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def test_split_help():
    result = runner.invoke(app, ['split', '--help'])
    assert result.exit_code == 0
    out = _clean(result.output)
    assert '--mode' in out
    assert '--train' in out
    assert '--val' in out
    assert '--seed' in out
    assert '--dry-run' in out


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def test_train_help():
    result = runner.invoke(app, ['train', '--help'])
    assert result.exit_code == 0
    out = _clean(result.output)
    assert '--model' in out
    assert '--mode' in out
    assert '--device' in out
    assert '--epochs' in out
    assert '--imagesz' in out
    assert '--save-period' in out
    assert '--overwrite' in out
    assert '--resume' in out
    assert '--no-split' in out
    assert '--train' in out
    assert '--val' in out
    assert '--seed' in out


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_help():
    result = runner.invoke(app, ['predict', '--help'])
    assert result.exit_code == 0
    out = _clean(result.output)
    assert '--model' in out
    assert '--tracker' in out
    assert '--tracker-config' in out
    assert '--device' in out
    assert '--conf-thresh' in out
    assert '--iou-thresh' in out
    assert '--skip-frames' in out
    assert '--one-object-per-label' in out
    assert '--opening-radius' in out
    assert '--overwrite' in out
    assert '--detailed' in out
    assert '--buffer-size' in out
    assert '--output-dir' in out
    assert '--local-cache' in out


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def test_render_help():
    result = runner.invoke(app, ['render', '--help'])
    assert result.exit_code == 0
    out = _clean(result.output)
    assert '--video' in out
    assert '--output' in out
    assert '--preset' in out
    assert '--start' in out
    assert '--end' in out
    assert '--alpha' in out
    assert '--masks' in out
    assert '--no-masks' in out
    assert '--boxes' in out
    assert '--no-boxes' in out
    assert '--labels' in out
    assert '--no-labels' in out
    assert '--tracklets' in out
    assert '--tracklet-size' in out
    assert '--tracklet-mask-centroids' in out
    assert '--tracklet-smooth-cutoff' in out
    assert '--tracklet-smooth-order' in out
    assert '--tracklet-interpolate' in out
    assert '--track-ids' in out
    assert '--min-observations' in out
    assert '--bbox-sizes' in out
    assert '--debug' in out


# ---------------------------------------------------------------------------
# transcode
# ---------------------------------------------------------------------------

def test_transcode_help():
    result = runner.invoke(app, ['transcode', '--help'])
    assert result.exit_code == 0
    out = _clean(result.output)
    assert '--output' in out
    assert '--crf' in out
    assert '--overwrite' in out


# ---------------------------------------------------------------------------
# auto_device
# ---------------------------------------------------------------------------

def test_auto_device_returns_valid():
    try:
        import torch  # noqa: F401
    except OSError:
        pytest.skip("torch DLLs could not be loaded in this environment")
    from octron.test_gpu import auto_device
    device = auto_device()
    assert device in ('cuda', 'mps', 'cpu')
