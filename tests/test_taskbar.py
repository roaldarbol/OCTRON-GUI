"""Tests for the Windows taskbar identity/icon helper (octron/_taskbar.py).

These guard the fix that gives OCTRON's napari window its own taskbar button
with napari's logo (branch fix/windows-taskbar-icon). They are deliberately
lightweight: the pure-ctypes helper is imported directly (no torch/napari), and
the Windows-only round-trip creates a bare QApplication rather than launching
the GUI.

What is and isn't covered
-------------------------
- COM correctness: the per-window AppUserModelID we write can be read back
  (catches a broken PROPVARIANT layout, wrong vtable slot, or bad GUID).
- The "must never break GUI startup" contract: the setter swallows all errors.
- A source-level guard that octron_gui() applies the Qt style BEFORE creating
  the window (the regression that caused the bug — setStyle after the window is
  shown wipes its taskbar identity).
- NOT covered: that the taskbar actually *renders* napari's logo. That is a
  Windows shell behaviour with no headless API to assert; it is verified
  manually.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import octron
from octron._taskbar import (get_windows_taskbar_app_id,
                             set_windows_taskbar_app_id)

WINDOWS = os.name == "nt"


class _ExplodingWindow:
    """A stand-in Qt window whose winId() raises, to prove errors are swallowed."""

    def winId(self):
        raise RuntimeError("winId boom")


def test_setter_never_raises_on_bad_window():
    # Taskbar icon is cosmetic: a broken window object must not crash startup.
    set_windows_taskbar_app_id(_ExplodingWindow(), "OCTRON.Test")


def test_setter_never_raises_with_missing_icon(tmp_path):
    missing = tmp_path / "does-not-exist.ico"
    set_windows_taskbar_app_id(_ExplodingWindow(), "OCTRON.Test", str(missing))


def test_getter_never_raises_on_bad_window():
    assert get_windows_taskbar_app_id(_ExplodingWindow()) is None


@pytest.mark.skipif(WINDOWS, reason="non-Windows no-op behaviour")
def test_noop_off_windows():
    # Off Windows the helpers return immediately without ever touching the window.
    class _NeverCalled:
        def winId(self):
            raise AssertionError("winId must not be called off Windows")

    set_windows_taskbar_app_id(_NeverCalled(), "OCTRON.Test")
    assert get_windows_taskbar_app_id(_NeverCalled()) is None


# The COM round-trip needs a real Qt application window (the shell property
# store does not work on a bare native window), but a QApplication hangs pytest
# at teardown. So we run it in an isolated subprocess that force-exits — the same
# isolation strategy tests/test_cli.py uses for fragile GUI imports.
_ROUNDTRIP_SUBPROCESS = textwrap.dedent(
    """
    import ctypes, os, sys
    from ctypes import wintypes
    from qtpy.QtWidgets import QApplication, QMainWindow
    from octron._taskbar import (set_windows_taskbar_app_id,
                                 get_windows_taskbar_app_id)

    ico = sys.argv[1]
    app = QApplication.instance() or QApplication([])  # must be kept alive
    win = QMainWindow()
    hwnd = wintypes.HWND(int(win.winId()))
    set_windows_taskbar_app_id(win, "OCTRON.Test.RoundTrip", ico)
    print("APPID:", get_windows_taskbar_app_id(win))
    WM_GETICON, ICON_BIG = 0x7F, 1
    print("ICONBIG_NONZERO:", bool(
        ctypes.windll.user32.SendMessageW(hwnd, WM_GETICON, ICON_BIG, 0)))
    sys.stdout.flush()
    os._exit(0)  # skip Qt teardown, which can hang the process
    """
)


@pytest.mark.skipif(not WINDOWS, reason="Windows-only shell property store")
def test_app_id_and_icon_roundtrip_on_windows(tmp_path):
    """The AppUserModelID we set reads back, and the .ico becomes the window icon.

    Exercises the real COM plumbing end to end: a wrong PROPVARIANT layout,
    vtable slot, or GUID would make the read-back wrong/None, and a broken
    WM_SETICON path would leave ICON_BIG null.
    """
    pytest.importorskip("qtpy")
    Image = pytest.importorskip("PIL.Image")

    ico = tmp_path / "tiny.ico"
    Image.new("RGBA", (32, 32), (255, 0, 255, 255)).save(ico, format="ICO")

    result = subprocess.run(
        [sys.executable, "-c", _ROUNDTRIP_SUBPROCESS, str(ico)],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + result.stderr
    assert "APPID: OCTRON.Test.RoundTrip" in output, output
    assert "ICONBIG_NONZERO: True" in output, output


def test_octron_gui_applies_style_before_creating_window():
    """Source-level guard for the root-cause regression.

    Changing the Qt application style after a window is shown wipes that window's
    Windows taskbar identity, so octron_gui() must call setStyle() BEFORE
    napari.Viewer(). This invariant has no runtime hook, so we assert it against
    the source rather than relying on a comment.
    """
    main_py = Path(octron.__file__).parent / "main.py"
    text = main_py.read_text(encoding="utf-8")

    body = text[text.index("def octron_gui"):]
    end = body.find("\nif __name__")
    if end != -1:
        body = body[:end]

    assert "setStyle" in body and "napari.Viewer(" in body
    assert body.index("setStyle") < body.index("napari.Viewer("), (
        "octron_gui() must apply app.setStyle() before creating the napari "
        "window, otherwise the window's Windows taskbar icon is lost."
    )
    # The window must be built hidden and given its taskbar identity before show.
    assert "napari.Viewer(show=False)" in body
    assert body.index("set_windows_taskbar_app_id") < body.index(".window.show()")
