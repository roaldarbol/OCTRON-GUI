"""Windows taskbar identity/icon helpers for OCTRON's main GUI window.

Pure ``ctypes`` / Win32 — no Qt, torch, or napari imports — so this module is
cheap to import and unit-testable (see ``tests/test_taskbar.py``). Everything is
a no-op on non-Windows platforms.

Why this exists
---------------
napari's OpenGL main window does not reliably surface its Qt window icon to the
Windows taskbar, and it shares the process-wide AppUserModelID with Qt/vispy's
hidden helper windows, so Windows collapses them into a single taskbar group
with a generic icon. To get napari's logo on its own taskbar button we, on the
window's shell property store, give it a per-window ``AppUserModelID`` (its own
taskbar group) plus a ``RelaunchIconResource`` pointing at a real ``.ico``, and
additionally force the native window icon via ``WM_SETICON``. All three are
needed together — dropping any one falls back to a generic icon.
"""

import ctypes
import os
from ctypes import (POINTER, Structure, byref, c_byte, c_int, c_ulong, c_ushort,
                    c_void_p, c_wchar_p, wintypes)

# ctypes Structures are safe to define on any platform; only ``ctypes.windll``
# and ``ctypes.WINFUNCTYPE`` are Windows-only, and those are referenced solely
# inside the function bodies below (guarded by an ``os.name == "nt"`` check).


class _GUID(Structure):
    _fields_ = [("Data1", c_ulong), ("Data2", c_ushort),
                ("Data3", c_ushort), ("Data4", c_byte * 8)]


class _PROPERTYKEY(Structure):
    _fields_ = [("fmtid", _GUID), ("pid", wintypes.DWORD)]


class _PROPVARIANT(Structure):
    # x64 layout: 8-byte header then the value union (we use the LPWSTR pointer).
    _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort),
                ("r3", c_ushort), ("p", c_void_p), ("pad", c_byte * 8)]


_VT_LPWSTR = 31
_IID_IPropertyStore = "{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}"
_FMTID_AppUserModel = "{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}"
_PID_APP_ID = 5          # PKEY_AppUserModel_ID
_PID_RELAUNCH_ICON = 3   # PKEY_AppUserModel_RelaunchIconResource

# IPropertyStore vtable slots.
_SLOT_RELEASE = 2
_SLOT_GETVALUE = 5
_SLOT_SETVALUE = 6
_SLOT_COMMIT = 7


def _guid(s):
    g = _GUID()
    if ctypes.windll.ole32.IIDFromString(c_wchar_p(s), byref(g)) != 0:
        raise OSError(f"IIDFromString({s}) failed")
    return g


def _vtable(store):
    return ctypes.cast(ctypes.cast(store, POINTER(c_void_p))[0], POINTER(c_void_p))


def _method(store, slot, restype, *argtypes):
    return ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(_vtable(store)[slot])


def _open_property_store(hwnd):
    store = c_void_p()
    iid = _guid(_IID_IPropertyStore)
    hr = ctypes.windll.shell32.SHGetPropertyStoreForWindow(
        wintypes.HWND(hwnd), byref(iid), byref(store))
    return store if hr == 0 else None


def set_windows_taskbar_app_id(qt_window, app_id, icon_ico_path=None):
    """Give a Qt window its own Windows taskbar identity and icon.

    Sets a per-window ``AppUserModelID`` (own taskbar group), and when
    ``icon_ico_path`` is given a ``RelaunchIconResource`` plus a forced native
    icon (``WM_SETICON``) loaded from that ``.ico``.

    No-op on non-Windows. Every failure (missing API, COM error, a window with
    no native handle) is swallowed: the taskbar icon is cosmetic and must never
    break GUI startup.
    """
    if os.name != "nt":
        return
    try:
        hwnd = int(qt_window.winId())
        ole32 = ctypes.windll.ole32
        ole32.CoTaskMemAlloc.restype = c_void_p
        fmtid = _guid(_FMTID_AppUserModel)

        store = _open_property_store(hwnd)
        if store is not None:
            try:
                set_value = _method(store, _SLOT_SETVALUE, c_int,
                                    POINTER(_PROPERTYKEY), POINTER(_PROPVARIANT))
                commit = _method(store, _SLOT_COMMIT, c_int)

                def _set_str(pid, value):
                    nbytes = (len(value) + 1) * 2
                    mem = ole32.CoTaskMemAlloc(nbytes)
                    ctypes.memmove(mem, ctypes.create_unicode_buffer(value), nbytes)
                    pv = _PROPVARIANT()
                    pv.vt = _VT_LPWSTR
                    pv.p = mem
                    set_value(store, byref(_PROPERTYKEY(fmtid, pid)), byref(pv))
                    ole32.PropVariantClear(byref(pv))

                _set_str(_PID_APP_ID, app_id)
                if icon_ico_path:
                    _set_str(_PID_RELAUNCH_ICON, f"{icon_ico_path},0")
                commit(store)
            finally:
                _method(store, _SLOT_RELEASE, c_int)(store)

        # Force the native window icon from the .ico — the napari window does not
        # surface its Qt icon to the taskbar reliably on its own.
        if icon_ico_path and os.path.exists(icon_ico_path):
            user32 = ctypes.windll.user32
            user32.LoadImageW.restype = c_void_p
            image_icon, lr_loadfromfile = 1, 0x10
            wm_seticon, icon_small, icon_big = 0x80, 0, 1
            for size, which in ((32, icon_big), (16, icon_small)):
                hicon = user32.LoadImageW(None, c_wchar_p(icon_ico_path),
                                          image_icon, size, size, lr_loadfromfile)
                if hicon:
                    user32.SendMessageW(wintypes.HWND(hwnd), wm_seticon,
                                        which, c_void_p(hicon))
    except Exception:
        pass


def get_windows_taskbar_app_id(qt_window):
    """Return the per-window ``AppUserModelID`` set on a Qt window, or ``None``.

    Companion to :func:`set_windows_taskbar_app_id` used to verify it in tests.
    Returns ``None`` on non-Windows, on failure, or when no per-window id is set.
    """
    if os.name != "nt":
        return None
    try:
        hwnd = int(qt_window.winId())
        store = _open_property_store(hwnd)
        if store is None:
            return None
        try:
            get_value = _method(store, _SLOT_GETVALUE, c_int,
                                POINTER(_PROPERTYKEY), POINTER(_PROPVARIANT))
            key = _PROPERTYKEY(_guid(_FMTID_AppUserModel), _PID_APP_ID)
            pv = _PROPVARIANT()
            if get_value(store, byref(key), byref(pv)) != 0:
                return None
            value = ctypes.wstring_at(pv.p) if pv.p else None
            ctypes.windll.ole32.PropVariantClear(byref(pv))
            return value
        finally:
            _method(store, _SLOT_RELEASE, c_int)(store)
    except Exception:
        return None
