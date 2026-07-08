"""
Shared filesystem locations for OCTRON.

Large model weights (YOLO detection/segmentation models and SAM2/SAM3
checkpoints) are cached in a common, environment-independent directory so
that installing OCTRON into a fresh virtual environment — or reinstalling —
does not trigger a re-download of the (multi-hundred-MB) weights.

Previously the weights were stored *inside* the installed package directory
(e.g. ``.../site-packages/octron/yolo_octron/models``), which meant every new
environment fetched and stored its own copy. They now live in a per-user
cache directory shared across environments.

The location can be overridden with the ``OCTRON_CACHE_DIR`` environment
variable — useful for shared/lab machines that keep the weights on a common
drive, or for pointing several installs at the same folder.
"""
import os
import shutil
from pathlib import Path

import platformdirs
from loguru import logger


def get_cache_dir() -> Path:
    """
    Return the root OCTRON cache directory, creating it if needed.

    Honours the ``OCTRON_CACHE_DIR`` environment variable; otherwise falls
    back to the platform-specific per-user cache location (e.g.
    ``%LOCALAPPDATA%\\octron\\Cache`` on Windows, ``~/.cache/octron`` on
    Linux, ``~/Library/Caches/octron`` on macOS).
    """
    override = os.environ.get("OCTRON_CACHE_DIR")
    if override:
        root = Path(override).expanduser()
    else:
        root = Path(platformdirs.user_cache_dir("octron", appauthor=False))
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_yolo_cache_dir() -> Path:
    """Directory holding the shared YOLO weights (``*.pt``)."""
    d = get_cache_dir() / "models" / "yolo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_sam_cache_dir() -> Path:
    """Directory holding the shared SAM2/SAM3 checkpoints."""
    d = get_cache_dir() / "models" / "sam"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reuse_legacy_weight(legacy_path, cache_path) -> bool:
    """
    Reuse a weight file previously downloaded into the package directory.

    Older OCTRON versions stored weights inside the installed package. When a
    user upgrades, copy any such file into the shared cache instead of
    re-downloading it. The legacy copy is left untouched (the package
    directory may be read-only, and leaving it does no harm).

    Parameters
    ----------
    legacy_path : str or Path
        Old package-local location of the weight file.
    cache_path : str or Path
        Destination in the shared cache.

    Returns
    -------
    bool
        True if ``cache_path`` exists after the call (already present, or
        successfully copied from the legacy location), False otherwise.
    """
    legacy_path = Path(legacy_path)
    cache_path = Path(cache_path)
    if cache_path.exists():
        return True
    try:
        if legacy_path.exists() and legacy_path.is_file():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, cache_path)
            logger.info(f"Reused existing weight {legacy_path.name} from {legacy_path.parent} -> {cache_path}")
            return True
    except Exception as e:
        logger.debug(f"Could not reuse legacy weight {legacy_path}: {e}")
    return False
