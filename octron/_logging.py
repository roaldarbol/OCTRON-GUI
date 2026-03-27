"""
Central logging configuration for OCTRON.

All modules import `from loguru import logger` and use it directly.
Call `setup_logging()` once at startup (from the GUI entry point or CLI callback).
The CLI branch can pass `debug=True` to enable DEBUG-level output.
"""

import sys
import logging
import warnings
from loguru import logger

from octron._version import version as octron_version


class _InterceptHandler(logging.Handler):
    """Route standard-library logging records through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Map stdlib level → loguru level name
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk the call stack to find the frame that called logger.warning() etc.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# Substrings found in known-noisy third-party WARNING messages.
# These are suppressed in BOTH normal and debug mode: they are per-frame or
# import-time side effects from third-party bugs and are never actionable.
_SUPPRESSED_WARNING_SUBSTRINGS = (
    "json_encoders",              # pydantic v2 deprecation at napari import time
    "argument 'device'",          # torch pin_memory() / is_pinned() — PyTorch 2.10 bug
    "torch.jit.trace",            # torch.jit.trace / trace_method DeprecationWarning
    "trace to be incorrect",      # TracerWarning fired during AMP initialisation
    "already a ScriptModule",     # torch.jit UserWarning during AMP initialisation
    "'rect=True' is incompatible",  # ultralytics dataloader shuffle warning (expected)
)

# Python logger name prefixes whose DEBUG/INFO records are always suppressed,
# even in --debug mode.  These libraries produce high-volume messages that are
# unrelated to octron behaviour and break in-place progress-bar display.
_NOISY_LIBRARY_PREFIXES = (
    "matplotlib",
    "PIL",
)

# Suppress the pydantic json_encoders DeprecationWarning at import time so it
# is filtered before any loguru handler (including any installed by ultralytics)
# can print it.  This fires when napari (imported by yolo_octron.py) loads
# pydantic — before we have a chance to evict ultralytics' loguru handler.
warnings.filterwarnings("ignore", message=r".*json_encoders.*")


def _make_loguru_filter(debug: bool):
    """Return a loguru filter callable for the given verbosity mode.

    Both modes suppress:
    - Known per-frame / import-time WARNING noise (pin_memory, TracerWarning,
      pydantic json_encoders, …) — these fire thousands of times and are not
      actionable; hiding them even in debug keeps progress bars readable.
    - DEBUG/INFO from high-volume library loggers (matplotlib, PIL) — unrelated
      to octron and would break ``\\r\\033[K`` in-place progress-bar display.

    Non-debug mode additionally:
    - Only passes octron.* records at INFO and above.
    - Blocks all other INFO/DEBUG — suppresses the ultralytics architecture
      table, training-config dump, TensorBoard URL, scan progress bars, etc.
    """
    substrings = _SUPPRESSED_WARNING_SUBSTRINGS
    noisy_prefixes = _NOISY_LIBRARY_PREFIXES

    def _filter(record) -> bool:
        level_no = record["level"].no
        name = record["name"]

        # Always suppress per-frame WARNING noise (both normal and debug mode).
        if level_no == 30:  # WARNING
            if any(s in record["message"] for s in substrings):
                return False

        # Always suppress high-volume library debug spam that breaks progress bars.
        if level_no < 30 and any(name.startswith(p) for p in noisy_prefixes):
            return False

        if debug:
            # Show everything else in debug mode.
            return True

        # Non-debug: pass octron messages at INFO and above.
        if name.startswith("octron") or name == "__main__":
            return level_no >= 20  # INFO = 20
        # Non-octron WARNING+ that wasn't suppressed above.
        return level_no >= 30

    return _filter


_LOG_FORMAT_NORMAL = (
    "<cyan>{time:HH:mm:ss}</cyan> | "
    "<level>{level: <8}</level> | "
    "{message}"
)

_LOG_FORMAT_DEBUG = (
    "<cyan>{time:HH:mm:ss.SSS}</cyan> | "
    "<level>{level: <8}</level> | "
    "<dim>{name}:{line}</dim> | "
    "{message}"
)


def setup_logging(debug: bool = False) -> None:
    """Configure loguru for the OCTRON application.

    Parameters
    ----------
    debug:
        When True, set log level to DEBUG and include source location in every
        message.  When False (default), use INFO level with a compact format.

        In non-debug mode a loguru filter is applied that:
        - Passes octron.* records at INFO and above (our own progress messages).
        - Passes non-octron WARNING records that are not known third-party noise
          (pydantic, torch pin_memory, torch.jit TracerWarning, …).
        - Blocks non-octron INFO, suppressing the ultralytics architecture table,
          training-config parameter dump, TensorBoard URL, and scan bars.

    This function is idempotent — calling it multiple times simply resets the
    handler to the current settings.  Call it again after ultralytics has been
    imported to evict any loguru handler ultralytics may have installed.
    """
    logger.remove()  # remove ALL handlers, including any installed by ultralytics

    level = "DEBUG" if debug else "INFO"
    fmt = _LOG_FORMAT_DEBUG if debug else _LOG_FORMAT_NORMAL

    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        colorize=True,
        filter=_make_loguru_filter(debug),
        enqueue=False,
    )

    # Route Python warnings and stdlib logging through loguru so all output
    # passes through a single, consistently-formatted channel.
    logging.captureWarnings(True)
    logging.root.handlers = [_InterceptHandler()]
    # Non-debug: only forward WARNING+ from stdlib logging to avoid paying the
    # cost of filtering high-volume ultralytics INFO records in _InterceptHandler.
    logging.root.setLevel(logging.DEBUG if debug else logging.WARNING)

    # Tame the ultralytics Python logger.  ultralytics installs a StreamHandler
    # directly on logging.getLogger('ultralytics') that prints architecture
    # tables, config dumps, TensorBoard URLs, etc. straight to stdout — bypassing
    # loguru entirely.  We remove those handlers and set the level so that:
    #   • non-debug: only WARNING+ propagates (via root) to our InterceptHandler
    #   • debug: everything propagates
    # tqdm epoch-progress bars are written directly to stderr by tqdm and are
    # unaffected by this change.
    try:
        _ult_logger = logging.getLogger("ultralytics")
        for _h in _ult_logger.handlers[:]:
            _ult_logger.removeHandler(_h)
        _ult_logger.setLevel(logging.DEBUG if debug else logging.WARNING)
        _ult_logger.propagate = True
    except Exception:
        pass  # ultralytics not yet imported — nothing to tame

    # Re-apply the pydantic filter in case it was cleared by a third party.
    if not debug:
        warnings.filterwarnings("ignore", message=r".*json_encoders.*")

    if debug:
        logger.debug("Debug logging enabled")


def print_welcome() -> None:
    """Print a short welcome banner to stderr via loguru."""
    logger.info("=" * 48)
    logger.info(f"  OCTRON  v{octron_version}")
    logger.info("  Segmentation & tracking for animal behavior")
    logger.info("=" * 48)
