"""
Central logging configuration for OCTRON.

All modules import `from loguru import logger` and use it directly.
Call `setup_logging()` once at startup (from the GUI entry point or CLI callback).
The CLI branch can pass `debug=True` to enable DEBUG-level output.
"""

import sys
import logging
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


# Substrings found in known-noisy third-party WARNING messages that should be
# hidden in normal (non-debug) operation.
_SUPPRESSED_WARNING_SUBSTRINGS = (
    "json_encoders",        # pydantic v2 deprecation at napari import time
    "argument 'device'",    # torch pin_memory() / is_pinned() — PyTorch 2.10 bug
    "torch.jit.trace",      # torch.jit.trace / trace_method DeprecationWarning
    "trace to be incorrect",  # TracerWarning fired during AMP initialisation
    "already a ScriptModule",  # torch.jit UserWarning during AMP initialisation
)


def _make_loguru_filter(debug: bool):
    """Return a loguru filter callable for the given verbosity mode.

    In non-debug mode the filter:
    - Passes any record from an ``octron.*`` module at INFO and above.
    - Passes non-octron WARNING records unless they match a known-noisy pattern
      (pydantic, torch pin_memory, torch.jit TracerWarning, …).
    - Blocks everything else — in particular ultralytics INFO output such as
      the model architecture table, training-config dump, TensorBoard URL, and
      dataset scan progress bars.

    In debug mode ``None`` is returned (loguru skips filtering entirely).
    """
    if debug:
        return None

    substrings = _SUPPRESSED_WARNING_SUBSTRINGS

    def _filter(record) -> bool:
        level_no = record["level"].no
        # Always pass octron messages at INFO and above
        if record["name"].startswith("octron") or record["name"] == "__main__":
            return level_no >= 20  # INFO = 20
        # Non-octron WARNING: pass unless it matches a suppressed pattern
        if level_no >= 30:  # WARNING = 30
            msg = record["message"]
            return not any(s in msg for s in substrings)
        # Non-octron INFO and below: suppress
        # (covers ultralytics architecture table, config dump, TensorBoard URL, …)
        return False

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

    if debug:
        logger.debug("Debug logging enabled")


def print_welcome() -> None:
    """Print a short welcome banner to stderr via loguru."""
    logger.info("=" * 48)
    logger.info(f"  OCTRON  v{octron_version}")
    logger.info("  Segmentation & tracking for animal behavior")
    logger.info("=" * 48)
