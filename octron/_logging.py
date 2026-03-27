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
        In non-debug mode, known noisy third-party warnings (pydantic,
        torch pin_memory, torch.jit) are suppressed.

    This function is idempotent — calling it multiple times simply resets the
    handler to the current settings.
    """
    logger.remove()  # drop default stderr handler

    level = "DEBUG" if debug else "INFO"
    fmt = _LOG_FORMAT_DEBUG if debug else _LOG_FORMAT_NORMAL

    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        colorize=True,
        enqueue=False,
    )

    # Route Python warnings and stdlib logging through loguru so all output
    # passes through a single, consistently-formatted channel.
    logging.captureWarnings(True)
    logging.root.handlers = [_InterceptHandler()]
    logging.root.setLevel(logging.DEBUG if debug else logging.WARNING)

    if not debug:
        # Suppress known noisy third-party DeprecationWarnings that are not
        # actionable by OCTRON users:
        #   • pydantic v1 json_encoders: fired at import time, not our code
        #   • torch pin_memory: PyTorch 2.10 bug, fires every DataLoader batch
        #   • torch.jit.trace: AMP initialisation noise at training startup
        warnings.filterwarnings("ignore", message=".*json_encoders.*", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*pin_memory.*", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.jit")
        warnings.filterwarnings("ignore", message=".*torch\.jit\.trace.*", category=DeprecationWarning)
    else:
        # In debug mode, restore default warning behaviour so everything shows.
        warnings.resetwarnings()
        logger.debug("Debug logging enabled")


def print_welcome() -> None:
    """Print a short welcome banner to stderr via loguru."""
    logger.info("=" * 48)
    logger.info(f"  OCTRON  v{octron_version}")
    logger.info("  Segmentation & tracking for animal behavior")
    logger.info("=" * 48)
