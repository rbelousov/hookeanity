"""Logging configuration.

Attributes:
    logger (loguru.logger): Configured `loguru` logger.
"""

import sys

from loguru import logger as _logger

_logger.remove()
_logger.add(
    sys.stdout,
    level="INFO",
    filter=lambda r: r["level"].name == "INFO",
    format="INFO {time:YYYY-MM-DD HH:mm:ss} | {message}",
)
logger = _logger
