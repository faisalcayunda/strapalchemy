"""
Logging module for StrapAlchemy.

This module provides:
- Thread-safe singleton logger with Rich console output
- INI-based configuration support
- Custom handlers with enhanced formatting

Usage:
    from strapalchemy.logging import logger

    logger.info("Application started")
    logger.error("An error occurred")

For service-specific loggers:
    from strapalchemy.logging import get_logger

    sql_logger = get_logger('sqlalchemy.engine.Engine')
    sql_logger.info("SELECT * FROM users")
"""

from strapalchemy.logging.handlers import RichConsoleHandler
from strapalchemy.logging.logger import AppLogger, SingletonMeta, get_logger, logger, setup_logging_from_ini

__all__ = [
    "logger",
    "get_logger",
    "setup_logging_from_ini",
    "AppLogger",
    "SingletonMeta",
    "RichConsoleHandler",
]
