"""
Logger module with thread-safe Singleton pattern and Rich console output.
Supports both INI-based configuration and programmatic setup.
"""

import logging
import logging.config
from pathlib import Path
from threading import Lock

# Import handler from separate module to avoid circular imports
from strapalchemy.logging.handlers import RichConsoleHandler


class SingletonMeta(type):
    """
    Thread-safe implementation of Singleton pattern.
    Ensures only one instance of the logger exists across the application.
    """

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


def setup_logging_from_ini(ini_path: str | Path = "logging.ini") -> bool:
    """
    Setup logging from INI configuration file.

    Args:
        ini_path: Path to logging.ini file (relative to project root or absolute)

    Returns:
        bool: True if successfully loaded, False otherwise
    """
    # Convert to Path object
    config_path = Path(ini_path)

    # If relative path, try from project root
    if not config_path.is_absolute():
        # Get project root (4 levels up from this file: app/platform/logging/logger.py -> ../../../)
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / config_path

    if config_path.exists():
        try:
            logging.config.fileConfig(config_path, disable_existing_loggers=False)
            return True
        except Exception as e:
            print(f"Warning: Failed to load logging config from {config_path}: {e}")
            return False
    return False


class AppLogger(metaclass=SingletonMeta):
    """
    Application logger with Singleton pattern.

    NOTE: Does NOT auto-load logging.ini to avoid circular imports.
    Call setup_logging_from_ini() explicitly before first use.
    """

    _logger = None
    _configured_from_ini = False

    def __init__(self):
        # Check if logging.ini was configured externally
        root_logger = logging.getLogger()
        self._configured_from_ini = len(root_logger.handlers) > 0

        if self._configured_from_ini:
            # Get root logger (configured by INI)
            self._logger = root_logger
        else:
            # Fallback to programmatic configuration
            self._logger = logging.getLogger("portal-data")
            self._logger.setLevel(logging.INFO)

            # Remove existing handlers to avoid duplicates
            self._logger.handlers.clear()

            # Add Rich console handler
            self._logger.addHandler(
                RichConsoleHandler(
                    markup=True,
                    rich_tracebacks=True,
                    tracebacks_show_locals=True,
                )
            )

    def get_logger(self):
        """Returns the singleton logger instance."""
        return self._logger

    def is_configured_from_ini(self) -> bool:
        """Check if logger was configured from INI file."""
        return self._configured_from_ini


# Global logger instance - import this in your modules
logger = AppLogger().get_logger()


# Export helper function to get specific loggers
def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance by name.

    Args:
        name: Logger name (e.g., "uvicorn.access", "sqlalchemy.engine")
              If None, returns the root logger

    Returns:
        logging.Logger: Logger instance
    """
    if name:
        return logging.getLogger(name)
    return logger
