"""
Custom log handlers for the application.
This module is separated to avoid circular imports when using logging.ini.
"""

from rich.console import Console
from rich.logging import RichHandler


class RichConsoleHandler(RichHandler):
    """
    Custom Rich handler with configurable console output.

    This class is designed to be used both programmatically and via logging.ini.

    Args:
        width (int): Console width for output (default: 200)
        style (str): Console style/color (e.g., "white", "magenta", "yellow")
        **kwargs: Additional arguments passed to RichHandler
    """

    def __init__(self, width=200, style=None, **kwargs):
        # Force disable time display
        kwargs.setdefault("show_time", False)

        super().__init__(
            console=Console(color_system="256", width=width, style=style),
            **kwargs,
        )
