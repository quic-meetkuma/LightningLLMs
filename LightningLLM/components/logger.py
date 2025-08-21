"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-21 22:51:08
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-21 23:15:43
# @ Description:
"""

"""
Custom logger module for the training system.
Supports logging to console and file, with DDP rank-0 logging capabilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from LightningLLM.utils.dist_utils import get_rank


class Logger:
    """Custom logger with console and file logging capabilities."""

    def __init__(
        self,
        name: str = "training_logger",
        log_file: Optional[str] = None,
        level: int = logging.INFO,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_file: Path to log file (if None, log only to console)
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if log_file is provided)
        if log_file:
            # Create directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

    def log_rank_zero(self, message: str, level: int = logging.INFO) -> None:
        """
        Log message only on rank 0 process.

        Args:
            message: Message to log
            level: Logging level
        """
        if get_rank() == 0:
            self.logger.log(level, message)

    def log_exception(
        self, message: str, exception: Exception, raise_exception: bool = True
    ) -> None:
        """
        Log exception message and optionally raise the exception.

        Args:
            message: Custom message to log
            exception: Exception to log
            raise_exception: Whether to raise the exception after logging
        """
        error_message = f"{message}: {str(exception)}"
        self.logger.error(error_message)

        if raise_exception:
            raise exception

    def prepare_for_logs(
        self, output_dir: str, save_metrics: bool = True, log_level: str = "INFO"
    ) -> None:
        """
        Prepare logger for training logs.

        Args:
            output_dir: Output directory for logs
            save_metrics: Whether to save metrics to file
            log_level: Logging level as string
        """
        # Convert string log level to logging constant
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Add file handler if saving metrics
        if save_metrics:
            log_file = Path(output_dir) / "training.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Check if file handler already exists
            file_handler_exists = any(
                isinstance(handler, logging.FileHandler)
                for handler in self.logger.handlers
            )

            if not file_handler_exists:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)


# Global logger instance
_logger: Optional[Logger] = None


def get_logger(name: str = "training_logger", log_file: Optional[str] = None) -> Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name
        log_file: Path to log file (if None, log only to console)

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = Logger(name, log_file)
    return _logger


def log_rank_zero(message: str, level: int = logging.INFO) -> None:
    """
    Log message only on rank 0 process.

    Args:
        message: Message to log
        level: Logging level
    """
    logger = get_logger()
    logger.log_rank_zero(message, level)


def log_exception(
    message: str, exception: Exception, raise_exception: bool = True
) -> None:
    """
    Log exception message and optionally raise the exception.

    Args:
        message: Custom message to log
        exception: Exception to log
        raise_exception: Whether to raise the exception after logging
    """
    logger = get_logger()
    logger.log_exception(message, exception, raise_exception)
