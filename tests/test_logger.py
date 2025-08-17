"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-16 12:38:56
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 12:38:58
# @ Description:
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# This below import will register all the components in the registry
import components  # noqa: E402

import pytest
import torch.nn as nn
from components.component_registry import ComponentFactory
from components.logger import get_logger
import logging


def test_logger():
    """Test the creation of logger."""
    # Create logger

    logger = get_logger("test_logger")
    logger.prepare_for_logs("./tmp/")
    logger.info("This is a test log message.")
    logger.log_rank_zero("This is a test log message.", logging.WARNING)
