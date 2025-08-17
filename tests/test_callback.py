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

# This below import will register all the components in the registry
import components  # noqa: E402

from lightning.pytorch.callbacks import Callback
import pytest
import torch.nn as nn
from components.component_registry import ComponentFactory

# Setup test data
CALLBACK_CONFIGS = {
    "early_stopping": {
        "name": "early_stopping",
        "monitor": "val_loss",
        "patience": 3,
        "stopping_threshold": 0.001,
        "log_rank_zero_only": True,
    },
    "model_summary": {
        "name": "model_summary",
        "max_depth": 1,
    },
    "model_checkpoint": {
        "name": "model_checkpoint",
        "dirpath": "tmp",
        "filename": "abc.model",
        "monitor": "test_loss",
        "save_top_k": 3,
        "mode": "min",
        "every_n_epochs": 1,
        "enable_version_counter": True,
    },
}


@pytest.mark.parametrize("callback_name", CALLBACK_CONFIGS.keys())
def test_callbacks(callback_name):
    """Test that all callbacks that can be created with their configs."""
    # Create callbacks using the factory
    config = CALLBACK_CONFIGS[callback_name]
    callback_inst = ComponentFactory.create_callback(**config)
    assert callback_inst is not None
    assert callback_inst.callback is not None
