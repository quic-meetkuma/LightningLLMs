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

import pytest
import torch
import torch.nn as nn

from LightningLLM.components.component_registry import ComponentFactory

# This below import will register all the components in the registry


# Setup test data
SCHEDULER_CONFIGS = {
    "constant_with_warmup": {
        "name": "constant_with_warmup",
        "num_warmup_steps": 10,
        "last_epoch": -1,
    },
    "linear_with_warmup": {
        "name": "linear_with_warmup",
        "num_warmup_steps": 10,
        "num_training_steps": 100,
        "last_epoch": -1,
    },
    "cosine_with_warmup": {
        "name": "cosine_with_warmup",
        "num_warmup_steps": 10,
        "num_training_steps": 100,
        "num_cycles": 0.5,
        "last_epoch": -1,
    },
}


@pytest.fixture
def dummy_optimizer():
    model = nn.Linear(10, 1)
    return torch.optim.Adam(model.parameters(), lr=1e-4)


@pytest.mark.parametrize("scheduler_name", SCHEDULER_CONFIGS.keys())
def test_schedulers(scheduler_name, dummy_optimizer):
    """Test that all schedulers that can be created with their configs."""
    # Create scheduler using the factory
    config = SCHEDULER_CONFIGS[scheduler_name]
    scheduler_inst = ComponentFactory.create_scheduler(
        **config, optimizer=dummy_optimizer
    )

    assert scheduler_inst is not None
    assert scheduler_inst.scheduler is not None
