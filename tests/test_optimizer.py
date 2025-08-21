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

import pytest
import torch
import torch.nn as nn

from LightningLLM.components.component_registry import ComponentFactory

# Setup test data
OPTIMIZER_CONFIGS = {
    "adam": {
        "name": "adam",
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    },
    "adamw": {
        "name": "adamw",
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
    },
    "sgd": {
        "name": "sgd",
        "lr": 1e-4,
        "momentum": 0.9,
        "weight_decay": 0.01,
        "dampening": 0.0,
        "nesterov": False,
    },
}


@pytest.fixture
def dummy_model():
    return nn.Linear(10, 1)


@pytest.mark.parametrize("opt_name", OPTIMIZER_CONFIGS.keys())
def test_optimizers(opt_name, dummy_model):
    """Test that all optimizers can be created with their configs."""
    # Create optimizer using the factory
    config = OPTIMIZER_CONFIGS[opt_name]
    opt_inst = ComponentFactory.create_optimizer(
        **config, model_params=dummy_model.parameters()
    )

    assert opt_inst is not None
    assert isinstance(opt_inst.optimizer, torch.optim.Optimizer)
    assert len(list(opt_inst.optimizer.param_groups)) == 1
