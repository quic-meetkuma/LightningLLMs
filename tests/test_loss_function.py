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

from components.component_registry import ComponentFactory

# Setup test data
LOSS_FN_CONFIGS = {
    "cross_entropy": {
        "name": "cross_entropy",
        "ignore_index": -100,
        "reduction": "mean",
        "weight": 1.0,
    },
    "kl_div": {
        "name": "kl_div",
        "temperature": 4.0,
        "reduction": "batchmean",
        "weight": 1.0,
    },
}


@pytest.mark.parametrize("loss_fn_name", LOSS_FN_CONFIGS.keys())
def test_loss_functions(loss_fn_name):
    """Test that all loss functions that can be created with their configs."""
    # Create loss functions using the factory
    config = LOSS_FN_CONFIGS[loss_fn_name]
    loss_fn = ComponentFactory.create_loss_function(**config)

    assert loss_fn is not None
