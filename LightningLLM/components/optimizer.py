"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-12 22:57:14
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:34:15
# @ Description:
"""

"""
Optimizer components for the training system.
"""

from typing import Type
import torch.optim as optim
from torch.optim import Optimizer
from LightningLLM.components.component_registry import registry


registry.optimizer("adam")(optim.Adam)
registry.optimizer("adamw")(optim.AdamW)
registry.optimizer("sgd")(optim.SGD)

def get_optimizer_cls(optimizer_name: str) -> Type[Optimizer]:
    optimizer_cls = registry.get_optimizer(optimizer_name)
    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer_cls
