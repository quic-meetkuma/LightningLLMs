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

from abc import ABC, abstractmethod

import torch.optim as optim

from LightningLLM.components.component_registry import registry


class BaseOptimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, model_params, **kwargs):
        self.model_params = model_params
        self.kwargs = kwargs
        self.optimizer = self._create_optimizer()

    @abstractmethod
    def _create_optimizer(self):
        """Create and return the optimizer instance."""
        pass

    def step(self):
        """Perform optimization step."""
        return self.optimizer.step()

    def zero_grad(self):
        """Zero gradients."""
        return self.optimizer.zero_grad()

    def state_dict(self):
        """Get optimizer state."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        return self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Get parameter groups."""
        return self.optimizer.param_groups


@registry.optimizer("adamw")
class AdamWOptimizer(BaseOptimizer):
    """AdamW optimizer wrapper."""

    def _create_optimizer(self):
        return optim.AdamW(
            self.model_params,
            lr=float(self.kwargs.get("lr", 3e-4)),
            weight_decay=self.kwargs.get("weight_decay", 0.0),
            betas=self.kwargs.get("betas", (0.9, 0.999)),
            eps=self.kwargs.get("eps", 1e-8),
            amsgrad=self.kwargs.get("amsgrad", False),
        )


@registry.optimizer("adam")
class AdamOptimizer(BaseOptimizer):
    """Adam optimizer wrapper."""

    def _create_optimizer(self):
        return optim.Adam(
            self.model_params,
            lr=float(self.kwargs.get("lr", 3e-4)),
            weight_decay=self.kwargs.get("weight_decay", 0.0),
            betas=self.kwargs.get("betas", (0.9, 0.999)),
            eps=self.kwargs.get("eps", 1e-8),
            amsgrad=self.kwargs.get("amsgrad", False),
        )


@registry.optimizer("sgd")
class SGDOptimizer(BaseOptimizer):
    """SGD optimizer wrapper."""

    def _create_optimizer(self):
        return optim.SGD(
            self.model_params,
            lr=float(self.kwargs.get("lr", 0.01)),
            momentum=self.kwargs.get("momentum", 0.0),
            weight_decay=self.kwargs.get("weight_decay", 0.0),
            dampening=self.kwargs.get("dampening", 0.0),
            nesterov=self.kwargs.get("nesterov", False),
        )


# @registry.optimizer("lion")
# class LionOptimizer(BaseOptimizer):
#     """Lion optimizer wrapper (requires lion-pytorch package)."""

#     def _create_optimizer(self):
#         try:
#             from lion_pytorch import Lion

#             return Lion(
#                 self.model_params,
#                 lr=self.kwargs.get("lr", 3e-4),
#                 weight_decay=self.kwargs.get("weight_decay", 0.0),
#                 betas=self.kwargs.get("betas", (0.9, 0.999)),
#             )
#         except ImportError:
#             raise ImportError(
#                 "Lion optimizer requires 'lion-pytorch' package. Install with: pip install lion-pytorch"
#             )
