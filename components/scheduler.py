"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-12 17:46:49
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:35:05
# @ Description:
"""

"""
Learning rate scheduler components for the training system.
"""

from abc import ABC, abstractmethod

import torch
from transformers import (get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

from components.component_registry import registry


class BaseScheduler(ABC):
    """Base class for all schedulers."""

    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.kwargs = kwargs
        self.scheduler = self._create_scheduler()

    @abstractmethod
    def _create_scheduler(self):
        """Create and return the scheduler instance."""
        pass

    def step(self, metrics=None):
        """Step the scheduler."""
        return self.scheduler.step(metrics)

    def state_dict(self):
        """Get scheduler state."""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        return self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self):
        """Get the last learning rate."""
        return self.scheduler.get_last_lr()


@registry.scheduler("constant_with_warmup")
class ConstantWarmupScheduler(BaseScheduler):
    """Cosine annealing with warmup learning rate scheduler from transformers."""

    def _create_scheduler(self):
        return get_constant_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.kwargs.get("num_warmup_steps", 0),
            last_epoch=self.kwargs.get("last_epoch ", -1),
        )


@registry.scheduler("linear_with_warmup")
class LinearWarmupScheduler(BaseScheduler):
    """Cosine annealing with warmup learning rate scheduler from transformers."""

    def _create_scheduler(self):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.kwargs.get("num_warmup_steps", 0),
            num_training_steps=self.kwargs.get("num_training_steps", 100),
            last_epoch=self.kwargs.get("last_epoch ", -1),
        )


@registry.scheduler("cosine_with_warmup")
class CosineAnnealingWarmupScheduler(BaseScheduler):
    """Cosine annealing with warmup learning rate scheduler from transformers."""

    def _create_scheduler(self):
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.kwargs.get("num_warmup_steps", 0),
            num_training_steps=self.kwargs.get("num_training_steps", 100),
            num_cycles=self.kwargs.get("num_cycles", 0.5),
            last_epoch=self.kwargs.get("last_epoch ", -1),
        )


if __name__ == "__main__":
    # Plot the learning rate schedules for testing purposes using:
    # python -m components.scheduler

    import matplotlib.pyplot as plt
    import torch.nn as nn

    # Create a dummy model and optimizer for testing
    dummy_model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)

    # Number of training steps for simulation
    num_steps = 100

    # Test ConstantWarmupScheduler
    optimizer1 = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    scheduler1 = ConstantWarmupScheduler(optimizer1, num_warmup_steps=10)
    lrs1 = []

    # Test LinearWarmupScheduler
    optimizer2 = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    scheduler2 = LinearWarmupScheduler(
        optimizer2, num_warmup_steps=10, num_training_steps=num_steps
    )
    lrs2 = []

    # Test CosineAnnealingWarmupScheduler
    optimizer3 = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)
    scheduler3 = CosineAnnealingWarmupScheduler(
        optimizer3, num_warmup_steps=10, num_training_steps=num_steps
    )
    lrs3 = []

    # Simulate training for each scheduler
    for i in range(num_steps):
        # Get current learning rate
        lrs1.append(scheduler1.get_last_lr()[0])
        lrs2.append(scheduler2.get_last_lr()[0])
        lrs3.append(scheduler3.get_last_lr()[0])

        # Perform dummy optimization step
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        # Step the schedulers
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

    # Plot learning rates
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_steps), lrs1, label="Constant Warmup")
    plt.plot(range(num_steps), lrs2, label="Linear Warmup")
    plt.plot(range(num_steps), lrs3, label="Cosine Warmup")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedulers Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
