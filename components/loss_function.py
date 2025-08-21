"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-12 17:27:09
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:33:24
# @ Description:
"""

"""
Loss function components for the modular training system.
Provides BaseLossFunction abstract class and concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.component_registry import registry


class BaseLossFunction(ABC, nn.Module):
    """
    Abstract base class for all loss functions in the training system.
    Provides common interface and functionality for different loss types.
    """

    def __init__(self, **kwargs):
        """
        Initialize the loss function.

        Args:
            **kwargs: Loss function specific parameters
        """
        super().__init__()
        self.config = kwargs

    @abstractmethod
    def forward(self, outputs: Any, targets: Any, **kwargs) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            outputs: Model outputs
            targets: Target values
            **kwargs: Additional arguments

        Returns:
            Computed loss tensor
        """
        pass

    def __call__(self, outputs: Any, targets: Any, **kwargs) -> torch.Tensor:
        """Call the loss function."""
        return self.forward(outputs, targets, **kwargs)


@registry.loss_function("cross_entropy")
class CrossEntropyLoss(BaseLossFunction):
    """
    Standard cross-entropy loss for classification tasks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ignore_index = kwargs.get("ignore_index", -100)
        self.reduction = kwargs.get("reduction", "mean")
        self.label_smoothing = kwargs.get("label_smoothing", 0.0)
        self.weight = kwargs.get("weight", 1.0)

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

    def forward(self, outputs: Any, targets: Any, **kwargs) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            outputs: Model outputs containing logits
            targets: Target labels
            **kwargs: Additional arguments

        Returns:
            Computed loss tensor
        """
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Handle sequence classification
        if len(logits.shape) == 3:
            # Language modeling: shift logits and targets
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            return self.weight * self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)
            )
        else:
            # Standard classification
            return self.weight * self.loss_fn(logits, targets)


@registry.loss_function("kl_div")
class KLDivergenceLoss(BaseLossFunction):
    """
    KL divergence loss for knowledge distillation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temperature = kwargs.get("temperature", 4.0)
        self.reduction = kwargs.get("reduction", "batchmean")
        self.weight = kwargs.get("weight", 1.0)

    def forward(self, outputs: Any, targets: Any, **kwargs) -> torch.Tensor:
        """
        Compute KL divergence loss.

        Args:
            outputs: Student model outputs containing logits
            targets: Teacher model outputs containing logits
            **kwargs: Additional arguments

        Returns:
            Computed loss tensor
        """
        student_logits = outputs.logits if hasattr(outputs, "logits") else outputs
        teacher_logits = targets.logits if hasattr(targets, "logits") else targets

        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Compute KL divergence
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction=self.reduction)

        return self.weight * kl_loss * (self.temperature**2)
