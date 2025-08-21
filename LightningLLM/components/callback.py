"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-16 11:58:28
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:58:30
# @ Description:
"""

from abc import ABC, abstractmethod

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary

from LightningLLM.components.component_registry import registry


class BaseCallback(ABC):
    """
    Abstract base class for all callbacks in the training system.
    Provides common interface and functionality for different callback types.
    """

    def __init__(self, **kwargs):
        """
        Initialize the callback.

        Args:
            **kwargs: Callback specific parameters
        """
        super().__init__()
        self.config = kwargs
        self.callback = self._create_callback()

    @abstractmethod
    def _create_callback(self):
        """Create and return the callback instance."""
        pass


@registry.callback("early_stopping")
class EarlyStoppingCallback(BaseCallback):
    """
    Wrapper over PyTorch Lightning's EarlyStopping callback.
    """

    def _create_callback(self):
        # Early stopping callback
        return EarlyStopping(**self.config)


@registry.callback("model_checkpoint")
class ModelCheckpointCallback(BaseCallback):
    """
    Wrapper over PyTorch Lightning's ModelCheckpoint callback.
    """

    def _create_callback(self):
        # Early stopping callback
        self.config["filename"] = "best-checkpoint-{epoch:02d}-{val_loss:.2f}"
        return ModelCheckpoint(**self.config)


@registry.callback("model_summary")
class ModelSummaryCallback(BaseCallback):
    """
    Wrapper over PyTorch Lightning's ModelSummary callback.
    """

    def _create_callback(self):
        # Model Summary callback
        return ModelSummary(**self.config)
