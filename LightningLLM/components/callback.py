"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-16 11:58:28
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:58:30
# @ Description:
"""

from abc import ABC, abstractmethod

from typing import Type
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary

from LightningLLM.components.component_registry import registry
from transformers.integrations.integration_utils import TensorBoardCallback
from transformers import EarlyStoppingCallback, ProgressCallback, PrinterCallback, DefaultFlowCallback
from transformers.trainer_callback import TrainerCallback

registry.callback("early_stopping")(EarlyStoppingCallback)
registry.callback("progressbar")(ProgressCallback)
registry.callback("printer")(PrinterCallback)
registry.callback("default_flow")(DefaultFlowCallback)
registry.callback("tensorboard")(TensorBoardCallback)


# default_callbacks = [DefaultFlowCallback(), ProgressCallback()]

def get_callback_cls(callback_name: str) -> type[TrainerCallback]:
    callback_cls = registry.get_callback(callback_name)
    if callback_cls is None:
        raise ValueError(f"Unknown optimizer: {callback_name}")
    return callback_cls
