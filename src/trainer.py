"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-12 17:22:13
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-17 16:28:07
# @ Description:
"""

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import set_seed
from components.component_registry import ComponentFactory


def create_trainer(config, output_dir):
    """Create PyTorch Lightning trainer with specified configuration."""

    # Set seed for reproducibility
    seed = config.training.seed
    set_seed(seed)

    # Callbacks
    callbacks = []

    callback_config = config.get_callback_config()
    for callback_name, callback_dict in callback_config["callbacks"].items():
        callback_inst = ComponentFactory.create_callback(callback_name, **callback_dict)
        callbacks.append(callback_inst.callback)

    # Logger
    tb_logger = TensorBoardLogger(output_dir, name="tensorboard_logs")

    training_config = config.get_training_config()
    kwargs = {}
    kwargs["callbacks"] = callbacks
    kwargs["max_steps"] = training_config.get("max_steps", -1)
    kwargs["accumulate_grad_batches"] = training_config.get(
        "gradient_accumulation_steps", 1
    )
    kwargs["deterministic"] = True
    kwargs["use_distributed_sampler"] = True
    kwargs["precision"] = training_config.get("precision", "fp32")
    kwargs["accelerator"] = training_config.get("device", "cuda")
    kwargs["devices"] = training_config.get("num_devices", 1)
    # if kwargs["precision"] == "16-mixed" and kwargs["accelerator"] == "cuda":
    #         kwargs["amp_backend"] = "apex"
    #         kwargs["amp_level"] = "O2"

    kwargs["logger"] = tb_logger
    kwargs["strategy"] = training_config.get("strategy", "auto")

    # Trainer
    trainer = pl.Trainer(**kwargs)

    return trainer
