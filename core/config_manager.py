"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-13 17:47:20
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:37:04
# @ Description:
"""

"""
Configuration manager for handling all training configurations.
Provides centralized configuration loading, validation, and management.
"""

import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import logging

from components.logger import get_logger
from components.data_collator import dynamic_padding_collate_fn


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""

    name: str = "adamw"
    lr: float = 3e-4
    # Additional optimizer-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """Configuration for loss functions."""

    name: str = "cross_entropy"
    # Additional loss-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers."""

    name: str = "cosine_with_warmup"
    # Additional scheduler-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for datasets."""

    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dataset_type: str = "seq_completion"
    dataset_name: str = "Arthur-LAGACHERIE/very-smollm-corpus-0.5M"
    dataset_subset: str = "default"
    train_split: str = "train"
    test_split: str = "test"
    train_batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 0
    max_length: Optional[int] = None
    split_ratio: float = (
        0.8  # Ratio for train/test split, used when only train_split is provided
    )
    seed: int = 42
    collate_fn: callable = dynamic_padding_collate_fn
    # Additional dataset-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for models."""

    model_name: str = "HuggingFaceTB/SmolLM-135M"
    model_type: str = "hf"  # 'hf' for Hugging Face, 'custom' for custom models
    auto_class_name: str = "AutoModelForCausalLM"
    torch_dtype: str = "float16"
    use_cache: bool = False
    attn_implementation: str = "sdpa"
    device_map: Optional[str] = None
    # Additional model-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackConfig:
    """Configuration for callbacks."""

    # Dictionary to hold all callback configurations
    # Key: callback name, Value: callback configuration dictionary
    callbacks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Additional callback-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Model trainer type
    type: str = "causal_lm"
    output_dir: str = "./results"
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    precision: str = "fp16-mixed"  # "fp16-mixed", "bf16"
    seed: int = 42
    device: str = "cuda"  # "cuda", "cpu", "qaic"
    num_devices: int = 1
    strategy: str = "ddp"

    # Additional training-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MasterConfig:
    """Main training configuration."""

    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Optimizer configuration
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Loss configuration
    loss: LossConfig = field(default_factory=LossConfig)

    # Callbacks configuration
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)

    # Callbacks configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages configuration loading, validation, and updates."""

    def __init__(
        self, config_source: Optional[Union[str, Path, Dict[str, Any]]] = None
    ):
        """
        Initialize ConfigManager with either:
        - Path to config file (str or Path)
        - Configuration dictionary
        - None (creates empty config)
        """
        self.config = MasterConfig()

        if config_source is not None:
            if isinstance(config_source, (str, Path)):
                self.load_config(config_source)
            elif isinstance(config_source, dict):
                self.update_config(config_source)
            else:
                raise TypeError(
                    f"config_source must be path (str/Path) or dict, got {type(config_source)}"
                )

    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {config_path.suffix}"
            )

        self.update_config(config_dict)

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary values."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                if isinstance(value, dict) and hasattr(
                    getattr(self.config, key), "__dataclass_fields__"
                ):
                    # Special handling for callbacks
                    if key == "callbacks":
                        nested_config = getattr(self.config, key)
                        for callback_name, callback_config in value.items():
                            if isinstance(callback_config, dict):
                                getattr(nested_config, "callbacks")[callback_name] = (
                                    callback_config
                                )
                            else:
                                getattr(nested_config, "extra_params")[
                                    callback_name
                                ] = callback_config
                    else:
                        # Update nested dataclass
                        nested_config = getattr(self.config, key)
                        for nested_key, nested_value in value.items():
                            if hasattr(nested_config, nested_key):
                                setattr(
                                    getattr(self.config, key), nested_key, nested_value
                                )
                            elif hasattr(nested_config, "extra_params"):
                                getattr(getattr(self.config, key), "extra_params")[
                                    nested_key
                                ] = nested_value
                else:
                    setattr(self.config, key, value)
            else:
                # Store unknown parameters in extra_params
                self.config.extra_params[key] = value

    def save_config(self, output_path: str) -> None:
        """Save current configuration to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self.config)

        if output_path.suffix.lower() in [".yaml", ".yml"]:
            with open(output_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == ".json":
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output file format: {output_path.suffix}")

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        errors = []

        # Validate model configuration
        if not self.config.model.model_name:
            errors.append("Model name is required")

        # Validate dataset configuration
        if not self.config.dataset.dataset_name:
            errors.append("Dataset name is required")

        # Validate training parameters
        if self.config.dataset.train_batch_size <= 0:
            errors.append("Train batch size must be positive")

        if self.config.dataset.eval_batch_size <= 0:
            errors.append("Validation batch size must be positive")

        if self.config.training.num_train_epochs <= 0:
            errors.append("Number of epochs must be positive")

        if self.config.training.gradient_accumulation_steps <= 0:
            errors.append("Gradient accumulation steps must be positive")

        # Validate optimizer configuration
        if float(self.config.optimizer.lr) <= 0:
            errors.append("Learning rate must be positive")

        # Validate device configuration
        valid_devices = ["cpu", "cuda", "qaic"]
        if self.config.training.device not in valid_devices:
            errors.append(f"Device must be one of {valid_devices}")

        if errors:
            raise ValueError(
                f"Configuration validation failed:\n"
                + "\n".join(f"- {error}" for error in errors)
            )

    def get_callback_config(self) -> Dict[str, Any]:
        """Get callback configuration as dictionary."""
        callback_dict = asdict(self.config.callbacks)
        callback_dict.update(callback_dict.pop("extra_params"))
        return callback_dict

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration as dictionary."""
        optimizer_dict = asdict(self.config.optimizer)
        optimizer_dict.update(optimizer_dict.pop("extra_params"))
        return optimizer_dict

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration as dictionary."""
        training_dict = asdict(self.config.training)
        training_dict.update(training_dict.pop("extra_params"))
        return training_dict

    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration as dictionary."""
        scheduler_dict = asdict(self.config.scheduler)
        scheduler_dict.update(scheduler_dict.pop("extra_params"))
        return scheduler_dict

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration as dictionary."""
        dataset_dict = asdict(self.config.dataset)
        return dataset_dict

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        model_dict = asdict(self.config.model)
        model_dict.update(model_dict.pop("extra_params"))
        return model_dict

    def get_loss_config(self) -> Dict[str, Any]:
        """Get loss configuration as dictionary."""
        loss_dict = asdict(self.config.loss)
        loss_dict.update(loss_dict.pop("extra_params"))
        return loss_dict

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self.config)

    def __getattr__(self, name: str) -> Any:
        """Allow direct access to config attributes."""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
