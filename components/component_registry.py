"""
Component registry for managing different training components.
This allows easy registration and retrieval of optimizers, schedulers, datasets, etc.
"""

import logging
from typing import Any, Dict, Type, Optional

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Registry for managing different training components."""

    def __init__(self):
        self._optimizers: Dict[str, Type] = {}
        self._schedulers: Dict[str, Type] = {}
        self._datasets: Dict[str, Type] = {}
        self._models: Dict[str, Type] = {}
        self._data_collators: Dict[str, Type] = {}
        self._metrics: Dict[str, Type] = {}
        self._loss_functions: Dict[str, Type] = {}
        self._callbacks: Dict[str, Type] = {}
        self._hooks: Dict[str, Type] = {}
        self._trainer_modules: Dict[str, Type] = {}

    def trainer_module(self, name: str):
        """Decorator to register a trainer module class."""

        def decorator(cls: Type):
            self._trainer_modules[name] = cls
            logger.info(f"Registered trainer module: {name}")
            return cls

        return decorator

    def optimizer(self, name: str):
        """Decorator to register an optimizer class."""

        def decorator(cls: Type):
            self._optimizers[name] = cls
            logger.info(f"Registered optimizer: {name}")
            return cls

        return decorator

    def scheduler(self, name: str):
        """Decorator to register a scheduler class."""

        def decorator(cls: Type):
            self._schedulers[name] = cls
            logger.info(f"Registered scheduler: {name}")
            return cls

        return decorator

    def dataset(self, name: str):
        """Decorator to register a dataset class."""

        def decorator(cls: Type):
            self._datasets[name] = cls
            logger.info(f"Registered dataset: {name}")
            return cls

        return decorator

    def model(self, name: str):
        """Decorator to register a model class."""

        def decorator(cls: Type):
            self._models[name] = cls
            logger.info(f"Registered model: {name}")
            return cls

        return decorator

    def data_collator(self, name: str):
        """Decorator to register a data collator class."""

        def decorator(fn_pointer: Type):
            self._data_collators[name] = fn_pointer
            logger.info(f"Registered data collator: {name}")
            return fn_pointer

        return decorator

    # def metric(self, name: str):
    #     """Decorator to register a metric class."""

    #     def decorator(cls: Type):
    #         self._metrics[name] = cls
    #         logger.info(f"Registered metric: {name}")
    #         return cls

    #     return decorator

    def loss_function(self, name: str):
        """Decorator to register a loss function class."""

        def decorator(cls: Type):
            self._loss_functions[name] = cls
            logger.info(f"Registered loss function: {name}")
            return cls

        return decorator

    def callback(self, name: str):
        """Decorator to register a callback class."""

        def decorator(cls: Type):
            self._callbacks[name] = cls
            logger.info(f"Registered callback: {name}")
            return cls

        return decorator

    # def hook(self, name: str):
    #     """Decorator to register a hook class."""

    #     def decorator(cls: Type):
    #         self._hooks[name] = cls
    #         logger.info(f"Registered hook: {name}")
    #         return cls

    #     return decorator

    def get_trainer_module(self, name: str) -> Optional[Type]:
        """Get trainer module class by name."""
        return self._trainer_modules.get(name)

    def get_optimizer(self, name: str) -> Optional[Type]:
        """Get optimizer class by name."""
        return self._optimizers.get(name)

    def get_scheduler(self, name: str) -> Optional[Type]:
        """Get scheduler class by name."""
        return self._schedulers.get(name)

    def get_dataset(self, name: str) -> Optional[Type]:
        """Get dataset class by name."""
        return self._datasets.get(name)

    def get_model(self, name: str) -> Optional[Type]:
        """Get model class by name."""
        return self._models.get(name)

    def get_data_collator(self, name: str) -> Optional[Type]:
        """Get data collator class by name."""
        return self._data_collators.get(name)

    # def get_metric(self, name: str) -> Optional[Type]:
    #     """Get metric class by name."""
    #     return self._metrics.get(name)

    def get_loss_function(self, name: str) -> Optional[Type]:
        """Get loss function class by name."""
        return self._loss_functions.get(name)

    def get_callback(self, name: str) -> Optional[Type]:
        """Get callback class by name."""
        return self._callbacks.get(name)

    # def get_hook(self, name: str) -> Optional[Type]:
    #     """Get hook class by name."""
    #     return self._hooks.get(name)

    def list_trainer_modules(self) -> list[str]:
        """List all registered trainer modules."""
        return list(self._trainer_modules.keys())

    def list_optimizers(self) -> list[str]:
        """List all registered optimizers."""
        return list(self._optimizers.keys())

    def list_schedulers(self) -> list[str]:
        """List all registered schedulers."""
        return list(self._schedulers.keys())

    def list_datasets(self) -> list[str]:
        """List all registered datasets."""
        return list(self._datasets.keys())

    def list_models(self) -> list[str]:
        """List all registered models."""
        return list(self._models.keys())

    def list_data_collators(self) -> list[str]:
        """List all registered data collators."""
        return list(self._data_collators.keys())

    # def list_metrics(self) -> list[str]:
    #     """List all registered metrics."""
    #     return list(self._metrics.keys())

    def list_loss_functions(self) -> list[str]:
        """List all registered loss functions."""
        return list(self._loss_functions.keys())

    def list_callbacks(self) -> list[str]:
        """List all registered callbacks."""
        return list(self._callbacks.keys())

    # def list_hooks(self) -> list[str]:
    #     """List all registered hooks."""
    #     return list(self._hooks.keys())


# Global registry instance
registry = ComponentRegistry()


class ComponentFactory:
    """Factory for creating components using the registry."""

    @staticmethod
    def create_trainer_module(name: str, **kwargs) -> Any:
        """Create a trainer module instance."""
        trainer_module_class = registry.get_trainer_module(name)
        if trainer_module_class is None:
            raise ValueError(
                f"Unknown trainer module: {name}. Available: {registry.list_trainer_modules()}"
            )
        return trainer_module_class(**kwargs)

    @staticmethod
    def create_optimizer(name: str, model_params, **kwargs) -> Any:
        """Create an optimizer instance."""
        optimizer_class = registry.get_optimizer(name)
        if optimizer_class is None:
            raise ValueError(
                f"Unknown optimizer: {name}. Available: {registry.list_optimizers()}"
            )
        return optimizer_class(model_params, **kwargs)

    @staticmethod
    def create_scheduler(name: str, optimizer, **kwargs) -> Any:
        """Create a scheduler instance."""
        scheduler_class = registry.get_scheduler(name)
        if scheduler_class is None:
            raise ValueError(
                f"Unknown scheduler: {name}. Available: {registry.list_schedulers()}"
            )
        return scheduler_class(optimizer, **kwargs)

    @staticmethod
    def create_dataset(name: str, **kwargs) -> Any:
        """Create a dataset instance."""
        dataset_class = registry.get_dataset(name)
        if dataset_class is None:
            raise ValueError(
                f"Unknown dataset: {name}. Available: {registry.list_datasets()}"
            )
        return dataset_class(**kwargs)

    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        """Create a model instance."""
        model_class = registry.get_model(model_type)
        if model_class is None:
            raise ValueError(
                f"Unknown model: {model_type}. Available: {registry.list_models()}"
            )
        return model_class(**kwargs)

    # @staticmethod
    # def create_metric(name: str, **kwargs) -> Any:
    #     """Create a metric instance."""
    #     metric_class = registry.get_metric(name)
    #     if metric_class is None:
    #         raise ValueError(
    #             f"Unknown metric: {name}. Available: {registry.list_metrics()}"
    #         )
    #     return metric_class(**kwargs)

    @staticmethod
    def create_loss_function(name: str, **kwargs) -> Any:
        """Create a loss function instance."""
        loss_function_class = registry.get_loss_function(name)
        if loss_function_class is None:
            raise ValueError(
                f"Unknown loss function: {name}. Available: {registry.list_loss_functions()}"
            )
        return loss_function_class(**kwargs)

    @staticmethod
    def create_callback(name: str, **kwargs) -> Any:
        """Create a callback instance."""
        callback_class = registry.get_callback(name)
        if callback_class is None:
            raise ValueError(
                f"Unknown callback: {name}. Available: {registry.list_callbacks()}"
            )
        return callback_class(**kwargs)

    # @staticmethod
    # def create_hook(name: str, **kwargs) -> Any:
    #     """Create a hook instance."""
    #     hook_class = registry.get_hook(name)
    #     if hook_class is None:
    #         raise ValueError(
    #             f"Unknown hook: {name}. Available: {registry.list_hooks()}"
    #         )
    #     return hook_class(**kwargs)
