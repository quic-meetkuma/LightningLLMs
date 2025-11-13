# In a new file: LightningLLM/trainers/base_trainer.py
from transformers import Trainer, TrainingArguments
from LightningLLM.components.component_registry import registry

@registry.trainer_module(
    name="base",
    args_cls=TrainingArguments,
    required_kwargs={}
)
class BaseTrainer(Trainer):
    pass  # Just using the standard Trainer