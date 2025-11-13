# In a new file: LightningLLM/trainers/kd_trainer.py
# from some_kd_library import KDTrainer, KDTrainingArguments
from transformers import Trainer, TrainingArguments
from LightningLLM.components.component_registry import registry

@registry.trainer_module(
    name="kd",
    args_cls=TrainingArguments,
    required_kwargs={"teacher_model": "REQUIRED", "temperature": 1.0}
)
class KDTrainerModule(Trainer):
    pass