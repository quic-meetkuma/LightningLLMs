# In a new file: LightningLLM/trainers/reward_trainer.py
from trl import RewardTrainer
from transformers import TrainingArguments
from LightningLLM.components.component_registry import registry

@registry.trainer_module(
    name="reward",
    args_cls=TrainingArguments,
    required_kwargs={"reward_tokenizer": "REQUIRED"}
)
class RewardTrainerModule(RewardTrainer):
    pass