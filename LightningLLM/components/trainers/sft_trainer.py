# In a new file: LightningLLM/trainers/sft_trainer.py
from trl import SFTTrainer, SFTConfig
from LightningLLM.components.component_registry import registry

@registry.trainer_module(
    name="sft",
    args_cls=SFTConfig,
    required_kwargs={"peft_config": "REQUIRED"}
)
class SFTTrainerModule(SFTTrainer):
    pass  # Just using the standard SFTTrainer