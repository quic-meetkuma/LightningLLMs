"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-12 17:22:21
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-18 00:21:11
# @ Description:
"""

"""
Main entry point for fine-tuning LLMs on the SAMSum dataset.
"""

from LightningLLM.components.component_registry import ComponentFactory
from LightningLLM.components.dataset import GenericDataModule
from LightningLLM.components.config_manager import ConfigManager, parse_arguments
from LightningLLM.components.optimizer import get_optimizer_cls
from LightningLLM.components.callback import get_callback_cls, default_callbacks
from transformers import Trainer
from transformers.training_args import TrainingArguments
import os
from trl import SFTConfig, SFTTrainer
from LightningLLM.components.dataset import SFTDataset
from LightningLLM.utils.helper import get_callbacks, get_optimizer, prepare_lora

default_training_args = {
    "overwrite_output_dir" : False,
    "use_cpu": False,
    "full_determinism": True,
    "torchdynamo": "eager",
    "remove_unused_columns": True,
    "skip_memory_metrics": True
}


def main():
    """Main entry point for training."""
    # Parse command line arguments using the new parser
    master_config = parse_arguments()

    # Create configuration manager
    config_manager = ConfigManager(master_config)
    # FIXME: Override missing values with defaults.
    # config_manager.load_config(args.config)

    # Validate configuration
    # config_manager.validate_config()


    # Initialize model
    trainer_config = config_manager.get_training_config()
    dataset_config = config_manager.get_dataset_config()
    model_config = config_manager.get_model_config()
    scheduler_config = config_manager.get_scheduler_config()
    
    output_dir = trainer_config.get("output_dir")
    
    # check ddp dict
    trainer_config = {**default_training_args, **trainer_config}
    
    # Initialize data module
    dataset_type = dataset_config.get("dataset_type", "sft_dataset")
    if dataset_type == "sft_dataset":
        train_dataset = SFTDataset(split="train", seed=trainer_config.get("seed", 42), **dataset_config)
        test_dataset = SFTDataset(split="test", seed=trainer_config.get("seed", 42), **dataset_config)


    # Initialize model and tokenizer
    model_dtype = trainer_config.get("dtype", "float16")
    if model_dtype == "fp16":
        model_dtype = "float16"
    elif model_dtype == "bf16":
        model_dtype = "bfloat16"
    else:
        model_dtype = "auto"
    model_config["dtype"] = model_dtype
    
    # Initialize model and tokenizer
    model_cls = ComponentFactory.create_model(**model_config)
    model = model_cls.load_model()
    peft_config = model_cls.load_peft_config()
    tokenizer = model_cls.load_tokenizer()

    # Initialize optimizer
    optimizer_cls_and_kwargs = get_optimizer(config_manager)

    # Initialize callbacks
    callbacks = get_callbacks(config_manager)

    # Initialize training arguments
    dtype = trainer_config.pop("dtype", "fp16")
    trainer_config[dtype] = True
    trainer_config["logging_dir"] = os.path.join(output_dir, "tb_logs")
    trainer_config["data_seed"] = trainer_config["seed"]
    
    # FIXME: You dont need a default value in case get method does not give answer. That should have been taken care in config value overriding phase above.
    trainer_config["lr_scheduler_type"] = scheduler_config.get("scheduler_name", "cosine")
    trainer_config["warmup_ratio"] = scheduler_config.get("warmup_ratio", None)
    trainer_config["warmup_steps"] = scheduler_config.get("warmup_steps", None)
    trainer_config["lr_scheduler_kwargs"] = scheduler_config.get("lr_scheduler_kwargs", None)
    
    # Set dataloader configurations
    trainer_config["dataloader_pin_memory"] = dataset_config.get("pin_memory", True)
    trainer_config["dataloader_persistent_workers"] = dataset_config.get("persistent_workers", True)
    trainer_config["dataloader_prefetch_factor"] = dataset_config.get("prefetch_factor", 1)
    trainer_config["dataloader_drop_last"] = dataset_config.get("drop_last", False)
    trainer_config["dataloader_num_workers"] = dataset_config.get("num_workers", 1)
    trainer_config["group_by_length"] = dataset_config.get("group_by_length", True)
    
    trainer_type = trainer_config.pop("type", "base")
    if trainer_type == "base":
        trainer_cls = Trainer
        args_cls = TrainingArguments
        kwargs = {}
    elif trainer_type == "sft":
        trainer_cls = SFTTrainer
        args_cls = SFTConfig
        kwargs = {"peft_config": peft_config} #, "max_seq_length": dataset_config['extra_params']["max_seq_length"]}
    else:
        raise ValueError(f"Invalid trainer type: {trainer_type}")
    
    args = args_cls(**trainer_config)
    
    def ce_loss(outputs, labels, num_items_in_batch):
        pass
    
    # Initialize trainer
    trainer = trainer_cls(model=model, 
                      processing_class=tokenizer, 
                      args=args, 
                      compute_loss_func=None,
                      train_dataset=train_dataset.dataset,
                      eval_dataset=test_dataset.dataset,
                      optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
                      callbacks=callbacks,
                      **kwargs)

    # Train model
    trainer.train()

if __name__ == "__main__":
    main()
