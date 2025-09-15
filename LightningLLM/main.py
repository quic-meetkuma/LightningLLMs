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
from LightningLLM.components.config_manager import ConfigManager, parse_arguments
from transformers import Trainer
from transformers.training_args import TrainingArguments, is_accelerate_available
import os
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from LightningLLM.components.dataset import SFTDataset
from LightningLLM.utils.helper import get_callbacks, get_optimizer
import yaml

def main():
    """Main entry point for training."""
    master_config = parse_arguments()
    config_manager = ConfigManager(master_config)

    trainer_config = config_manager.get_training_config()
    dataset_config = config_manager.get_dataset_config()
    model_config = config_manager.get_model_config()
    scheduler_config = config_manager.get_scheduler_config()
    
    output_dir = trainer_config.get("output_dir", "./training_results") # Provide a default string value
    
    # Ensure default training arguments are present
    trainer_config.setdefault("overwrite_output_dir", False)
    trainer_config.setdefault("use_cpu", False)
    trainer_config.setdefault("full_determinism", True)
    trainer_config.setdefault("torchdynamo", "eager")
    trainer_config.setdefault("remove_unused_columns", True)
    trainer_config.setdefault("skip_memory_metrics", True)
    
    # Initialize dataset
    dataset_type = dataset_config.get("dataset_type", "sft_dataset")
    if dataset_type == "sft_dataset":
        train_dataset = SFTDataset(split="train", seed=trainer_config.get("seed", 42), **dataset_config)
        test_dataset = SFTDataset(split="test", seed=trainer_config.get("seed", 42), **dataset_config)
    else:
        # TODO: Implement ComponentFactory for datasets if other types are needed
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Only 'sft_dataset' is currently supported.")

    # Initialize model and tokenizer
    model_dtype = trainer_config.get("dtype", "float16")
    model_config["dtype"] = {
        "fp16": "float16",
        "bf16": "bfloat16"
    }.get(model_dtype, "auto")
    
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
    
    # Ensure scheduler config is correctly applied
    trainer_config.setdefault("lr_scheduler_type", scheduler_config.get("name", "cosine"))
    trainer_config.setdefault("warmup_ratio", scheduler_config.get("warmup_ratio", None))
    trainer_config.setdefault("warmup_steps", scheduler_config.get("warmup_steps", None))
    # trainer_config.setdefault("lr_scheduler_kwargs", scheduler_config.get("lr_scheduler_kwargs", None)) # This field doesn't exist in config_manager.py
    
    # Set dataloader configurations
    trainer_config.setdefault("dataloader_pin_memory", dataset_config.get("dataloader_pin_memory", True))
    trainer_config.setdefault("dataloader_persistent_workers", dataset_config.get("dataloader_persistent_workers", True))
    trainer_config.setdefault("dataloader_prefetch_factor", dataset_config.get("dataloader_prefetch_factor", 1))
    trainer_config.setdefault("dataloader_drop_last", dataset_config.get("dataloader_drop_last", False))
    trainer_config.setdefault("dataloader_num_workers", dataset_config.get("dataloader_num_workers", 1))
    trainer_config.setdefault("group_by_length", dataset_config.get("group_by_length", True))
    
    # Set DDP configurations
    if trainer_config.get("ddp_config", None) is not None:
        ddp_config = trainer_config.pop('ddp_config')
        trainer_config = {**trainer_config, **ddp_config}
            
    trainer_type = trainer_config.pop("type", "base")
    if trainer_type == "base":
        trainer_cls = Trainer
        args_cls = TrainingArguments
        kwargs = {}
    elif trainer_type == "sft":
        trainer_cls = SFTTrainer
        args_cls = SFTConfig
        kwargs = {"peft_config": peft_config}
    else:
        raise ValueError(f"Invalid trainer type: {trainer_type}")
    
    fsdp_config_path = trainer_config.get("fsdp_config", None)
    if fsdp_config_path:
        # Open and load the YAML file
        with open(fsdp_config_path, 'r') as file:
            fsdp_config = yaml.safe_load(file)

        parallelism_dict = fsdp_config.get("parallelism_config", None)
        if parallelism_dict is not None:
            if is_accelerate_available("1.10.1"):
                from accelerate.parallelism_config import ParallelismConfig
            else:
                raise RuntimeError("Accelerate package with 1.10.1 version not found.")
            parallelism_config = ParallelismConfig(
                dp_replicate_size=parallelism_dict.get("dp_replicate_size", 1),
                dp_shard_size=parallelism_dict.get("dp_shard_size", 1),
                tp_size=parallelism_dict.get("tp_size", 1),
                cp_size=parallelism_dict.get("cp_size", 1),
            )
        trainer_config["parallelism_config"] = parallelism_config
    
    args = args_cls(**trainer_config)
    
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
