"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-16 12:38:56
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 12:38:58
# @ Description:
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# This below import will register all the components in the registry

import pytest
import torch

from LightningLLM.components.dataset import GenericDataModule

# Setup test data
DATASET_CONFIGS = {
    "smollm_dataset": {
        "tokenizer_name": "HuggingFaceTB/SmolLM-135M",
        "dataset_type": "seq_completion",
        "dataset_name": "Arthur-LAGACHERIE/very-smollm-corpus-0.5M",
        "dataset_subset": "default",
        "train_split": "train",
        "test_split": "test",
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "num_workers": 4,
        "max_length": 512,
        "split_ratio": 0.8,
        "seed": 42,
        "collate_fn": "dynamic_padding",
        "extra_params": {
            "input_columns": ["text"],
        },
    },
    "samsum_dataset": {
        "tokenizer_name": "HuggingFaceTB/SmolLM-135M",
        "dataset_type": "chatml_instruction_following",
        "dataset_name": "knkarthick/samsum",
        "dataset_subset": "default",
        "train_split": "train",
        "test_split": "test",
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "num_workers": 4,
        "max_length": None,
        "split_ratio": 0.8,
        "seed": 42,
        "collate_fn": "dynamic_padding",
        "extra_params": {
            "input_columns": ["dialogue", "summary"],
            "target_column": "summary",
            "prompt_template": """Dialogue:\n{dialogue}\n\nGenerate a concise summary in one sentence:{summary}""",
            "ignore_index": -100,
        },
    },
    "alpaca_dataset": {
        "tokenizer_name": "HuggingFaceTB/SmolLM-135M",
        "dataset_type": "chatml_instruction_following",
        "dataset_name": "tatsu-lab/alpaca",
        "dataset_subset": "default",
        "train_split": "train",
        "test_split": "test",
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "num_workers": 1,
        "max_length": None,
        "split_ratio": 0.8,
        "seed": 42,
        "collate_fn": "dynamic_padding",
        "extra_params": {
            "input_columns": ["instruction", "input", "output"],
            "target_column": "output",
            "prompt_template": """[INST] {instruction}\n{input} [/INST]\n{output}""",
            "ignore_index": -100,
        },
    },
    "gsm8k_dataset": {
        "tokenizer_name": "HuggingFaceTB/SmolLM-135M",
        "dataset_type": "chatml_instruction_following",
        "dataset_name": "openai/gsm8k",
        "dataset_subset": "main",
        "train_split": "train",
        "test_split": "test",
        "train_batch_size": 4,
        "eval_batch_size": 4,
        "num_workers": 4,
        "max_length": None,
        "split_ratio": 0.8,
        "seed": 42,
        "collate_fn": "dynamic_padding",
        "extra_params": {
            "input_columns": ["question", "answer"],
            "target_column": "answer",
            "prompt_template": """Solve this step-by-step:\n{question}\n\nReasoning: Let's think step by step. The answer is: {answer}""",
            "ignore_index": -100,
        },
    },
}


@pytest.mark.parametrize("dataset_name", DATASET_CONFIGS.keys())
def test_datasets(dataset_name):
    """Test that all datasets can be created with their configs."""
    # Create datasets using the factory
    config = DATASET_CONFIGS[dataset_name]
    data_module = GenericDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    # Get a batch from train loader
    print("Getting a batch from train loader...")
    for i, batch in enumerate(train_loader):
        assert batch.keys() == {"input_ids", "attention_mask", "labels"}
        assert batch["input_ids"].shape[0] == config["train_batch_size"]
        assert batch["attention_mask"].shape[0] == config["train_batch_size"]
        assert batch["labels"].shape[0] == config["train_batch_size"]
        assert batch["input_ids"].dtype == torch.int64
        assert batch["attention_mask"].dtype == torch.int64
        assert batch["labels"].dtype == torch.int64
        if config["max_length"] is not None:
            assert batch["input_ids"].shape[1] <= config["max_length"]
        break

    # Get a batch from test loader
    print("Getting a batch from test loader...")
    for batch in test_loader:
        assert batch["input_ids"].shape[0] == config["eval_batch_size"]
        break

    print("Dataset loading test completed successfully!")
