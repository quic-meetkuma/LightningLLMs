"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-12 18:31:33
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:32:58
# @ Description:
"""

"""
Dataset components for the training system.
"""

from typing import Dict, Any, Optional
import re
import torch.nn.functional as F
from datasets import load_dataset, load_dataset_builder
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from LightningLLM.components.component_registry import registry
from LightningLLM.utils.dataset_helper import insert_pad_token


# Used for pretraining
@registry.dataset("seq_completion")
class SentenceCompletionDataset(Dataset):
    """Generic dataset class which can be used for autoregressive training."""

    def __init__(self, dataset, tokenizer, max_length, split, **kwargs):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kwargs = kwargs
        self.split = split
        self.input_column = kwargs.get("extra_params", {}).get("input_column", "text")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        source_text = item[self.input_column]

        # Tokenize source text without padding
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding=False,  # No padding here
            truncation=True,
            return_tensors="pt",
        )

        # Extract input_ids and attention_mask
        source_ids = source_encoding["input_ids"].squeeze()
        source_mask = source_encoding["attention_mask"].squeeze()

        # Get pad token id from tokenizer
        pad_token_id = self.tokenizer.pad_token_id

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": source_ids,
            "pad_token_id": pad_token_id,
        }


# Used for SFT
@registry.dataset("chatml_instruction_following")
class ChatMLInstructionFollowingDataset(Dataset):
    """Generic dataset class which can be used for autoregressive training."""

    def __init__(self, dataset, tokenizer, max_length, split, **kwargs):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kwargs = kwargs
        self.split = split
        self.extra_params = kwargs.get("extra_params", {})

        if "input_columns" not in self.extra_params:
            raise ValueError(
                "input_columns must be provided in extra_params for chatml_instruction_following dataset."
            )
        if "target_column" not in self.extra_params:
            raise ValueError(
                "target_column must be provided in extra_params for chatml_instruction_following dataset."
            )
        if "prompt_template" not in self.extra_params:
            raise ValueError(
                "prompt_template must be provided in extra_params for chatml_instruction_following dataset."
            )
        self.input_columns = self.extra_params["input_columns"]
        self.target_column = self.extra_params["target_column"]
        self.prompt_template = self.extra_params["prompt_template"]
        self.ignore_index = self.extra_params["ignore_index"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        template_dict = {}
        for input_column in self.input_columns:
            if input_column not in item:
                raise KeyError(
                    f"Input column '{input_column}' not found in dataset item."
                )
            template_dict[input_column] = item[input_column]

        input_text = self.prompt_template.format(**template_dict)
        input_text = input_text.replace("\n ", " ")  # Cleanup if input is empty.

        if self.target_column not in item:
            raise KeyError(
                f"Target column '{self.target_column}' not found in dataset item."
            )
        target_text = item[self.target_column]

        # Tokenize source text without padding
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=False,  # No padding here
            truncation=True,
            return_tensors="pt",
        )

        # Extract input_ids and attention_mask
        input_ids = input_encoding["input_ids"].squeeze(dim=0)
        attn_mask = input_encoding["attention_mask"].squeeze(dim=0)

        label_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding=False,  # No padding here
            truncation=True,
            return_tensors="pt",
        )

        labels = label_encoding["input_ids"].squeeze(dim=0)
        labels = F.pad(
            labels,
            (input_ids.shape[0] - labels.shape[0], 0),
            "constant",
            self.ignore_index,
        )

        # Get pad token id from tokenizer
        pad_token_id = self.tokenizer.pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "pad_token_id": pad_token_id,
        }


@registry.dataset("sft_dataset")
class SFTDataset(Dataset):
    """
    A Supervised Fine-Tuning (SFT) dataset class for text data.

    This class handles loading data from Hugging Face datasets, filtering out invalid samples,
    and applying a prompt/completion templating for SFT tasks.

    Args:
        dataset_name (str): The name of the dataset to load from Hugging Face datasets.
        split (str): The dataset split to use (e.g., "train", "validation", "test").
        prompt_template (str): A string template for constructing the prompt. Variables in the
                                template should be enclosed in curly braces, e.g., "Answer the question: {question}".
        completion_template (str): A string template for constructing the completion (target).
                                   Variables should be enclosed in curly braces, e.g., "{answer}".

    Raises:
        RuntimeError: If any variables specified in `prompt_template` or `completion_template`
                      are not found as columns in the loaded dataset.
    """
    def __init__(
        self,
        dataset_name: str,
        split: str,
        prompt_template: str,
        completion_template: str,
        split_ratio: float = 0.8,
        seed: int = 42,
        **kwargs,
    ):
        db = load_dataset_builder(dataset_name)
        available_splits = []
        if db.info.splits is not None:
            available_splits = list(db.info.splits.keys())

        if split not in available_splits and "train" not in available_splits:
            raise ValueError(f"Split {split} is not available for dataset {dataset_name}.")

        # FIXME: Add streaming support for larger datasets.
        self.dataset = load_dataset(
            dataset_name, split=split
        )
        if split == "test" and len(available_splits) == 1:
            split_ratio = split_ratio
            splitted_dataset = self.dataset.train_test_split(
                test_size=(1 - split_ratio), seed=seed
            )
            self.dataset = splitted_dataset["test"]

        self.prompt_template = prompt_template
        self.completion_template = completion_template
        self.dataset_columns = self.dataset.column_names
        
        # Extract variables from templates and check if they exist in dataset columns
        prompt_variables = re.findall(r"\{(.*?)\}", self.prompt_template)
        completion_variables = re.findall(r"\{(.*?)\}", self.completion_template)
        
        for var in prompt_variables:
            if var not in self.dataset_columns:
                raise RuntimeError(f"Prompt template variable '{var}' not found in dataset columns: {self.dataset_columns}.")
        for var in completion_variables:
            if var not in self.dataset_columns:
                raise RuntimeError(f"Completion template variable '{var}' not found in dataset columns: {self.dataset_columns}.")
        
        # Filter out samples with None or empty strings in relevant columns
        # Only filter columns that are actually used in the templates
        self.relevant_columns = list(set(prompt_variables + completion_variables))
        self.dataset = self.dataset.filter(self._filter_empty_or_none_samples)
        self.dataset = self.dataset.map(self._preprocess_sample)

    def _filter_empty_or_none_samples(self, example: Dict[str, Any]) -> bool:
        """
        Filters out samples where any of the relevant columns are None or contain only whitespace.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.

        Returns:
            bool: True if the sample should be kept, False otherwise.
        """
        for column in self.relevant_columns:
            value = example.get(column)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True

    def _preprocess_sample(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Applies the prompt and completion templates to a single example.

        Args:
            example (Dict[str, Any]): A single sample from the dataset.

        Returns:
            Dict[str, str]: A dictionary containing the 'prompt' and 'completion' strings.
        """
        return {
            "prompt": self.prompt_template.format(**example),
            "completion": self.completion_template.format(**example),
        }
        
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Retrieves a processed sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, str]: A dictionary containing the processed 'prompt' and 'completion' for the sample.
        """
        # Get the raw example using .select and access the first element
        example = self.dataset.select(indices=[int(idx)])[0]

        # Apply preprocessing (templating) on the fly
        processed_example = self._preprocess_sample(example)
        
        return processed_example
