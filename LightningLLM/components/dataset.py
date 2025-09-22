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
from typing import Callable
import importlib
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
        split_ratio: float = 0.8,
        seed: int = 42,
        **kwargs,
    ):
        
        prompt_template = kwargs.get("prompt_template", None)
        completion_template = kwargs.get("completion_template", None)
        prompt_func = kwargs.get("prompt_func", None)
        completion_func = kwargs.get("completion_func", None)
        remove_samples_with_empty_columns = kwargs.get("remove_samples_with_empty_columns", True)
        
        if (prompt_template is None and prompt_func is None) and (prompt_template is not None and prompt_func is not None):
            raise RuntimeError("Either provide prompt_template or prompt_func in the config.")
        if (completion_template is None and completion_func is None) and (completion_template is not None and completion_func is not None):
            raise RuntimeError("Either provide completion_template or completion_func in the config.")
        
        db = load_dataset_builder(dataset_name)
        available_splits = []
        if db.info.splits is not None:
            available_splits = list(db.info.splits.keys())

        if split not in available_splits and split == "train":
            raise ValueError(f"Split {split} is not available for dataset {dataset_name}.")

        load_split = split
        if split not in available_splits:
            load_split = "train"

        # FIXME: Add streaming support for larger datasets.
        self.dataset = load_dataset(
            dataset_name, split=load_split
        )
        if len(available_splits) == 1:
            split_ratio = split_ratio
            splitted_dataset = self.dataset.train_test_split(
                test_size=(1 - split_ratio), seed=seed
            )
            if split == "test":
                self.dataset = splitted_dataset["test"]
            else:
                self.dataset = splitted_dataset["train"]

        self.dataset_columns = self.dataset.column_names
        if prompt_template:
            self.prompt_template = prompt_template
            self.prompt_func = None
            # Extract variables from templates and check if they exist in dataset columns
            prompt_variables = re.findall(r"\{(.*?)\}", self.prompt_template)
            for var in prompt_variables:
                if var not in self.dataset_columns:
                    raise RuntimeError(f"Prompt template variable '{var}' not found in dataset columns: {self.dataset_columns}.")
        else:
            prompt_variables = self.dataset_columns
            self.prompt_func = self.import_func(prompt_func)
            
        if completion_template:
            self.completion_template = completion_template
            self.completion_func = None
            # Extract variables from templates and check if they exist in dataset columns
            completion_variables = re.findall(r"\{(.*?)\}", self.completion_template)
            for var in completion_variables:
                if var not in self.dataset_columns:
                    raise RuntimeError(f"Completion template variable '{var}' not found in dataset columns: {self.dataset_columns}.")
        else:
            completion_variables = self.dataset_columns
            self.completion_func = self.import_func(completion_func)
            
        # Filter out samples with None or empty strings in relevant columns
        # Only filter columns that are actually used in the templates
        self.relevant_columns = list(set(prompt_variables + completion_variables))
        if remove_samples_with_empty_columns:
            self.dataset = self.dataset.filter(self._filter_empty_or_none_samples)
        self.dataset = self.dataset.map(self._preprocess_sample)

    def import_func(self, func_path: str) -> Callable:
        if ":" not in func_path:
            raise ValueError("func_path must be in the format 'module_file_path:function_name'.")
        module_file_path, function_name = func_path.split(":")

        try:
            module = importlib.import_module(module_file_path)
        except Exception as e:
            raise RuntimeError(f"Unable to import module : {module_file_path}.")
        if not hasattr(module, function_name):
            raise ValueError(f"Function {function_name} not found in module {module_file_path}.")
        return getattr(module, function_name)


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
        prompt_text = self.prompt_func(example) if self.prompt_func is not None else self.prompt_template.format(**example)
        completion_text = self.completion_func(example) if self.completion_func is not None else self.completion_template.format(**example)
        return {
            "prompt": prompt_text,
            "completion": completion_text,
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
