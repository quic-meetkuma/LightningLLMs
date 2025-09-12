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
        available_splits = list(db.info.splits.keys())

        if split not in available_splits and "train" not in available_splits:
            raise ValueError(f"Split {split} is not available.")

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




class GenericDataModule(LightningDataModule):
    """Data module for all the datasets."""

    def __init__(self, dataset_config: Dict):
        super().__init__()
        self.tokenizer = None
        self.train_dataset = None
        self.test_dataset = None

        self.dataset_config = dataset_config
        self.seed = self.dataset_config["seed"]
        self.dataset_type = self.dataset_config["dataset_type"]
        self.tokenizer_name = self.dataset_config["tokenizer_name"]
        self.dataset_name = self.dataset_config["dataset_name"]
        self.dataset_subset = self.dataset_config["dataset_subset"]
        self.train_split_name = self.dataset_config["train_split"]
        self.test_split_name = self.dataset_config["test_split"]
        self.train_batch_size = self.dataset_config["train_batch_size"]
        self.eval_batch_size = self.dataset_config["eval_batch_size"]
        self.num_workers = self.dataset_config["num_workers"]
        self.max_length = self.dataset_config["max_length"]
        self.split_ratio = None
        self.extra_params = self.dataset_config.get("extra_params", {})
        self.collate_fn_name = self.dataset_config.get("collate_fn", "dynamic_padding")
        self.collate_fn = registry.get_data_collator(self.collate_fn_name)
        self.prepare_data()

    def prepare_data(self):
        """Download and prepare data."""
        # Load dataset
        db = load_dataset_builder(self.dataset_name, name=self.dataset_subset)
        available_splits = list(db.info.splits.keys())

        load_dataset(
            self.dataset_name, name=self.dataset_subset, split=self.train_split_name
        )

        if len(available_splits) == 1:
            self.split_ratio = self.dataset_config["split_ratio"]
        else:
            load_dataset(
                self.dataset_name, name=self.dataset_subset, split=self.test_split_name
            )

    def setup(self, stage=None):
        """Setup datasets for trainin and testing."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        insert_pad_token(self.tokenizer)

        # Load datasets
        train_data = load_dataset(
            self.dataset_name, name=self.dataset_subset, split=self.train_split_name
        )
        if self.split_ratio:
            splitted_dataset = train_data.train_test_split(
                test_size=(1 - self.split_ratio), seed=self.seed
            )
            train_data = splitted_dataset["train"]
            test_data = splitted_dataset["test"]
        else:
            test_data = load_dataset(
                self.dataset_name,
                name=self.dataset_subset,
                split=self.test_split_name,
            )

        # self.train_dataset = ComponentFactory.create_dataset(
        #     self.dataset_type,
        #     dataset=train_data,
        #     tokenizer=self.tokenizer,
        #     max_length=self.max_length,
        #     split="train",
        #     extra_params=self.extra_params,
        # )
        # from torch.utils.data import Subset
        # self.train_dataset = Subset(self.train_dataset, indices=range(0, 320))

        # self.test_dataset = ComponentFactory.create_dataset(
        #     self.dataset_type,
        #     dataset=test_data,
        #     tokenizer=self.tokenizer,
        #     max_length=self.max_length,
        #     split="test",
        #     extra_params=self.extra_params,
        # )
        # self.test_dataset = Subset(self.test_dataset, indices=range(0, 320))

        prompt_completion_example = {"prompt": "The sky is", "completion": " blue."}

        def preprocess_function(example):
            return {
                "prompt": example['dialogue'],
                "completion": example['summary'],
            }

        dataset = train_data.map(preprocess_function, remove_columns=["dialogue", "summary"])
        import pdb; pdb.set_trace()
        print(next(iter(dataset["train"])))

    # def train_dataloader(self):
    #     """Create training dataloader."""
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.train_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=True,
    #         pin_memory=True,
    #         collate_fn=self.collate_fn,
    #         persistent_workers=True,
    #     )

    # def test_dataloader(self):
    #     """Create test dataloader."""
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.eval_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=True,
    #         collate_fn=self.collate_fn,
    #     )


if __name__ == "__main__":
    import yaml

    from LightningLLM.components.config_manager import ConfigManager

    sample_config = """
    # Sample Dataset configuration

    train_batch_size: 4
    seed: 42

    model:
        model_name: 'bert-base-uncased'

    dataset:
        name: 'cornell-movie-review-data/rotten_tomatoes'
        train_split: 'train'
        test_split: 'test'
        num_workers: 0
        max_length: null
        input_column: 'text'
    """
    config_data = yaml.safe_load(sample_config)

    config_manager = ConfigManager(config_data)
    # Test the dataset loading
    print("Testing dataset loading...")

    # Initialize the data module
    data_module = GenericDataModule(config_manager)

    try:
        # Prepare data
        print("Preparing data...")
        data_module.prepare_data()

        # Setup datasets
        print("Setting up datasets...")
        data_module.setup()

        # Test data loaders
        print("Testing data loaders...")
        train_loader = data_module.train_dataloader()
        test_loader = data_module.test_dataloader()

        # Get a batch from train loader
        print("Getting a batch from train loader...")
        for i, batch in enumerate(train_loader):
            print(f"Batch keys: {batch.keys()}")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Attention mask shape: {batch['attention_mask'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
            break

        # Get a batch from test loader
        print("Getting a batch from test loader...")
        for batch in test_loader:
            print(f"Test batch input IDs shape: {batch['input_ids'].shape}")
            break

        print("Dataset loading test completed successfully!")

    except Exception as e:
        print(f"Error during dataset loading test: {e}")
        import traceback

        traceback.print_exc()
