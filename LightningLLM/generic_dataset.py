from datasets import load_dataset, load_dataset_builder
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, Dict, Any, List
import re

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
            splitted_dataset = dataset.train_test_split(
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


class SFTDataLoader:
    """
    A Supervised Fine-Tuning (SFT) data loader manager, providing DataLoader instances
    for train and test splits using the SFTDataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        train_split (str): The name of the training split (e.g., "train").
        test_split (str): The name of the test split (e.g., "test" or "validation").
        prompt_template (str): The template string for prompts.
        completion_template (str): The template string for completions.
        batch_size (int): Batch size for the DataLoader. Defaults to 1.
        num_workers (int): Number of workers for DataLoader. Defaults to 0.
        shuffle_train (bool): Whether to shuffle the training dataset. Defaults to True.
    """
    def __init__(
        self,
        dataset_name: str,
        train_split: str,
        test_split: str,
        prompt_template: str,
        completion_template: str,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle_train: bool = True
    ):
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.test_split = test_split
        self.prompt_template = prompt_template
        self.completion_template = completion_template
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train

        self.train_dataset = None
        self.test_dataset = None

    def _create_dataset(self, split: str) -> SFTDataset:
        """
        Helper method to create an SFTDataset instance for a given split.

        Args:
            split (str): The dataset split to create (e.g., "train", "test").

        Returns:
            SFTDataset: An instance of SFTDataset.
        """
        return SFTDataset(
            dataset_name=self.dataset_name,
            split=split,
            prompt_template=self.prompt_template,
            completion_template=self.completion_template,
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training set.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        if self.train_dataset is None:
            self.train_dataset = self._create_dataset(self.train_split)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers
        )

    def get_test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the test/validation set.

        Returns:
            DataLoader: DataLoader for the test/validation dataset.
        """
        if self.test_dataset is None:
            self.test_dataset = self._create_dataset(self.test_split)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False, # Typically no shuffle for test data
            num_workers=self.num_workers
        )

if __name__ == "__main__":
    # Example usage with the samsum dataset
    print("--- Testing SFTDataset ---")
    sft_dataset = SFTDataset(
        dataset_name="knkarthick/samsum",
        split="train",
        prompt_template="Summarize this dialog:\n{dialogue}\n---\nSummary:\n",
        completion_template="{summary}",
    )
    
    print(f"Number of samples in SFTDataset: {len(sft_dataset)}")
    print("First 3 samples:")
    for i in range(3):
        sample = sft_dataset[i]
        print(f"Sample {i+1} - Prompt: {sample['prompt']}")
        print(f"Sample {i+1} - Completion: {sample['completion']}\n")

    print("\n--- Testing SFTDataLoader ---")
    # Initialize the data loader manager
    sft_data_loader_manager = SFTDataLoader(
        dataset_name="knkarthick/samsum",
        train_split="train",
        test_split="test",
        prompt_template="Summarize this dialog:\n{dialogue}\n---\nSummary:\n",
        completion_template="{summary}",
        batch_size=4,
        num_workers=2
    )

    train_loader = sft_data_loader_manager.get_train_dataloader()
    test_loader = sft_data_loader_manager.get_test_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Iterate through a few examples from the training loader
    print("\nFirst batch from train_loader:")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"Prompts (first example): {batch['prompt'][0]}")
        print(f"Completions (first example): {batch['completion'][0]}")
        if i >= 0: # Print only 1 batch
            break

    # Iterate through a few examples from the test loader
    print("\nFirst batch from test_loader:")
    for i, batch in enumerate(test_loader):
        print(f"Batch {i+1}:")
        print(f"Prompts (first example): {batch['prompt'][0]}")
        print(f"Completions (first example): {batch['completion'][0]}")
        if i >= 0: # Print only 1 batch
            break
