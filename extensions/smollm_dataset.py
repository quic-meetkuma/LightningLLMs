"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-21 23:16:52
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-21 23:16:55
# @ Description:
"""

from LightningLLM.components.component_registry import registry

from torch.utils.data import Dataset
import random
from collections import defaultdict


@registry.dataset("smollm_dataset")
class SmolLMDataset(Dataset):
    """Generic dataset class which can be used for autoregressive training."""

    def __init__(self, dataset, tokenizer, max_length, split, **kwargs):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kwargs = kwargs
        self.split = split
        self.seed = kwargs.get("extra_params", {}).get("seed", 42)
        self.input_column = kwargs.get("extra_params", {}).get("input_column", "text")
        self.sources = kwargs.get("extra_params", {}).get("sources", None)
        if self.sources is None:
            self.sources = list(set(self.dataset["source"]))
        if self.split == "train":
            self.samples_per_source = kwargs.get("extra_params", {}).get(
                "train_samples_per_source", len(self.dataset)
            )
        else:
            self.samples_per_source = kwargs.get("extra_params", {}).get(
                "test_samples_per_source", len(self.dataset)
            )
        self.sample_dataset()

    def sample_dataset(self):
        # Group indices by source
        source_indices = defaultdict(list)

        for i, source in enumerate(self.dataset["source"]):
            source_indices[source].append(i)

        # Sample from each source
        sampled_indices = []

        for source, indices in source_indices.items():
            sample_size = min(self.samples_per_source, len(indices))
            sampled_source_indices = random.sample(indices, sample_size)
            sampled_indices.extend(sampled_source_indices)

        # Create the final dataset
        self.dataset = self.dataset.select(sampled_indices)

        # Shuffle the result
        self.dataset = self.dataset.shuffle(seed=self.seed)

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
