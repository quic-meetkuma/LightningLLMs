"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-12 23:04:46
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-16 11:46:07
# @ Description:
"""

import torch
from components.component_registry import registry


@registry.data_collator("dynamic_padding")
def dynamic_padding_collate_fn(batch):
    """Custom collate function that pads sequences to the longest sequence in the batch."""
    # Find the maximum sequence length in this batch
    max_length = max(len(item["input_ids"]) for item in batch)

    # Get pad token id from the tokenizer (default to 0 if not available)
    pad_token_id = batch[0].get("pad_token_id", 0)

    # Pad sequences to max_length
    padded_batch = []
    for item in batch:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]

        # Calculate padding length
        padding_length = max_length - len(input_ids)

        # Pad input_ids
        padded_input_ids = torch.cat(
            [
                input_ids,
                torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype),
            ]
        )

        # Pad attention_mask
        padded_attention_mask = torch.cat(
            [attention_mask, torch.zeros(padding_length, dtype=attention_mask.dtype)]
        )

        # Pad labels (using -100 to ignore in loss computation)
        padded_labels = torch.cat(
            [labels, torch.full((padding_length,), -100, dtype=labels.dtype)]
        )

        padded_batch.append(
            {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "labels": padded_labels,
            }
        )

    # Stack the padded sequences
    stacked_batch = {
        "input_ids": torch.stack([item["input_ids"] for item in padded_batch]),
        "attention_mask": torch.stack(
            [item["attention_mask"] for item in padded_batch]
        ),
        "labels": torch.stack([item["labels"] for item in padded_batch]),
    }

    return stacked_batch
