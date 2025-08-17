"""
# @ Author: Meet Patel
# @ Create Time: 2025-08-17 13:17:51
# @ Modified by: Meet Patel
# @ Modified time: 2025-08-17 13:17:53
# @ Description:
"""

import torch


def insert_pad_token(tokenizer):
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        # Try to use existing special token as pad token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
