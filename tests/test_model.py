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
import torch.nn as nn

from components.component_registry import ComponentFactory

# Setup test data
MODEL_CONFIGS = {
    "hf": {
        "model_type": "hf",
        "model_name": "HuggingFaceTB/SmolLM-135M",
        "auto_class_name": "AutoModelForCausalLM",
        "tokenizer_max_length": 1024,
        "attn_implementation": "sdpa",
        "torch_dtype": "float16",
    },
}


@pytest.mark.parametrize("model_name", MODEL_CONFIGS.keys())
def test_models(model_name):
    """Test that all models that can be created with their configs."""
    # Create models using the factory
    config = MODEL_CONFIGS[model_name]
    model_inst = ComponentFactory.create_model(**config)
    assert model_inst is not None

    model = model_inst.load_model()
    assert isinstance(model, nn.Module)

    tokenizer = model_inst.load_tokenizer()
    assert tokenizer is not None
    assert hasattr(tokenizer, "pad_token_id")
